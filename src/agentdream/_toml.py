"""TOML loading without a hard dependency.

Python 3.11+ ships :mod:`tomllib`. On 3.10 we use ``tomli`` when it is installed and
otherwise fall back to :func:`mini_loads`, a small parser for the subset of TOML that
agentdream config files need: ``[table]`` / ``[a.b]`` headers, ``key = value`` with
bare/quoted (dotted) keys, basic and literal strings (single-line, plus basic ``\"\"\"``
multi-line), integers, floats, booleans, arrays (possibly multi-line, nested) and inline
tables, and ``#`` comments. Anything else raises :class:`TOMLError` with a line number.
"""

from __future__ import annotations

import re
from typing import Any


class TOMLError(ValueError):
    pass


def loads(text: str) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found,unused-ignore]
    except ModuleNotFoundError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef,unused-ignore]
        except ModuleNotFoundError:
            return mini_loads(text)
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise TOMLError(str(exc)) from exc


# ------------------------------------------------------------------ fallback parser

_BARE = re.compile(r"[A-Za-z0-9_\-]+")
_ESC = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", '"': '"', "\\": "\\"}


class _Parser:
    def __init__(self, text: str):
        self.s = text.replace("\r\n", "\n")
        self.i = 0
        self.line = 1

    # -- low level
    def err(self, msg: str) -> TOMLError:
        return TOMLError(f"line {self.line}: {msg}")

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def advance(self, n: int = 1) -> None:
        for _ in range(n):
            if self.i < len(self.s) and self.s[self.i] == "\n":
                self.line += 1
            self.i += 1

    def skip_ws(self, newlines: bool = False) -> None:
        while self.i < len(self.s):
            ch = self.s[self.i]
            if ch in " \t" or (newlines and ch == "\n"):
                self.advance()
            elif ch == "#":
                while self.i < len(self.s) and self.s[self.i] != "\n":
                    self.advance()
            else:
                break

    def expect_eol(self) -> None:
        self.skip_ws()
        if self.peek() not in ("", "\n"):
            raise self.err(f"unexpected {self.peek()!r}")

    # -- grammar
    def parse(self) -> dict[str, Any]:
        root: dict[str, Any] = {}
        cur = root
        while True:
            self.skip_ws(newlines=True)
            if self.i >= len(self.s):
                return root
            if self.peek() == "[":
                if self.s.startswith("[[", self.i):
                    raise self.err("arrays of tables are not supported by the fallback parser (pip install tomli)")
                self.advance()
                self.skip_ws()
                keys = self.key()
                self.skip_ws()
                if self.peek() != "]":
                    raise self.err("expected ']'")
                self.advance()
                self.expect_eol()
                cur = root
                for k in keys:
                    nxt = cur.setdefault(k, {})
                    if not isinstance(nxt, dict):
                        raise self.err(f"{k!r} is not a table")
                    cur = nxt
                continue
            keys = self.key()
            self.skip_ws()
            if self.peek() != "=":
                raise self.err("expected '='")
            self.advance()
            self.skip_ws()
            val = self.value()
            tgt = cur
            for k in keys[:-1]:
                tgt = tgt.setdefault(k, {})
                if not isinstance(tgt, dict):
                    raise self.err(f"{k!r} is not a table")
            if keys[-1] in tgt:
                raise self.err(f"duplicate key {keys[-1]!r}")
            tgt[keys[-1]] = val
            self.expect_eol()

    def key(self) -> list[str]:
        parts = []
        while True:
            self.skip_ws()
            ch = self.peek()
            if ch == '"':
                parts.append(self.basic_string())
            elif ch == "'":
                parts.append(self.literal_string())
            else:
                m = _BARE.match(self.s, self.i)
                if not m:
                    raise self.err("expected a key")
                parts.append(m.group(0))
                self.advance(len(m.group(0)))
            self.skip_ws()
            if self.peek() == ".":
                self.advance()
                continue
            return parts

    def value(self) -> Any:
        ch = self.peek()
        if self.s.startswith('"""', self.i):
            return self.multiline_basic()
        if ch == '"':
            return self.basic_string()
        if ch == "'":
            return self.literal_string()
        if ch == "[":
            return self.array()
        if ch == "{":
            return self.inline_table()
        m = re.compile(r"true|false|[+\-]?(?:inf|nan)|[0-9A-Za-z_+\-.:]+").match(self.s, self.i)
        if not m:
            raise self.err("expected a value")
        tok = m.group(0)
        self.advance(len(tok))
        if tok == "true":
            return True
        if tok == "false":
            return False
        t = tok.replace("_", "")
        try:
            if re.fullmatch(r"[+\-]?\d+", t):
                return int(t)
            if re.fullmatch(r"0x[0-9a-fA-F]+", t):
                return int(t, 16)
            return float(t)
        except ValueError:
            raise self.err(f"unsupported value {tok!r} (dates are not supported by the fallback parser)") from None

    def basic_string(self) -> str:
        self.advance()  # opening quote
        out: list[str] = []
        while True:
            ch = self.peek()
            if ch in ("", "\n"):
                raise self.err("unterminated string")
            self.advance()
            if ch == '"':
                return "".join(out)
            if ch == "\\":
                out.append(self.escape())
            else:
                out.append(ch)

    def escape(self) -> str:
        e = self.peek()
        self.advance()
        if e in _ESC:
            return _ESC[e]
        if e in ("u", "U"):
            n = 4 if e == "u" else 8
            code = self.s[self.i:self.i + n]
            self.advance(n)
            try:
                return chr(int(code, 16))
            except ValueError:
                raise self.err("bad unicode escape") from None
        raise self.err(f"bad escape \\{e}")

    def multiline_basic(self) -> str:
        self.advance(3)
        if self.peek() == "\n":
            self.advance()
        out: list[str] = []
        while True:
            if self.i >= len(self.s):
                raise self.err("unterminated multi-line string")
            if self.s.startswith('"""', self.i):
                self.advance(3)
                return "".join(out)
            ch = self.peek()
            self.advance()
            if ch == "\\":
                if self.peek() == "\n":  # line-ending backslash: trim newline + leading whitespace
                    self.skip_ws(newlines=True)
                    continue
                out.append(self.escape())
            else:
                out.append(ch)

    def literal_string(self) -> str:
        if self.s.startswith("'''", self.i):
            end = self.s.find("'''", self.i + 3)
            if end < 0:
                raise self.err("unterminated multi-line string")
            body = self.s[self.i + 3:end]
            self.advance(end + 3 - self.i)
            return body[1:] if body.startswith("\n") else body
        end = self.s.find("'", self.i + 1)
        nl = self.s.find("\n", self.i + 1)
        if end < 0 or (0 <= nl < end):
            raise self.err("unterminated string")
        body = self.s[self.i + 1:end]
        self.advance(end + 1 - self.i)
        return body

    def array(self) -> list[Any]:
        self.advance()
        out: list[Any] = []
        while True:
            self.skip_ws(newlines=True)
            if self.peek() == "]":
                self.advance()
                return out
            out.append(self.value())
            self.skip_ws(newlines=True)
            if self.peek() == ",":
                self.advance()
            elif self.peek() != "]":
                raise self.err("expected ',' or ']' in array")

    def inline_table(self) -> dict[str, Any]:
        self.advance()
        out: dict[str, Any] = {}
        self.skip_ws()
        if self.peek() == "}":
            self.advance()
            return out
        while True:
            keys = self.key()
            self.skip_ws()
            if self.peek() != "=":
                raise self.err("expected '=' in inline table")
            self.advance()
            self.skip_ws()
            tgt = out
            for k in keys[:-1]:
                tgt = tgt.setdefault(k, {})
            tgt[keys[-1]] = self.value()
            self.skip_ws()
            if self.peek() == ",":
                self.advance()
                continue
            if self.peek() == "}":
                self.advance()
                return out
            raise self.err("expected ',' or '}' in inline table")


def mini_loads(text: str) -> dict[str, Any]:
    """Parse the TOML subset described in the module docstring."""
    return _Parser(text).parse()


__all__ = ["TOMLError", "loads", "mini_loads"]
