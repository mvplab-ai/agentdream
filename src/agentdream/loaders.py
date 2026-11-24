"""Transcript loaders with format auto-detection.

Formats:

* ``claude-code``: Claude Code session JSONL (``~/.claude/projects/<proj>/*.jsonl``);
  lines with ``type`` user/assistant and ``message.content`` str or block list.
  Tool noise is skipped: thinking blocks and successful tool results are dropped,
  tool calls are reduced to a one-line ``[tool] Name: command`` message, failed
  tool results are kept (trimmed) because they reveal recurring mistakes.
* ``jsonl``: generic chat log, one ``{"role", "content"}`` object per line.
* ``openai``: OpenAI-style messages JSON (a list or ``{"messages": [...]}``).
* ``text``: ``User: ...`` / ``Assistant: ...`` turns, or raw text (one message per
  non-empty line, role ``log``).

Each :class:`Message` keeps ``source_path`` + ``line_no`` (1-based) or ``index`` so
evidence can be cited as ``file:line``. Lines are counted on CRLF / LF / lone CR only (see
:func:`agentdream.util.split_lines`), so Windows files and JSON strings containing U+2028
cite the line an editor shows. Files are decoded as UTF-8 (a BOM is ignored; invalid bytes
become U+FFFD).

Errors: every problem with a transcript file surfaces as :class:`LoaderError` (missing or
unreadable files, a forced format the content can't satisfy, unknown formats). Malformed
lines inside a JSONL transcript are skipped and reported in ``metadata["skipped_lines"]``
and as a notice; they never abort the load.
"""

from __future__ import annotations

import glob as _glob
import json
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .models import Message, Transcript
from .util import parse_timestamp, split_lines

TRANSCRIPT_EXTS = {".jsonl", ".json", ".txt", ".log", ".md", ".chat", ".transcript"}
_TURN = re.compile(r"^\s*(user|human|assistant|ai|agent|system|tool|bot|model|claude)\s*[:>]\s?(.*)$", re.I)
_ROLE_MAP = {"human": "user", "ai": "assistant", "agent": "assistant", "bot": "assistant", "model": "assistant",
             "claude": "assistant"}
_NOISE = re.compile(r"^\s*<(command-name|command-message|command-args|local-command-stdout|system-reminder)>")


class LoaderError(Exception):
    pass


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") in (None, "text", "input_text", "output_text"):
                parts.append(str(b.get("text", "")))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)


# ---------------------------------------------------------------- format parsers

def _ts(value: Any) -> str | None:
    return None if value is None or isinstance(value, (dict, list)) else str(value)


def parse_claude_code(path: str, text: str) -> list[Message]:
    msgs: list[Message] = []
    for ln, line in enumerate(split_lines(text), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict) or obj.get("type") not in ("user", "assistant"):
            continue
        if obj.get("isMeta"):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or obj["type"])
        ts = _ts(obj.get("timestamp"))
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip() and not _NOISE.match(content):
                msgs.append(Message(role, content.strip(), ts, ln, path))
            continue
        if not isinstance(content, list):
            continue
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "text":
                t = str(block.get("text", "")).strip()
                if t and not _NOISE.match(t):
                    texts.append(t)
            elif bt == "tool_use":
                inp = block.get("input") or {}
                if not isinstance(inp, dict):
                    inp = {"input": inp}
                detail = inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp.get("url") or ""
                if not detail and inp:
                    detail = json.dumps(inp, ensure_ascii=False, default=str)[:160]
                msgs.append(Message("tool", f"[tool] {block.get('name', 'tool')}: {str(detail)[:300]}", ts, ln, path))
            elif bt == "tool_result" and block.get("is_error"):
                t = _text_of(block.get("content")).strip()
                if t:
                    msgs.append(Message("tool", f"[tool error] {t[:400]}", ts, ln, path))
        if texts:
            msgs.append(Message(role, "\n".join(texts), ts, ln, path))
    return msgs


def parse_jsonl(path: str, text: str) -> list[Message]:
    msgs: list[Message] = []
    for ln, line in enumerate(split_lines(text), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        role = obj.get("role") or obj.get("speaker") or obj.get("author") or "unknown"
        content = _text_of(obj.get("content", obj.get("text", obj.get("message"))))
        if not content.strip():
            continue
        ts = obj.get("timestamp") or obj.get("time") or obj.get("ts") or obj.get("created_at")
        msgs.append(Message(_ROLE_MAP.get(str(role).lower(), str(role).lower()), content.strip(),
                            _ts(ts), ln, path))
    return msgs


def _line_index(text: str) -> Callable[[int], int]:
    starts = [0]
    for m in re.finditer(r"\r\n|\r|\n", text):
        starts.append(m.end())

    def line_of(offset: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    return line_of


def parse_openai(path: str, text: str) -> list[Message]:
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise LoaderError(f"{path}: not valid JSON ({str(exc)[:120]})") from exc
    items = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise LoaderError(f"{path}: no messages list")
    line_of = _line_index(text)
    # Approximate line numbers by locating each "role" key in order.
    role_offsets = [m.start() for m in re.finditer(r'"role"\s*:', text)]
    msgs: list[Message] = []
    for i, obj in enumerate(items):
        if not isinstance(obj, dict):
            continue
        content = _text_of(obj.get("content"))
        if obj.get("tool_calls") and not content:
            calls = obj["tool_calls"] if isinstance(obj["tool_calls"], list) else []
            funcs = [c.get("function") for c in calls if isinstance(c, dict)]
            names = ", ".join(str(f.get("name", "tool")) if isinstance(f, dict) else "tool" for f in funcs)
            content = f"[tool] {names or 'tool'}"
        if not content.strip():
            continue
        ln = line_of(role_offsets[i]) if i < len(role_offsets) else None
        role = str(obj.get("role", "unknown"))
        if role == "tool" and len(content) > 400:
            content = content[:400] + "..."
        m = Message(role, content.strip(), _ts(obj.get("timestamp")), ln, path, index=i)
        msgs.append(m)
    return msgs


def parse_text(path: str, text: str) -> list[Message]:
    lines = split_lines(text)
    turns = [i for i, ln in enumerate(lines) if _TURN.match(ln)]
    msgs: list[Message] = []
    if turns:
        cur_role, cur_lines, cur_ln = None, [], None
        for i, line in enumerate(lines):
            m = _TURN.match(line)
            if m:
                if cur_role is not None and "\n".join(cur_lines).strip():
                    msgs.append(Message(cur_role, "\n".join(cur_lines).strip(), None, cur_ln, path))
                role = m.group(1).lower()
                cur_role, cur_lines, cur_ln = _ROLE_MAP.get(role, role), [m.group(2)], i + 1
            elif cur_role is not None:
                cur_lines.append(line)
        if cur_role is not None and "\n".join(cur_lines).strip():
            msgs.append(Message(cur_role, "\n".join(cur_lines).strip(), None, cur_ln, path))
        return msgs
    ts_re = re.compile(r"^\s*\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)\]?\s*")
    for i, line in enumerate(lines):
        if line.strip():
            ts = None
            m = ts_re.match(line)
            if m:
                ts = m.group(1)
            msgs.append(Message("log", line.strip(), ts, i + 1, path))
    return msgs


PARSERS: dict[str, Callable[[str, str], list[Message]]] = {
    "claude-code": parse_claude_code,
    "jsonl": parse_jsonl,
    "openai": parse_openai,
    "text": parse_text,
}


def register_loader(name: str, parser: Callable[[str, str], list[Message]]) -> None:
    PARSERS[name] = parser


def detect_format(text: str, path: str = "") -> str:
    text = text.lstrip("\ufeff")
    stripped = text.lstrip()
    if not stripped:
        return "text"
    if stripped[0] in "[{":
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            data = None
        if data is not None:
            if isinstance(data, list) or (isinstance(data, dict) and isinstance(data.get("messages"), list)):
                return "openai"
            if isinstance(data, dict) and ("role" in data or "type" in data):
                # a single-line JSONL file
                return "claude-code" if data.get("type") in ("user", "assistant", "summary", "system") \
                    and "message" in data else "jsonl"
    # JSONL? Sample the first lines even if the very first one isn't JSON (a truncated write, a banner):
    # it's JSONL when most sampled lines are JSON objects.
    objs = []
    sampled = 0
    for line in split_lines(text)[:50]:
        if not line.strip():
            continue
        sampled += 1
        try:
            o = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(o, dict):
            objs.append(o)
    if objs and len(objs) * 2 >= sampled:
        if any(o.get("type") in ("user", "assistant") and isinstance(o.get("message"), dict) for o in objs):
            return "claude-code"
        if any("role" in o or "speaker" in o for o in objs):
            return "jsonl"
    return "text"


def _malformed_lines(text: str) -> list[int]:
    """1-based numbers of non-blank JSONL lines that aren't JSON objects."""
    bad = []
    for ln, line in enumerate(split_lines(text), start=1):
        if not line.strip():
            continue
        try:
            ok = isinstance(json.loads(line), dict)
        except (ValueError, RecursionError):
            ok = False
        if not ok:
            bad.append(ln)
    return bad


def load_transcript(path: str | os.PathLike, format: str | None = None) -> Transcript:
    p = Path(path)
    try:
        raw = p.read_bytes()
        mtime = p.stat().st_mtime
    except OSError as exc:
        raise LoaderError(f"cannot read transcript {p}: {exc.strerror or exc}") from exc
    text = raw.decode("utf-8-sig", errors="replace")
    fmt = format or detect_format(text, str(p))
    if fmt not in PARSERS:
        raise LoaderError(f"unknown transcript format {fmt!r}")
    try:
        msgs = PARSERS[fmt](str(p), text)
    except LoaderError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, RecursionError) as exc:
        raise LoaderError(f"{p}: could not parse as {fmt} ({type(exc).__name__}: {str(exc)[:120]})") from exc
    if fmt != "openai":  # openai parser already stores the original message index
        for i, m in enumerate(msgs):
            m.index = i
    meta: dict[str, Any] = {"mtime": mtime, "bytes": len(raw)}
    if fmt in ("jsonl", "claude-code"):
        skipped = _malformed_lines(text)
        if skipped:
            meta["skipped_lines"] = skipped
    return Transcript(id=p.stem, messages=msgs, source_path=str(p), format=fmt, metadata=meta)


def expand_sources(sources: Iterable[str | os.PathLike] | str | os.PathLike) -> list[Path]:
    """Files, directories (recursive) and glob patterns -> sorted unique transcript files."""
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    out: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)

    for s in sources:
        s = os.path.expanduser(str(s))
        if any(ch in s for ch in "*?["):
            for g in sorted(_glob.glob(s, recursive=True)):
                gp = Path(g)
                if gp.is_file():
                    add(gp)
                elif gp.is_dir():
                    for f in _dir_files(gp):
                        add(f)
            continue
        p = Path(s)
        if p.is_dir():
            for f in _dir_files(p):
                add(f)
        elif p.is_file():
            add(p)
        else:
            raise LoaderError(f"transcript source not found: {s}")
    return out


def _dir_files(d: Path) -> list[Path]:
    return sorted(f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in TRANSCRIPT_EXTS
                  and not any(part.startswith(".") for part in f.relative_to(d).parts))


def _natural(s: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


def first_timestamp(t: Transcript) -> float | None:
    """Epoch seconds of the first message with a *parsable* timestamp (junk timestamps are skipped)."""
    for m in t.messages:
        ts = parse_timestamp(m.timestamp)
        if ts is not None:
            return ts
    return None


def _sort_key(t: Transcript) -> tuple:
    ts = first_timestamp(t)
    return (0 if ts is not None else 1, ts or 0.0, _natural(t.source_path or ""))


def load_transcripts(sources: Iterable[str | os.PathLike[str]] | str | os.PathLike[str] | None,
                     format: str | None = None, max_sessions: int | None = None,
                     min_sessions: int = 0) -> tuple[list[Transcript], list[str]]:
    """Load and chronologically order transcripts. Returns (transcripts, notices).

    Ordering: first message timestamp when every session has one, else file mtime,
    then path. When more than ``max_sessions`` are found the most recent are kept.
    """
    files = expand_sources(sources) if sources else []
    notices: list[str] = []
    ts: list[Transcript] = []
    empty = 0
    for f in files:
        t = load_transcript(f, format)
        bad = t.metadata.get("skipped_lines")
        if bad:
            shown = ", ".join(map(str, bad[:10])) + (", ..." if len(bad) > 10 else "")
            notices.append(f"skipped {len(bad)} malformed line(s) in {f} (line {shown})")
        if not t.messages:
            notices.append(f"skipped empty transcript {f}")
            empty += 1
            continue
        ts.append(t)
    all_have_ts = all(first_timestamp(t) is not None for t in ts)
    if all_have_ts:
        ts.sort(key=_sort_key)
    else:
        # mtimes are unreliable (git checkouts, copies); natural path order is predictable
        ts.sort(key=lambda t: _natural(t.source_path or ""))
    if max_sessions is not None and len(ts) > max_sessions:
        notices.append(f"{len(ts)} sessions found; keeping the most recent {max_sessions} (max_sessions)")
        ts = ts[-max_sessions:]
    if len(ts) < min_sessions:
        raise LoaderError(f"need at least {min_sessions} session transcript(s), found {len(ts)}"
                          + (f" ({empty} empty transcript(s) skipped)" if empty else ""))
    seen_ids: dict[str, int] = {}
    for t in ts:
        if t.id in seen_ids:
            seen_ids[t.id] += 1
            t.id = f"{t.id}-{seen_ids[t.id]}"
        else:
            seen_ids[t.id] = 1
    return ts, notices
