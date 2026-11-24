"""Helpers for the edge-case suite (``tests/edge``).

Deliberately *not* a ``conftest.py``: a second module named ``conftest`` would shadow
``tests/conftest.py`` for the ``from conftest import ...`` lines in the main suite. The
``edge`` marker is applied to everything under ``tests/edge`` by ``tests/conftest.py``.

Conventions (``scripts/edge_case_table.py`` relies on them): one file per area,
``test_<area>.py``; every test has a docstring whose first line is the case description.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentdream import LLMBackend

POKEMON_MEMORY = (
    "- The PokéCenter in Viridian City is north of the gym.\n"
    "- Use pytest for tests.\n"
    "- Use pytest for tests!\n"
    "- The default branch is master. (2026-01-01)\n"
    "- The default branch is main. (2026-02-01)\n"
    "- Untouched note about llamas.\n"
)

POKEMON_SESSIONS = {
    "1.jsonl": [{"role": "assistant", "content": "Walked north of the gym. No PokéCenter here."},
                {"role": "assistant", "content": "Still no PokéCenter north of the gym."}],
    "2.jsonl": [{"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."}],
}


def write(path: Path, text: str, encoding: str = "utf-8", newline: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline=newline) as f:
        f.write(text)
    return path


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_jsonl(path: Path, rows: list[Any], eol: str = "\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(r, ensure_ascii=False) + eol for r in rows).encode("utf-8"))
    return path


def tree_digest(*paths: Path) -> str:
    """One sha256 over names + bytes of every file (and symlink target) under ``paths``."""
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        files = [p] if not p.is_dir() else sorted(x for x in p.rglob("*") if x.is_file() or x.is_symlink())
        for f in files:
            h.update(str(f).encode())
            h.update(b"\0")
            if f.is_symlink():
                h.update(b"link:" + os.readlink(f).encode())
            elif f.exists():
                h.update(f.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def workspace(tmp_path: Path, memory: str = POKEMON_MEMORY,
              sessions: dict[str, list[dict[str, Any]]] | None = None) -> tuple[Path, Path]:
    """A markdown memory folder + a sessions folder that produce several operations."""
    mem = write(tmp_path / "mem" / "MEMORY.md", memory).parent
    ses = tmp_path / "sessions"
    for name, rows in (sessions or POKEMON_SESSIONS).items():
        write_jsonl(ses / name, rows)
    return mem, ses


def phase_of(prompt: str) -> str:
    if "Task: CONSOLIDATE" in prompt:
        return "consolidate"
    if "Task: VERIFY" in prompt:
        return "verify"
    return "insights"


Responder = Callable[[str, str], Any]


class FakeLLM(LLMBackend):
    """Scriptable backend. ``respond(phase, prompt)`` returns a string, a dict/list (sent as
    JSON) or raises. Every (system, prompt) pair is recorded in ``calls``."""

    name = "fake"

    def __init__(self, respond: Responder | dict[str, Any] | str | None = None, **kw: Any):
        super().__init__(model="fake-1", **kw)
        self.respond = respond
        self.calls: list[tuple[str, str]] = []

    @property
    def sent(self) -> str:
        """Everything that left the machine, concatenated."""
        return "\n".join(s + "\n" + p for s, p in self.calls)

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        self.calls.append((system, prompt))
        phase = phase_of(prompt)
        r: Any = self.respond
        if callable(r):
            r = r(phase, prompt)
        elif isinstance(r, dict) and set(r) <= {"consolidate", "verify", "insights"}:
            r = r.get(phase, {"operations": []})
        if isinstance(r, BaseException):
            raise r
        if r is None:
            r = {"operations": []}
        return r if isinstance(r, str) else json.dumps(r)


class FakeHTTPResponse:
    """Minimal stand-in for what ``urllib.request.urlopen`` returns."""

    def __init__(self, payload: Any, status: int = 200):
        self._data = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *a: Any) -> None:
        return None


def http_error(url: str, code: int, body: str = "", headers: dict[str, str] | None = None) -> Exception:
    import email.message
    import io
    import urllib.error

    msg = email.message.Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError(url, code, f"HTTP {code}", msg, io.BytesIO(body.encode()))
