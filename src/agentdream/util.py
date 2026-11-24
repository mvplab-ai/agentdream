"""Small stdlib helpers: atomic writes, hashing, durations, time."""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def hash_path(path: str | os.PathLike) -> str:
    """Content hash of a file or a whole directory tree (names + bytes)."""
    p = Path(path)
    h = hashlib.sha256()
    if p.is_file():
        h.update(p.read_bytes())
        return h.hexdigest()
    if p.is_dir():
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(str(f.relative_to(p)).encode())
            h.update(b"\x00")
            h.update(f.read_bytes())
            h.update(b"\x00")
        return h.hexdigest()
    return ""


def copy_path(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    s, d = Path(src), Path(dst)
    if s.is_dir():
        shutil.copytree(s, d)
    else:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)


def remove_path(p: str | os.PathLike) -> None:
    pp = Path(p)
    if pp.is_dir() and not pp.is_symlink():
        shutil.rmtree(pp)
    elif pp.exists() or pp.is_symlink():
        pp.unlink()


_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*"
                  r"(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?|w|weeks?)?\s*$", re.I)
_DUR_MULT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str | int | float) -> float:
    """Parse '30m', '6h', '1d', '90s', '45' (minutes default for bare numbers? no: seconds)."""
    if isinstance(text, (int, float)):
        return float(text)
    total = 0.0
    parts = re.findall(r"\d+(?:\.\d+)?\s*[a-zA-Z]*", text)
    if not parts or "".join(p.replace(" ", "") for p in parts) != text.replace(" ", ""):
        raise ValueError(f"invalid duration: {text!r}")
    for part in parts:
        m = _DUR.match(part)
        if not m:
            raise ValueError(f"invalid duration: {text!r}")
        unit = (m.group(2) or "s").lower()[0]
        total += float(m.group(1)) * _DUR_MULT[unit]
    return total


def parse_timestamp(value: Any) -> float | None:
    """Best-effort conversion of ISO strings / epoch numbers to epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1000.0 if v > 1e11 else v
    s = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return parse_timestamp(float(s))
    s = s.replace("Z", "+00:00")
    try:
        dt = _dt.datetime.fromisoformat(s)
    except ValueError:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
        if not m:
            return None
        dt = _dt.datetime.fromisoformat(m.group(1))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.timestamp()


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4
