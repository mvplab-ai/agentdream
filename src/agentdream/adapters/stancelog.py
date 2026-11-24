"""Optional adapter for stancelog (https://github.com/systemagents-ai/stancelog).

stancelog is imported lazily; when it is not installed the adapter is simply
not registered. Because stancelog's API may evolve, this adapter is duck-typed:
it looks for a store/log object exposing one of the usual read methods and a
way to append records. Paths may be given as ``stancelog:<path>``.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any

from ..models import MemoryEntry
from .base import AdapterError, MemoryAdapter


def available() -> bool:
    try:
        importlib.import_module("stancelog")
        return True
    except Exception:
        return False


def _open(mod: Any, path: Path) -> Any:
    for name in ("open", "load", "Store", "StanceLog", "Log", "Memory"):
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                return fn(str(path))
            except TypeError:
                continue
    raise AdapterError("stancelog: could not find an open()/Store() entry point")


def _records(store: Any) -> list[Any]:
    for name in ("all", "entries", "records", "list", "items", "stances"):
        attr = getattr(store, name, None)
        if attr is None:
            continue
        val = attr() if callable(attr) else attr
        return list(val.values() if isinstance(val, dict) else val)
    try:
        return list(iter(store))
    except TypeError as exc:
        raise AdapterError("stancelog: store is not iterable") from exc


def _get(rec: Any, *names: str) -> Any:
    for n in names:
        if isinstance(rec, dict) and n in rec:
            return rec[n]
        if hasattr(rec, n):
            return getattr(rec, n)
    return None


class StancelogAdapter(MemoryAdapter):
    name = "stancelog"

    def __init__(self, path, **options):
        p = str(path)
        if p.startswith("stancelog:"):
            p = p[len("stancelog:"):]
        super().__init__(Path(p), **options)
        self.single_file = Path(p).is_file()

    @classmethod
    def detect(cls, path: Path) -> bool:
        return str(path).startswith("stancelog:")

    def load(self) -> list[MemoryEntry]:
        mod = importlib.import_module("stancelog")
        store = _open(mod, self.path)
        out = []
        for i, rec in enumerate(_records(store)):
            content = _get(rec, "content", "text", "stance", "claim", "value", "body")
            if content is None:
                continue
            meta = {"_order": i}
            for k in ("subject", "topic", "confidence", "tags", "status"):
                v = _get(rec, k)
                if v is not None:
                    meta[k] = v
            out.append(MemoryEntry(
                id=str(_get(rec, "id", "uid", "key") or f"stance-{i + 1}"), content=str(content),
                path=self.path.name, key=_get(rec, "subject", "topic", "key"), metadata=meta,
                created_at=_str(_get(rec, "created_at", "created", "timestamp", "ts")),
                updated_at=_str(_get(rec, "updated_at", "updated")), source="stancelog",
            ))
        return out

    def write(self, entries: list[MemoryEntry], dest) -> Path:
        dest = Path(dest)
        self._check_dest(dest)
        mod = importlib.import_module("stancelog")
        if dest.exists():
            raise AdapterError(f"destination exists: {dest}")
        if self.path.is_dir():
            dest.mkdir(parents=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
        store = _open(mod, dest)
        adder = next((getattr(store, n) for n in ("add", "append", "record", "write", "put") if hasattr(store, n)), None)
        if adder is None:
            shutil.rmtree(dest, ignore_errors=True)
            raise AdapterError("stancelog: store has no add()/append() method")
        for e in entries:
            try:
                adder(e.content, id=e.id, **{k: v for k, v in e.metadata.items() if not k.startswith("_")})
            except TypeError:
                adder({"id": e.id, "content": e.content, **{k: v for k, v in e.metadata.items()
                                                             if not k.startswith("_")}})
        for closer in ("save", "flush", "close"):
            fn = getattr(store, closer, None)
            if callable(fn):
                fn()
        return dest


def _str(v: Any) -> str | None:
    return None if v is None else str(v)
