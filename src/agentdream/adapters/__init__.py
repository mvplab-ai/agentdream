"""Memory store adapters + registry.

Third parties can register adapters either at runtime::

    from agentdream.adapters import register_adapter
    register_adapter("mystore", MyAdapter)

or via the ``agentdream.adapters`` entry-point group in their package metadata.
"""

from __future__ import annotations

from pathlib import Path

from .base import AdapterError, MemoryAdapter
from .jsonstore import JSONAdapter
from .markdown import MarkdownAdapter
from .sqlite import SQLiteAdapter, create_sqlite_store

_REGISTRY: dict[str, type[MemoryAdapter]] = {}
_ENTRY_POINTS_LOADED = False


def register_adapter(name: str, cls: type[MemoryAdapter]) -> None:
    _REGISTRY[name] = cls


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="agentdream.adapters"):
            try:
                register_adapter(ep.name, ep.load())
            except Exception:  # pragma: no cover - third party failure
                continue
    except Exception:  # pragma: no cover
        pass


def _register_builtins() -> None:
    register_adapter("stancelog", _LazyStancelog)  # detection is prefix-only, so safe without the package
    register_adapter("sqlite", SQLiteAdapter)
    register_adapter("json", JSONAdapter)
    register_adapter("markdown", MarkdownAdapter)


class _LazyStancelog(MemoryAdapter):
    """Proxy that imports the stancelog adapter only when actually used."""

    name = "stancelog"

    def __new__(cls, path, **options):
        from .stancelog import StancelogAdapter, available

        if not available():
            raise AdapterError("the stancelog adapter needs the 'stancelog' package (pip install stancelog)")
        return StancelogAdapter(path, **options)

    @classmethod
    def detect(cls, path: Path) -> bool:
        return str(path).startswith("stancelog:")

    def load(self):  # pragma: no cover - never instantiated
        raise NotImplementedError

    def write(self, entries, dest):  # pragma: no cover
        raise NotImplementedError


_register_builtins()


def available_adapters() -> list[str]:
    _load_entry_points()
    return list(_REGISTRY)


def get_adapter(path: str | Path, format: str | None = None, **options) -> MemoryAdapter:
    """Return an adapter instance for ``path``, auto-detecting the format when not given."""
    _load_entry_points()
    if format:
        if format in ("md", "text", "txt"):
            format = "markdown"
        if format in ("jsonl",):
            format = "json"
        cls = _REGISTRY.get(format)
        if cls is None:
            raise AdapterError(f"unknown memory format {format!r}; available: {', '.join(_REGISTRY)}")
        return cls(path, **options)
    p = Path(str(path))
    if str(path).startswith("stancelog:"):
        return _REGISTRY["stancelog"](path, **options)
    if not p.exists():
        raise AdapterError(f"memory store not found: {path}")
    for name, cls in _REGISTRY.items():
        try:
            if cls.detect(p):
                return cls(p, **options)
        except Exception:
            continue
    if p.is_dir():
        return MarkdownAdapter(p, **options)  # empty folder -> markdown store
    raise AdapterError(f"could not detect memory store format for {path}")


__all__ = [
    "AdapterError", "MemoryAdapter", "MarkdownAdapter", "JSONAdapter", "SQLiteAdapter", "create_sqlite_store",
    "register_adapter", "get_adapter", "available_adapters",
]
