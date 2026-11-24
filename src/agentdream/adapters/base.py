"""Memory store adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import MemoryEntry


class AdapterError(Exception):
    pass


class MemoryAdapter(ABC):
    """A memory store.

    ``load()`` reads entries without modifying the store. ``write(entries, dest)``
    creates a NEW store at ``dest`` (which must not be the input location) using
    the input store as a layout template where that makes sense.
    """

    name: str = "base"
    #: True when the store is a single file (json/sqlite) rather than a folder.
    single_file: bool = False

    def __init__(self, path: str | Path, **options):
        self.path = Path(path) if not isinstance(path, Path) else path
        self.options = options

    @classmethod
    def detect(cls, path: Path) -> bool:  # pragma: no cover - overridden
        return False

    @abstractmethod
    def load(self) -> list[MemoryEntry]: ...

    @abstractmethod
    def write(self, entries: list[MemoryEntry], dest: str | Path) -> Path: ...

    def sidecars(self, path: str | Path) -> list[Path]:
        """Extra files that belong to the store at ``path`` (journals, WAL...).

        ``accept`` backs them up and removes them when it swaps a new store in, so
        stale state from the old store cannot leak into the new one.
        """
        return []

    def output_path(self, store_dir: str | Path) -> Path:
        """Where to put the new store inside a staging ``store_dir``."""
        d = Path(store_dir)
        return d / self.path.name if self.single_file else d

    def _check_dest(self, dest: Path) -> None:
        try:
            same = dest.resolve() == self.path.resolve()
        except OSError:
            same = False
        if same:
            raise AdapterError(f"refusing to write over the input store {self.path}; write to a new location")
