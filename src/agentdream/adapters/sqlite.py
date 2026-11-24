"""SQLite memory store: one table of entries.

Default table ``memories`` with columns ``id, content, key, metadata, created_at,
updated_at, source`` (only ``id`` and a text column are required; the text
column may be named content/text/memory/value/note). The input database is
opened read-only; ``write`` copies it with the SQLite backup API to the new
location and rewrites the table there, so other tables survive untouched.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from ..models import MemoryEntry
from .base import AdapterError, MemoryAdapter

_TEXT_COLS = ("content", "text", "memory", "value", "note", "body")
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _ro_connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


class SQLiteAdapter(MemoryAdapter):
    name = "sqlite"
    single_file = True

    def __init__(self, path: str | Path, table: str | None = None, **options: Any):
        super().__init__(path, **options)
        self.table = table

    @classmethod
    def detect(cls, path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            return True
        try:
            with open(path, "rb") as f:
                return f.read(16) == _SQLITE_MAGIC
        except OSError:
            return False

    def _resolve_table(self, con: sqlite3.Connection) -> tuple[str, list[str], str]:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        candidates = [self.table] if self.table else (["memories", "memory", "entries", "notes", *tables])
        for t in candidates:
            if t not in tables:
                continue
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            text_col = next((c for c in _TEXT_COLS if c in cols), None)
            if "id" in cols and text_col:
                return t, cols, text_col
        raise AdapterError(f"{self.path}: no table with id + content columns found")

    def load(self) -> list[MemoryEntry]:
        if not self.path.exists():
            raise AdapterError(f"memory store not found: {self.path}")
        con = _ro_connect(self.path)
        try:
            table, cols, text_col = self._resolve_table(con)
            con.row_factory = sqlite3.Row
            order = "rowid" if "rowid" not in cols else "id"
            rows = con.execute(f'SELECT * FROM "{table}" ORDER BY {order}').fetchall()
        finally:
            con.close()
        entries = []
        for i, r in enumerate(rows):
            d = dict(r)
            meta = {}
            if d.get("metadata"):
                try:
                    meta = json.loads(d["metadata"])
                except (TypeError, ValueError):
                    meta = {"raw_metadata": d["metadata"]}
            extras = {k: v for k, v in d.items()
                      if k not in ("id", text_col, "key", "metadata", "created_at", "updated_at", "source", "path")}
            if extras:
                meta["_extra_columns"] = extras
            meta["_order"] = i
            entries.append(MemoryEntry(
                id=str(d["id"]), content=str(d[text_col] or ""), path=d.get("path") or table, key=d.get("key"),
                metadata=meta, created_at=d.get("created_at"), updated_at=d.get("updated_at"),
                source=d.get("source") or f"{self.path.name}:{table}:{d['id']}",
            ))
        return entries

    def write(self, entries: list[MemoryEntry], dest: str | Path) -> Path:
        dest = Path(dest)
        self._check_dest(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        src = _ro_connect(self.path)
        out = sqlite3.connect(str(tmp))
        try:
            src.backup(out)
            table, cols, text_col = self._resolve_table(src)
        finally:
            src.close()
        try:
            out.execute(f'DELETE FROM "{table}"')
            for e in entries:
                meta = {k: v for k, v in e.metadata.items() if not k.startswith("_")}
                row = {"id": e.id, text_col: e.content, "key": e.key,
                       "metadata": json.dumps(meta, ensure_ascii=False) if meta else None,
                       "created_at": e.created_at, "updated_at": e.updated_at, "source": e.source,
                       "path": e.path}
                row.update(e.metadata.get("_extra_columns") or {})
                row = {k: v for k, v in row.items() if k in cols}
                names = ", ".join(f'"{k}"' for k in row)
                marks = ", ".join("?" for _ in row)
                out.execute(f'INSERT INTO "{table}" ({names}) VALUES ({marks})', list(row.values()))
            out.commit()
        finally:
            out.close()
        os.replace(tmp, dest)
        return dest


def create_sqlite_store(path: str | Path, entries: list[MemoryEntry], table: str = "memories") -> Path:
    """Helper to create a fresh SQLite memory store (used by tests/examples)."""
    p = Path(path)
    con = sqlite3.connect(str(p))
    try:
        con.execute(f'CREATE TABLE IF NOT EXISTS "{table}" (id TEXT PRIMARY KEY, content TEXT NOT NULL, key TEXT, '
                    'metadata TEXT, created_at TEXT, updated_at TEXT, source TEXT)')
        for e in entries:
            con.execute(f'INSERT INTO "{table}" VALUES (?,?,?,?,?,?,?)',
                        (e.id, e.content, e.key, json.dumps(e.metadata) if e.metadata else None,
                         e.created_at, e.updated_at, e.source))
        con.commit()
    finally:
        con.close()
    return p
