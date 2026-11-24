"""JSON / JSONL memory store (a single file or a folder of files).

Accepted shapes per file:

* ``.jsonl``: one entry object per line.
* ``.json``: a list of entry objects, ``{"entries": [...]}`` / ``{"memories": [...]}``,
  or a flat ``{"key": "value", ...}`` mapping (each pair becomes an entry with id=key).

Entry objects may use ``content`` / ``text`` / ``memory`` / ``value`` / ``note`` for the
text. Unknown fields are preserved in ``metadata`` and written back.
The output keeps the same shape as the input.

Files are UTF-8 (a BOM is accepted). A memory store is never repaired silently: invalid UTF-8,
invalid JSON (trailing commas, a truncated last line) or a non-object JSONL line raise
:class:`AdapterError` naming ``file:line``, because dropping the entry would lose it on accept.
A folder store keeps its non-JSON files when written.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path
from typing import Any

from ..models import MemoryEntry
from ..util import atomic_write_text, split_lines
from .base import AdapterError, MemoryAdapter

_TEXT_KEYS = ("content", "text", "memory", "value", "note", "body")
_KNOWN = {"id", "key", "metadata", "created_at", "updated_at", "source", "created", "updated", "timestamp"}


def _entry_from_obj(obj: Any, rel: str, idx: int) -> MemoryEntry:
    if isinstance(obj, str):
        return MemoryEntry(id=f"{rel}#{idx + 1}", content=obj, path=rel,
                           metadata={"_shape_item": "string"}, source=f"{rel}#{idx}")
    if not isinstance(obj, dict):
        raise AdapterError(f"{rel}: entry {idx} is not an object")
    text_key = next((k for k in _TEXT_KEYS if k in obj), None)
    if text_key is None:
        raise AdapterError(f"{rel}: entry {idx} has no content field ({', '.join(_TEXT_KEYS)})")
    content = obj[text_key]
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    meta = dict(obj.get("metadata") or {})
    extras = {k: v for k, v in obj.items() if k not in _KNOWN and k != text_key}
    if extras:
        meta["_extra_fields"] = extras
    meta["_text_key"] = text_key
    return MemoryEntry(
        id=str(obj.get("id") or f"{rel}#{idx + 1}"), content=content, path=rel,
        key=obj.get("key"), metadata=meta,
        created_at=obj.get("created_at") or obj.get("created"),
        updated_at=obj.get("updated_at") or obj.get("updated") or obj.get("timestamp"),
        source=obj.get("source") or f"{rel}#{idx}",
    )


def _obj_from_entry(e: MemoryEntry) -> Any:
    meta = {k: v for k, v in e.metadata.items() if not k.startswith("_")}
    if e.metadata.get("_shape_item") == "string":
        return e.content
    text_key = e.metadata.get("_text_key", "content")
    obj: dict[str, Any] = {"id": e.id, text_key: e.content}
    if e.key is not None:
        obj["key"] = e.key
    if meta:
        obj["metadata"] = meta
    if e.created_at:
        obj["created_at"] = e.created_at
    if e.updated_at:
        obj["updated_at"] = e.updated_at
    if e.source and not str(e.source).startswith(str(e.path or "")):
        obj["source"] = e.source
    obj.update(e.metadata.get("_extra_fields") or {})
    return obj


class JSONAdapter(MemoryAdapter):
    name = "json"

    def __init__(self, path: str | Path, **options: Any):
        super().__init__(path, **options)
        self.single_file = self.path.is_file() or self.path.suffix.lower() in (".json", ".jsonl")
        self._shapes: dict[str, dict[str, Any]] = {}

    @classmethod
    def detect(cls, path: Path) -> bool:
        if path.is_file():
            return path.suffix.lower() in (".json", ".jsonl")
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")]
            return bool(files) and all(p.suffix.lower() in (".json", ".jsonl") for p in files)
        return False

    def _files(self) -> list[tuple[str, Path]]:
        if not self.path.exists():
            raise AdapterError(f"memory store not found: {self.path}")
        if self.path.is_file():
            return [(self.path.name, self.path)]
        return [(p.relative_to(self.path).as_posix(), p) for p in sorted(self.path.rglob("*"))
                if p.is_file() and p.suffix.lower() in (".json", ".jsonl") and not p.name.startswith(".")]

    @staticmethod
    def _read(rel: str, p: Path) -> str:
        try:
            raw = p.read_bytes()
        except OSError as exc:
            raise AdapterError(f"cannot read memory file {p}: {exc.strerror or exc}") from exc
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AdapterError(f"{rel}: not valid UTF-8 (byte 0x{raw[exc.start]:02x} at offset {exc.start}); "
                               "re-save the file as UTF-8") from exc

    def load(self) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for rel, p in self._files():
            text = self._read(rel, p)
            if p.suffix.lower() == ".jsonl":
                self._shapes[rel] = {"kind": "jsonl"}
                i = 0
                for ln, line in enumerate(split_lines(text), start=1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except (ValueError, RecursionError) as exc:
                        raise AdapterError(f"{rel}:{ln}: invalid JSON ({str(exc)[:100]}); fix or remove the line "
                                           "(a cut-off last line usually means an interrupted write)") from exc
                    if not isinstance(obj, (dict, str)):
                        raise AdapterError(f"{rel}:{ln}: entry is not an object ({type(obj).__name__})")
                    entries.append(_entry_from_obj(obj, rel, i))
                    i += 1
                continue
            try:
                data = json.loads(text) if text.strip() else []
            except json.JSONDecodeError as exc:
                raise AdapterError(f"{rel}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}") from exc
            except (ValueError, RecursionError) as exc:
                raise AdapterError(f"{rel}: invalid JSON ({str(exc)[:100]})") from exc
            if isinstance(data, list):
                self._shapes[rel] = {"kind": "list"}
                items = data
            elif isinstance(data, dict):
                list_key = next((k for k in ("entries", "memories", "items", "notes") if isinstance(data.get(k), list)),
                                None)
                if list_key:
                    self._shapes[rel] = {"kind": "wrapped", "key": list_key,
                                         "rest": {k: v for k, v in data.items() if k != list_key}}
                    items = data[list_key]
                else:
                    self._shapes[rel] = {"kind": "map"}
                    for k, v in data.items():
                        content = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                        entries.append(MemoryEntry(id=str(k), content=content, path=rel, key=str(k),
                                                   metadata={"_map_raw": not isinstance(v, str)}, source=rel))
                    continue
            else:
                raise AdapterError(f"{rel}: unsupported JSON top-level type")
            for i, obj in enumerate(items):
                entries.append(_entry_from_obj(obj, rel, i))
        for i, e in enumerate(entries):
            e.metadata["_order"] = i
        return entries

    def write(self, entries: list[MemoryEntry], dest: str | Path) -> Path:
        dest = Path(dest)
        self._check_dest(dest)
        if not self._shapes:
            self.load()
        files = [rel for rel, _ in self._files()]
        default_rel = files[0] if files else "memory.json"
        groups: dict[str, list[MemoryEntry]] = {rel: [] for rel in files}
        for e in entries:
            rel = e.path if e.path in groups else default_rel
            groups.setdefault(rel, []).append(e)

        def render(rel: str, ents: list[MemoryEntry]) -> str:
            shape = self._shapes.get(rel, {"kind": "jsonl" if rel.endswith(".jsonl") else "list"})
            if shape["kind"] == "jsonl":
                return "".join(json.dumps(_obj_from_entry(e), ensure_ascii=False) + "\n" for e in ents)
            if shape["kind"] == "map":
                out = {}
                for e in ents:
                    k = e.key or e.id
                    out[k] = e.content
                    if e.metadata.get("_map_raw"):
                        # structured value; an operation may have rewritten it as plain text
                        with contextlib.suppress(ValueError):
                            out[k] = json.loads(e.content)
                return json.dumps(out, indent=2, ensure_ascii=False) + "\n"
            items = [_obj_from_entry(e) for e in ents]
            if shape["kind"] == "wrapped":
                return json.dumps({**shape["rest"], shape["key"]: items}, indent=2, ensure_ascii=False) + "\n"
            return json.dumps(items, indent=2, ensure_ascii=False) + "\n"

        if self.path.is_file():
            atomic_write_text(dest, render(self.path.name, groups.get(self.path.name, [])))
            return dest
        tmp = dest.with_name(dest.name + ".tmp-write")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        # a folder store may hold other files (README, notes...): keep them
        for p in sorted(self.path.rglob("*")):
            rel = p.relative_to(self.path).as_posix()
            if p.is_file() and rel not in groups and not any(part.startswith(".") for part in rel.split("/")):
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tmp / rel)
        for rel, ents in groups.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_text(render(rel, ents), encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.replace(dest)
        return dest
