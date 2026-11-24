"""Optional adapter for stancelog (https://github.com/systemagents-ai/stancelog).

stancelog is an append-only, contradiction-aware belief log on SQLite. Its public
API (``stancelog.Memory``) is used directly:

* ``load()`` returns one :class:`MemoryEntry` per *current* belief
  (``Memory.all_current()``), keyed by subject, with the stance, polarity,
  confidence and valid/recorded times as metadata and the subject's closed
  (superseded/forgotten) beliefs from ``Memory.history()`` under
  ``metadata["stancelog_history"]``.
* ``write(entries, dest)`` creates a NEW stancelog database at ``dest`` by
  calling ``Memory.remember()`` once per entry, in order.

Opening a stancelog database is not read-only (it switches to WAL, applies the
schema, reconciles the journal and creates a lock file), so ``load()`` opens a
private temporary copy and the input database is never touched.

stancelog is imported lazily; paths are given as ``stancelog:<path>``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..models import MemoryEntry
from .base import AdapterError, MemoryAdapter

PREFIX = "stancelog:"

#: metadata keys owned by this adapter (everything else is passed through to stancelog)
_OWN_KEYS = {"subject", "stance", "polarity", "confidence", "valid_from", "recorded_at", "source",
             "belief_id", "value", "stancelog_history", "stancelog_metadata"}


def available() -> bool:
    try:
        importlib.import_module("stancelog")
        return True
    except Exception:
        return False


def strip_prefix(path: str | os.PathLike) -> str:
    p = os.fspath(path)
    return p[len(PREFIX):] if p.startswith(PREFIX) else p


def _stancelog():
    try:
        return importlib.import_module("stancelog")
    except ImportError as exc:  # pragma: no cover - guarded by the lazy proxy
        raise AdapterError("the stancelog adapter needs the 'stancelog' package") from exc


def _belief_record(b: Any) -> dict[str, Any]:
    return {
        "id": b.id, "content": b.text or b.value, "value": b.value, "stance": b.stance,
        "polarity": b.polarity, "confidence": b.confidence, "status": b.status,
        "valid_from": b.valid_from, "valid_to": b.valid_to, "recorded_at": b.recorded_at,
        "source": b.source,
    }


class StancelogAdapter(MemoryAdapter):
    name = "stancelog"
    single_file = True

    def __init__(self, path, **options):
        super().__init__(Path(strip_prefix(path)).expanduser(), **options)

    @classmethod
    def detect(cls, path: Path) -> bool:
        return str(path).startswith(PREFIX)

    def sidecars(self, path) -> list[Path]:
        # the journal matters most: stancelog would otherwise "recover" the old
        # events from it into the newly written database
        p = strip_prefix(path)
        return [Path(p + s) for s in (".journal.jsonl", "-wal", "-shm", ".lock")]

    # ------------------------------------------------------------------ load
    def load(self) -> list[MemoryEntry]:
        if not self.path.is_file():
            raise AdapterError(f"stancelog database not found: {self.path}")
        sl = _stancelog()
        with tempfile.TemporaryDirectory(prefix="agentdream-stancelog-") as tmp:
            copy = Path(tmp) / self.path.name
            shutil.copy2(self.path, copy)
            wal = Path(str(self.path) + "-wal")
            if wal.is_file():  # committed-but-not-checkpointed writes of a live process
                shutil.copy2(wal, str(copy) + "-wal")
            mem = sl.Memory(str(copy), journal=False)
            try:
                beliefs = list(reversed(mem.all_current()))  # oldest recorded first
                histories = {b.subject: [_belief_record(h) for h in mem.history(b.subject)
                                         if h.id != b.id] for b in beliefs}
            finally:
                mem.close()

        out = []
        for i, b in enumerate(beliefs):
            meta: dict[str, Any] = {
                "_order": i, "subject": b.subject, "stance": b.stance, "polarity": b.polarity,
                "confidence": b.confidence, "valid_from": b.valid_from, "recorded_at": b.recorded_at,
                "source": b.source, "belief_id": b.id, "value": b.value,
                # remember the exact text so write() can tell whether the dream rewrote it
                "_text": b.text or b.value,
            }
            if b.metadata:
                meta["stancelog_metadata"] = dict(b.metadata)
            if histories.get(b.subject):
                meta["stancelog_history"] = histories[b.subject]
            first = histories[b.subject][0]["valid_from"] if histories.get(b.subject) else b.valid_from
            out.append(MemoryEntry(
                id=b.id, content=b.text or b.value, path=self.path.name, key=b.subject, metadata=meta,
                created_at=min(first, b.valid_from), updated_at=b.valid_from, source="stancelog",
            ))
        return out

    # ----------------------------------------------------------------- write
    def write(self, entries: list[MemoryEntry], dest) -> Path:
        dest = Path(strip_prefix(dest))
        self._check_dest(dest)
        if dest.exists():
            raise AdapterError(f"destination exists: {dest}")
        sl = _stancelog()
        dest.parent.mkdir(parents=True, exist_ok=True)
        mem = sl.Memory(str(dest), journal=False)
        try:
            for e in entries:
                self._remember(mem, e)
        except Exception:
            mem.close()
            for p in (dest, Path(f"{dest}-wal"), Path(f"{dest}-shm"), Path(f"{dest}.lock")):
                p.unlink(missing_ok=True)
            raise
        mem.close()
        Path(f"{dest}.lock").unlink(missing_ok=True)
        return dest

    @staticmethod
    def _remember(mem: Any, e: MemoryEntry) -> None:
        md = e.metadata or {}
        content = e.content.strip()
        if not content:
            return
        unchanged = md.get("_text") == e.content
        # Polarity is kept from the surviving belief (a resolved contradiction keeps the
        # newest stance; its rewritten text also quotes the old one, which would confuse
        # re-extraction). Confidence/value are only reused when the text is untouched;
        # entries without a stored stance (new insights) are extracted by stancelog.
        polarity = md.get("polarity")
        confidence = md.get("confidence") if unchanged else None
        value = md.get("value") if unchanged else None
        extra = dict(md.get("stancelog_metadata") or {})
        extra.update({k: v for k, v in md.items() if k not in _OWN_KEYS and not k.startswith("_")})
        if e.id and not unchanged:
            extra.setdefault("agentdream_id", e.id)
        kwargs: dict[str, Any] = dict(
            subject=md.get("subject") or e.key, polarity=polarity, confidence=confidence, value=value,
            source=md.get("source") or "agentdream", valid_from=_valid_time(e.updated_at or md.get("valid_from")),
            metadata=extra or None,
        )
        if not kwargs["subject"]:
            kwargs.pop("subject")
        try:
            mem.remember(content, **kwargs)
        except ValueError:
            if "subject" in kwargs:
                raise
            # no subject could be extracted (e.g. a free-form insight): key it by its own text
            mem.remember(content, subject=content[:80], **kwargs)


def _valid_time(value: Any) -> str | None:
    """A timestamp stancelog accepts (ISO-8601), or None (= now) when unparseable."""
    if not value:
        return None
    from datetime import datetime

    s = str(value).strip()
    try:
        datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s)
    except ValueError:
        return None
    return s
