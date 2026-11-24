"""Core data model: memory entries, transcripts, evidence and operations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

OP_TYPES = ("merge", "replace", "correct", "insight", "contradiction")


def stable_hash(*parts: Any, length: int = 10) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:length]


@dataclass
class MemoryEntry:
    """A single memory note.

    ``path`` is the location inside the store (relative file for folder stores,
    file name / table for others). ``key`` is an optional finer key (section
    heading, JSON key...). ``metadata`` is free-form; keys starting with ``_``
    are adapter-private layout hints and are not shown in reports.
    """

    id: str
    content: str
    path: str | None = None
    key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        return cls(
            id=str(d["id"]),
            content=str(d.get("content", "")),
            path=d.get("path"),
            key=d.get("key"),
            metadata=dict(d.get("metadata") or {}),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            source=d.get("source"),
        )

    def copy(self) -> MemoryEntry:
        return copy.deepcopy(self)

    @property
    def timestamp(self) -> str | None:
        return self.updated_at or self.created_at


@dataclass
class Message:
    role: str
    content: str
    timestamp: str | None = None
    line_no: int | None = None
    source_path: str | None = None
    index: int = 0

    @property
    def citation(self) -> str:
        src = self.source_path or "<memory>"
        if self.line_no is not None:
            return f"{src}:{self.line_no}"
        return f"{src}#{self.index}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citation"] = self.citation
        return d


@dataclass
class Transcript:
    id: str
    messages: list[Message] = field(default_factory=list)
    source_path: str | None = None
    format: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def started_at(self) -> str | None:
        for m in self.messages:
            if m.timestamp:
                return m.timestamp
        return None

    def text_size(self) -> int:
        return sum(len(m.content.encode("utf-8")) for m in self.messages)


@dataclass
class Evidence:
    citation: str
    quote: str
    source_path: str | None = None
    line_no: int | None = None
    role: str | None = None
    stance: str = "supports"  # supports | refutes | mentions
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        return cls(**{k: d.get(k) for k in ("citation", "quote", "source_path", "line_no", "role", "timestamp")},
                   stance=d.get("stance") or "supports")

    @classmethod
    def from_message(cls, msg: Message, quote: str, stance: str = "supports") -> Evidence:
        q = " ".join(quote.split())
        if len(q) > 240:
            q = q[:237] + "..."
        return cls(citation=msg.citation, quote=q, source_path=msg.source_path, line_no=msg.line_no,
                   role=msg.role, stance=stance, timestamp=msg.timestamp)


@dataclass
class Operation:
    """One reviewable change. Applying an operation removes ``remove`` ids and
    upserts ``upsert`` entries. ``targets`` are the original ids it is about."""

    id: str
    type: str
    summary: str
    reason: str = ""
    targets: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    upsert: list[MemoryEntry] = field(default_factory=list)
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    kind: str | None = None  # contradictions: temporal | conflict | evidence
    confidence: float = 0.5
    phase: str = ""
    origin: str = "deterministic"
    history: list[str] = field(default_factory=list)

    @property
    def changes_store(self) -> bool:
        return bool(self.remove or self.upsert)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["upsert"] = [e.to_dict() for e in self.upsert]
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Operation:
        return cls(
            id=d["id"], type=d["type"], summary=d.get("summary", ""), reason=d.get("reason", ""),
            targets=list(d.get("targets") or []), remove=list(d.get("remove") or []),
            upsert=[MemoryEntry.from_dict(e) for e in d.get("upsert") or []],
            before=list(d.get("before") or []), after=list(d.get("after") or []),
            evidence=[Evidence.from_dict(e) for e in d.get("evidence") or []],
            depends_on=list(d.get("depends_on") or []), kind=d.get("kind"),
            confidence=float(d.get("confidence", 0.5)), phase=d.get("phase", ""),
            origin=d.get("origin", "deterministic"), history=list(d.get("history") or []),
        )

    @staticmethod
    def make_id(type_: str, targets: Iterable[str], after: Iterable[str]) -> str:
        return f"{type_}-{stable_hash(type_, sorted(targets), list(after), length=8)}"


def apply_operations(entries: list[MemoryEntry], ops: Iterable[Operation], *,
                     copy_untouched: bool = True) -> list[MemoryEntry]:
    """Return a new list of entries with ``ops`` applied in order. Never mutates input.

    With ``copy_untouched=False`` entries that no operation touches are shared with
    the input list instead of deep-copied (callers must not mutate them in place).
    """
    order: list[str] = []
    by_id: dict[str, MemoryEntry] = {}
    for e in entries:
        if e.id not in by_id:
            order.append(e.id)
        by_id[e.id] = e.copy() if copy_untouched else e
    for op in ops:
        for rid in op.remove:
            by_id.pop(rid, None)
        for up in op.upsert:
            if up.id not in by_id and up.id not in order:
                order.append(up.id)
            by_id[up.id] = up.copy()
    return [by_id[i] for i in order if i in by_id]


def resolve_dependencies(ops: list[Operation], selected: Iterable[str]) -> list[str]:
    """Expand a selection of op ids with their (transitive) dependencies."""
    by_id = {o.id: o for o in ops}
    out: set[str] = set()
    stack = list(selected)
    while stack:
        cur = stack.pop()
        if cur in out or cur not in by_id:
            continue
        out.add(cur)
        stack.extend(by_id[cur].depends_on)
    return [o.id for o in ops if o.id in out]
