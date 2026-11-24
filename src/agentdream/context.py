"""Shared state passed through the dream pipeline phases."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import MemoryEntry, Operation, Transcript, apply_operations
from .text import content_tokens, fold, split_sentences
from .util import parse_timestamp


class DreamCanceled(Exception):
    pass


@dataclass
class DreamConfig:
    memory: str | None = None
    sessions: list[str] = field(default_factory=list)
    backend: str = "deterministic"
    model: str | None = None
    instructions: str | None = None
    memory_format: str | None = None
    transcript_format: str | None = None
    out: str | None = None
    jobs_dir: str | None = None
    # limits
    max_sessions: int = 100
    min_sessions: int = 1
    max_bytes_per_transcript: int = 2_000_000
    max_total_tokens: int = 400_000
    on_limit: str = "truncate"  # truncate | fail
    # redaction
    redact: bool = True
    redact_emails: bool = False
    # heuristics
    insight_min_sessions: int = 2
    max_insights: int = 12
    backend_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DreamConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SentenceRec:
    t_index: int
    m_index: int
    s_index: int
    msg: Any
    text: str
    folded: str
    toks: frozenset[str]
    session: str
    order: int


class DreamContext:
    def __init__(self, config: DreamConfig, backend: Any = None,
                 progress: Callable[[str, float], None] | None = None,
                 cancel_event: threading.Event | None = None):
        self.config = config
        self.backend = backend
        self.adapter: Any = None
        self.original_entries: list[MemoryEntry] = []
        self.entries: list[MemoryEntry] = []  # redacted working view, updated as ops are added
        self.transcripts: list[Transcript] = []
        self.ops: list[Operation] = []
        self.warnings: list[str] = []
        self.notices: list[str] = []
        self.stats: dict[str, Any] = {}
        self.redaction_counts: dict[str, int] = {}
        self.output_dir: Path | None = None
        self.store_path: Path | None = None
        self.final_entries: list[MemoryEntry] = []
        self._producer: dict[str, str] = {}
        self._progress = progress
        self._cancel = cancel_event
        self._sentences: list[SentenceRec] | None = None
        self.skip_phases: set[str] = set()

    # -- control
    def check_cancel(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise DreamCanceled("dream canceled")

    def progress(self, phase: str, fraction: float) -> None:
        if self._progress:
            self._progress(phase, fraction)

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    # -- operations
    def entry(self, entry_id: str) -> MemoryEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def add_op(self, op: Operation) -> Operation:
        deps = {self._producer[t] for t in op.targets + op.remove if t in self._producer}
        deps |= {self._producer[u.id] for u in op.upsert if u.id in self._producer}
        op.depends_on = sorted(set(op.depends_on) | deps)
        existing = {o.id for o in self.ops}
        base, n = op.id, 2
        while op.id in existing:
            op.id = f"{base}-{n}"
            n += 1
        self.ops.append(op)
        if op.changes_store:
            self.entries = apply_operations(self.entries, [op])
            for u in op.upsert:
                self._producer[u.id] = op.id
        return op

    # -- recency helpers
    @staticmethod
    def entry_time(e: MemoryEntry) -> float | None:
        return parse_timestamp(e.updated_at or e.created_at)

    @staticmethod
    def recency_key(e: MemoryEntry) -> tuple:
        ts = parse_timestamp(e.updated_at or e.created_at)
        return (ts if ts is not None else float("-inf"), e.metadata.get("_order", 0))

    # -- transcript sentence index
    def sentences(self) -> list[SentenceRec]:
        if self._sentences is not None:
            return self._sentences
        out: list[SentenceRec] = []
        order = 0
        for ti, t in enumerate(self.transcripts):
            for mi, m in enumerate(t.messages):
                for si, s in enumerate(split_sentences(m.content)):
                    out.append(SentenceRec(ti, mi, si, m, s, fold(s), frozenset(content_tokens(s)), t.id, order))
                    order += 1
        self._sentences = out
        return out
