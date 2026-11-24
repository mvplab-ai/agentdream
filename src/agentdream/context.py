"""Shared state passed through the dream pipeline phases."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .blocking import TokenIndex
from .models import MemoryEntry, Operation, Transcript, apply_operations
from .text import content_tokens, fold, split_sentences
from .util import parse_timestamp


class DreamCanceled(Exception):
    pass


class DreamLimitError(Exception):
    """A configured size limit was exceeded and ``on_limit="fail"`` (or truncation could not help)."""


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
    #: memory entries are never truncated (that would delete notes on accept): over the limit is an error
    max_entries: int = 20_000
    on_limit: str = "truncate"  # truncate | fail
    # chunking (LLM backends): token budget per model call, transcript overlap, cap on calls per phase
    max_chunk_tokens: int = 60_000
    chunk_overlap: int = 400
    max_chunks: int = 64
    # LLM call retries for transient errors (timeouts, HTTP 408/429/5xx): attempts = 1 + llm_retries;
    # wait = Retry-After if the server sent one, else llm_retry_backoff * 2**attempt; capped at llm_retry_max_wait
    llm_retries: int = 2
    llm_retry_backoff: float = 2.0
    llm_retry_max_wait: float = 60.0
    # redaction
    redact: bool = True
    redact_emails: bool = False
    redact_entropy: bool = True
    entropy_threshold: float = 4.0
    entropy_min_length: int = 24
    redact_cards: bool = True
    redact_patterns: list[str] = field(default_factory=list)
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

    def validate(self) -> None:
        """Reject nonsense limits up front (ValueError) instead of misbehaving later."""
        positive = ("max_sessions", "max_bytes_per_transcript", "max_total_tokens", "max_entries", "max_chunk_tokens",
                    "max_chunks")
        for name in positive:
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")
        for name in ("min_sessions", "chunk_overlap", "llm_retries"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {v!r}")
        for name in ("llm_retry_backoff", "llm_retry_max_wait"):
            if not isinstance(getattr(self, name), (int, float)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if self.on_limit not in ("truncate", "fail"):
            raise ValueError(f"on_limit must be 'truncate' or 'fail', got {self.on_limit!r}")


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
        self.redactor: Any = None  # set by the redact phase; LLM prompts get a final pass through it
        self.output_dir: Path | None = None
        self.store_path: Path | None = None
        self.final_entries: list[MemoryEntry] = []
        self._producer: dict[str, str] = {}
        self._progress = progress
        self._cancel = cancel_event
        self._sentences: list[SentenceRec] | None = None
        self._sentence_index: TokenIndex | None = None
        self._by_id: tuple[list[MemoryEntry], int, dict[str, MemoryEntry]] | None = None
        self._op_ids: set[str] = set()
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
        """Current (working-view) entry by id. O(1): the id index is rebuilt only when
        ``self.entries`` is replaced (``add_op`` always replaces it) or changes length. Custom
        phases that overwrite ``ctx.entries[i]`` in place should assign a new list instead."""
        cached = self._by_id
        if cached is None or cached[0] is not self.entries or cached[1] != len(self.entries):
            idx: dict[str, MemoryEntry] = {}
            for e in self.entries:
                idx.setdefault(e.id, e)
            # holding a reference to the list (not its id()) means a recycled id can't alias it
            cached = self._by_id = (self.entries, len(self.entries), idx)
        return cached[2].get(entry_id)

    def add_op(self, op: Operation) -> Operation:
        deps = {self._producer[t] for t in op.targets + op.remove if t in self._producer}
        deps |= {self._producer[u.id] for u in op.upsert if u.id in self._producer}
        op.depends_on = sorted(set(op.depends_on) | deps)
        if len(self._op_ids) != len(self.ops):  # ops list was edited by a custom phase
            self._op_ids = {o.id for o in self.ops}
        base, n = op.id, 2
        while op.id in self._op_ids:
            op.id = f"{base}-{n}"
            n += 1
        self.ops.append(op)
        self._op_ids.add(op.id)
        if op.changes_store:
            # untouched entries are shared, not deep-copied: O(n) per op instead of O(n * entry size)
            self.entries = apply_operations(self.entries, [op], copy_untouched=False)
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
        self._sentence_index = None
        return out

    def sentence_index(self) -> TokenIndex:
        """Inverted index: content token -> ascending positions in :meth:`sentences`."""
        recs = self.sentences()
        if self._sentence_index is None or self._sentence_index.size != len(recs):
            self._sentence_index = TokenIndex(r.toks for r in recs)
        return self._sentence_index
