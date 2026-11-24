"""Token-budgeted chunking of memory stores and transcripts (stdlib only).

Used by the LLM engine's map-reduce: when the memory store and transcripts
don't fit one model call, each phase runs once per chunk (map) and the proposed
operations are merged (reduce, see :mod:`agentdream.engine.llm`).

Rules:

* Token counts are an approximation (``~4 characters per token``, see
  :func:`estimate_tokens`); budgets therefore keep a safety margin.
* Items (memory entries, transcript messages) are packed greedily, in order.
  An item is never split across chunks unless it alone exceeds the budget; then
  it is cut at paragraph, line, sentence, word or - last resort - character
  boundaries, and every part keeps the original id / citation plus a
  ``(part k/n)`` marker.
* Transcript chunks overlap: a new chunk starts with the trailing messages of the
  previous chunk *of the same session*, up to ``overlap_tokens`` (capped at a
  quarter of the budget so every chunk makes progress). Overlap never crosses
  sessions.
* Everything is deterministic: the same input and budget give the same chunks.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .models import MemoryEntry, Message, Transcript
from .util import estimate_tokens

T = TypeVar("T")

#: chars per token assumed by :func:`estimate_tokens`
CHARS_PER_TOKEN = 4
#: tokens charged for a "### Session ..." header each time a session starts inside a chunk
SESSION_HEADER_TOKENS = 24


class ChunkingError(ValueError):
    pass


@dataclass
class Chunk(Generic[T]):
    index: int
    items: list[T]
    tokens: int
    #: number of leading items repeated from the previous chunk (transcripts only)
    overlap: int = 0


# ------------------------------------------------------------------ text splitting

# Preferred cut points, best first: after a blank line, a line, a sentence, whitespace.
_CUTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n[ \t]*\n\s*"),
    re.compile(r"\n"),
    re.compile(r"[.!?][\"')\]]*\s+"),
    re.compile(r"\s+"),
)


def split_text(text: str, max_tokens: int) -> list[str]:
    """Split ``text`` into consecutive pieces of at most ``max_tokens`` (estimated) each,
    preferring natural boundaries. ``"".join(pieces) == text`` always holds."""
    if max_tokens < 1:
        raise ChunkingError("max_tokens must be >= 1")
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]
    cuts = [sorted({m.end() for m in pat.finditer(text)}) for pat in _CUTS]
    pieces: list[str] = []
    start, n = 0, len(text)
    while n - start > max_chars:
        limit = start + max_chars
        best = None
        for level in cuts:
            i = bisect.bisect_right(level, limit) - 1
            if i >= 0 and level[i] > start:
                if level[i] - start >= max_chars // 2:  # good boundary that isn't wastefully early
                    best = level[i]
                    break
                best = max(best or 0, level[i])
        cut = best if best and best > start else limit
        pieces.append(text[start:cut])
        start = cut
    pieces.append(text[start:])
    return pieces


# ------------------------------------------------------------------ generic packing

def pack(items: Sequence[T], cost: Callable[[T], int], budget: int, *, overlap_tokens: int = 0,
         same_group: Callable[[T, T], bool] | None = None, group_cost: int = 0) -> list[Chunk[T]]:
    """Greedy, order-preserving packing of items (each must fit: ``cost(item) + group_cost <= budget``).

    ``same_group(a, b)`` says whether consecutive items belong to the same group (session);
    each group run inside a chunk costs ``group_cost`` extra (its header). With
    ``overlap_tokens`` a new chunk first repeats trailing items of the previous chunk that are
    in the same group as the next item."""
    if budget < 1:
        raise ChunkingError("budget must be >= 1")
    overlap_tokens = max(0, min(overlap_tokens, budget // 4))

    def joins(prev: T | None, it: T) -> int:
        if prev is None or same_group is None or not same_group(prev, it):
            return group_cost
        return 0

    chunks: list[Chunk[T]] = []
    cur: list[T] = []
    cur_tok = 0
    cur_overlap = 0
    for it in items:
        c = cost(it)
        if c + group_cost > budget:
            raise ChunkingError(f"item of {c} tokens exceeds the chunk budget of {budget}; split it first")
        step = c + joins(cur[-1] if cur else None, it)
        if cur and cur_tok + step > budget:
            chunks.append(Chunk(len(chunks), cur, cur_tok, cur_overlap))
            carried: list[T] = []
            carried_tok = 0
            if overlap_tokens and same_group is not None:
                for prev in reversed(cur[cur_overlap:]):
                    pc = cost(prev)
                    if not same_group(prev, it) or carried_tok + pc > overlap_tokens \
                            or group_cost + carried_tok + pc + c > budget:
                        break
                    carried.insert(0, prev)
                    carried_tok += pc
            cur, cur_overlap = carried, len(carried)
            cur_tok = carried_tok + (group_cost if carried else 0)
            step = c + joins(cur[-1] if cur else None, it)
        cur.append(it)
        cur_tok += step
    if len(cur) > cur_overlap:
        chunks.append(Chunk(len(chunks), cur, cur_tok, cur_overlap))
    return chunks


# ------------------------------------------------------------------ memory entries

def entry_cost(e: MemoryEntry) -> int:
    """Approximate prompt tokens for one rendered memory entry (content + id/metadata line)."""
    return estimate_tokens(e.content) + 10 + estimate_tokens(f"{e.id}{e.path or ''}{e.key or ''}")


def split_entry(e: MemoryEntry, budget: int) -> list[MemoryEntry]:
    """An entry larger than ``budget`` becomes excerpts with the same id and
    ``metadata["_part"] = [k, n]``. (The LLM engine never lets a model rewrite an entry it
    only saw part of.)"""
    if entry_cost(e) <= budget:
        return [e]
    overhead = entry_cost(MemoryEntry(id=e.id, content="", path=e.path, key=e.key)) + 8
    texts = split_text(e.content, max(1, budget - overhead))
    return [MemoryEntry(id=e.id, content=t, path=e.path, key=e.key,
                        metadata={**e.metadata, "_part": [k, len(texts)]},
                        created_at=e.created_at, updated_at=e.updated_at, source=e.source)
            for k, t in enumerate(texts, start=1)]


def chunk_memory(entries: Sequence[MemoryEntry], max_tokens: int) -> list[Chunk[MemoryEntry]]:
    """Pack memory entries into chunks of at most ``max_tokens`` (no overlap)."""
    units = [p for e in entries for p in split_entry(e, max_tokens)]
    return pack(units, entry_cost, max_tokens)


# ------------------------------------------------------------------ transcripts

@dataclass
class TranscriptSlice:
    """Consecutive messages of one transcript inside a chunk."""

    transcript: Transcript
    messages: list[Message] = field(default_factory=list)
    #: leading messages repeated from the previous chunk
    overlap: int = 0


@dataclass
class _Unit:
    t_index: int
    transcript: Transcript
    message: Message


def message_cost(m: Message) -> int:
    """Approximate prompt tokens for one rendered message line (citation + role + content)."""
    return estimate_tokens(m.content) + estimate_tokens(m.citation) + 8


def split_message(m: Message, budget: int) -> list[Message]:
    """A message larger than ``budget`` becomes parts that keep its citation, prefixed ``(part k/n)``."""
    if message_cost(m) <= budget:
        return [m]
    overhead = message_cost(Message(m.role, "", m.timestamp, m.line_no, m.source_path, m.index)) + 8
    texts = split_text(m.content, max(1, budget - overhead))
    return [Message(m.role, f"(part {k}/{len(texts)}) {t}", m.timestamp, m.line_no, m.source_path, m.index)
            for k, t in enumerate(texts, start=1)]


def chunk_transcripts(transcripts: Sequence[Transcript], max_tokens: int,
                      overlap_tokens: int = 0) -> list[Chunk[TranscriptSlice]]:
    """Pack transcript messages (in the given, chronological order) into chunks of at most
    ``max_tokens`` including per-session headers."""
    budget = max_tokens - SESSION_HEADER_TOKENS
    if budget < 16:
        raise ChunkingError(f"transcript chunk budget too small ({max_tokens} tokens)")
    units = [_Unit(ti, t, part) for ti, t in enumerate(transcripts) for m in t.messages
             for part in split_message(m, budget)]
    packed = pack(units, lambda u: message_cost(u.message), max_tokens, overlap_tokens=overlap_tokens,
                  same_group=lambda a, b: a.t_index == b.t_index, group_cost=SESSION_HEADER_TOKENS)
    out: list[Chunk[TranscriptSlice]] = []
    for ch in packed:
        slices: list[TranscriptSlice] = []
        for k, u in enumerate(ch.items):
            if not slices or slices[-1].transcript is not u.transcript:
                slices.append(TranscriptSlice(u.transcript))
            slices[-1].messages.append(u.message)
            if k < ch.overlap:
                slices[-1].overlap += 1
        out.append(Chunk(ch.index, slices, ch.tokens, ch.overlap))
    return out


__all__ = ["CHARS_PER_TOKEN", "SESSION_HEADER_TOKENS", "Chunk", "ChunkingError", "TranscriptSlice",
           "chunk_memory", "chunk_transcripts", "entry_cost", "estimate_tokens", "message_cost", "pack",
           "split_entry", "split_message", "split_text"]
