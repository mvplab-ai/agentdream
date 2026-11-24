"""LLM-backed phases: prompts, chunked map-reduce, tolerant JSON parsing and validation.

A phase whose input fits ``max_chunk_tokens`` is one model call. Larger inputs are
planned into several calls (:func:`plan_calls`, using :mod:`agentdream.chunking`);
every call's proposals are validated against the *whole* store and transcripts,
then merged by :func:`reduce_operations`.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from ..blocking import TokenIndex, cluster_order, similar_pairs
from ..chunking import (
    SESSION_HEADER_TOKENS,
    TranscriptSlice,
    chunk_memory,
    chunk_transcripts,
    entry_cost,
    message_cost,
)
from ..context import DreamCanceled, DreamContext, DreamLimitError
from ..models import Evidence, MemoryEntry, Message, Operation, Transcript, stable_hash
from ..text import fold, jaccard, shorten, token_set
from ..util import estimate_tokens

SYSTEM_PROMPT = """You are agentdream, a careful memory-consolidation engine for an AI agent ("dreaming").
You receive the agent's current memory store (entries with ids) and, for some tasks, session transcripts in which
every message is prefixed with a citation like [path/to/session.jsonl:42].

Your job is to propose a small set of high-value, reviewable operations on the memory store:
- merge: entries that say the same thing. targets = all duplicate ids, FIRST id is kept; new_content = best wording.
- replace: a stale value superseded by a newer one. targets = [kept_id, stale_id, ...]; new_content = the current value.
- correct: an entry that transcripts prove wrong. targets = [id]; new_content = the corrected note. Evidence REQUIRED.
- insight: a new note (recurring mistake, repeated preference, converged workflow). targets = []; Evidence REQUIRED.
- contradiction: conflicting statements. kind = "temporal" when a stance changed over time (latest wins, keep a short
  history in new_content, e.g. "Use X. (History: earlier rejected ...)"), "conflict" otherwise. targets = involved ids.

Rules:
- Transcripts are ground truth about what actually happened; memory may be outdated or wrong.
- Every evidence item must cite an exact citation from the transcripts and quote text copied verbatim from it.
- Never invent ids or citations. Prefer fewer, correct operations over many speculative ones.
- Keep notes concise, self-contained and in the store's existing style.
- Output ONLY a JSON object, no prose, matching:
{"operations": [{"type": "merge|replace|correct|insight|contradiction", "targets": ["id", ...],
  "new_content": "string or null", "kind": "temporal|conflict|null", "summary": "one line",
  "reason": "why", "confidence": 0.0-1.0, "evidence": [{"citation": "path:line", "quote": "verbatim text"}]}]}
If nothing needs to change, return {"operations": []}."""

OPERATIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"operations": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "type": {"enum": ["merge", "replace", "correct", "insight", "contradiction"]},
            "targets": {"type": "array", "items": {"type": "string"}},
            "new_content": {"type": ["string", "null"]},
            "kind": {"type": ["string", "null"]},
            "summary": {"type": "string"}, "reason": {"type": "string"},
            "confidence": {"type": "number"},
            "evidence": {"type": "array", "items": {"type": "object", "properties": {
                "citation": {"type": "string"}, "quote": {"type": "string"}}}},
        },
        "required": ["type"],
    }}},
    "required": ["operations"],
}

PHASE_TASKS = {
    "consolidate": ("Task: CONSOLIDATE the memory store. Find duplicates / near-duplicates (merge), stale values "
                    "that a newer entry supersedes (replace, judge recency by timestamps or later position), and stance "
                    "reversals between entries (contradiction, kind=temporal). Only use operation types merge, "
                    "replace, contradiction."),
    "verify": ("Task: VERIFY each memory entry against the transcripts. If transcripts show an entry is wrong "
               "(e.g. the agent repeatedly failed acting on it and later found the real answer), emit a correct "
               "operation with the transcript-proven value and cite evidence. If a stated preference was reversed "
               "in later sessions, emit contradiction kind=temporal with the latest stance as new_content and a short "
               "history. Only use operation types correct, contradiction."),
    "insights": ("Task: surface NEW INSIGHTS from the transcripts that are not already in memory: recurring mistakes "
                 "(the same failure in several places), repeated user preferences, converged workflows. Each insight "
                 "needs evidence from at least two places when possible. Only use operation type insight (or "
                 "contradiction kind=temporal for preferences that flipped across sessions)."),
}


# ------------------------------------------------------------------ JSON repair

def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", text, re.S)
    return m.group(1) if m else text


def _balanced(text: str) -> str | None:
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    stack, in_str, esc = [], False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return text[start:i + 1]
    # unterminated: close what is open (truncated output)
    return text[start:] + "".join(reversed(stack)) if stack and not in_str else None


def parse_json_loose(text: str) -> Any:
    """Parse model output into JSON, repairing common problems. Raises ValueError."""
    if text is None:
        raise ValueError("empty model output")
    candidates = []
    t = text.strip()
    candidates.append(t)
    fenced = _strip_fences(t).strip()
    candidates.append(fenced)
    bal = _balanced(fenced)
    if bal:
        candidates.append(bal)
    for c in list(candidates):
        fixed = re.sub(r",\s*([}\]])", r"\1", c)
        fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        candidates.append(fixed)
    for c in candidates:
        try:
            return json.loads(c)
        except ValueError:
            continue
    for c in candidates:
        try:
            py = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", c)))
            val = ast.literal_eval(py)
            if isinstance(val, (dict, list)):
                return val
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
    raise ValueError(f"could not parse JSON from model output: {shorten(t, 120)!r}")


def extract_operations(data: Any) -> list[dict]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("operations", data.get("ops", data.get("changes")))
        if items is None and "type" in data:
            items = [data]
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("model output has no 'operations' list")
    return [i for i in items if isinstance(i, dict)]


# ------------------------------------------------------------------ prompt rendering

#: messages longer than this are cut in prompts (long tool dumps rarely carry memory-worthy facts)
MAX_MESSAGE_CHARS = 8000


def render_memory(entries: list[MemoryEntry]) -> str:
    lines = []
    for e in entries:
        when = e.updated_at or e.created_at
        meta = f" path={e.path}" if e.path else ""
        meta += f" section={e.key!r}" if e.key else ""
        meta += f" updated={when}" if when else ""
        meta += f" order={e.metadata.get('_order')}" if "_order" in e.metadata else ""
        part = e.metadata.get("_part")
        if part:
            meta += f" EXCERPT {part[0]}/{part[1]} of a long entry (read-only: do not propose new_content for it)"
        lines.append(f"- id={e.id!r}{meta}\n  {e.content.replace(chr(10), chr(10) + '  ')}")
    return "\n".join(lines) or "(empty store)"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_MESSAGE_CHARS else text[:MAX_MESSAGE_CHARS] + " …[truncated]"


def render_slices(slices: list[TranscriptSlice]) -> str:
    out = []
    for sl in slices:
        t = sl.transcript
        head = f"### Session {t.id} ({t.source_path}; format={t.format})"
        if sl.overlap:
            head += f" (continued: the first {sl.overlap} message(s) repeat the end of the previous part)"
        out.append(head)
        for m in sl.messages:
            ts = f" @{m.timestamp}" if m.timestamp else ""
            out.append(f"[{m.citation}]{ts} {m.role}: {_clip(m.content)}")
        out.append("")
    return "\n".join(out)


def render_transcripts(ctx: DreamContext) -> str:
    return render_slices([TranscriptSlice(t, list(t.messages)) for t in ctx.transcripts])


@dataclass
class Call:
    """One model call of a (possibly chunked) phase."""

    index: int
    total: int
    entries: list[MemoryEntry]
    slices: list[TranscriptSlice] | None  # None: the phase doesn't look at transcripts

    @property
    def chunked(self) -> bool:
        return self.total > 1

    def visible_ids(self) -> set[str]:
        return {e.id for e in self.entries}

    def excerpt_ids(self) -> set[str]:
        return {e.id for e in self.entries if e.metadata.get("_part")}


def build_prompt(ctx: DreamContext, phase: str, call: Call | None = None) -> str:
    if call is None:
        call = Call(0, 1, list(ctx.entries),
                    None if phase == "consolidate" else [TranscriptSlice(t, list(t.messages)) for t in ctx.transcripts])
    parts = [PHASE_TASKS[phase]]
    if call.chunked:
        seen = "the memory store" + (" and the transcripts" if call.slices is not None else "")
        parts.append(f"Input part {call.index + 1} of {call.total}: you see only part of {seen}. Propose operations "
                     "only about entries shown here and cite only messages shown here; the other parts are "
                     "processed separately and all results are merged.")
    if ctx.config.instructions:
        parts.append(f"User instructions to focus this dream (follow them): {ctx.config.instructions}")
    parts.append("## Memory store\n" + render_memory(call.entries))
    if call.slices is not None:
        parts.append("## Session transcripts (oldest first)\n" + render_slices(call.slices))
    parts.append("Return the JSON object now.")
    return "\n\n".join(parts)


def prompt_overhead(ctx: DreamContext, phase: str) -> int:
    """Tokens a call needs besides memory + transcripts (system prompt, task, instructions, headers)."""
    fixed = SYSTEM_PROMPT + PHASE_TASKS[phase] + (ctx.config.instructions or "")
    return estimate_tokens(fixed) + 160


# ------------------------------------------------------------------ planning (map)

def _clipped_transcripts(ctx: DreamContext) -> list[Transcript]:
    out = []
    for t in ctx.transcripts:
        if any(len(m.content) > MAX_MESSAGE_CHARS for m in t.messages):
            msgs = [Message(m.role, _clip(m.content), m.timestamp, m.line_no, m.source_path, m.index)
                    for m in t.messages]
            t = Transcript(t.id, msgs, t.source_path, t.format, t.metadata)
        out.append(t)
    return out


def similarity_order(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    """Store order, but with likely duplicates / same-subject notes moved next to each other so
    consolidation chunks see them together."""
    sets = [token_set(e.content) for e in entries]
    claims = claims_in_entries(entries)
    subjects = [c.subject if c else frozenset() for c in claims]
    pairs = [(i, j) for i, j in similar_pairs(sets, 0.3) if jaccard(sets[i], sets[j]) >= 0.3]
    pairs += [(i, j) for i, j in similar_pairs(subjects, 0.5) if jaccard(subjects[i], subjects[j]) >= 0.5]
    return [entries[i] for i in cluster_order(len(entries), pairs)]


def claims_in_entries(entries: list[MemoryEntry]) -> list[Any]:
    from ..text import claims_in

    out = []
    for e in entries:
        cs = claims_in(e.content)
        out.append(cs[0] if cs else None)
    return out


def plan_calls(ctx: DreamContext, phase: str) -> tuple[list[Call], list[str]]:
    """Split a phase into model calls that each fit ``max_chunk_tokens``. Returns (calls, notices)."""
    cfg = ctx.config
    overhead = prompt_overhead(ctx, phase)
    budget = cfg.max_chunk_tokens - overhead
    if budget < 256:
        raise DreamLimitError(f"max_chunk_tokens={cfg.max_chunk_tokens} is too small: the prompt scaffolding alone "
                              f"needs ~{overhead} tokens (use at least {overhead + 1024})")
    entries = list(ctx.entries)
    mem_tok = sum(entry_cost(e) for e in entries)
    notices: list[str] = []

    if phase == "consolidate":
        if mem_tok <= budget:
            return [Call(0, 1, entries, None)], notices
        chunks = chunk_memory(similarity_order(entries), budget)
        calls = [Call(i, len(chunks), ch.items, None) for i, ch in enumerate(chunks)]
        return _cap(ctx, phase, calls, notices, keep="first"), notices

    transcripts = _clipped_transcripts(ctx)
    tx_tok = sum(message_cost(m) for t in transcripts for m in t.messages) + SESSION_HEADER_TOKENS * len(transcripts)
    if mem_tok + tx_tok <= budget:
        return [Call(0, 1, entries, [TranscriptSlice(t, list(t.messages)) for t in transcripts])], notices

    if mem_tok <= budget // 2:
        # the whole store goes along with every transcript chunk
        tx_chunks = chunk_transcripts(transcripts, budget - mem_tok, cfg.chunk_overlap)
        plan = [(entries, ch.items) for ch in tx_chunks]
    else:
        # big store: each transcript chunk is paired with the entries it is relevant to
        mem_budget = budget * 2 // 5
        tx_chunks = chunk_transcripts(transcripts, budget - mem_budget, cfg.chunk_overlap)
        sets = [token_set(e.content) for e in entries]
        # Relevance = sharing a *distinctive* token (rare across the store: names, ids, numbers),
        # not "service" or "port". Entries without any distinctive token need two shared tokens.
        df = TokenIndex(sets)
        cap = max(3, len(entries) // 50)
        rare_sets = [frozenset(t for t in s if df.df(t) <= cap) for s in sets]
        rare_index = TokenIndex(rare_sets)
        plan = []
        for ch in tx_chunks:
            toks: set[str] = set()
            for sl in ch.items:
                for m in sl.messages:
                    toks |= token_set(m.content)
            rare_hits = rare_index.overlap_counts(toks)
            all_hits = df.overlap_counts(toks)
            relevant = sorted(set(rare_hits) | {i for i, c in all_hits.items() if not rare_sets[i] and c >= 2})
            counts = {i: (rare_hits.get(i, 0), all_hits.get(i, 0)) for i in relevant}
            if phase == "verify":
                if not relevant:
                    continue  # nothing in memory is about this part of the transcripts
                for mch in chunk_memory([entries[i] for i in sorted(relevant)], mem_budget):
                    plan.append((mch.items, ch.items))
            else:  # insights: memory is context only (avoid re-discovering known facts); most relevant first
                picked: list[int] = []
                used = 0
                for i in sorted(relevant, key=lambda i: (-counts[i][0], -counts[i][1], i)):
                    c = entry_cost(entries[i])
                    if used + c <= mem_budget:
                        picked.append(i)
                        used += c
                plan.append(([entries[i] for i in sorted(picked)], ch.items))
        if not plan:
            notices.append(f"{phase}: no transcript part mentions any memory entry; nothing to send to the model")
    calls = [Call(i, len(plan), mem, sl) for i, (mem, sl) in enumerate(plan)]
    return _cap(ctx, phase, calls, notices, keep="last"), notices


def _cap(ctx: DreamContext, phase: str, calls: list[Call], notices: list[str], keep: str) -> list[Call]:
    cfg = ctx.config
    if len(calls) <= cfg.max_chunks:
        return calls
    msg = (f"{phase}: the input needs {len(calls)} model calls of <= {cfg.max_chunk_tokens} tokens, more than "
           f"max_chunks={cfg.max_chunks}")
    if cfg.on_limit == "fail":
        raise DreamLimitError(msg + " (raise max_chunks / max_chunk_tokens, or reduce the input)")
    kept = calls[:cfg.max_chunks] if keep == "first" else calls[-cfg.max_chunks:]
    which = "first" if keep == "first" else "most recent"
    notices.append(msg + f"; sent the {which} {cfg.max_chunks} and covered the rest with deterministic heuristics")
    return [Call(i, len(kept), c.entries, c.slices) for i, c in enumerate(kept)]


# ------------------------------------------------------------------ validation / conversion

def _citation_index(ctx: DreamContext) -> dict[str, Message]:
    idx: dict[str, Message] = {}
    for t in ctx.transcripts:
        for m in t.messages:
            idx[m.citation] = m
            base = m.citation.replace("\\", "/").rsplit("/", 1)[-1]
            idx.setdefault(base, m)
    return idx


def _norm_ws(s: str) -> str:
    return " ".join(fold(s).split())


def validate_evidence(ctx: DreamContext, items: Any, cites: dict[str, Message]) -> tuple[list[Evidence], int]:
    out, bad = [], 0
    for it in items or []:
        if not isinstance(it, dict):
            bad += 1
            continue
        cit = str(it.get("citation", "")).strip().strip("[]")
        msg = cites.get(cit) or cites.get(cit.replace("\\", "/").rsplit("/", 1)[-1])
        if msg is None:
            bad += 1
            continue
        quote = str(it.get("quote") or "").strip().strip('"')
        if quote and _norm_ws(quote.rstrip(".…")) in _norm_ws(msg.content):
            out.append(Evidence.from_message(msg, quote, str(it.get("stance") or "supports")))
        else:
            bad += 1
            out.append(Evidence.from_message(msg, msg.content, "supports"))
    return out, bad


def convert_operations(ctx: DreamContext, phase: str, raw_ops: list[dict], origin: str,
                       call: Call | None = None, cites: dict[str, Message] | None = None) -> list[Operation]:
    cites = cites if cites is not None else _citation_index(ctx)
    allowed = {"consolidate": {"merge", "replace", "contradiction"},
               "verify": {"correct", "contradiction"},
               "insights": {"insight", "contradiction"}}[phase]
    where = f" (part {call.index + 1}/{call.total})" if call is not None and call.chunked else ""
    visible = call.visible_ids() if call is not None and call.chunked else None
    excerpts = call.excerpt_ids() if call is not None else set()
    live = {e.id: e for e in ctx.entries}
    out: list[Operation] = []
    for i, r in enumerate(raw_ops):
        typ = str(r.get("type", "")).lower().strip()
        if typ not in allowed:
            ctx.warn(f"{phase}{where}: dropped LLM operation #{i} with unsupported type {typ!r}")
            continue
        targets = [str(t) for t in (r.get("targets") or [])]
        unknown = [t for t in targets if t not in live]
        if unknown:
            ctx.warn(f"{phase}{where}: dropped LLM operation #{i}: unknown entry id(s) {unknown}")
            continue
        if visible is not None and any(t not in visible for t in targets):
            ctx.warn(f"{phase}{where}: dropped LLM operation #{i}: targets entries not shown in this part")
            continue
        new_content = r.get("new_content")
        new_content = str(new_content).strip() if new_content not in (None, "") else None
        evidence, bad = validate_evidence(ctx, r.get("evidence"), cites)
        if bad:
            ctx.warn(f"{phase}{where}: LLM operation #{i} had {bad} unverifiable evidence item(s)")
        summary = str(r.get("summary") or "").strip() or f"{typ} proposed by {origin}"
        reason = str(r.get("reason") or "").strip()
        try:
            confidence = max(0.0, min(1.0, float(r.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6
        kind = r.get("kind") or None
        remove: list[str] = []
        upsert: list[MemoryEntry] = []
        before = [live[t].content for t in targets]

        if typ in ("merge", "replace") and len(targets) < (2 if typ == "merge" else 1):
            ctx.warn(f"{phase}{where}: dropped LLM {typ} #{i}: needs more targets")
            continue
        if targets and excerpts & set(targets) and (new_content or typ in ("merge", "replace")):
            ctx.warn(f"{phase}{where}: LLM {typ} #{i} would rewrite an entry it only saw an excerpt of; "
                     "downgraded to a flag")
            typ, kind, new_content = "contradiction", kind or "conflict", None
        if typ == "correct":
            if not targets or not new_content:
                ctx.warn(f"{phase}{where}: dropped LLM correct #{i}: needs a target and new_content")
                continue
            if not evidence:
                ctx.warn(f"{phase}{where}: LLM correction #{i} had no verifiable evidence; downgraded to a flag")
                typ, kind, new_content = "contradiction", "evidence", None
        if typ == "insight":
            if not new_content:
                continue
            if not evidence:
                ctx.warn(f"{phase}{where}: dropped LLM insight #{i} without verifiable evidence")
                continue
        if typ == "contradiction" and not targets and not new_content:
            continue

        if targets:
            keep = live[targets[0]].copy()
            if new_content:
                keep.content = new_content
            remove = targets[1:] if typ != "contradiction" or new_content else []
            if new_content or remove:
                upsert = [keep]
        elif new_content:
            upsert = [MemoryEntry(id=f"insight-{stable_hash(new_content, length=8)}", content=new_content,
                                  key={"insight": "Insights"}.get(typ, "Preferences"),
                                  metadata={"origin": origin, "evidence": [e.citation for e in evidence]},
                                  source="agentdream")]
        out.append(Operation(
            id=Operation.make_id(typ, targets, [new_content or ""]), type=typ, summary=summary, reason=reason,
            targets=targets, remove=remove, upsert=upsert, before=before,
            after=[new_content] if new_content else [], evidence=evidence, kind=kind,
            confidence=confidence, phase=phase, origin=origin,
        ))
    return out


# ------------------------------------------------------------------ reduce

_TYPE_RANK = {"correct": 5, "replace": 4, "merge": 3, "contradiction": 2, "insight": 1}


def _ev_key(e: Evidence) -> tuple[str, str]:
    return (e.citation, _norm_ws(e.quote))


def _union_evidence(into: Operation, other: Operation) -> None:
    seen = {_ev_key(e) for e in into.evidence}
    for e in other.evidence:
        if _ev_key(e) not in seen:
            seen.add(_ev_key(e))
            into.evidence.append(e)


def reduce_operations(ctx: DreamContext, per_call: list[list[Operation]]) -> list[Operation]:
    """Merge the operations proposed for each chunk into one consistent, deterministic list.

    1. identical proposals (same type/kind/targets/new content) collapse into one; their
       evidence is unioned and the highest confidence kept;
    2. merges whose target sets overlap are unified into one merge (duplication is transitive);
    3. remaining store-changing operations that touch the same entry conflict: the one with the
       most verified evidence wins (then confidence, then type, then first seen); the others
       are dropped and recorded in the winner's history with their citations;
    4. flags (operations that change nothing) are kept, deduplicated.

    Output order is the order in which each surviving proposal was first seen; ids are
    content-derived, so they don't depend on how the input was chunked.
    """
    seq: list[tuple[tuple[int, int], Operation]] = [((ci, pi), op) for ci, ops in enumerate(per_call)
                                                     for pi, op in enumerate(ops)]
    if not seq:
        return []
    # 1. exact duplicates
    by_key: dict[tuple, tuple[tuple[int, int], Operation]] = {}
    for pos, op in seq:
        key = (op.type, op.kind, tuple(sorted(op.targets)), tuple(_norm_ws(a) for a in op.after))
        if key in by_key:
            kept = by_key[key][1]
            _union_evidence(kept, op)
            kept.confidence = max(kept.confidence, op.confidence)
        else:
            by_key[key] = (pos, op)
    items = sorted(by_key.values(), key=lambda x: x[0])

    # 2. unify overlapping merges
    merges = [(pos, op) for pos, op in items if op.type == "merge"]
    others = [(pos, op) for pos, op in items if op.type != "merge"]
    if len(merges) > 1:
        uf = list(range(len(merges)))

        def find(x: int) -> int:
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        owner: dict[str, int] = {}
        for k, (_pos, op) in enumerate(merges):
            for t in op.targets:
                if t in owner:
                    ra, rb = find(owner[t]), find(k)
                    if ra != rb:
                        uf[max(ra, rb)] = min(ra, rb)
                else:
                    owner[t] = k
        groups: dict[int, list[int]] = {}
        for k in range(len(merges)):
            groups.setdefault(find(k), []).append(k)
        unified = []
        for ks in groups.values():
            if len(ks) == 1:
                unified.append(merges[ks[0]])
                continue
            members = [merges[k] for k in ks]
            pos0, first = members[0]
            best = max(members, key=lambda m: (m[1].confidence, len(m[1].evidence), -m[0][0], -m[0][1]))[1]
            targets: list[str] = []
            for _p, op in members:
                targets += [t for t in op.targets if t not in targets]
            live = ctx.entry(targets[0])
            keep = (live.copy() if live is not None else first.upsert[0].copy())
            keep.content = best.after[0] if best.after else keep.content
            merged = Operation(
                id=Operation.make_id("merge", targets, [keep.content]), type="merge",
                summary=f"Merged {len(targets)} duplicate notes into {targets[0]} (combined from "
                        f"{len(members)} chunk proposals)",
                reason=best.reason, targets=targets, remove=targets[1:], upsert=[keep],
                before=[e.content for e in (ctx.entry(t) for t in targets) if e is not None],
                after=[keep.content], evidence=[], kind=None, confidence=best.confidence, phase=best.phase,
                origin=best.origin)
            for _p, op in members:
                _union_evidence(merged, op)
            unified.append((pos0, merged))
        merges = unified
    items = sorted(merges + others, key=lambda x: x[0])

    # 3. conflicts between store-changing operations on the same entry
    changing = [(pos, op) for pos, op in items if op.changes_store]
    flags = [(pos, op) for pos, op in items if not op.changes_store]

    def touched(op: Operation) -> set[str]:
        return set(op.targets) | set(op.remove) | {u.id for u in op.upsert}

    uf2 = list(range(len(changing)))

    def find2(x: int) -> int:
        while uf2[x] != x:
            uf2[x] = uf2[uf2[x]]
            x = uf2[x]
        return x

    owner2: dict[str, int] = {}
    for k, (_pos, op) in enumerate(changing):
        for t in sorted(touched(op)):
            if t in owner2:
                ra, rb = find2(owner2[t]), find2(k)
                if ra != rb:
                    uf2[max(ra, rb)] = min(ra, rb)
            else:
                owner2[t] = k
    groups2: dict[int, list[int]] = {}
    for k in range(len(changing)):
        groups2.setdefault(find2(k), []).append(k)
    survivors: list[tuple[tuple[int, int], Operation]] = []
    for ks in groups2.values():
        members = [changing[k] for k in ks]
        if len(members) == 1:
            survivors.append(members[0])
            continue
        _win_pos, win = max(members, key=lambda m: (len(m[1].evidence), m[1].confidence,
                                                   _TYPE_RANK.get(m[1].type, 0), -m[0][0], -m[0][1]))
        for (cpos, op) in members:
            if op is win:
                continue
            cites = ", ".join(e.citation for e in op.evidence) or "no evidence"
            alt = op.after[0] if op.after else op.summary
            win.history.append(f"alternative proposal from part {cpos[0] + 1} ({op.type}): {alt} [{cites}]")
        ids = sorted(set().union(*(touched(op) for _p, op in members)))
        ctx.warn(f"{win.phase}: {len(members)} chunk proposals conflict on {', '.join(ids[:4])}"
                 f"{'…' if len(ids) > 4 else ''}; kept {win.type} {win.id} (most evidence / confidence)")
        survivors.append((min(p for p, _ in members), win))
    return [op for _pos, op in sorted(survivors + flags, key=lambda x: x[0])]


# ------------------------------------------------------------------ phase runner

@dataclass
class PhaseResult:
    operations: list[Operation]
    calls: int
    failed: int
    #: some input was not analysed by the model (failed or capped calls): fill in deterministically
    partial: bool


def run_llm_phase(ctx: DreamContext, phase: str) -> PhaseResult:
    """Map: one backend call per planned chunk. Reduce: :func:`reduce_operations`.

    Raises when *every* call failed (the caller then falls back to the deterministic heuristics);
    individual chunk failures are recorded as warnings and mark the result ``partial``."""
    backend = ctx.backend
    calls, notices = plan_calls(ctx, phase)
    for n in notices:
        ctx.notices.append(n)
    capped = any("max_chunks" in n for n in notices)
    if len(calls) > 1:
        ctx.notices.append(f"{phase}: input split into {len(calls)} chunks of <= {ctx.config.max_chunk_tokens} "
                           "tokens (map-reduce)")
    ctx.stats.setdefault("chunks", {})[phase] = len(calls)
    cites = _citation_index(ctx)
    origin = backend.describe()
    per_call: list[list[Operation]] = []
    failed = 0
    last_exc: Exception | None = None
    for call in calls:
        ctx.check_cancel()
        ctx.progress(phase, call.index / max(1, len(calls)))
        try:
            raw = backend.complete(SYSTEM_PROMPT, build_prompt(ctx, phase, call), OPERATIONS_SCHEMA)
            ops = extract_operations(parse_json_loose(raw))
        except DreamCanceled:
            raise
        except Exception as exc:
            failed += 1
            last_exc = exc
            if len(calls) > 1:
                ctx.warn(f"{phase}: part {call.index + 1}/{len(calls)} failed ({type(exc).__name__}: {exc})")
            per_call.append([])
            continue
        per_call.append(convert_operations(ctx, phase, ops, origin, call, cites))
    if calls and failed == len(calls) and last_exc is not None:
        raise last_exc
    return PhaseResult(reduce_operations(ctx, per_call), len(calls), failed, partial=bool(failed) or capped)
