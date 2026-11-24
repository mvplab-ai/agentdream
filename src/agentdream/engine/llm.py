"""LLM-backed phases: prompts, tolerant JSON parsing and validation of proposed operations."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from ..context import DreamContext
from ..models import Evidence, MemoryEntry, Message, Operation, stable_hash
from ..text import fold, shorten

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
    "consolidate": ("Task: CONSOLIDATE the memory store. Find duplicates / near-duplicates (merge), stale values that a "
                    "newer entry supersedes (replace, judge recency by timestamps or later position), and stance "
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

def render_memory(entries: list[MemoryEntry]) -> str:
    lines = []
    for e in entries:
        when = e.updated_at or e.created_at
        meta = f" path={e.path}" if e.path else ""
        meta += f" section={e.key!r}" if e.key else ""
        meta += f" updated={when}" if when else ""
        meta += f" order={e.metadata.get('_order')}" if "_order" in e.metadata else ""
        lines.append(f"- id={e.id!r}{meta}\n  {e.content.replace(chr(10), chr(10) + '  ')}")
    return "\n".join(lines) or "(empty store)"


def render_transcripts(ctx: DreamContext, max_chars_per_message: int = 1500) -> str:
    out = []
    for t in ctx.transcripts:
        out.append(f"### Session {t.id} ({t.source_path}; format={t.format})")
        for m in t.messages:
            content = m.content if len(m.content) <= max_chars_per_message else \
                m.content[:max_chars_per_message] + " …[truncated]"
            ts = f" @{m.timestamp}" if m.timestamp else ""
            out.append(f"[{m.citation}]{ts} {m.role}: {content}")
        out.append("")
    return "\n".join(out)


def build_prompt(ctx: DreamContext, phase: str) -> str:
    parts = [PHASE_TASKS[phase]]
    if ctx.config.instructions:
        parts.append(f"User instructions to focus this dream (follow them): {ctx.config.instructions}")
    parts.append("## Memory store\n" + render_memory(ctx.entries))
    if phase != "consolidate":
        parts.append("## Session transcripts (oldest first)\n" + render_transcripts(ctx))
    parts.append("Return the JSON object now.")
    return "\n\n".join(parts)


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


def convert_operations(ctx: DreamContext, phase: str, raw_ops: list[dict], origin: str) -> list[Operation]:
    cites = _citation_index(ctx)
    allowed = {"consolidate": {"merge", "replace", "contradiction"},
               "verify": {"correct", "contradiction"},
               "insights": {"insight", "contradiction"}}[phase]
    out: list[Operation] = []
    for i, r in enumerate(raw_ops):
        typ = str(r.get("type", "")).lower().strip()
        if typ not in allowed:
            ctx.warn(f"{phase}: dropped LLM operation #{i} with unsupported type {typ!r}")
            continue
        targets = [str(t) for t in (r.get("targets") or [])]
        live = {e.id: e for e in ctx.entries}
        unknown = [t for t in targets if t not in live]
        if unknown:
            ctx.warn(f"{phase}: dropped LLM operation #{i}: unknown entry id(s) {unknown}")
            continue
        new_content = r.get("new_content")
        new_content = str(new_content).strip() if new_content not in (None, "") else None
        evidence, bad = validate_evidence(ctx, r.get("evidence"), cites)
        if bad:
            ctx.warn(f"{phase}: LLM operation #{i} had {bad} unverifiable evidence item(s)")
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
            ctx.warn(f"{phase}: dropped LLM {typ} #{i}: needs more targets")
            continue
        if typ == "correct":
            if not targets or not new_content:
                ctx.warn(f"{phase}: dropped LLM correct #{i}: needs a target and new_content")
                continue
            if not evidence:
                ctx.warn(f"{phase}: LLM correction #{i} had no verifiable evidence; downgraded to a flag")
                typ, kind, new_content = "contradiction", "evidence", None
        if typ == "insight":
            if not new_content:
                continue
            if not evidence:
                ctx.warn(f"{phase}: dropped LLM insight #{i} without verifiable evidence")
                continue
        if typ == "contradiction" and not targets and not new_content:
            continue

        if targets:
            keep = live[targets[0]].copy()
            if new_content:
                keep.content = new_content
            remove = targets[1:]
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


def run_llm_phase(ctx: DreamContext, phase: str) -> list[Operation]:
    """Ask the backend for operations. Raises on backend/parse failure (caller falls back)."""
    backend = ctx.backend
    prompt = build_prompt(ctx, phase)
    raw = backend.complete(SYSTEM_PROMPT, prompt, OPERATIONS_SCHEMA)
    data = parse_json_loose(raw)
    ops = extract_operations(data)
    return convert_operations(ctx, phase, ops, origin=backend.describe())
