"""The dream pipeline. Each phase is a ``fn(ctx)`` and can be replaced:

    pipe = Pipeline.default()
    pipe.replace("insights", my_insights)
    pipe.insert_after("verify", "lint", my_lint_phase)
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path

from .adapters import get_adapter
from .backends import get_backend
from .context import DreamCanceled, DreamConfig, DreamContext, DreamLimitError
from .engine import run_analysis_phase
from .loaders import load_transcripts
from .models import MemoryEntry, Operation, apply_operations
from .redact import Redactor
from .report import build_report, render_markdown, store_diff
from .text import token_set
from .util import atomic_write_json, atomic_write_text, estimate_tokens, now_iso

Phase = Callable[[DreamContext], None]


# ----------------------------------------------------------------- phases

def phase_load(ctx: DreamContext) -> None:
    cfg = ctx.config
    if not cfg.memory:
        raise ValueError("no memory store given")
    ctx.adapter = get_adapter(cfg.memory, cfg.memory_format)
    ctx.original_entries = ctx.adapter.load()
    transcripts, notices = load_transcripts(cfg.sessions, cfg.transcript_format, max_sessions=None,
                                            min_sessions=cfg.min_sessions)
    ctx.transcripts = transcripts
    ctx.notices.extend(notices)
    ctx.stats["sessions_found"] = len(transcripts)


def make_redactor(cfg: DreamConfig) -> Redactor:
    return Redactor(emails=cfg.redact_emails, enabled=cfg.redact, extra_patterns=cfg.redact_patterns,
                    entropy=cfg.redact_entropy, entropy_threshold=cfg.entropy_threshold,
                    entropy_min_length=cfg.entropy_min_length, cards=cfg.redact_cards)


def phase_redact(ctx: DreamContext) -> None:
    red = make_redactor(ctx.config)
    ctx.entries = []
    for e in ctx.original_entries:
        r = e.copy()
        r.content = red.redact(e.content)
        ctx.entries.append(r)
    for t in ctx.transcripts:
        for m in t.messages:
            m.content = red.redact(m.content)
    ctx.redaction_counts = dict(red.report.counts)
    if red.report.total:
        ctx.notices.append(f"redacted {red.report.total} secret(s) before analysis")
    ctx.redactor = red


def phase_limits(ctx: DreamContext) -> None:
    cfg = ctx.config
    fail = cfg.on_limit == "fail"
    if len(ctx.transcripts) > cfg.max_sessions:
        msg = f"{len(ctx.transcripts)} sessions exceed max_sessions={cfg.max_sessions}"
        if fail:
            raise DreamLimitError(msg)
        ctx.transcripts = ctx.transcripts[-cfg.max_sessions:]
        ctx.notices.append(msg + f"; kept the most recent {cfg.max_sessions}")
    for t in ctx.transcripts:
        size = t.text_size()
        if size <= cfg.max_bytes_per_transcript:
            continue
        msg = f"transcript {t.source_path} is {size} bytes (> max_bytes_per_transcript={cfg.max_bytes_per_transcript})"
        if fail:
            raise DreamLimitError(msg)
        kept, total = [], 0
        for m in reversed(t.messages):
            b = len(m.content.encode("utf-8"))
            if total + b > cfg.max_bytes_per_transcript:
                break
            kept.append(m)
            total += b
        dropped = len(t.messages) - len(kept)
        t.messages = list(reversed(kept))
        t.metadata["truncated_messages"] = dropped
        ctx.notices.append(msg + f"; truncated to the most recent {len(kept)} message(s) ({dropped} dropped)")

    # running totals (one pass), so truncation is linear even with thousands of sessions
    mem_tok = sum(estimate_tokens(e.content) for e in ctx.entries)
    per_t = [sum(estimate_tokens(m.content) for m in t.messages) for t in ctx.transcripts]
    tok = mem_tok + sum(per_t)
    ctx.stats["estimated_tokens"] = tok
    if tok > cfg.max_total_tokens:
        msg = f"estimated {tok} tokens exceed max_total_tokens={cfg.max_total_tokens}"
        if fail:
            raise DreamLimitError(msg)
        dropped = 0
        while len(ctx.transcripts) > 1 and tok > cfg.max_total_tokens:
            ctx.transcripts.pop(0)
            tok -= per_t.pop(0)
            dropped += 1
        trimmed = 0
        if tok > cfg.max_total_tokens and ctx.transcripts:
            t = ctx.transcripts[0]
            while len(t.messages) - trimmed > 1 and tok > cfg.max_total_tokens:
                tok -= estimate_tokens(t.messages[trimmed].content)
                trimmed += 1
            if trimmed:
                t.messages = t.messages[trimmed:]
                t.metadata["truncated_messages"] = t.metadata.get("truncated_messages", 0) + trimmed
        if tok > cfg.max_total_tokens:
            raise DreamLimitError(msg + " even after truncation (the memory store alone is too large; raise "
                                        "max_total_tokens)")
        ctx.notices.append(msg + f"; dropped {dropped} oldest session(s)"
                           + (f" and {trimmed} oldest message(s) of the next one" if trimmed else "") + " to fit")
        ctx.stats["estimated_tokens"] = tok
    if len(ctx.transcripts) < cfg.min_sessions:
        raise DreamLimitError(f"need at least {cfg.min_sessions} session(s) after limits, have {len(ctx.transcripts)}")
    ctx._sentences = None


_PHASE_WORDS = {"consolidate": r"merg\w*|dedup\w*|consolidat\w*|duplicates?",
                "verify": r"verif\w*|correct\w*|fact.?check\w*",
                "insights": r"insights?|patterns?"}


def apply_instructions(ctx: DreamContext) -> None:
    """Deterministic interpretation of free-text instructions: phase toggles + focus keywords."""
    text = (ctx.config.instructions or "").lower()
    if not text:
        return
    for phase, words in _PHASE_WORDS.items():
        if re.search(rf"\b(?:skip|no|don'?t|do not|without|ignore)\s+(?:\w+\s+)?(?:{words})", text):
            ctx.skip_phases.add(phase)
    m = re.search(r"\bonly\s+(?:do\s+|run\s+)?(\w+)", text)
    if m:
        for phase, words in _PHASE_WORDS.items():
            if re.fullmatch(words, m.group(1)):
                ctx.skip_phases |= set(_PHASE_WORDS) - {phase}


def _analysis(name: str) -> Phase:
    def run(ctx: DreamContext) -> None:
        if name in ctx.skip_phases:
            ctx.notices.append(f"phase '{name}' skipped per instructions")
            return
        run_analysis_phase(ctx, name)
    run.__name__ = f"phase_{name}"
    return run


def _focus_notice(ctx: DreamContext) -> None:
    if not ctx.config.instructions or not ctx.ops:
        return
    kws = token_set(ctx.config.instructions) - token_set("focus only on about dream memory notes skip insights "
                                                         "merge verify correct please")
    if not kws:
        return

    def hit(op: Operation) -> bool:
        blob = " ".join(op.before + op.after + [op.summary])
        return bool(kws & token_set(blob))

    # stable: focused operations first, dependencies preserved because sorting is stable within groups
    focused = [o for o in ctx.ops if hit(o)]
    if focused:
        ctx.notices.append(f"{len(focused)} operation(s) match the instruction focus ({', '.join(sorted(kws))})")


def materialize(original: list[MemoryEntry], ops: list[Operation], redactor: Redactor) -> list[MemoryEntry]:
    """Apply ops to the ORIGINAL (unredacted) entries. Upserts that only carry the redacted
    form of an original entry's content get the original content back, so secrets that were
    redacted for analysis are not destroyed in untouched text."""
    redacted = {e.id: redactor.redact(e.content) for e in original}
    orig = {e.id: e.content for e in original}
    out = apply_operations(original, ops)
    for e in out:
        if e.id in orig and e.content == redacted[e.id]:
            e.content = orig[e.id]
    return out


def phase_write(ctx: DreamContext) -> None:
    out = ctx.output_dir
    if out is None:
        raise ValueError("no output directory")
    out.mkdir(parents=True, exist_ok=True)
    _focus_notice(ctx)
    red = make_redactor(ctx.config)
    final = materialize(ctx.original_entries, ctx.ops, red)
    ctx.final_entries = final
    red.report.counts.clear()
    _sanitize_report_text(ctx, red)
    _warn_rewritten_secrets(ctx, red)
    store_path = ctx.adapter.output_path(out / "store")
    if (out / "store").exists():
        import shutil

        shutil.rmtree(out / "store")
    (out / "store").mkdir(parents=True, exist_ok=True)
    if store_path == out / "store":
        (out / "store").rmdir()
    ctx.adapter.write(final, store_path)
    ctx.store_path = store_path
    input_path = Path(str(ctx.config.memory))
    # diffs use redacted renderings so reports never echo secrets
    diff = store_diff(input_path, store_path, ctx.original_entries, final)
    diff = red.redact(diff)
    sessions = [{"id": t.id, "path": t.source_path, "format": t.format, "messages": len(t.messages)}
                for t in ctx.transcripts]
    shown_config = ctx.config.to_dict()
    if shown_config.get("instructions"):
        shown_config["instructions"] = red.redact(shown_config["instructions"])
    report = build_report(
        job_id=ctx.stats.get("job_id", "adhoc"), config=shown_config, ops=ctx.ops,
        original=ctx.original_entries, final=final, transcripts=sessions, warnings=ctx.warnings,
        notices=ctx.notices, redactions=ctx.redaction_counts, diff=diff,
        backend=ctx.backend.describe() if ctx.backend else "deterministic", store_path=str(store_path),
        created_at=ctx.stats.get("created_at", now_iso()),
    )
    ctx.stats["report"] = report
    atomic_write_json(out / "report.json", report)
    atomic_write_text(out / "report.md", render_markdown(report, ctx.ops))


def _sanitize_report_text(ctx: DreamContext, red: Redactor) -> None:
    """Defense in depth: everything a report shows (evidence quotes, before/after, history,
    summaries) goes through the redactor again, so a custom phase that built evidence from
    unredacted text still can't leak a secret into report.md / report.json."""
    if not ctx.config.redact:
        return
    for op in ctx.ops:
        op.summary, op.reason = red.redact(op.summary), red.redact(op.reason)
        op.before = [red.redact(x) for x in op.before]
        op.after = [red.redact(x) for x in op.after]
        op.history = [red.redact(x) for x in op.history]
        for ev in op.evidence:
            ev.quote = red.redact(ev.quote)
    for k, v in red.report.counts.items():  # count anything the second pass caught
        ctx.redaction_counts[k] = ctx.redaction_counts.get(k, 0) + v
    red.report.counts.clear()


def _warn_rewritten_secrets(ctx: DreamContext, red: Redactor) -> None:
    """An operation that rewrites an entry containing a secret writes the ``[REDACTED:...]``
    placeholder (the model never saw the secret). Say so before anyone accepts it."""
    if not ctx.config.redact:
        return
    with_secret = {e.id for e in ctx.original_entries if "[REDACTED:" in red.redact(e.content)}
    red.report.counts.clear()
    for op in ctx.ops:
        hit = [u.id for u in op.upsert if u.id in with_secret and "[REDACTED:" in u.content]
        if hit:
            ctx.warn(f"{op.id} rewrites {', '.join(hit)}, which contained a redacted secret; the new text keeps "
                     "the [REDACTED:...] placeholder - restore the value by hand or exclude this operation")


class Pipeline:
    def __init__(self, phases: list[tuple[str, Phase]]):
        self.phases = list(phases)

    @classmethod
    def default(cls) -> Pipeline:
        return cls([
            ("load", phase_load),
            ("redact", phase_redact),
            ("limits", phase_limits),
            ("consolidate", _analysis("consolidate")),
            ("verify", _analysis("verify")),
            ("insights", _analysis("insights")),
            ("write", phase_write),
        ])

    def names(self) -> list[str]:
        return [n for n, _ in self.phases]

    def replace(self, name: str, fn: Phase) -> Pipeline:
        self.phases = [(n, fn if n == name else f) for n, f in self.phases]
        return self

    def remove(self, name: str) -> Pipeline:
        self.phases = [(n, f) for n, f in self.phases if n != name]
        return self

    def insert_after(self, after: str, name: str, fn: Phase) -> Pipeline:
        i = self.names().index(after)
        self.phases.insert(i + 1, (name, fn))
        return self

    def run(self, ctx: DreamContext, on_phase: Callable[[str, int, int], None] | None = None) -> DreamContext:
        apply_instructions(ctx)
        total = len(self.phases)
        for i, (name, fn) in enumerate(self.phases):
            ctx.check_cancel()
            if on_phase:
                on_phase(name, i, total)
            fn(ctx)
        ctx.check_cancel()
        return ctx


def run_dream(config: DreamConfig, output_dir: str | Path, *, pipeline: Pipeline | None = None,
              backend=None, progress=None, cancel_event: threading.Event | None = None,
              on_phase=None, job_id: str = "adhoc", created_at: str | None = None) -> DreamContext:
    be = backend or get_backend(config.backend, config.model, **(config.backend_options or {}))
    ctx = DreamContext(config, be, progress=progress, cancel_event=cancel_event)
    ctx.output_dir = Path(output_dir)
    ctx.stats["job_id"] = job_id
    ctx.stats["created_at"] = created_at or now_iso()
    (pipeline or Pipeline.default()).run(ctx, on_phase=on_phase)
    return ctx


__all__ = [
    "DreamCanceled",
    "DreamLimitError",
    "Pipeline",
    "make_redactor",
    "materialize",
    "phase_limits",
    "phase_load",
    "phase_redact",
    "phase_write",
    "run_dream",
]
