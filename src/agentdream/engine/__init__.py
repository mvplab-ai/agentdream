"""Analysis engines: deterministic heuristics and LLM-backed phases with fallback."""

from __future__ import annotations

from ..context import DreamCanceled, DreamContext
from . import deterministic
from .llm import parse_json_loose, run_llm_phase

HEURISTICS = {
    "consolidate": deterministic.consolidate,
    "verify": deterministic.verify,
    "insights": deterministic.insights,
}


def run_analysis_phase(ctx: DreamContext, phase: str) -> None:
    """Run one analysis phase with the configured backend, falling back to heuristics."""
    backend = ctx.backend
    if backend is None or not getattr(backend, "is_llm", False):
        HEURISTICS[phase](ctx)
        return
    try:
        ops = run_llm_phase(ctx, phase)
    except DreamCanceled:
        raise
    except Exception as exc:  # backend down, bad JSON, ... -> deterministic fallback
        ctx.warn(f"{phase}: {backend.describe()} failed ({type(exc).__name__}: {exc}); "
                 "fell back to deterministic heuristics")
        HEURISTICS[phase](ctx)
        return
    for op in ops:
        ctx.add_op(op)


__all__ = ["HEURISTICS", "parse_json_loose", "run_analysis_phase"]
