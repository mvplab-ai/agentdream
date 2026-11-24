"""Analysis engines: deterministic heuristics and LLM-backed phases with fallback."""

from __future__ import annotations

from ..context import DreamCanceled, DreamContext, DreamLimitError
from . import deterministic
from .llm import parse_json_loose, run_llm_phase

HEURISTICS = {
    "consolidate": deterministic.consolidate,
    "verify": deterministic.verify,
    "insights": deterministic.insights,
}


def run_analysis_phase(ctx: DreamContext, phase: str) -> None:
    """Run one analysis phase with the configured backend, falling back to heuristics.

    * deterministic backend: heuristics only;
    * LLM backend: map-reduce over chunks; if every call fails (backend down, unparseable
      output) the heuristics run instead; if only some chunks failed or were capped by
      ``max_chunks``, the LLM operations are applied first and the heuristics then fill in
      on the updated store;
    * limit errors (``on_limit="fail"``) and cancellation propagate.
    """
    backend = ctx.backend
    if backend is None or not getattr(backend, "is_llm", False):
        HEURISTICS[phase](ctx)
        return
    try:
        result = run_llm_phase(ctx, phase)
    except (DreamCanceled, DreamLimitError):
        raise
    except Exception as exc:  # backend down, bad JSON, ... -> deterministic fallback
        ctx.warn(f"{phase}: {backend.describe()} failed ({type(exc).__name__}: {exc}); "
                 "fell back to deterministic heuristics")
        HEURISTICS[phase](ctx)
        return
    for op in result.operations:
        ctx.add_op(op)
    if result.partial:
        ctx.warn(f"{phase}: {result.failed} of {result.calls} model call(s) failed or input was capped; "
                 "deterministic heuristics filled in for the rest")
        HEURISTICS[phase](ctx)


__all__ = ["HEURISTICS", "parse_json_loose", "run_analysis_phase"]
