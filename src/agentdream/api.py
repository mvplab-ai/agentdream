"""High-level Python API."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .backends import LLMBackend
from .context import DreamConfig
from .jobs import DreamJob


def _sessions_list(sessions: Iterable[str | os.PathLike] | str | os.PathLike | None) -> list[str]:
    if sessions is None:
        return []
    if isinstance(sessions, (str, os.PathLike)):
        return [str(sessions)]
    return [str(s) for s in sessions]


def make_config(memory: str | os.PathLike, sessions=None, backend: str | LLMBackend = "deterministic",
                **options: Any) -> DreamConfig:
    known = set(DreamConfig.__dataclass_fields__)
    unknown = {k: v for k, v in options.items() if k not in known}
    if unknown:
        raise TypeError(f"unknown dream option(s): {', '.join(sorted(unknown))}")
    name = backend.name if isinstance(backend, LLMBackend) else (backend or "deterministic")
    opts = dict(options)
    for k in ("out", "jobs_dir"):
        if opts.get(k) is not None:
            opts[k] = str(Path(opts[k]).absolute())
    mem = str(memory)
    if mem.startswith("stancelog:"):  # the shell doesn't expand ~ after the prefix
        mem = "stancelog:" + str(Path(mem[len("stancelog:"):]).expanduser().absolute())
    else:
        mem = str(Path(mem).absolute())
    return DreamConfig(memory=mem, sessions=_sessions_list(sessions), backend=name, **opts)


def dream(memory: str | os.PathLike, sessions=None, backend: str | LLMBackend = "deterministic", *,
          wait: bool = True, apply: bool = False, pipeline=None, **options: Any) -> DreamJob:
    """Run a dream. Dry-run by default: produces a new store + report in a staging dir.

    ``dream(memory="./mem", sessions=["./logs"]).report`` -> report dict.
    With ``wait=False`` the job runs in a background thread (``job.status``, ``job.cancel()``,
    ``job.wait()``). ``apply=True`` accepts all operations afterwards (with a backup).
    """
    cfg = make_config(memory, sessions, backend, **options)
    job = DreamJob(cfg, pipeline=pipeline, backend=backend if isinstance(backend, LLMBackend) else None)
    if not wait:
        if apply:
            raise ValueError("apply=True requires wait=True")
        return job.start()
    job.run()
    if apply and job.status == "completed":
        from .review import accept

        accept(job)
    return job


def dream_async(memory, sessions=None, backend="deterministic", **options) -> DreamJob:
    return dream(memory, sessions, backend, wait=False, **options)
