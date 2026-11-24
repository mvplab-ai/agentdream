"""Review: accept (all / partial) or discard a completed dream."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .adapters import get_adapter
from .context import DreamConfig
from .jobs import DreamJob
from .models import Operation
from .pipeline import make_redactor, materialize
from .util import copy_path, hash_path, now_iso, remove_path


class ReviewError(Exception):
    pass


@dataclass
class AcceptResult:
    job_id: str
    applied_ops: list[str]
    skipped_ops: list[str]
    target: str
    backup: str | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _split(v: Iterable[str] | str | None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    out = []
    for item in v:
        out.extend(x.strip() for x in str(item).split(",") if x.strip())
    return out


def _match_ids(ops: list[Operation], wanted: list[str]) -> list[str]:
    ids = [o.id for o in ops]
    out = []
    for w in wanted:
        if w in ids:
            out.append(w)
            continue
        pref = [i for i in ids if i.startswith(w)]
        if len(pref) == 1:
            out.append(pref[0])
        else:
            raise ReviewError(f"unknown or ambiguous operation id {w!r}")
    return out


def select_operations(ops: list[Operation], only=None, exclude=None) -> tuple[list[Operation], list[str]]:
    """Return (selected ops in original order, warnings). Dependencies are pulled in for --only and
    dependents are dropped for --exclude so the result is always consistent."""
    warnings: list[str] = []
    by_id = {o.id: o for o in ops}
    only_ids, excl_ids = _match_ids(ops, _split(only)), _match_ids(ops, _split(exclude))
    if only_ids:
        chosen: set[str] = set()
        stack = list(only_ids)
        while stack:
            cur = stack.pop()
            if cur in chosen:
                continue
            chosen.add(cur)
            for d in by_id[cur].depends_on:
                if d in by_id and d not in chosen:
                    if d not in only_ids:
                        warnings.append(f"{cur} depends on {d}; including it")
                    stack.append(d)
    else:
        chosen = set(by_id)
    if excl_ids:
        removed = set(excl_ids)
        changed = True
        while changed:
            changed = False
            for o in ops:
                if o.id in chosen and o.id not in removed and any(d in removed for d in o.depends_on):
                    removed.add(o.id)
                    warnings.append(f"{o.id} depends on excluded {', '.join(d for d in o.depends_on if d in removed)}; "
                                    "excluding it too")
                    changed = True
        chosen -= removed
    return [o for o in ops if o.id in chosen], warnings


def accept(job: DreamJob | str, only=None, exclude=None, to: str | os.PathLike | None = None,
           backup: bool = True, force: bool = False, jobs_dir=None) -> AcceptResult:
    if isinstance(job, str):
        job = DreamJob.load(job, jobs_dir)
    if job.status != "completed":
        raise ReviewError(f"job {job.id} is {job.status}; only completed dreams can be accepted")
    if job.review == "discarded":
        raise ReviewError(f"job {job.id} was discarded")
    if job.review in ("accepted", "partially_accepted") and not force:
        raise ReviewError(f"job {job.id} was already {job.review} (use force to apply again)")
    cfg: DreamConfig = job.config
    ops = job.operations
    selected, warnings = select_operations(ops, only, exclude)
    original = job.snapshot()
    final = materialize(original, selected, make_redactor(cfg))

    src = Path(str(cfg.memory))
    target = Path(to) if to else src
    if not src.exists():
        raise ReviewError(f"original memory store {src} no longer exists (needed as layout template)")
    if not force and job.input_hash and hash_path(src) != job.input_hash:
        raise ReviewError(f"memory store {src} changed since the dream ran; re-run the dream or use force")
    adapter = get_adapter(str(src), cfg.memory_format)

    backup_path: str | None = None
    if target.exists() and backup:
        bdir = job.dir / "backups" / now_iso().replace(":", "")
        bdir.mkdir(parents=True, exist_ok=True)
        dest = bdir / target.name
        copy_path(target, dest)
        backup_path = str(dest)

    # write next to the target, then swap in atomically-ish (rename)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.agentdream-new")
    old = target.with_name(f".{target.name}.agentdream-old")
    remove_path(tmp)
    remove_path(old)
    # the adapter refuses to write onto its own input, so write to tmp (a different path)
    adapter.write(final, tmp)
    if target.exists():
        os.replace(target, old) if target.is_file() else target.rename(old)
    os.replace(tmp, target) if tmp.is_file() else tmp.rename(target)
    remove_path(old)

    applied = [o.id for o in selected]
    skipped = [o.id for o in ops if o.id not in applied]
    job.review = "accepted" if not skipped else "partially_accepted"
    job.accepted_ops = applied
    job.applied_to = str(target)
    job.backup_path = backup_path
    job.save()
    return AcceptResult(job.id, applied, skipped, str(target), backup_path, warnings)


def discard(job: DreamJob | str, jobs_dir=None) -> DreamJob:
    if isinstance(job, str):
        job = DreamJob.load(job, jobs_dir)
    if job.review in ("accepted", "partially_accepted"):
        raise ReviewError(f"job {job.id} was already {job.review}; restore from the backup instead")
    if not job.done:
        job.cancel()
    store = job.out_dir / "store"
    if store.exists():
        shutil.rmtree(store)
    job.review = "discarded"
    job.store_path = None
    job.save()
    return job
