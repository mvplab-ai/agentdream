"""Review: accept (all / partial), discard, or restore a dream.

``accept`` is crash-safe: the new store is written next to the target, the target is moved
aside, the new store renamed in, and only then the old one deleted. Any failure on the way
puts the original back (and removes the half-written copy), so the target is either the
original or the complete new store. A backup is taken first (unless ``backup=False``) and
can be put back with :func:`restore` / ``agentdream restore JOB``. Concurrent accepts onto one
target are serialised by a lock in the jobs dir; the loser then sees the store changed.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .adapters import get_adapter
from .adapters.stancelog import strip_prefix
from .context import DreamConfig
from .jobs import DreamJob
from .models import Operation
from .pipeline import make_redactor, materialize
from .util import copy_path, hash_path, now_iso, remove_path

#: operation ids: a list, or a comma-separated string ("op1,op2"); unique prefixes are accepted
Selection = Iterable[str] | str | None


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
    out: list[str] = []
    for item in v:
        out.extend(x.strip() for x in str(item).split(",") if x.strip())
    return out


def _match_ids(ops: list[Operation], wanted: list[str]) -> list[str]:
    ids = [o.id for o in ops]
    out: list[str] = []
    for w in wanted:
        if w in ids:
            hit = w
        else:
            pref = [i for i in ids if i.startswith(w)]
            if len(pref) != 1:
                raise ReviewError(f"unknown or ambiguous operation id {w!r}"
                                  + (f" (ambiguous: {', '.join(pref)})" if pref else ""))
            hit = pref[0]
        if hit not in out:  # "--only a,a" applies a once
            out.append(hit)
    return out


def select_operations(ops: list[Operation], only: Selection = None, exclude: Selection = None
                      ) -> tuple[list[Operation], list[str]]:
    """Return (selected ops in original order, warnings). Dependencies are pulled in for --only and
    dependents are dropped for --exclude so the result is always consistent.

    ``only=None`` means "all"; an explicitly empty ``only`` (``""``, ``[]``) is an error rather than
    silently applying everything. (``accept`` refuses a selection that leaves nothing to apply.)"""
    warnings: list[str] = []
    by_id = {o.id: o for o in ops}
    only_list = _split(only)
    if only is not None and not only_list:
        raise ReviewError("--only was given but contains no operation ids (omit it to apply everything)")
    only_ids, excl_ids = _match_ids(ops, only_list), _match_ids(ops, _split(exclude))
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


def _move(src: Path, dst: Path) -> None:
    if src.is_dir() and not src.is_symlink():
        src.rename(dst)
    else:
        os.replace(src, dst)


def _kind(p: Path) -> str:
    return "a directory" if p.is_dir() else "a file"


def _check_kind(target: Path, like: Path) -> None:
    if target.exists() and like.exists() and target.is_dir() != like.is_dir():
        raise ReviewError(f"target {target} is {_kind(target)} but the store is {_kind(like)}; refusing to replace "
                          "one with the other")


def _swap_in(new: Path, target: Path, sidecars: list[Path]) -> None:
    """Atomically-ish replace ``target`` with the already written ``new``; on any failure restore
    the original target and sidecars and remove ``new`` before re-raising."""
    old = target.with_name(f".{target.name}.agentdream-old")
    remove_path(old)
    moved_target = False
    moved_sides: list[tuple[Path, Path]] = []
    try:
        if target.exists() or target.is_symlink():
            _move(target, old)
            moved_target = True
        for side in sidecars:
            if side.exists():
                aside = side.with_name(side.name + ".agentdream-old")
                remove_path(aside)
                _move(side, aside)
                moved_sides.append((side, aside))
        _move(new, target)
    except BaseException:
        with contextlib.suppress(OSError):
            if moved_target and not (target.exists() or target.is_symlink()):
                _move(old, target)
        for side, aside in moved_sides:
            with contextlib.suppress(OSError):
                if not side.exists():
                    _move(aside, side)
        with contextlib.suppress(OSError):
            remove_path(new)
        raise
    remove_path(old)
    for _side, aside in moved_sides:
        with contextlib.suppress(OSError):
            remove_path(aside)


def _cleanup_partial(new: Path) -> None:
    for p in new.parent.glob(glob.escape(new.name) + "*"):
        with contextlib.suppress(OSError):
            remove_path(p)


def _target_lock(job: DreamJob, target: Path):  # type: ignore[no-untyped-def]
    from .claude_hook import FileLock

    key = hashlib.sha256(str(target.absolute()).encode()).hexdigest()[:16]
    return FileLock(job.jobs_dir / ".locks" / f"accept-{key}.lock")


def accept(job: DreamJob | str, only: Selection = None, exclude: Selection = None,
           to: str | os.PathLike[str] | None = None, backup: bool = True, force: bool = False,
           jobs_dir: str | os.PathLike[str] | None = None) -> AcceptResult:
    if isinstance(job, str):
        job = DreamJob.load(job, jobs_dir)
    if job.status != "completed":
        raise ReviewError(f"job {job.id} is {job.status}; only completed dreams can be accepted")
    if job.review == "discarded":
        raise ReviewError(f"job {job.id} was discarded")
    if job.review in ("accepted", "partially_accepted") and not force:
        raise ReviewError(f"job {job.id} was already {job.review} (use force to apply again)")
    cfg: DreamConfig = job.config
    try:
        ops = job.operations
        original = job.snapshot()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReviewError(f"job {job.id}: its operations.json / snapshot.json is missing or corrupt ({exc}); "
                          "re-run the dream") from exc
    selected, warnings = select_operations(ops, only, exclude)
    if ops and not selected:
        raise ReviewError("no operations selected by --only/--exclude; nothing to apply (to drop the whole dream "
                          "use `agentdream discard JOB`)")
    final = materialize(original, selected, make_redactor(cfg))

    src = Path(strip_prefix(str(cfg.memory)))
    target = Path(strip_prefix(to)) if to else src
    if not src.exists():
        raise ReviewError(f"original memory store {src} no longer exists (needed as layout template)")
    _check_kind(target, src)
    adapter = get_adapter(str(cfg.memory), cfg.memory_format)  # keeps a stancelog: prefix

    lock = _target_lock(job, target)
    if not lock.acquire(timeout=120):
        raise ReviewError(f"another accept is writing to {target}; try again")
    try:
        if not force and job.input_hash and hash_path(src) != job.input_hash:
            raise ReviewError(f"memory store {src} changed since the dream ran; re-run the dream, or pass --force "
                              "(force=True) to apply it anyway")
        backup_path: str | None = None
        if (target.exists() or target.is_symlink()) and backup:
            stamp = now_iso().replace(":", "")
            bdir = job.dir / "backups" / stamp
            n = 2
            while bdir.exists():  # a second accept (after restore) within the same second
                bdir = job.dir / "backups" / f"{stamp}-{n}"
                n += 1
            try:
                bdir.mkdir(parents=True, exist_ok=True)
                dest = bdir / target.name
                copy_path(target, dest)
                for side in adapter.sidecars(target):
                    if side.is_file():
                        copy_path(side, bdir / side.name)
            except OSError as exc:
                raise ReviewError(f"could not back up {target} ({exc}); nothing was changed") from exc
            backup_path = str(dest)

        # write next to the target, then swap it in; failures leave the original in place
        tmp = target.with_name(f".{target.name}.agentdream-new")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _cleanup_partial(tmp)
            # the adapter refuses to write onto its own input, so write to tmp (a different path)
            adapter.write(final, tmp)
            _swap_in(tmp, target, adapter.sidecars(target))
        except Exception as exc:
            with contextlib.suppress(OSError):
                _cleanup_partial(tmp)
            raise ReviewError(f"accept failed ({type(exc).__name__}: {exc}); the original store {target} is intact"
                              + (f" (backup: {backup_path})" if backup_path else "")) from exc
    finally:
        lock.release()

    applied = [o.id for o in selected]
    skipped = [o.id for o in ops if o.id not in applied]
    job.review = "accepted" if not skipped else "partially_accepted"
    job.accepted_ops = applied
    job.applied_to = str(target)
    job.backup_path = backup_path
    job.save()
    return AcceptResult(job.id, applied, skipped, str(target), backup_path, warnings)


def restore(job: DreamJob | str, to: str | os.PathLike[str] | None = None,
            jobs_dir: str | os.PathLike[str] | None = None) -> Path:
    """Put the backup taken by ``accept`` back in place (or at ``to``). Returns the restored path.

    Restoring over the accepted target marks the job ``review="restored"``; since the store then
    matches the dream's input again, the dream can be accepted again."""
    if isinstance(job, str):
        job = DreamJob.load(job, jobs_dir)
    if not job.backup_path:
        raise ReviewError(f"job {job.id} has no backup to restore (it was never accepted, or was accepted with "
                          "--no-backup)")
    backup = Path(job.backup_path)
    if not (backup.exists() or backup.is_symlink()):
        raise ReviewError(f"backup {backup} is missing; nothing to restore")
    target = Path(strip_prefix(str(to))) if to else Path(job.applied_to or strip_prefix(str(job.config.memory)))
    _check_kind(target, backup)
    tmp = target.with_name(f".{target.name}.agentdream-new")
    sides = [p for p in backup.parent.iterdir() if p.name != backup.name]
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _cleanup_partial(tmp)
        copy_path(backup, tmp)
        _swap_in(tmp, target, [target.with_name(s.name) for s in sides])
        for s in sides:
            copy_path(s, target.with_name(s.name))
    except Exception as exc:
        with contextlib.suppress(OSError):
            _cleanup_partial(tmp)
        raise ReviewError(f"restore failed ({type(exc).__name__}: {exc}); {target} is unchanged") from exc
    if not to:
        job.review = "restored"
        job.save()
    return target


def discard(job: DreamJob | str, jobs_dir: str | os.PathLike[str] | None = None) -> DreamJob:
    if isinstance(job, str):
        job = DreamJob.load(job, jobs_dir)
    if job.review in ("accepted", "partially_accepted"):
        raise ReviewError(f"job {job.id} was already {job.review}; restore from the backup instead "
                          f"(agentdream restore {job.id})")
    if not job.done:
        job.cancel()
    store = job.out_dir / "store"
    if store.exists():
        shutil.rmtree(store)
    job.review = "discarded"
    job.store_path = None
    job.save()
    return job
