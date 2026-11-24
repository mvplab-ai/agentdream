"""Dream jobs: async execution, status lifecycle, persisted state.

State lives in ``<jobs_dir>/<job_id>/job.json`` (default jobs dir:
``$AGENTDREAM_HOME/jobs`` or ``./.agentdream/jobs``) so ``status`` survives
process restarts. Statuses: pending -> running -> completed | failed | canceled.
Review state (separate): pending | accepted | partially_accepted | discarded.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .context import DreamCanceled, DreamConfig
from .models import MemoryEntry, Operation
from .util import atomic_write_json, hash_path, now_iso

STATUSES = ("pending", "running", "completed", "failed", "canceled")
TERMINAL = ("completed", "failed", "canceled")


def default_jobs_dir() -> Path:
    home = os.environ.get("AGENTDREAM_HOME")
    if home:
        return Path(home) / "jobs"
    return Path.cwd() / ".agentdream" / "jobs"


def _new_id() -> str:
    return time.strftime("dream-%Y%m%d-%H%M%S-", time.gmtime()) + secrets.token_hex(3)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class JobNotFound(Exception):
    pass


class DreamJob:
    def __init__(self, config: DreamConfig, job_id: str | None = None, jobs_dir: str | Path | None = None,
                 pipeline=None, backend=None):
        self.config = config
        self.id = job_id or _new_id()
        self.jobs_dir = Path(jobs_dir or config.jobs_dir or default_jobs_dir())
        self.status = "pending"
        self.review = "pending"
        self.created_at = now_iso()
        # ISO timestamps have 1s resolution; this orders jobs created within the same second
        self.created_ns = time.time_ns()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.progress = 0.0
        self.phase: str | None = None
        self.error: str | None = None
        self.warnings: list[str] = []
        self.counts: dict[str, Any] = {}
        self.input_hash: str | None = None
        self.out_dir = Path(config.out) if config.out else self.dir / "output"
        self.store_path: str | None = None
        self.accepted_ops: list[str] = []
        self.applied_to: str | None = None
        self.backup_path: str | None = None
        self.pid: int | None = None
        self._pipeline = pipeline
        self._backend = backend
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- persistence
    @property
    def dir(self) -> Path:
        return self.jobs_dir / self.id

    @property
    def path(self) -> Path:
        return self.dir / "job.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "status": self.status, "review": self.review, "created_at": self.created_at,
            "created_ns": self.created_ns, "started_at": self.started_at, "finished_at": self.finished_at,
            "progress": round(self.progress, 3),
            "phase": self.phase, "error": self.error, "warnings": self.warnings, "counts": self.counts,
            "input_hash": self.input_hash, "out_dir": str(self.out_dir), "store_path": self.store_path,
            "accepted_ops": self.accepted_ops, "applied_to": self.applied_to, "backup_path": self.backup_path,
            "pid": self.pid, "cancel_requested": self._cancel.is_set() or self._file_cancel_flag(),
            "config": self.config.to_dict(),
        }

    def _file_cancel_flag(self) -> bool:
        try:
            return bool(json.loads(self.path.read_text()).get("cancel_requested"))
        except (OSError, ValueError):
            return False

    def save(self) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, self.to_dict())

    @classmethod
    def load(cls, job_id: str, jobs_dir: str | Path | None = None) -> DreamJob:
        jd = Path(jobs_dir or default_jobs_dir())
        p = jd / job_id / "job.json"
        if not p.exists():
            matches = sorted(d.name for d in jd.glob(f"{job_id}*") if (d / "job.json").exists()) if jd.exists() else []
            if len(matches) == 1:
                p = jd / matches[0] / "job.json"
            else:
                raise JobNotFound(f"no such dream job: {job_id}" + (f" (ambiguous: {matches})" if matches else ""))
        d = json.loads(p.read_text(encoding="utf-8"))
        job = cls(DreamConfig.from_dict(d.get("config") or {}), job_id=d["id"], jobs_dir=jd)
        for k in ("status", "review", "created_at", "created_ns", "started_at", "finished_at", "progress", "phase",
                  "error", "warnings", "counts", "input_hash", "store_path", "accepted_ops", "applied_to",
                  "backup_path", "pid"):
            if k in d:
                setattr(job, k, d[k])
        job.out_dir = Path(d.get("out_dir") or job.dir / "output")
        if d.get("cancel_requested"):
            job._cancel.set()
        # A job left "running" by a dead process can never finish: mark it failed.
        if job.status in ("running", "pending") and job.pid and job.pid != os.getpid() and not _pid_alive(job.pid):
            job.status = "failed"
            job.error = f"interrupted: worker process {job.pid} exited before finishing"
            job.finished_at = job.finished_at or now_iso()
            job.save()
        return job

    @classmethod
    def list(cls, jobs_dir: str | Path | None = None) -> list[DreamJob]:
        jd = Path(jobs_dir or default_jobs_dir())
        if not jd.exists():
            return []
        out = []
        for d in sorted(jd.iterdir()):
            if (d / "job.json").exists():
                try:
                    out.append(cls.load(d.name, jd))
                except (ValueError, KeyError, OSError):
                    continue
        out.sort(key=lambda j: (j.created_at, j.created_ns or 0, j.id))
        return out

    def refresh(self) -> DreamJob:
        """Reload persisted state (useful when another process runs the job)."""
        other = DreamJob.load(self.id, self.jobs_dir)
        for k, v in other.__dict__.items():
            if not k.startswith("_"):
                setattr(self, k, v)
        if other._cancel.is_set():
            self._cancel.set()
        return self

    # ---------------------------------------------------------------- execution
    def start(self) -> DreamJob:
        """Run in a background thread; returns immediately."""
        if self.status != "pending":
            raise RuntimeError(f"job {self.id} already {self.status}")
        self.pid = os.getpid()
        self.save()
        self._thread = threading.Thread(target=self.run, name=f"agentdream-{self.id}", daemon=True)
        self._thread.start()
        return self

    def wait(self, timeout: float | None = None) -> DreamJob:
        if self._thread is not None:
            self._thread.join(timeout)
        return self

    def cancel(self) -> DreamJob:
        self._cancel.set()
        if self.status == "pending":
            self.status = "canceled"
            self.finished_at = now_iso()
        self.save()
        return self

    @property
    def done(self) -> bool:
        return self.status in TERMINAL

    def _on_progress(self, phase: str, frac: float) -> None:
        if self._file_cancel_flag():
            self._cancel.set()
        self.phase = phase
        self.save()

    def run(self) -> DreamJob:
        """Run synchronously in this thread."""
        from .pipeline import run_dream

        if self.status == "canceled" or self._cancel.is_set():
            self.status = "canceled"
            self.finished_at = self.finished_at or now_iso()
            self.save()
            return self
        self.status = "running"
        self.started_at = now_iso()
        self.pid = os.getpid()
        try:
            from .adapters.stancelog import strip_prefix

            self.input_hash = hash_path(strip_prefix(self.config.memory)) if self.config.memory else None
        except OSError:
            self.input_hash = None
        self.save()

        def on_phase(name: str, i: int, total: int) -> None:
            if self._file_cancel_flag():
                self._cancel.set()
            self.phase = name
            self.progress = i / total
            self.save()

        try:
            ctx = run_dream(self.config, self.out_dir, pipeline=self._pipeline, backend=self._backend,
                            progress=self._on_progress, cancel_event=self._cancel, on_phase=on_phase,
                            job_id=self.id, created_at=self.created_at)
        except DreamCanceled:
            self.status = "canceled"
            self.error = "canceled by user"
        except Exception as exc:
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            (self.dir / "error.log").parent.mkdir(parents=True, exist_ok=True)
            (self.dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        else:
            self.warnings = list(ctx.warnings)
            self.counts = dict(ctx.stats.get("report", {}).get("counts", {}))
            self.store_path = str(ctx.store_path) if ctx.store_path else None
            atomic_write_json(self.dir / "snapshot.json", [e.to_dict() for e in ctx.original_entries])
            atomic_write_json(self.dir / "operations.json", [o.to_dict() for o in ctx.ops])
            self.status = "completed"
            self.progress = 1.0
            self.phase = None
        self.finished_at = now_iso()
        self.save()
        return self

    # ---------------------------------------------------------------- results
    @property
    def report(self) -> dict[str, Any]:
        p = self.out_dir / "report.json"
        if not p.exists():
            raise FileNotFoundError(f"job {self.id} has no report (status={self.status}, review={self.review})")
        return json.loads(p.read_text(encoding="utf-8"))

    @property
    def report_md(self) -> str:
        p = self.out_dir / "report.md"
        if not p.exists():
            raise FileNotFoundError(f"job {self.id} has no report (status={self.status}, review={self.review})")
        return p.read_text(encoding="utf-8")

    @property
    def operations(self) -> list[Operation]:
        p = self.dir / "operations.json"
        if not p.exists():
            return []
        return [Operation.from_dict(d) for d in json.loads(p.read_text(encoding="utf-8"))]

    def snapshot(self) -> list[MemoryEntry]:
        p = self.dir / "snapshot.json"
        return [MemoryEntry.from_dict(d) for d in json.loads(p.read_text(encoding="utf-8"))]

    def summary_line(self) -> str:
        c = self.counts or {}
        bits = ", ".join(f"{k}={c[k]}" for k in ("merged", "replaced", "corrected", "insights", "contradictions",
                                                  "unchanged") if k in c)
        return f"{self.id}  {self.status:<9} review={self.review:<18} {bits}"

    def __repr__(self) -> str:
        return f"DreamJob(id={self.id!r}, status={self.status!r}, review={self.review!r})"
