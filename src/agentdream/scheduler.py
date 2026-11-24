"""Scheduler: dream when idle, on an interval, or every N new sessions.

State (which sessions were already dreamed over, last run time) is kept in a
JSON file so a restarted scheduler doesn't re-dream old sessions. Clock and
sleep are injectable for tests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loaders import expand_sources
from .util import atomic_write_json, parse_duration


@dataclass
class Trigger:
    reason: str
    new_sessions: list[str] = field(default_factory=list)


def _sig(p: Path) -> str:
    # nanosecond mtimes: a session rewritten within the same second (same size) still counts as new.
    # (HFS+ only has 1s resolution; APFS/ext4/NTFS are fine.)
    st = p.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


class Scheduler:
    def __init__(self, memory: str, sessions: list[str], *, every: str | float | None = None,
                 idle: str | float | None = None, new_sessions: int | None = None,
                 state_file: str | Path | None = None, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, dream_fn: Callable[..., Any] | None = None,
                 dream_options: dict[str, Any] | None = None, apply: bool = False, only_new: bool = True):
        if every is None and idle is None and new_sessions is None:
            raise ValueError("give at least one trigger: every, idle or new_sessions")
        self.memory = memory
        self.sessions = sessions
        self.every = parse_duration(every) if every is not None else None
        self.idle = parse_duration(idle) if idle is not None else None
        self.new_sessions = new_sessions
        self.clock = clock
        self.sleep = sleep
        self.apply = apply
        self.only_new = only_new
        self.dream_options = dict(dream_options or {})
        if dream_fn is None:
            from .api import dream as dream_fn
        self.dream_fn = dream_fn
        jobs_dir = self.dream_options.get("jobs_dir")
        base = Path(jobs_dir).parent if jobs_dir else Path.cwd() / ".agentdream"
        self.state_file = Path(state_file) if state_file else base / "scheduler-state.json"
        self.state = self._load_state()
        self.runs: list[Any] = []

    def _load_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"dreamed": {}, "last_run": None, "started": None}

    def _save_state(self) -> None:
        atomic_write_json(self.state_file, self.state)

    def pending_sessions(self) -> list[Path]:
        try:
            files = expand_sources(self.sessions)
        except Exception:
            return []
        dreamed = self.state.get("dreamed", {})
        return [f for f in files if dreamed.get(str(f.resolve())) != _sig(f)]

    def check(self) -> Trigger | None:
        now = self.clock()
        if self.state.get("started") is None:
            self.state["started"] = now
            self._save_state()
        pending = self.pending_sessions()
        names = [str(p) for p in pending]
        if self.new_sessions is not None and len(pending) >= self.new_sessions:
            return Trigger(f"{len(pending)} new session(s) >= {self.new_sessions}", names)
        if self.idle is not None and pending:
            latest = max(p.stat().st_mtime for p in pending)
            if now - latest >= self.idle:
                return Trigger(f"idle for {int(now - latest)}s (>= {int(self.idle)}s) with {len(pending)} new "
                               "session(s)", names)
        if self.every is not None:
            last = self.state.get("last_run") or self.state.get("started")
            if now - last >= self.every and (pending or not self.only_new):
                return Trigger(f"interval of {int(self.every)}s elapsed", names)
        return None

    def run_once(self) -> Any | None:
        trig = self.check()
        if trig is None:
            return None
        sessions = trig.new_sessions if (self.only_new and trig.new_sessions) else self.sessions
        job = self.dream_fn(self.memory, sessions, apply=self.apply, **self.dream_options)
        status = getattr(job, "status", "completed")
        if status == "completed":
            for s in trig.new_sessions:
                p = Path(s)
                if p.exists():
                    self.state.setdefault("dreamed", {})[str(p.resolve())] = _sig(p)
        self.state["last_run"] = self.clock()
        self.state.setdefault("history", []).append(
            {"at": self.state["last_run"], "reason": trig.reason, "job": getattr(job, "id", None), "status": status})
        self.state["history"] = self.state["history"][-50:]
        self._save_state()
        self.runs.append(job)
        return job

    def run_forever(self, poll: float = 60.0, once: bool = False, max_iterations: int | None = None,
                    log: Callable[[str], None] | None = None) -> None:
        it = 0
        while True:
            job = self.run_once()
            if log:
                log(f"dreamed: {getattr(job, 'id', job)} ({getattr(job, 'status', '')})" if job else "no trigger")
            it += 1
            if once or (max_iterations is not None and it >= max_iterations):
                return
            self.sleep(poll)
