"""Claude Code integration: dream automatically after sessions (opt-in, dry-run by default).

``agentdream install-claude-hook`` adds a ``SessionEnd`` (or ``Stop``) hook to
``.claude/settings.json`` (project), ``.claude/settings.local.json`` (``--local``) or
``~/.claude/settings.json`` (``--user``)::

    {"hooks": {"SessionEnd": [{"hooks": [{"type": "command",
                                           "command": "/path/python -m agentdream hook --every 3",
                                           "timeout": 30}]}]}}

When Claude Code runs the hook it passes JSON on stdin (``session_id``,
``transcript_path``, ``cwd``, ``hook_event_name``...). ``agentdream hook``:

1. records the session in ``<project>/.agentdream/hook-state.json`` (under a short
   state lock, so concurrent hooks don't lose updates);
2. returns immediately unless at least ``--every N`` sessions are waiting;
3. otherwise starts a **detached** worker process (``agentdream hook-worker``) and exits.

The worker takes a non-blocking dream lock (a second worker just exits), dreams over the
waiting sessions into ``<project>/.agentdream/jobs`` and marks them done. Without
``--apply`` (or ``[hook] apply = true``) nothing is applied: review with
``agentdream show <job>`` and ``agentdream accept <job>``.

The hook never blocks or fails the Claude session: every error is caught and written to
``<project>/.agentdream/hook.log`` and the exit code is always 0.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from .util import atomic_write_json, atomic_write_text

EVENTS = ("SessionEnd", "Stop")
STATE_DIRNAME = ".agentdream"
LOG_NAME = "hook.log"
LOG_MAX_BYTES = 1_000_000
MAX_ATTEMPTS = 3
#: default hook timeout written to settings.json (seconds; the hook itself returns in well under 1s)
HOOK_TIMEOUT = 30
_OURS = re.compile(r"agentdream(?:\.exe)?[\"']?\s+hook(?:\s|$)")


# ------------------------------------------------------------------ small utilities

def read_payload(stream: IO[str] | None = None) -> dict[str, Any]:
    """Hook input from stdin. Never raises: no input, a TTY or invalid JSON give ``{}``."""
    stream = sys.stdin if stream is None else stream
    try:
        if stream is None or (hasattr(stream, "isatty") and stream.isatty()):
            return {}
        raw = stream.read()
    except (OSError, ValueError):
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {"_invalid_payload": raw[:200]}
    return data if isinstance(data, dict) else {"_invalid_payload": raw[:200]}


def project_dir(payload: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    for cand in (payload.get("cwd"), env.get("CLAUDE_PROJECT_DIR")):
        if cand and os.path.isdir(str(cand)):
            return Path(str(cand)).absolute()
    return Path.cwd()


class HookLog:
    """Append-only log with one-file rotation; never raises."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > LOG_MAX_BYTES:
                os.replace(self.path, self.path.with_name(self.path.name + ".1"))
            stamp = _dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{stamp} [{os.getpid()}] {msg}\n")
        except Exception:  # logging must never break the hook
            with contextlib.suppress(Exception):
                sys.stderr.write(f"agentdream hook: {msg}\n")


class FileLock:
    """Inter-process lock on a file (``fcntl.flock`` / ``msvcrt.locking``; O_EXCL as a last resort).

    The OS releases flock/msvcrt locks when the holder dies, so a crashed worker never leaves
    a stale lock behind."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._fh: IO[str] | None = None
        self._excl = False

    def acquire(self, timeout: float = 0.0, poll: float = 0.05) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self._try():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)

    def _try(self) -> bool:
        fh = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115 - held open while locked
        try:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:  # pragma: no cover - Windows
                try:
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                except ImportError:
                    fh.close()
                    return self._try_excl()
        except OSError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return True

    def _try_excl(self) -> bool:  # pragma: no cover - platforms without fcntl/msvcrt
        excl = self.path.with_name(self.path.name + ".excl")
        try:
            fd = os.open(excl, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        self._excl = True
        return True

    def release(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(Exception):
                try:
                    import fcntl

                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover
                    import msvcrt

                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            with contextlib.suppress(Exception):
                self._fh.close()
            self._fh = None
        if self._excl:  # pragma: no cover
            with contextlib.suppress(OSError):
                os.unlink(self.path.with_name(self.path.name + ".excl"))
            self._excl = False

    @property
    def held(self) -> bool:
        return self._fh is not None or self._excl

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise TimeoutError(f"could not lock {self.path}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


@contextlib.contextmanager
def _state_lock(sdir: Path, timeout: float = 3.0) -> Iterator[None]:
    lock = FileLock(sdir / "hook-state.lock")
    if not lock.acquire(timeout=timeout):
        raise TimeoutError("hook state is locked by another process")
    try:
        yield
    finally:
        lock.release()


def _load_state(sdir: Path) -> dict[str, Any]:
    try:
        data = json.loads((sdir / "hook-state.json").read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("sessions", {})
            data.setdefault("runs", [])
            return data
    except (OSError, ValueError):
        pass
    return {"sessions": {}, "runs": []}


def _save_state(sdir: Path, state: dict[str, Any]) -> None:
    state["runs"] = state.get("runs", [])[-50:]
    atomic_write_json(sdir / "hook-state.json", state)


def pending_sessions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sessions recorded but not dreamed over yet (and not given up on), oldest first."""
    out = [dict(v, id=k) for k, v in state.get("sessions", {}).items()
           if not v.get("dreamed") and v.get("attempts", 0) < MAX_ATTEMPTS]
    out.sort(key=lambda s: (s.get("first_seen", 0), s["id"]))
    return out


def claude_project_store(transcript_path: str | None) -> Path | None:
    """Claude Code keeps a project's sessions in ``~/.claude/projects/<project>/``."""
    if not transcript_path:
        return None
    p = Path(transcript_path).expanduser()
    return p.parent if p.parent.name else None


def default_memory(project: Path, transcript_path: str | None = None) -> Path | None:
    """Where the project's Claude memory lives, first match wins:

    1. ``~/.claude/projects/<project>/memory/`` (Claude Code auto-memory folder, next to the transcripts)
    2. ``<project>/CLAUDE.md``
    3. ``<project>/.claude/CLAUDE.md``
    """
    store = claude_project_store(transcript_path)
    candidates = [store / "memory"] if store is not None else []
    candidates += [project / "CLAUDE.md", project / ".claude" / "CLAUDE.md"]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
        if c.is_file():
            return c
    return None


# ------------------------------------------------------------------ hook entry points

@dataclass
class HookOptions:
    memory: str | None = None
    every: int | None = None
    apply: bool | None = None
    config: str | None = None
    jobs_dir: str | None = None
    backend: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def argv(self) -> list[str]:
        out: list[str] = []
        if self.memory:
            out += ["--memory", self.memory]
        if self.every is not None:
            out += ["--every", str(self.every)]
        if self.apply:
            out += ["--apply"]
        if self.config:
            out += ["--config", self.config]
        if self.jobs_dir:
            out += ["--jobs-dir", self.jobs_dir]
        if self.backend:
            out += ["--backend", self.backend]
        if self.model:
            out += ["--model", self.model]
        return out


def _settings(opts: HookOptions, project: Path) -> Any:
    from .config import resolve

    return resolve({"backend": opts.backend, "model": opts.model, "jobs_dir": opts.jobs_dir},
                   cwd=project, config_path=opts.config,
                   cli_hook={"memory": opts.memory, "every": opts.every, "apply": opts.apply})


def spawn_worker(project: Path, opts: HookOptions, log: HookLog) -> None:
    """Start ``agentdream hook-worker`` fully detached from Claude Code's process group."""
    cmd = [sys.executable, "-m", "agentdream", "hook-worker", "--project", str(project), *opts.argv()]
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "cwd": str(project), "close_fds": True}
    log_fh = open(log.path, "a", encoding="utf-8")  # noqa: SIM115 - handed to the child
    kwargs["stdout"] = log_fh
    kwargs["stderr"] = log_fh
    if os.name == "nt":  # pragma: no cover
        kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0x8)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                                   | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **kwargs)
    finally:
        log_fh.close()
    log(f"started background dream worker pid={proc.pid}")


def run_hook(opts: HookOptions, stdin: IO[str] | None = None, environ: Mapping[str, str] | None = None,
             spawn: Callable[[Path, HookOptions, HookLog], None] | None = None,
             now: Callable[[], float] = time.time) -> int:
    """Entry point for ``agentdream hook``. Always returns 0."""
    env = os.environ if environ is None else environ
    payload = read_payload(stdin)
    try:
        project = project_dir(payload, env)
    except Exception:  # pragma: no cover - cwd vanished
        return 0
    log = HookLog(project / STATE_DIRNAME / LOG_NAME)
    try:
        return _run_hook(opts, payload, project, log, env, spawn or spawn_worker, now)
    except Exception as exc:
        log(f"error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 0


def _run_hook(opts: HookOptions, payload: dict[str, Any], project: Path, log: HookLog,
              env: Mapping[str, str], spawn: Callable[[Path, HookOptions, HookLog], None],
              now: Callable[[], float]) -> int:
    if "_invalid_payload" in payload:
        log(f"ignoring invalid hook input: {payload['_invalid_payload']!r}")
        payload = {}
    settings = _settings(opts, project)
    every = max(1, int(settings.hook.get("every") or 1))
    sdir = project / STATE_DIRNAME
    event = payload.get("hook_event_name") or "manual"
    sid = str(payload.get("session_id") or "")
    tpath = payload.get("transcript_path")
    with _state_lock(sdir):
        state = _load_state(sdir)
        if sid or tpath:
            key = sid or str(tpath)
            rec = state["sessions"].setdefault(key, {"first_seen": now(), "dreamed": False})
            rec["last_seen"] = now()
            rec["event"] = event
            if tpath:
                rec["transcript"] = str(Path(str(tpath)).expanduser())
        pending = [s for s in pending_sessions(state) if s.get("transcript")]
        _save_state(sdir, state)
    if len(pending) < every:
        log(f"{event}: session {sid or '?'} recorded; {len(pending)}/{every} session(s) waiting, not dreaming yet")
        return 0
    if env.get("AGENTDREAM_HOOK_SYNC") == "1":
        log(f"{event}: {len(pending)} session(s) waiting; dreaming in the foreground (AGENTDREAM_HOOK_SYNC=1)")
        return run_worker(opts, project)
    log(f"{event}: {len(pending)} session(s) waiting (every={every}); starting a background dream")
    spawn(project, opts, log)
    return 0


def run_worker(opts: HookOptions, project: Path) -> int:
    """Entry point for ``agentdream hook-worker``. Always returns 0."""
    log = HookLog(project / STATE_DIRNAME / LOG_NAME)
    lock = FileLock(project / STATE_DIRNAME / "dream.lock")
    try:
        if not lock.acquire(timeout=0):
            log("another dream is already running for this project; skipping (it re-checks when done)")
            return 0
        # sessions that end while we dream are picked up by another round (bounded)
        for _round in range(3):
            if not _run_worker(opts, project, log):
                break
    except Exception as exc:
        log(f"worker error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    finally:
        lock.release()
    return 0


def _run_worker(opts: HookOptions, project: Path, log: HookLog) -> bool:
    """One dream over the waiting sessions. Returns True when a dream completed."""
    from .api import dream
    from .review import accept

    settings = _settings(opts, project)
    for w in settings.warnings:
        log(f"config: {w}")
    every = max(1, int(settings.hook.get("every") or 1))
    apply = bool(settings.hook.get("apply"))
    sdir = project / STATE_DIRNAME
    with _state_lock(sdir):
        state = _load_state(sdir)
        pending = [s for s in pending_sessions(state) if s.get("transcript") and Path(s["transcript"]).exists()]
    if len(pending) < every:
        log(f"worker: {len(pending)}/{every} session(s) waiting; nothing (more) to do")
        return False
    latest = pending[-1].get("transcript")
    memory = (settings.hook.get("memory") or settings.dream.get("memory")
              or default_memory(project, latest))
    if not memory:
        log("worker: no memory store found (looked for ~/.claude/projects/<project>/memory/, CLAUDE.md, "
            ".claude/CLAUDE.md); set --memory or [hook] memory")
        _mark(sdir, pending, job_id=None, ok=False)
        return False
    options = dict(settings.dream)
    options.pop("memory", None)
    options.pop("sessions", None)
    backend = options.pop("backend", "deterministic")
    options.setdefault("jobs_dir", str(sdir / "jobs"))
    max_s = int(options.get("max_sessions") or 100)
    sessions = [s["transcript"] for s in pending][-max_s:]
    log(f"worker: dreaming over {len(sessions)} session(s) into memory {memory} (backend={backend}, "
        f"{'APPLY' if apply else 'dry run'})")
    job = dream(str(memory), sessions, backend, **options)
    ok = job.status == "completed"
    if ok:
        c = job.counts
        log(f"worker: job {job.id} completed: merged={c.get('merged')} replaced={c.get('replaced')} "
            f"corrected={c.get('corrected')} insights={c.get('insights')} contradictions={c.get('contradictions')}; "
            f"report: {job.out_dir / 'report.md'}")
        if apply:
            res = accept(job)
            log(f"worker: applied {len(res.applied_ops)} operation(s) to {res.target} (backup: {res.backup})")
        else:
            log(f"worker: nothing applied. Review: agentdream show {job.id} --jobs-dir {options['jobs_dir']}; "
                f"then agentdream accept {job.id} --jobs-dir {options['jobs_dir']}")
    else:
        log(f"worker: job {job.id} {job.status}: {job.error}")
    _mark(sdir, pending, job.id, ok)
    return ok


def _mark(sdir: Path, sessions: list[dict[str, Any]], job_id: str | None, ok: bool) -> None:
    with _state_lock(sdir):
        state = _load_state(sdir)
        for s in sessions:
            rec = state["sessions"].get(s["id"])
            if rec is None:
                continue
            if ok:
                rec["dreamed"] = True
                rec["job"] = job_id
            else:
                rec["attempts"] = rec.get("attempts", 0) + 1
        state["runs"].append({"at": time.time(), "job": job_id, "ok": ok, "sessions": [s["id"] for s in sessions]})
        _save_state(sdir, state)


# ------------------------------------------------------------------ installer

@dataclass
class InstallResult:
    settings_path: str
    changed: bool
    backup: str | None
    command: str | None = None
    removed: int = 0
    message: str = ""


def settings_path(scope: str, project: str | os.PathLike[str] | None = None) -> Path:
    if scope == "user":
        return Path(os.path.expanduser("~")) / ".claude" / "settings.json"
    base = Path(project or os.getcwd())
    return base / ".claude" / ("settings.local.json" if scope == "local" else "settings.json")


def hook_command(opts: HookOptions, python: str | None = None) -> str:
    argv = [python or sys.executable, "-m", "agentdream", "hook", *opts.argv()]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def is_agentdream_hook(command: str) -> bool:
    return bool(_OURS.search(command or ""))


def _read_settings(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}, text
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc}); fix it first - nothing was changed") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object; nothing was changed")
    return data, text


def _write_settings(path: Path, data: dict[str, Any], old_text: str | None) -> tuple[bool, str | None]:
    new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if old_text is not None and old_text == new_text:
        return False, None
    if old_text is not None and old_text.strip():
        try:
            if json.loads(old_text) == data:
                return False, None
        except ValueError:  # pragma: no cover - _read_settings validated it
            pass
    backup = None
    if path.exists():
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        b = path.with_name(f"{path.name}.agentdream-backup-{stamp}")
        n = 1
        while b.exists():
            n += 1
            b = path.with_name(f"{path.name}.agentdream-backup-{stamp}-{n}")
        b.write_bytes(path.read_bytes())
        backup = str(b)
    atomic_write_text(path, new_text)
    return True, backup


def _strip_ours(hooks: dict[str, Any], events: tuple[str, ...]) -> int:
    removed = 0
    for event in events:
        groups = hooks.get(event)
        if not isinstance(groups, list):
            continue
        kept_groups = []
        for g in groups:
            if not isinstance(g, dict) or not isinstance(g.get("hooks"), list):
                kept_groups.append(g)
                continue
            before = len(g["hooks"])
            g["hooks"] = [h for h in g["hooks"]
                          if not (isinstance(h, dict) and is_agentdream_hook(str(h.get("command", ""))))]
            removed += before - len(g["hooks"])
            if g["hooks"]:
                kept_groups.append(g)
        if kept_groups:
            hooks[event] = kept_groups
        else:
            hooks.pop(event, None)
    return removed


def install(scope: str = "project", project: str | os.PathLike[str] | None = None, opts: HookOptions | None = None,
            event: str = "SessionEnd", path: str | os.PathLike[str] | None = None,
            python: str | None = None) -> InstallResult:
    """Add (or update) the agentdream hook in a Claude Code settings file. Idempotent; backs the
    file up before changing it; other settings and hooks are left untouched."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {', '.join(EVENTS)}")
    opts = opts or HookOptions()
    p = Path(path) if path else settings_path(scope, project)
    data, old = _read_settings(p)
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"{p}: 'hooks' must be an object; nothing was changed")
    command = hook_command(opts, python)
    _strip_ours(hooks, EVENTS)  # one agentdream hook in total, for the chosen event
    hooks.setdefault(event, []).append({"hooks": [{"type": "command", "command": command,
                                                   "timeout": HOOK_TIMEOUT}]})
    changed, backup = _write_settings(p, data, old)
    msg = (f"{'installed' if changed else 'already installed'}: {event} hook in {p}"
           + (f" (backup: {backup})" if backup else ""))
    return InstallResult(str(p), changed, backup, command, 0, msg)


def uninstall(scope: str = "project", project: str | os.PathLike[str] | None = None,
              path: str | os.PathLike[str] | None = None) -> InstallResult:
    p = Path(path) if path else settings_path(scope, project)
    data, old = _read_settings(p)
    hooks = data.get("hooks")
    removed = _strip_ours(hooks, tuple(hooks)) if isinstance(hooks, dict) else 0
    if isinstance(hooks, dict) and not hooks:
        data.pop("hooks")
    if not removed:
        return InstallResult(str(p), False, None, None, 0, f"no agentdream hook in {p}")
    changed, backup = _write_settings(p, data, old)
    return InstallResult(str(p), changed, backup, None, removed,
                         f"removed {removed} agentdream hook(s) from {p}" + (f" (backup: {backup})" if backup else ""))


__all__ = ["EVENTS", "FileLock", "HookLog", "HookOptions", "InstallResult", "default_memory", "hook_command",
           "install", "is_agentdream_hook", "pending_sessions", "read_payload", "run_hook", "run_worker",
           "settings_path", "uninstall"]
