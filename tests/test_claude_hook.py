"""Claude Code hook: stdin parsing, every-N gating, locking, never-raises, background worker, installer."""

import io
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agentdream import claude_hook as ch
from agentdream.cli import main
from agentdream.jobs import DreamJob

from conftest import hash_tree, write

NOTE = "- The PokéCenter in Viridian City is north of the gym.\n- Run `pytest -q` before committing.\n"


def cc_line(role, text, ts="2026-11-24T10:00:00Z"):
    return json.dumps({"type": role, "timestamp": ts, "message": {"role": role, "content": text}})


def make_project(tmp_path, *, memory="claude_md", n_sessions=3):
    """A project dir + a fake ~/.claude/projects/<p>/ folder with Claude Code transcripts."""
    proj = tmp_path / "proj"
    proj.mkdir()
    store = tmp_path / "claude-home" / "projects" / "-tmp-proj"
    store.mkdir(parents=True)
    if memory == "claude_md":
        write(proj / "CLAUDE.md", NOTE)
    elif memory == "auto":
        write(store / "memory" / "MEMORY.md", NOTE)
    paths = []
    for i in range(n_sessions):
        p = store / f"session-{i}.jsonl"
        p.write_text("\n".join([
            cc_line("assistant", "Walked north of the gym. No PokéCenter here."),
            cc_line("assistant", "Found the PokéCenter! It is south of the gym.", "2026-11-24T10:05:00Z"),
        ]) + "\n", encoding="utf-8")
        paths.append(p)
    return proj, store, paths


def payload(proj, path, sid, event="SessionEnd"):
    return io.StringIO(json.dumps({"session_id": sid, "transcript_path": str(path), "cwd": str(proj),
                                   "hook_event_name": event, "reason": "exit"}))


def log_text(proj):
    p = proj / ".agentdream" / "hook.log"
    return p.read_text() if p.exists() else ""


# ------------------------------------------------------------------ stdin parsing


class TTY(io.StringIO):
    def isatty(self):
        return True


class Broken:
    def isatty(self):
        return False

    def read(self):
        raise OSError("closed")


def test_read_payload_variants():
    assert ch.read_payload(io.StringIO('{"session_id": "s", "transcript_path": "/t.jsonl", "cwd": "/"}')) == \
        {"session_id": "s", "transcript_path": "/t.jsonl", "cwd": "/"}
    assert ch.read_payload(io.StringIO("")) == {}
    assert ch.read_payload(io.StringIO("   \n")) == {}
    assert ch.read_payload(TTY("{}")) == {}  # interactive: don't block waiting for input
    assert ch.read_payload(Broken()) == {}
    assert "_invalid_payload" in ch.read_payload(io.StringIO("{not json"))
    assert "_invalid_payload" in ch.read_payload(io.StringIO("[1, 2]"))


def test_project_dir_resolution(tmp_path, monkeypatch):
    assert ch.project_dir({"cwd": str(tmp_path)}) == tmp_path
    assert ch.project_dir({}, {"CLAUDE_PROJECT_DIR": str(tmp_path)}) == tmp_path
    monkeypatch.chdir(tmp_path)
    assert ch.project_dir({"cwd": str(tmp_path / "gone")}, {}) == tmp_path


# ------------------------------------------------------------------ gating


def test_every_n_sessions_gating_and_dedup(tmp_path):
    proj, _store, paths = make_project(tmp_path)
    spawned = []
    opts = ch.HookOptions(every=3)

    def spawn(project, o, log):
        spawned.append((project, o.every))

    for i in range(2):
        assert ch.run_hook(opts, payload(proj, paths[i], f"s{i}"), spawn=spawn) == 0
        # a Stop hook fires after every response: the same session must not count twice
        assert ch.run_hook(opts, payload(proj, paths[i], f"s{i}", "Stop"), spawn=spawn) == 0
    assert spawned == []
    assert "2/3 session(s) waiting" in log_text(proj)
    assert ch.run_hook(opts, payload(proj, paths[2], "s2"), spawn=spawn) == 0
    assert spawned == [(proj, 3)]
    state = json.loads((proj / ".agentdream" / "hook-state.json").read_text())
    assert sorted(state["sessions"]) == ["s0", "s1", "s2"]
    assert state["sessions"]["s0"]["transcript"] == str(paths[0])


def test_every_comes_from_config_hook_section(tmp_path):
    proj, _store, paths = make_project(tmp_path)
    write(proj / ".agentdream.toml", "[hook]\nevery = 2\n")
    spawned = []
    ch.run_hook(ch.HookOptions(), payload(proj, paths[0], "a"), spawn=lambda *a: spawned.append(1))
    assert spawned == []
    ch.run_hook(ch.HookOptions(), payload(proj, paths[1], "b"), spawn=lambda *a: spawned.append(1))
    assert spawned == [1]


def test_concurrent_hooks_do_not_lose_sessions(tmp_path):
    proj, store, _paths = make_project(tmp_path, n_sessions=0)
    opts = ch.HookOptions(every=100)
    errors = []

    def fire(i):
        try:
            ch.run_hook(opts, payload(proj, store / f"s{i}.jsonl", f"s{i}"), spawn=lambda *a: None)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    state = json.loads((proj / ".agentdream" / "hook-state.json").read_text())
    assert len(state["sessions"]) == 12


# ------------------------------------------------------------------ never raises


def test_hook_never_raises_and_logs(tmp_path, monkeypatch, capsys):
    proj, _store, paths = make_project(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("config exploded")

    monkeypatch.setattr(ch, "_settings", boom)
    assert ch.run_hook(ch.HookOptions(), payload(proj, paths[0], "s0")) == 0
    assert "config exploded" in log_text(proj)
    monkeypatch.undo()
    assert ch.run_hook(ch.HookOptions(), io.StringIO("{garbage")) == 0  # invalid stdin
    # an unwritable state dir: .agentdream is a *file*
    proj2 = tmp_path / "p2"
    write(proj2 / ".agentdream", "not a dir")
    assert ch.run_hook(ch.HookOptions(), payload(proj2, paths[0], "x")) == 0
    assert "agentdream hook:" in capsys.readouterr().err  # fell back to stderr
    # bad arguments from a future settings.json must not exit 2 (that would block a Stop hook)
    monkeypatch.chdir(proj)
    assert main(["hook", "--definitely-not-a-flag"]) == 0
    assert "invalid hook arguments" in log_text(proj)


def test_worker_never_raises(tmp_path, monkeypatch):
    proj, _store, _paths = make_project(tmp_path)
    monkeypatch.setattr(ch, "_run_worker", lambda *a: (_ for _ in ()).throw(RuntimeError("worker boom")))
    assert ch.run_worker(ch.HookOptions(), proj) == 0
    assert "worker boom" in log_text(proj)


# ------------------------------------------------------------------ locking


def test_file_lock_is_exclusive(tmp_path):
    a, b = ch.FileLock(tmp_path / "x.lock"), ch.FileLock(tmp_path / "x.lock")
    assert a.acquire() and a.held
    assert not b.acquire(timeout=0.1)
    a.release()
    assert b.acquire() and not a.held
    b.release()
    with ch.FileLock(tmp_path / "y.lock") as lk:
        assert lk.held
    assert not lk.held


def test_lock_held_by_another_process(tmp_path):
    lockfile = tmp_path / "proj" / ".agentdream" / "dream.lock"
    ready = tmp_path / "ready"
    code = ("import sys, time; from agentdream.claude_hook import FileLock; l = FileLock(sys.argv[1]); "
            "assert l.acquire(); open(sys.argv[2], 'w').close(); time.sleep(30)")
    proc = subprocess.Popen([sys.executable, "-c", code, str(lockfile), str(ready)])
    try:
        deadline = time.time() + 20
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        assert not ch.FileLock(lockfile).acquire(timeout=0.2)
        # a second worker for the same project backs off immediately
        t0 = time.time()
        assert ch.run_worker(ch.HookOptions(), tmp_path / "proj") == 0
        assert time.time() - t0 < 2
        assert "already running" in log_text(tmp_path / "proj")
    finally:
        proc.kill()
        proc.wait()
    relock = ch.FileLock(lockfile)
    assert relock.acquire(timeout=2)  # released by the OS when the holder died
    relock.release()


# ------------------------------------------------------------------ worker end to end


def test_sync_worker_dry_run_then_apply(tmp_path, monkeypatch):
    proj, _store, paths = make_project(tmp_path)
    monkeypatch.setenv("AGENTDREAM_HOOK_SYNC", "1")
    before = hash_tree(proj / "CLAUDE.md")
    assert ch.run_hook(ch.HookOptions(every=1), payload(proj, paths[0], "s0")) == 0
    assert hash_tree(proj / "CLAUDE.md") == before  # dry run: memory untouched
    jobs = DreamJob.list(proj / ".agentdream" / "jobs")
    assert len(jobs) == 1 and jobs[0].status == "completed" and jobs[0].review == "pending"
    assert jobs[0].config.memory == str(proj / "CLAUDE.md")
    assert any(o.type == "correct" for o in jobs[0].operations)
    log = log_text(proj)
    assert "nothing applied" in log and jobs[0].id in log
    state = json.loads((proj / ".agentdream" / "hook-state.json").read_text())
    assert state["sessions"]["s0"]["dreamed"] is True and state["sessions"]["s0"]["job"] == jobs[0].id
    # the same session ending again (e.g. a resumed session) doesn't re-dream
    assert ch.run_hook(ch.HookOptions(every=1), payload(proj, paths[0], "s0")) == 0
    assert len(DreamJob.list(proj / ".agentdream" / "jobs")) == 1
    # --apply: the next session is applied, with a backup
    assert ch.run_hook(ch.HookOptions(every=1, apply=True), payload(proj, paths[1], "s1")) == 0
    assert "south of the gym" in (proj / "CLAUDE.md").read_text()
    applied = [j for j in DreamJob.list(proj / ".agentdream" / "jobs") if j.review == "accepted"]
    assert len(applied) == 1 and applied[0].backup_path and Path(applied[0].backup_path).exists()


def test_default_memory_resolution(tmp_path):
    proj, store, paths = make_project(tmp_path, memory="auto")
    assert ch.default_memory(proj, str(paths[0])) == store / "memory"
    (store / "memory" / "MEMORY.md").unlink()  # empty folder doesn't count
    assert ch.default_memory(proj, str(paths[0])) is None
    write(proj / ".claude" / "CLAUDE.md", NOTE)
    assert ch.default_memory(proj, str(paths[0])) == proj / ".claude" / "CLAUDE.md"
    write(proj / "CLAUDE.md", NOTE)
    assert ch.default_memory(proj, None) == proj / "CLAUDE.md"
    assert ch.claude_project_store(None) is None


def test_worker_without_memory_gives_up_after_retries(tmp_path, monkeypatch):
    proj, _store, paths = make_project(tmp_path, memory=None)
    monkeypatch.setenv("AGENTDREAM_HOOK_SYNC", "1")
    for _ in range(ch.MAX_ATTEMPTS + 1):
        assert ch.run_hook(ch.HookOptions(), payload(proj, paths[0], "s0")) == 0
    assert "no memory store found" in log_text(proj)
    state = json.loads((proj / ".agentdream" / "hook-state.json").read_text())
    assert state["sessions"]["s0"]["attempts"] == ch.MAX_ATTEMPTS
    assert ch.pending_sessions(state) == []


def test_worker_uses_explicit_memory_and_handles_failed_dreams(tmp_path, monkeypatch):
    proj, _store, paths = make_project(tmp_path, memory=None)
    mem = write(tmp_path / "elsewhere" / "notes.md", NOTE)
    monkeypatch.setenv("AGENTDREAM_HOOK_SYNC", "1")
    assert ch.run_hook(ch.HookOptions(memory=str(mem)), payload(proj, paths[0], "s0")) == 0
    (job,) = DreamJob.list(proj / ".agentdream" / "jobs")
    assert job.config.memory == str(mem)
    # a failing dream (e.g. limits) is logged and retried later, not marked done
    assert ch.run_hook(ch.HookOptions(memory=str(mem)), payload(proj, paths[1], "s1")) == 0
    write(proj / ".agentdream.toml", "[limits]\nmax_total_tokens = 1\non_limit = 'fail'\n")
    assert ch.run_hook(ch.HookOptions(memory=str(mem)), payload(proj, paths[2], "s2")) == 0
    state = json.loads((proj / ".agentdream" / "hook-state.json").read_text())
    assert state["sessions"]["s2"].get("attempts") == 1 and not state["sessions"]["s2"]["dreamed"]
    assert "failed" in log_text(proj)


def test_real_hook_returns_fast_and_dreams_in_background(tmp_path):
    proj, _store, paths = make_project(tmp_path)
    body = json.dumps({"session_id": "bg1", "transcript_path": str(paths[0]), "cwd": str(proj),
                       "hook_event_name": "SessionEnd"})
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "agentdream", "hook", "--every", "1"], input=body, text=True,
                       capture_output=True, cwd=proj, timeout=60)
    elapsed = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr
    assert elapsed < 10, elapsed  # interpreter start-up dominates; the dream itself runs detached
    state_file = proj / ".agentdream" / "hook-state.json"
    deadline = time.time() + 60
    while time.time() < deadline:
        if state_file.exists() and json.loads(state_file.read_text())["sessions"]["bg1"].get("dreamed"):
            break
        time.sleep(0.2)
    else:  # pragma: no cover
        pytest.fail("background worker did not finish:\n" + log_text(proj))
    log = log_text(proj)
    assert "started background dream worker" in log and "completed" in log
    pids = {line.split("[", 1)[1].split("]", 1)[0] for line in log.splitlines() if line.count("[") and "] " in line}
    assert len(pids) >= 2  # hook and worker are different processes


# ------------------------------------------------------------------ installer


EXISTING = {
    "permissions": {"allow": ["Bash(pytest:*)"]},
    "hooks": {
        "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo pre"}]}],
        "SessionEnd": [{"hooks": [{"type": "command", "command": "notify-send bye"}]}],
    },
}


def test_install_is_idempotent_and_preserves_other_settings(tmp_path):
    proj = tmp_path / "proj"
    sp = write(proj / ".claude" / "settings.json", json.dumps(EXISTING, indent=2))
    res = ch.install("project", proj, ch.HookOptions(every=3), python="/usr/bin/python3")
    assert res.changed and res.backup and Path(res.backup).read_text() == json.dumps(EXISTING, indent=2)
    data = json.loads(sp.read_text())
    assert data["permissions"] == EXISTING["permissions"]
    assert data["hooks"]["PreToolUse"] == EXISTING["hooks"]["PreToolUse"]
    ends = data["hooks"]["SessionEnd"]
    assert ends[0] == EXISTING["hooks"]["SessionEnd"][0]  # someone else's hook kept
    ours = ends[1]["hooks"][0]
    assert ours["type"] == "command" and ours["timeout"] == ch.HOOK_TIMEOUT
    assert shlex.split(ours["command"]) == ["/usr/bin/python3", "-m", "agentdream", "hook", "--every", "3"]
    again = ch.install("project", proj, ch.HookOptions(every=3), python="/usr/bin/python3")
    assert not again.changed and again.backup is None and "already installed" in again.message
    assert len(list(sp.parent.glob("settings.json.agentdream-backup-*"))) == 1
    # changing options updates the one entry instead of adding another
    ch.install("project", proj, ch.HookOptions(every=5, apply=True), python="/usr/bin/python3")
    data = json.loads(sp.read_text())
    cmds = [h["command"] for g in data["hooks"]["SessionEnd"] for h in g["hooks"]]
    assert sum(ch.is_agentdream_hook(c) for c in cmds) == 1 and any("--every 5 --apply" in c for c in cmds)
    # switching to Stop moves it
    ch.install("project", proj, ch.HookOptions(), event="Stop", python="/usr/bin/python3")
    data = json.loads(sp.read_text())
    assert not any(ch.is_agentdream_hook(h["command"]) for g in data["hooks"]["SessionEnd"] for h in g["hooks"])
    assert ch.is_agentdream_hook(data["hooks"]["Stop"][0]["hooks"][0]["command"])


def test_uninstall_removes_only_ours(tmp_path):
    proj = tmp_path / "proj"
    sp = write(proj / ".claude" / "settings.json", json.dumps(EXISTING))
    ch.install("project", proj, ch.HookOptions())
    res = ch.uninstall("project", proj)
    assert res.removed == 1 and res.backup
    data = json.loads(sp.read_text())
    assert data == EXISTING
    assert "no agentdream hook" in ch.uninstall("project", proj).message
    # a settings file that only had our hook ends up without an empty "hooks" object
    sp2 = tmp_path / "solo" / "settings.json"
    ch.install(path=sp2)
    assert not ch.install(path=sp2).changed
    ch.uninstall(path=sp2)
    assert json.loads(sp2.read_text()) == {}


def test_install_scopes_and_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    assert ch.settings_path("user") == tmp_path / "home" / ".claude" / "settings.json"
    assert ch.settings_path("local", tmp_path) == tmp_path / ".claude" / "settings.local.json"
    res = ch.install("user")
    assert res.changed and res.backup is None  # new file: nothing to back up
    assert (tmp_path / "home" / ".claude" / "settings.json").exists()
    bad = write(tmp_path / "bad" / "settings.json", "{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        ch.install(path=bad)
    assert bad.read_text() == "{ not json"
    write(bad, "[1]")
    with pytest.raises(ValueError, match="JSON object"):
        ch.install(path=bad)
    write(bad, '{"hooks": []}')
    with pytest.raises(ValueError, match="'hooks' must be an object"):
        ch.install(path=bad)
    with pytest.raises(ValueError, match="event"):
        ch.install(path=bad, event="PreToolUse")
    empty = write(tmp_path / "empty" / "settings.json", "")
    assert ch.install(path=empty).changed


def test_cli_install_uninstall_and_the_command_runs(tmp_path, monkeypatch, capsys):
    proj, _store, paths = make_project(tmp_path)
    monkeypatch.chdir(proj)
    assert main(["install-claude-hook", "--every", "2", "--memory", "CLAUDE.md"]) == 0
    out = capsys.readouterr().out
    assert "installed" in out and "dry run" in out
    data = json.loads((proj / ".claude" / "settings.json").read_text())
    cmd = data["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
    assert str(proj / "CLAUDE.md") in cmd and "--every 2" in cmd and "--apply" not in cmd
    body = json.dumps({"session_id": "c1", "transcript_path": str(paths[0]), "cwd": str(proj),
                       "hook_event_name": "SessionEnd"})
    r = subprocess.run(shlex.split(cmd), input=body, text=True, capture_output=True, cwd=proj, timeout=60)
    assert r.returncode == 0 and "1/2 session(s) waiting" in log_text(proj)
    assert main(["install-claude-hook", "--local", "--apply", "--event", "Stop"]) == 0
    assert "APPLY" in capsys.readouterr().out
    local = json.loads((proj / ".claude" / "settings.local.json").read_text())
    assert "--apply" in local["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert main(["uninstall-claude-hook"]) == 0 and "removed 1" in capsys.readouterr().out
    assert main(["uninstall-claude-hook", "--local"]) == 0
    write(proj / ".claude" / "settings.json", "{oops")
    assert main(["install-claude-hook"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_hook_worker_cli_entry(tmp_path, monkeypatch):
    proj, _store, paths = make_project(tmp_path)
    monkeypatch.chdir(proj)
    ch.run_hook(ch.HookOptions(every=1), payload(proj, paths[0], "w1"), spawn=lambda *a: None)
    assert main(["hook-worker", "--project", str(proj)]) == 0
    assert len(DreamJob.list(proj / ".agentdream" / "jobs")) == 1
    assert main(["hook-worker"]) == 0  # missing --project: still exit 0


def test_log_rotation(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "LOG_MAX_BYTES", 100)
    log = ch.HookLog(tmp_path / "h.log")
    for i in range(20):
        log(f"line {i} " + "x" * 20)
    assert (tmp_path / "h.log.1").exists() and os.path.getsize(tmp_path / "h.log") < 400
