"""Scheduler triggers with a fake clock."""

import os

import pytest

from agentdream import Scheduler
from agentdream.cli import main
from agentdream.util import parse_duration

from conftest import write


class FakeJob:
    def __init__(self, sessions):
        self.sessions = sessions
        self.status = "completed"
        self.id = f"fake-{len(sessions)}"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _sched(tmp_path, clock, **kw):
    calls = []

    def fake_dream(memory, sessions, apply=False, **opts):
        calls.append(list(sessions))
        return FakeJob(sessions)

    mem = write(tmp_path / "m" / "MEMORY.md", "- x\n").parent
    s = tmp_path / "s"
    s.mkdir(exist_ok=True)
    sch = Scheduler(str(mem), [str(s)], clock=clock, sleep=lambda _: None, dream_fn=fake_dream,
                    state_file=tmp_path / "state.json", **kw)
    return sch, s, calls


def _session(s, name, mtime):
    p = write(s / name, "User: hi\n")
    os.utime(p, (mtime, mtime))
    return p


def test_parse_duration():
    assert parse_duration("30m") == 1800 and parse_duration("6h") == 21600 and parse_duration("1d") == 86400
    assert parse_duration("90s") == 90 and parse_duration("1h30m") == 5400 and parse_duration(45) == 45
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_every_n_sessions(tmp_path):
    clock = Clock()
    sch, s, calls = _sched(tmp_path, clock, new_sessions=2)
    _session(s, "a.txt", clock.t)
    assert sch.run_once() is None
    _session(s, "b.txt", clock.t)
    assert sch.run_once() is not None
    assert len(calls) == 1 and len(calls[0]) == 2
    assert sch.run_once() is None  # already dreamed over
    _session(s, "c.txt", clock.t)
    assert sch.check() is None
    # state persists across scheduler restarts
    sch2, _, _calls2 = _sched(tmp_path, clock, new_sessions=2)
    assert [p.name for p in sch2.pending_sessions()] == ["c.txt"]


def test_idle_trigger(tmp_path):
    clock = Clock()
    sch, s, calls = _sched(tmp_path, clock, idle="30m")
    _session(s, "a.txt", clock.t - 60)
    assert sch.check() is None  # active a minute ago
    clock.t += 31 * 60
    trig = sch.check()
    assert trig and "idle" in trig.reason
    sch.run_once()
    assert calls and sch.check() is None  # nothing new


def test_interval_trigger_with_once_loop(tmp_path):
    clock = Clock()
    sch, s, calls = _sched(tmp_path, clock, every="6h")
    _session(s, "a.txt", clock.t)
    sch.run_forever(once=True)
    assert calls == []  # interval not elapsed yet
    clock.t += 6 * 3600 + 1
    sch.run_forever(once=True)
    assert len(calls) == 1
    clock.t += 6 * 3600 + 1
    sch.run_forever(poll=1, max_iterations=2)
    assert len(calls) == 1  # no new sessions -> nothing to dream over


def test_requires_trigger(tmp_path):
    with pytest.raises(ValueError):
        Scheduler("m", ["s"])


def test_cli_schedule_once_real_dream(tmp_path, capsys):
    mem = write(tmp_path / "m" / "MEMORY.md", "- Use pytest.\n- Use pytest!\n").parent
    s = tmp_path / "s"
    write(s / "1.txt", "User: hello\n")
    rc = main(["schedule", "--memory", str(mem), "--sessions", str(s), "--new-sessions", "1", "--once",
               "--state-file", str(tmp_path / "st.json")])
    assert rc == 0
    assert "dreamed:" in capsys.readouterr().out
    rc = main(["schedule", "--memory", str(mem), "--sessions", str(s), "--new-sessions", "1", "--once",
               "--state-file", str(tmp_path / "st.json")])
    assert "no trigger" in capsys.readouterr().out
