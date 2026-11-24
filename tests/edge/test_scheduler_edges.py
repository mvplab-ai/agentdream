"""Scheduler boundaries: idle threshold, every-N-sessions, intervals, clock going backwards, corrupt state."""

from __future__ import annotations

import json
import os

import pytest

from agentdream import Scheduler

from edgekit import write, write_bytes

T0 = 1_000_000.0


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


class FakeJob:
    def __init__(self, sessions, status="completed"):
        self.sessions, self.status, self.id = sessions, status, f"fake-{len(sessions)}"


def _sched(tmp_path, clock, **kw):
    calls: list = []

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


@pytest.mark.parametrize("age,fires", [(1800.0, True), (1799.0, False), (1800.5, True), (1799.999, False)])
def test_idle_threshold_boundary(tmp_path, age, fires):
    """--idle 30m fires at exactly 30 min of inactivity (and after), not at 29m59s."""
    clock = Clock()
    sch, s, _calls = _sched(tmp_path, clock, idle="30m")
    _session(s, "a.txt", T0 - age)
    assert (sch.run_once() is not None) is fires


def test_idle_uses_latest_of_several_sessions(tmp_path):
    """Idle time is measured from the most recently modified pending session."""
    clock = Clock()
    sch, s, calls = _sched(tmp_path, clock, idle=60)
    _session(s, "old.txt", T0 - 10_000)
    _session(s, "new.txt", T0 - 59)
    assert sch.run_once() is None
    clock.t += 1
    assert sch.run_once() is not None and len(calls[0]) == 2


@pytest.mark.parametrize("n,fires", [(2, False), (3, True), (4, True)])
def test_every_n_sessions_boundary(tmp_path, n, fires):
    """--new-sessions 3 with N-1 / N / N+1 new sessions."""
    sch, s, calls = _sched(tmp_path, Clock(), new_sessions=3)
    for i in range(n):
        _session(s, f"{i}.txt", T0 - 5)
    job = sch.run_once()
    assert (job is not None) is fires
    if fires:
        assert len(calls[0]) == n


def test_every_n_sessions_counts_only_new_ones(tmp_path):
    """After a dream, the counter restarts: sessions already dreamed over don't count again."""
    sch, s, calls = _sched(tmp_path, Clock(), new_sessions=2)
    _session(s, "a.txt", T0)
    _session(s, "b.txt", T0)
    assert sch.run_once() is not None
    _session(s, "c.txt", T0)
    assert sch.run_once() is None
    _session(s, "d.txt", T0)
    assert sch.run_once() is not None and len(calls[1]) == 2


@pytest.mark.parametrize("elapsed,fires", [(3600.0, True), (3599.0, False), (3601.0, True)])
def test_interval_boundary(tmp_path, elapsed, fires):
    """--every 1h fires at exactly one hour after the last run (with pending sessions), not a second earlier."""
    clock = Clock()
    sch, s, _calls = _sched(tmp_path, clock, every="1h")
    sch.check()  # records the start time
    _session(s, "a.txt", T0)
    clock.t = T0 + elapsed
    assert (sch.run_once() is not None) is fires


def test_interval_without_new_sessions_does_not_fire(tmp_path):
    """With only_new (default), an elapsed interval and no new sessions does nothing."""
    clock = Clock()
    sch, _s, _calls = _sched(tmp_path, clock, every="1m")
    sch.check()
    clock.t += 3600
    assert sch.run_once() is None


def test_clock_going_backwards_resets_interval_baseline(tmp_path):
    """Clock jumps back a day: the interval restarts from 'now' instead of stalling for a day."""
    clock = Clock()
    sch, s, _calls = _sched(tmp_path, clock, every="1h")
    sch.check()
    _session(s, "a.txt", T0 - 100_000)
    clock.t = T0 + 3600
    assert sch.run_once() is not None
    clock.t = T0 - 86_400  # NTP step / VM restore
    _session(s, "b.txt", T0 - 200_000)
    assert sch.run_once() is None
    assert sch.state["last_run"] == clock.t  # baseline reset to the new 'now'
    clock.t += 3600
    assert sch.run_once() is not None


def test_session_mtime_in_the_future_is_not_idle(tmp_path):
    """A session modified 'in the future' (clock went backwards) is not considered idle; no crash."""
    clock = Clock()
    sch, s, _calls = _sched(tmp_path, clock, idle=60)
    _session(s, "a.txt", T0 + 10_000)
    assert sch.run_once() is None


@pytest.mark.parametrize("raw", [b"\x00\xff garbage", b"[]", b'"str"', b"42", b"null", b'{"dreamed": []}',
                                 b'{"dreamed": {"x": 1}, "last_run": "abc", "started": [1]}', b'{"history": 5}',
                                 b'{"dreamed": {"/a": "1:2"', b""])
def test_corrupt_state_file_starts_fresh(tmp_path, raw):
    """A corrupt / truncated / wrong-typed state file doesn't crash the scheduler; it is rewritten valid."""
    clock = Clock()
    write_bytes(tmp_path / "state.json", raw)
    sch, s, _calls = _sched(tmp_path, clock, new_sessions=1)
    _session(s, "a.txt", T0)
    assert sch.run_once() is not None
    state = json.loads((tmp_path / "state.json").read_text())
    assert isinstance(state["dreamed"], dict) and len(state["dreamed"]) == 1
    assert isinstance(state["history"], list)


def test_state_survives_restart(tmp_path):
    """A restarted scheduler (same state file) doesn't re-dream sessions already handled."""
    clock = Clock()
    sch, s, _calls = _sched(tmp_path, clock, new_sessions=1)
    _session(s, "a.txt", T0)
    assert sch.run_once() is not None
    sch2, _s, calls2 = _sched(tmp_path, clock, new_sessions=1)
    assert sch2.run_once() is None and calls2 == []


def test_failed_dream_keeps_sessions_pending(tmp_path):
    """A failed dream doesn't mark its sessions as dreamed; the next check retries them."""
    clock = Clock()
    mem = write(tmp_path / "m" / "MEMORY.md", "- x\n").parent
    s = tmp_path / "s"
    _session(s, "a.txt", T0)
    statuses = ["failed", "completed"]
    sch = Scheduler(str(mem), [str(s)], clock=clock, sleep=lambda _: None, new_sessions=1,
                    state_file=tmp_path / "state.json",
                    dream_fn=lambda m, ss, apply=False, **o: FakeJob(ss, statuses.pop(0)))
    assert sch.run_once().status == "failed"
    assert sch.run_once().status == "completed"
    assert sch.run_once() is None


def test_sessions_dir_disappears(tmp_path):
    """If the sessions directory is deleted while scheduling, checks return no trigger instead of raising."""
    sch, s, _calls = _sched(tmp_path, Clock(), new_sessions=1)
    s.rmdir()
    assert sch.run_once() is None


def test_negative_or_zero_triggers_rejected(tmp_path):
    """Nonsense triggers (--every 0, --idle -5m, --new-sessions 0) are rejected with ValueError."""
    for kw in ({"every": "0s"}, {"idle": -300}, {"new_sessions": 0}):
        with pytest.raises(ValueError):
            _sched(tmp_path, Clock(), **kw)
