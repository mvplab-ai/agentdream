"""Concurrent dreams (threads and processes) on one store; concurrent accepts; job id collisions."""

from __future__ import annotations

import json
import subprocess
import sys
import threading

from agentdream import DreamJob, ReviewError, accept, dream, dream_async, get_adapter
from agentdream import jobs as jobs_mod

from edgekit import tree_digest, workspace


def _check_job(job):
    assert job.status == "completed", job.error
    assert json.loads((job.out_dir / "report.json").read_text())["job_id"] == job.id
    get_adapter(job.store_path).load()


def test_threads_dream_same_store(tmp_path):
    """Six threads dreaming over the same store at once: all complete, distinct ids, input intact."""
    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem, ses)
    barrier = threading.Barrier(6)
    results: list = []

    def run():
        barrier.wait()
        results.append(dream(mem, [ses]))

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert len(results) == 6 and len({j.id for j in results}) == 6
    for j in results:
        _check_job(j)
    assert tree_digest(mem, ses) == digest
    assert len(DreamJob.list()) == 6


def test_async_jobs_same_store(tmp_path):
    """Several dream_async jobs started back to back all complete with consistent reports."""
    mem, ses = workspace(tmp_path)
    jobs = [dream_async(mem, [ses]) for _ in range(4)]
    for j in jobs:
        j.wait(60)
        _check_job(j)
    counts = {json.dumps(j.counts, sort_keys=True) for j in jobs}
    assert len(counts) == 1  # deterministic: same input -> same result


def test_processes_dream_same_store(tmp_path):
    """Three CLI processes dreaming over the same store and jobs dir at the same time: all succeed."""
    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem, ses)
    procs = [subprocess.Popen([sys.executable, "-m", "agentdream", "dream", "--memory", str(mem), "--sessions",
                               str(ses), "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(3)]
    outs = [p.communicate(timeout=120) for p in procs]
    assert [p.returncode for p in procs] == [0, 0, 0], [o[1] for o in outs]
    ids = [json.loads(o[0])["id"] for o in outs]
    assert len(set(ids)) == 3
    for i in ids:
        _check_job(DreamJob.load(i))
    assert tree_digest(mem, ses) == digest


def test_job_id_collision_is_avoided(tmp_path, monkeypatch):
    """Two jobs created in the same second with the same random suffix still get distinct ids and dirs."""
    mem, ses = workspace(tmp_path)
    monkeypatch.setattr(jobs_mod.secrets, "token_hex", lambda n: "abcdef")
    monkeypatch.setattr(jobs_mod.time, "strftime", lambda fmt, t=None: "dream-20260101-000000-")
    a, b = dream(mem, [ses]), dream(mem, [ses])
    assert a.id != b.id and a.dir != b.dir
    _check_job(a)
    _check_job(b)


def test_concurrent_accepts_one_wins_other_refuses(tmp_path):
    """Two different dreams accepted onto the same store at once: exactly one applies, the other is refused
    because the store changed under it (no interleaved / corrupted result)."""
    mem, ses = workspace(tmp_path)
    j1, j2 = dream(mem, [ses]), dream(mem, [ses])
    barrier = threading.Barrier(2)
    outcome: dict = {}

    def run(job):
        barrier.wait()
        try:
            accept(job)
            outcome[job.id] = "ok"
        except ReviewError as exc:
            outcome[job.id] = str(exc)

    ts = [threading.Thread(target=run, args=(j,)) for j in (j1, j2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    oks = [k for k, v in outcome.items() if v == "ok"]
    assert len(oks) == 1, outcome
    other = next(v for k, v in outcome.items() if v != "ok")
    assert "changed since the dream" in other
    # the store equals exactly the winner's staged output
    winner = DreamJob.load(oks[0])
    assert [e.content for e in get_adapter(mem).load()] == [e.content for e in get_adapter(winner.store_path).load()]
