"""Job lifecycle, persistence, cancel; accept / partial accept / discard."""

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from agentdream import DreamJob, Pipeline, ReviewError, accept, discard, dream, get_adapter, make_config
from agentdream.review import select_operations

from conftest import hash_tree, ops_of, write, write_jsonl


def _setup(tmp_path):
    mem = write(tmp_path / "mem" / "MEMORY.md",
                "- The PokéCenter in Viridian City is north of the gym.\n"
                "- Use pytest for tests.\n- Use pytest for tests!\n"
                "- The default branch is master.\n- The default branch is main.\n"
                "- Untouched note about llamas.\n").parent
    s = tmp_path / "s"
    write_jsonl(s / "1.jsonl", [{"role": "assistant", "content": "Walked north of the gym. No PokéCenter here."},
                                {"role": "assistant", "content": "Still no PokéCenter north of the gym."}])
    write_jsonl(s / "2.jsonl", [{"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."}])
    return mem, s


def test_status_lifecycle_and_persistence(tmp_path, isolated_home):
    mem, s = _setup(tmp_path)
    seen = []
    pipe = Pipeline.default()
    orig_verify = dict(pipe.phases)["verify"]

    def spy(ctx):
        seen.append(DreamJob.load(job.id).status)
        orig_verify(ctx)

    pipe.replace("verify", spy)
    job = DreamJob(make_config(mem, [s]), pipeline=pipe)
    assert job.status == "pending"
    job.run()
    assert seen == ["running"]
    assert job.status == "completed" and job.progress == 1.0
    assert (isolated_home / "jobs" / job.id / "job.json").exists()
    again = DreamJob.load(job.id)
    assert again.status == "completed" and again.counts == job.counts
    assert again.report["counts"]["corrected"] == 1
    assert [j.id for j in DreamJob.list()] == [job.id]
    assert DreamJob.load(job.id[:-2]).id == job.id  # unique prefix


def test_status_survives_process_restart(tmp_path, isolated_home):
    mem, s = _setup(tmp_path)
    code = ("import sys; from agentdream import dream; j = dream(sys.argv[1], [sys.argv[2]]); print(j.id)")
    out = subprocess.run([sys.executable, "-c", code, str(mem), str(s)], capture_output=True, text=True,
                         check=True, cwd=tmp_path)
    job = DreamJob.load(out.stdout.strip())
    assert job.status == "completed" and job.counts["corrected"] == 1


def test_dead_worker_marks_job_failed(tmp_path, isolated_home):
    mem, s = _setup(tmp_path)
    job = DreamJob(make_config(mem, [s]))
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    job.status, job.pid = "running", p.pid
    job.save()
    loaded = DreamJob.load(job.id)
    assert loaded.status == "failed" and "interrupted" in loaded.error


def test_async_and_cancel(tmp_path):
    mem, s = _setup(tmp_path)
    gate, entered = threading.Event(), threading.Event()
    pipe = Pipeline.default()

    def slow(ctx):
        entered.set()
        gate.wait(5)
        ctx.check_cancel()

    pipe.insert_after("limits", "slow", slow)
    job = dream(mem, [s], wait=False, pipeline=pipe)
    assert entered.wait(5)
    assert job.status == "running"
    assert DreamJob.load(job.id).phase == "slow"
    job.cancel()
    gate.set()
    job.wait(5)
    assert job.status == "canceled"
    assert DreamJob.load(job.id).status == "canceled"
    assert not (job.out_dir / "report.json").exists()


def test_cancel_from_another_handle(tmp_path):
    """Cancellation via the persisted flag (what `agentdream cancel` does from another process)."""
    mem, s = _setup(tmp_path)
    gate, entered = threading.Event(), threading.Event()
    pipe = Pipeline.default()
    pipe.insert_after("redact", "wait", lambda ctx: (entered.set(), gate.wait(5)))
    job = dream(mem, [s], wait=False, pipeline=pipe)
    assert entered.wait(5)
    DreamJob.load(job.id).cancel()  # separate object, shares only the job file
    gate.set()
    job.wait(5)
    assert job.status == "canceled"


def test_failed_job(tmp_path):
    job = dream(tmp_path / "nope", [tmp_path])
    assert job.status == "failed" and "not found" in job.error
    with pytest.raises(ReviewError):
        accept(job)


def test_accept_all_with_backup(tmp_path):
    mem, s = _setup(tmp_path)
    original = (mem / "MEMORY.md").read_text()
    job = dream(mem, [s])
    assert (mem / "MEMORY.md").read_text() == original  # dry run
    res = accept(job.id)
    assert res.backup and Path(res.backup, "MEMORY.md").read_text() == original
    new = (mem / "MEMORY.md").read_text()
    assert "south of the gym" in new and "master" not in new and new.count("pytest") == 1
    assert "Untouched note about llamas." in new
    assert DreamJob.load(job.id).review == "accepted"
    assert set(res.applied_ops) == {o.id for o in job.operations} and not res.skipped_ops
    with pytest.raises(ReviewError):
        accept(job.id)  # already accepted


def test_partial_accept_only_and_exclude(tmp_path):
    mem, s = _setup(tmp_path)
    job = dream(mem, [s])
    corr = ops_of(job, "correct")[0]
    merge = ops_of(job, "merge")[0]
    out1 = tmp_path / "only"
    res = accept(job, only=corr.id, to=out1)
    assert res.applied_ops == [corr.id]
    text = (out1 / "MEMORY.md").read_text()
    assert "south of the gym" in text and "master" in text and text.count("pytest") == 2
    assert DreamJob.load(job.id).review == "partially_accepted"

    out2 = tmp_path / "excl"
    res2 = accept(job, exclude=[merge.id], to=out2, force=True)
    assert merge.id in res2.skipped_ops
    text2 = (out2 / "MEMORY.md").read_text()
    assert text2.count("pytest") == 2 and "south of the gym" in text2 and "master" not in text2


def test_partial_accept_respects_dependencies(tmp_path):
    store = tmp_path / "m.json"
    store.write_text(json.dumps([
        {"id": "a", "content": "The PokéCenter in Viridian City is north of the gym.", "updated_at": "2026-01-01"},
        {"id": "b", "content": "The PokéCenter in Viridian City is north of the gym!", "updated_at": "2026-01-02"},
    ]))
    s = tmp_path / "s"
    write_jsonl(s / "1.jsonl", [{"role": "assistant", "content": "No PokéCenter north of the gym."},
                                {"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."}])
    job = dream(store, [s])
    merge, corr = ops_of(job, "merge")[0], ops_of(job, "correct")[0]
    assert corr.depends_on == [merge.id]
    sel, warns = select_operations(job.operations, only=[corr.id])
    assert [o.id for o in sel] == [merge.id, corr.id] and warns
    sel, warns = select_operations(job.operations, exclude=[merge.id])
    assert sel == [] and warns
    res = accept(job, only=corr.id[:12], to=tmp_path / "o.json")  # prefix ids work too
    data = json.loads((tmp_path / "o.json").read_text())
    assert len(data) == 1 and "south" in data[0]["content"]
    assert res.warnings


def test_accept_refuses_if_store_changed(tmp_path):
    mem, s = _setup(tmp_path)
    job = dream(mem, [s])
    (mem / "MEMORY.md").write_text("- Someone edited this meanwhile.\n")
    with pytest.raises(ReviewError, match="changed"):
        accept(job)
    accept(job, force=True, to=tmp_path / "elsewhere")


def test_discard(tmp_path):
    mem, s = _setup(tmp_path)
    before = hash_tree(mem)
    job = dream(mem, [s])
    assert Path(job.store_path).exists()
    discard(job.id)
    j = DreamJob.load(job.id)
    assert j.review == "discarded" and not (job.out_dir / "store").exists()
    assert hash_tree(mem) == before
    with pytest.raises(ReviewError):
        accept(job.id)


def test_apply_flag_and_custom_out(tmp_path):
    mem, s = _setup(tmp_path)
    out = tmp_path / "staging"
    job = dream(mem, [s], out=out, apply=True)
    assert (out / "report.md").exists() and (out / "report.json").exists() and (out / "store" / "MEMORY.md").exists()
    assert "south of the gym" in (mem / "MEMORY.md").read_text()
    assert job.review == "accepted"


def test_report_structure(tmp_path):
    mem, s = _setup(tmp_path)
    job = dream(mem, [s])
    rep = job.report
    assert set(rep["sections"]) == {"merged", "replaced", "corrected", "insights", "contradictions"}
    assert rep["counts"]["unchanged"] == 1
    assert rep["diff"].startswith("--- a/MEMORY.md")
    md = job.report_md
    for title in ("## Merged", "## Replaced (stale)", "## Corrected", "## New insights", "## Contradictions",
                  "## Unchanged", "## Diff"):
        assert title in md
    ids = [o["id"] for o in rep["operations"]]
    assert len(ids) == len(set(ids))
    # op ids are stable across runs on the same input
    assert [o.id for o in dream(mem, [s]).operations] == ids


def test_sqlite_diff_rendered(tmp_path):
    from agentdream import MemoryEntry
    from agentdream.adapters import create_sqlite_store

    db = create_sqlite_store(tmp_path / "m.db", [MemoryEntry("1", "The default branch is master."),
                                                 MemoryEntry("2", "The default branch is main.")])
    write(tmp_path / "s.txt", "User: hi\n")
    job = dream(db, [tmp_path / "s.txt"])
    assert "-[1] The default branch is master." in job.report["diff"]
    assert [e.content for e in get_adapter(job.store_path).load()] == ["The default branch is main."]
