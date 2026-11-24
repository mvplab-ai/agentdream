"""Crashes and kills mid-dream and mid-accept: input intact, job failed, nothing half-written treated as
complete, originals restored from backup (and `agentdream restore JOB`)."""

from __future__ import annotations

import os
import pathlib
import re
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from agentdream import DreamJob, Pipeline, ReviewError, accept, dream, get_adapter, make_config
from agentdream.adapters.markdown import MarkdownAdapter
from agentdream.cli import main
from agentdream.review import restore

from edgekit import tree_digest, workspace, write

PHASES = Pipeline.default().names()


def _crashing(phase: str, after: bool) -> Pipeline:
    pipe = Pipeline.default()
    orig = dict(pipe.phases)[phase]

    def boom(ctx):
        if after:
            orig(ctx)
        raise RuntimeError(f"injected crash in {phase}")

    return pipe.replace(phase, boom)


def _assert_failed_cleanly(job, mem_digest, mem):
    assert job.status == "failed" and "injected crash" in (job.error or "")
    assert tree_digest(mem) == mem_digest
    again = DreamJob.load(job.id)
    assert again.status == "failed"
    with pytest.raises(FileNotFoundError):
        _ = again.report
    with pytest.raises(ReviewError, match="only completed"):
        accept(again)
    assert (job.dir / "error.log").exists()


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("after", [False, True], ids=["before", "after"])
def test_exception_at_each_phase(tmp_path, phase, after):
    """An exception before/after each pipeline phase: job failed, input hash unchanged, no report served."""
    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem)
    job = dream(mem, [ses], pipeline=_crashing(phase, after))
    _assert_failed_cleanly(job, digest, mem)


def test_crash_between_store_and_report_write(tmp_path, monkeypatch):
    """Crash after the staged store is written but while writing report.md: job failed, store not presented."""
    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem)
    import agentdream.pipeline as pl

    real = pl.atomic_write_text

    def flaky(path, text, encoding="utf-8"):
        if str(path).endswith("report.md"):
            raise OSError("disk full")
        return real(path, text, encoding)

    monkeypatch.setattr(pl, "atomic_write_text", flaky)
    job = dream(mem, [ses])
    assert job.status == "failed" and "disk full" in job.error
    assert tree_digest(mem) == digest
    assert job.store_path is None
    with pytest.raises(FileNotFoundError):
        _ = job.report_md
    code = main(["show", job.id])
    assert code == 2


def test_show_cli_refuses_failed_job(tmp_path, capsys):
    """`agentdream show` on a failed job prints one clean error line (exit 2)."""
    mem, ses = workspace(tmp_path)
    job = dream(mem, [ses], pipeline=_crashing("write", after=True))
    assert main(["show", job.id]) == 2
    err = capsys.readouterr().err
    assert "failed" in err and "Traceback" not in err


_KILL_SCRIPT = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from agentdream import DreamJob, Pipeline, make_config

    mem, ses, marker, where = sys.argv[1:5]

    def stall(ctx):
        Path(marker).write_text(ctx.stats["job_id"])
        time.sleep(120)

    pipe = Pipeline.default().insert_after(where, "stall", stall)
    DreamJob(make_config(mem, [ses]), pipeline=pipe).run()
""")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX-only")
@pytest.mark.parametrize("where", ["consolidate", "write"])
def test_sigkill_mid_dream(tmp_path, where):
    """A real worker process SIGKILLed mid-dream (mid-analysis, or after write but before 'completed'):
    the next status shows failed, the input is intact, partial output is not accepted or shown."""
    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem)
    script = write(tmp_path / "worker.py", _KILL_SCRIPT)
    marker = tmp_path / "marker"
    proc = subprocess.Popen([sys.executable, str(script), str(mem), str(ses), str(marker), where])
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "worker never reached the stall phase"
        time.sleep(0.05)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.wait(timeout=30)
    job_id = marker.read_text()
    job = DreamJob.load(job_id)
    assert job.status == "failed" and "interrupted" in job.error
    assert tree_digest(mem) == digest
    if where == "write":
        assert (job.out_dir / "report.json").exists()  # on disk, but must not be served as complete
    with pytest.raises(FileNotFoundError):
        _ = job.report
    with pytest.raises(ReviewError):
        accept(job)
    assert main(["status", job_id]) == 0


def _accepted_setup(tmp_path):
    mem, ses = workspace(tmp_path)
    job = dream(mem, [ses])
    assert job.status == "completed" and job.operations
    return mem, job


def _leftovers(parent: pathlib.Path):
    return [p.name for p in parent.iterdir() if re.search(r"agentdream-(new|old)|\.tmp(-write)?$", p.name)]


def test_crash_during_write_after_backup(tmp_path, monkeypatch):
    """Exception inside the adapter write (after the backup exists): original intact, clear error, no temp leftovers."""
    mem, job = _accepted_setup(tmp_path)
    digest = tree_digest(mem)
    real = MarkdownAdapter.write

    def half_write(self, entries, dest):
        real(self, entries[:1], dest)  # something half-written on disk ...
        raise OSError("disk full")       # ... then the crash

    monkeypatch.setattr(MarkdownAdapter, "write", half_write)
    with pytest.raises(ReviewError, match="disk full") as ei:
        accept(job)
    assert "intact" in str(ei.value)
    assert tree_digest(mem) == digest
    assert _leftovers(mem.parent) == []
    assert DreamJob.load(job.id).review == "pending"


def test_crash_during_swap_restores_original(tmp_path, monkeypatch):
    """Exception while swapping the new store in (old store already moved aside): the original is put back."""
    mem, job = _accepted_setup(tmp_path)
    digest = tree_digest(mem)
    real_rename = pathlib.Path.rename

    def flaky_rename(self, target):
        if self.name.endswith("agentdream-new"):
            raise OSError("power loss")
        return real_rename(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", flaky_rename)
    with pytest.raises(ReviewError, match="power loss"):
        accept(job)
    assert tree_digest(mem) == digest
    assert _leftovers(mem.parent) == []


def test_crash_during_swap_single_file_store(tmp_path, monkeypatch):
    """Same as above for a single-file (JSON) store swapped with os.replace."""
    store = write(tmp_path / "m.json", '[{"id": "a", "content": "Use pytest."}, {"id": "b", "content": "Use pytest!"}]')
    _, ses = workspace(tmp_path / "w")
    job = dream(store, [ses])
    digest = tree_digest(store)
    import agentdream.review as rv

    real_replace = os.replace

    def flaky(src, dst):
        if str(src).endswith("agentdream-new"):
            raise OSError("power loss")
        return real_replace(src, dst)

    monkeypatch.setattr(rv.os, "replace", flaky)
    with pytest.raises(ReviewError, match="power loss"):
        accept(job)
    assert tree_digest(store) == digest
    assert _leftovers(tmp_path) == []


def test_restore_command_puts_backup_back(tmp_path, capsys):
    """After accept, `agentdream restore JOB` restores the backup byte-for-byte; the dream can be re-accepted."""
    mem, job = _accepted_setup(tmp_path)
    digest = tree_digest(mem)
    res = accept(job)
    assert res.backup and tree_digest(mem) != digest
    assert main(["restore", job.id]) == 0
    assert "restored" in capsys.readouterr().out
    assert tree_digest(mem) == digest
    again = DreamJob.load(job.id)
    assert again.review == "restored"
    accept(again)  # store matches the dream's input again
    assert tree_digest(mem) != digest


def test_restore_python_api_and_to(tmp_path):
    """restore(job, to=DIR) writes the backup to another location instead of the accepted target."""
    mem, job = _accepted_setup(tmp_path)
    original = {e.content for e in get_adapter(mem).load()}
    accept(job)
    elsewhere = tmp_path / "restored-copy"
    restore(job.id, to=elsewhere)
    assert {e.content for e in get_adapter(elsewhere).load()} == original


def test_restore_without_backup_is_a_clean_error(tmp_path, capsys):
    """restore on a job accepted with --no-backup (or never accepted) fails with a clear message."""
    _mem, job = _accepted_setup(tmp_path)
    assert main(["restore", job.id]) == 2
    assert "no backup" in capsys.readouterr().err
    accept(job, backup=False)
    with pytest.raises(ReviewError, match="no backup"):
        restore(job.id)


def test_restore_with_missing_backup_files(tmp_path):
    """If the backup directory was deleted, restore says so instead of crashing."""
    import shutil

    _mem, job = _accepted_setup(tmp_path)
    res = accept(job)
    shutil.rmtree(pathlib.Path(res.backup).parent)
    with pytest.raises(ReviewError, match=r"backup .* is missing"):
        restore(job.id)


def test_job_never_started_then_process_died(tmp_path):
    """A 'pending' job.json whose creator died is reported failed on the next load."""
    mem, ses = workspace(tmp_path)
    job = DreamJob(make_config(mem, [ses]))
    job.pid = 2 ** 22 + 12345  # no such process
    job.save()
    assert DreamJob.load(job.id).status == "failed"
