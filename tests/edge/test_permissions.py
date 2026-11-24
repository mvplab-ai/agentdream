"""Read-only directories and permission errors: clean errors, no traceback spam, input untouched."""

from __future__ import annotations

import contextlib
import os
import stat

import pytest

from agentdream import DreamJob, dream
from agentdream.cli import main

from edgekit import tree_digest, workspace, write

pytestmark = pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                                reason="permission checks are meaningless as root / on Windows")


@contextlib.contextmanager
def read_only(path):
    """chmod 0o500 (r-x) for the duration, always restored so tmp cleanup works."""
    old = stat.S_IMODE(path.stat().st_mode)
    path.chmod(0o500)
    try:
        yield path
    finally:
        path.chmod(old)


def _no_traceback(capsys):
    cap = capsys.readouterr()
    assert "Traceback" not in cap.err and "Traceback" not in cap.out
    return cap


def test_read_only_output_dir(tmp_path, capsys):
    """--out under a read-only directory: job failed with 'Permission denied', CLI exit 1, input untouched."""
    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem, ses)
    ro = tmp_path / "ro"
    ro.mkdir()
    with read_only(ro):
        code = main(["dream", "--memory", str(mem), "--sessions", str(ses), "--out", str(ro / "out")])
    cap = _no_traceback(capsys)
    assert code == 1 and "Permission denied" in cap.err
    assert tree_digest(mem, ses) == digest
    job = DreamJob.list()[-1]
    assert job.status == "failed"


def test_read_only_jobs_dir(tmp_path, capsys, isolated_home):
    """Jobs dir whose parent is read-only: one-line error (exit 2), no traceback, input untouched."""
    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem, ses)
    isolated_home.mkdir(parents=True, exist_ok=True)
    with read_only(isolated_home):
        code = main(["dream", "--memory", str(mem), "--sessions", str(ses)])
    cap = _no_traceback(capsys)
    assert code == 2 and "Permission denied" in cap.err
    assert len(cap.err.strip().splitlines()) == 1
    assert tree_digest(mem, ses) == digest


def test_jobs_dir_becomes_read_only_mid_run(tmp_path, isolated_home):
    """If the job dir turns read-only while the dream runs, the API raises a clean OSError; input untouched."""
    from agentdream import Pipeline

    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem, ses)
    holder: dict = {}

    def lock_down(ctx):
        d = isolated_home / "jobs" / ctx.stats["job_id"]
        holder["d"] = d
        d.chmod(0o500)

    pipe = Pipeline.default().insert_after("load", "lock", lock_down)
    try:
        with pytest.raises(PermissionError):
            dream(mem, [ses], pipeline=pipe)
    finally:
        holder["d"].chmod(0o700)
    assert tree_digest(mem, ses) == digest


def test_accept_into_read_only_target_dir(tmp_path, capsys):
    """accept --to inside a read-only directory: exit 2, clean message, review stays pending."""
    mem, ses = workspace(tmp_path)
    job = dream(mem, [ses])
    digest = tree_digest(mem)
    ro = tmp_path / "ro"
    ro.mkdir()
    with read_only(ro):
        code = main(["accept", job.id, "--to", str(ro / "dest")])
    cap = _no_traceback(capsys)
    assert code == 2 and "Permission denied" in cap.err
    assert tree_digest(mem) == digest
    assert DreamJob.load(job.id).review == "pending"


def test_accept_over_store_in_read_only_parent(tmp_path, capsys):
    """accept over the original store when its parent dir is read-only (can't swap): original intact."""
    mem, ses = workspace(tmp_path)
    job = dream(mem, [ses])
    digest = tree_digest(mem)
    with read_only(mem.parent):
        code = main(["accept", job.id])
    _no_traceback(capsys)
    assert code == 2
    assert tree_digest(mem) == digest
    assert DreamJob.load(job.id).review == "pending"


def test_unreadable_memory_file(tmp_path):
    """A memory file with mode 000: job failed with a clear error naming the file."""
    mem, ses = workspace(tmp_path)
    f = mem / "MEMORY.md"
    f.chmod(0o000)
    try:
        job = dream(mem, [ses])
    finally:
        f.chmod(0o600)
    assert job.status == "failed" and "Permission denied" in job.error and "MEMORY.md" in job.error


def test_unreadable_session_file(tmp_path):
    """An unreadable transcript: job failed with LoaderError naming the file (not a bare traceback)."""
    mem, ses = workspace(tmp_path)
    f = ses / "1.jsonl"
    f.chmod(0o000)
    try:
        job = dream(mem, [ses])
    finally:
        f.chmod(0o600)
    assert job.status == "failed" and "LoaderError" in job.error and "1.jsonl" in job.error


def test_read_only_store_dir_is_fine_for_dreaming(tmp_path):
    """A read-only memory store can still be dreamed over (input is only read)."""
    mem, ses = workspace(tmp_path)
    write(mem / "notes.md", "- another\n")
    with read_only(mem):
        job = dream(mem, [ses])
    assert job.status == "completed", job.error


def test_scheduler_state_in_read_only_dir(tmp_path, capsys):
    """`schedule --once` with a state file in a read-only dir: clean error (exit 2), no traceback."""
    mem, ses = workspace(tmp_path)
    ro = tmp_path / "ro"
    ro.mkdir()
    with read_only(ro):
        code = main(["schedule", "--memory", str(mem), "--sessions", str(ses), "--new-sessions", "1", "--once",
                     "--state-file", str(ro / "state.json")])
    cap = _no_traceback(capsys)
    assert code == 2 and "Permission denied" in cap.err
