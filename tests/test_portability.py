"""Cross-platform robustness: symlinked temp dirs (macOS /var -> /private/var), coarse clocks,
same-second file rewrites."""

import os

import pytest

from agentdream import Scheduler, accept, dream
from agentdream.jobs import DreamJob

from conftest import hash_tree, write, write_jsonl


def test_jobs_created_in_same_second_keep_creation_order(tmp_path):
    from agentdream.context import DreamConfig

    jd = tmp_path / "jobs"
    ids = []
    for _ in range(5):
        j = DreamJob(DreamConfig(memory=str(tmp_path)), jobs_dir=jd)
        j.created_at = "2026-01-01T00:00:00+00:00"  # simulate a 1-second clock
        j.save()
        ids.append(j.id)
    assert [j.id for j in DreamJob.list(jd)] == ids


def test_scheduler_sees_same_second_rewrite(tmp_path):
    s = tmp_path / "s"
    p = write(s / "a.txt", "User: hi\n")
    os.utime(p, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    class Done:
        status = "completed"

    sch = Scheduler(str(tmp_path), [str(s)], new_sessions=1, state_file=tmp_path / "st.json",
                    dream_fn=lambda m, ss, apply=False, **o: Done())
    assert sch.run_once() is not None
    assert sch.pending_sessions() == []
    p.write_text("User: yo\n")  # same size, rewritten "within the same second"
    os.utime(p, ns=(1_000_000_000_000_000_000, 1_000_000_000_500_000_000))
    assert [x.name for x in sch.pending_sessions()] == ["a.txt"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_paths_through_a_symlinked_directory(tmp_path):
    """macOS tmp dirs live behind /var -> /private/var; resolving one side only must not break anything."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:  # pragma: no cover - e.g. Windows without privileges
        pytest.skip("cannot create symlinks here")
    mem = write(real / "mem" / "MEMORY.md", "- Use 4 spaces.\n- Use 4 spaces.\n").parent
    write_jsonl(real / "s" / "1.jsonl", [{"role": "user", "content": "hello"}])
    before = hash_tree(mem)
    job = dream(link / "mem", [link / "s"], jobs_dir=link / "jobs")
    assert job.status == "completed", job.error
    assert hash_tree(mem) == before
    res = accept(job.id, jobs_dir=link / "jobs")
    assert res.backup and os.path.exists(res.backup)
    assert (real / "mem" / "MEMORY.md").read_text().count("Use 4 spaces") == 1
