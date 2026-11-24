"""Accept on stale dreams, partial accept with bad / duplicate / empty selections, --only with --exclude."""

from __future__ import annotations

import pytest

from agentdream import DreamJob, ReviewError, accept, discard, dream
from agentdream.cli import main

from edgekit import tree_digest, workspace, write


@pytest.fixture
def done(tmp_path):
    mem, ses = workspace(tmp_path)
    job = dream(mem, [ses])
    assert job.status == "completed" and len(job.operations) >= 3, [o.id for o in job.operations]
    return mem, job


def _independent(job):
    return [o.id for o in job.operations if not o.depends_on and o.changes_store]


@pytest.mark.parametrize("change", ["edit", "add", "delete"])
def test_stale_accept_refused_by_default(done, capsys, change):
    """Store edited / file added / file deleted after the dream: accept refuses with a clear --force hint."""
    mem, job = done
    if change == "edit":
        (mem / "MEMORY.md").write_text((mem / "MEMORY.md").read_text() + "- new note\n")
    elif change == "add":
        write(mem / "extra.md", "- extra\n")
    else:
        write(mem / "extra.md", "- extra\n")
        job = dream(mem, [job.config.sessions[0]])
        (mem / "extra.md").unlink()
    digest = tree_digest(mem)
    with pytest.raises(ReviewError, match=r"changed since the dream.*--force"):
        accept(job)
    assert main(["accept", job.id]) == 2
    err = capsys.readouterr().err
    assert "--force" in err and "Traceback" not in err
    assert tree_digest(mem) == digest
    assert DreamJob.load(job.id).review == "pending"


def test_stale_accept_force_overrides(done):
    """--force applies a stale dream anyway (with a backup)."""
    mem, job = done
    (mem / "MEMORY.md").write_text((mem / "MEMORY.md").read_text() + "- new note\n")
    res = accept(job, force=True)
    assert res.backup and res.applied_ops


def test_unknown_op_id_rejected_nothing_written(done):
    """--only with an unknown id: ReviewError naming it; store untouched, review stays pending."""
    mem, job = done
    digest = tree_digest(mem)
    with pytest.raises(ReviewError, match="unknown or ambiguous operation id 'nope-123'"):
        accept(job, only="nope-123")
    with pytest.raises(ReviewError, match="'zzz'"):
        accept(job, exclude=["zzz"])
    assert tree_digest(mem) == digest and DreamJob.load(job.id).review == "pending"


def test_invalid_op_id_strings(done):
    """Ids with path separators, globs or control characters are rejected like unknown ids."""
    _mem, job = done
    for bad in ("../x", "*", "merge-*", "\x00", "a\nb"):
        with pytest.raises(ReviewError):
            accept(job, only=bad)


def test_ambiguous_prefix_lists_candidates():
    """A prefix matching several ops is refused and the message lists the matches; a unique prefix works."""
    from agentdream.models import Operation
    from agentdream.review import select_operations

    ops = [Operation(id="merge-aaa1", type="merge", summary=""), Operation(id="merge-aab2", type="merge", summary="")]
    with pytest.raises(ReviewError, match=r"ambiguous.*merge-aaa1.*merge-aab2"):
        select_operations(ops, only="merge-aa")
    assert [o.id for o in select_operations(ops, only="merge-aab")[0]] == ["merge-aab2"]


def test_duplicate_ids_applied_once(done):
    """--only a,a,a applies a once; no error, no duplicate in accepted_ops."""
    _mem, job = done
    op = _independent(job)[0]
    res = accept(job, only=f"{op},{op}, {op}")
    assert res.applied_ops == [op]
    assert DreamJob.load(job.id).accepted_ops == [op]


def test_empty_only_list_is_an_error(done, capsys):
    """--only '' (or only=[]) is refused instead of silently applying everything."""
    mem, job = done
    digest = tree_digest(mem)
    for empty in ("", " , ", []):
        with pytest.raises(ReviewError, match="no operation ids"):
            accept(job, only=empty)
    assert main(["accept", job.id, "--only", ""]) == 2
    assert tree_digest(mem) == digest


def test_only_and_exclude_together(done):
    """--only A,B --exclude B applies exactly A."""
    _mem, job = done
    a, b = _independent(job)[:2]
    res = accept(job, only=[a, b], exclude=b)
    assert res.applied_ops == [a]
    assert b in res.skipped_ops


def test_only_and_exclude_cancel_out(done):
    """--only A --exclude A leaves nothing to apply: refused, store untouched."""
    mem, job = done
    a = _independent(job)[0]
    digest = tree_digest(mem)
    with pytest.raises(ReviewError, match="no operations selected"):
        accept(job, only=a, exclude=a)
    assert tree_digest(mem) == digest


def test_exclude_everything_is_refused(done):
    """Excluding every operation is refused with a hint to use discard."""
    _mem, job = done
    with pytest.raises(ReviewError, match="discard"):
        accept(job, exclude=[o.id for o in job.operations])


def test_flag_only_selection_is_allowed(tmp_path):
    """Selecting only flag operations (no store change) is fine: the store is rewritten identically."""
    mem, ses = workspace(tmp_path, memory="- Don't use Poetry for dependency management. (2026-06-01)\n")
    write(ses / "3.jsonl", '{"role": "user", "content": "Let\'s use Poetry for dependency management.", '
                           '"timestamp": "2026-02-01T00:00:00Z"}\n')
    job = dream(mem, [ses])
    flags = [o.id for o in job.operations if not o.changes_store]
    assert flags
    digest = tree_digest(mem)
    accept(job, only=flags, backup=False)
    assert tree_digest(mem) == digest


def test_accept_twice_discard_after_accept(done):
    """Accepting twice, discarding after accept and accepting after discard are all refused."""
    _mem, job = done
    accept(job)
    with pytest.raises(ReviewError, match="already accepted"):
        accept(job)
    with pytest.raises(ReviewError, match="restore"):
        discard(job)


def test_accept_after_discard(done):
    """A discarded dream cannot be accepted."""
    _mem, job = done
    discard(job)
    with pytest.raises(ReviewError, match="discarded"):
        accept(job)


def test_corrupt_operations_file(done, capsys):
    """A truncated operations.json / snapshot.json gives a clean ReviewError, not a JSON traceback."""
    mem, job = done
    digest = tree_digest(mem)
    (job.dir / "operations.json").write_text('[{"id": "x", "type": ')
    with pytest.raises(ReviewError, match=r"operations\.json"):
        accept(job)
    assert main(["accept", job.id]) == 2
    assert "Traceback" not in capsys.readouterr().err
    assert tree_digest(mem) == digest


def test_corrupt_job_json(done, capsys):
    """A corrupt job.json: `status JOB` fails cleanly; `list` skips it."""
    _mem, job = done
    job.path.write_text("{not json")
    assert main(["status", job.id]) == 2
    assert "Traceback" not in capsys.readouterr().err
    assert main(["list"]) == 0
