"""Path traversal: hostile entry ids / paths / keys, symlinks out of the store, job ids with ../, --out overlap."""

from __future__ import annotations

import json
import os

import pytest

from agentdream import DreamJob, accept, dream, get_adapter
from agentdream.adapters.markdown import MarkdownAdapter
from agentdream.cli import main
from agentdream.jobs import JobNotFound
from agentdream.mcp_server import MCPServer
from agentdream.models import MemoryEntry

from edgekit import FakeLLM, tree_digest, workspace, write

HOSTILE = ["../../outside.md", "../outside.md", "/tmp/agentdream-abs-outside.md", "a/../../outside.md",
           "..\\..\\outside.md", "C:\\evil.md", "notes/../../outside.md", "", ".", "..", "./../x.md", "~/x.md"]


def _all_files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("bad", HOSTILE)
def test_markdown_write_keeps_hostile_paths_inside_dest(tmp_path, bad):
    """Entries whose path is ../.., absolute, Windows-style or '..' are written inside dest, never outside."""
    mem = write(tmp_path / "box" / "mem" / "MEMORY.md", "- keep me\n").parent
    ad = MarkdownAdapter(mem)
    entries = [*ad.load(), MemoryEntry(id="x", content="hostile note", path=bad)]
    dest = tmp_path / "box" / "out"
    ad.write(entries, dest)
    assert not (tmp_path / "box" / "outside.md").exists() and not (tmp_path / "outside.md").exists()
    assert not os.path.exists("/tmp/agentdream-abs-outside.md")
    for p in dest.rglob("*"):
        assert dest.resolve() in p.resolve().parents
    contents = [e.content for e in MarkdownAdapter(dest).load()]
    assert "hostile note" in contents and "keep me" in contents


def test_json_store_with_traversal_ids_and_keys(tmp_path):
    """A JSON map store whose keys / ids look like ../../etc/passwd: kept as data, all writes inside dest."""
    store = write(tmp_path / "box" / "m.json",
                  json.dumps({"../../etc/passwd": "Use pytest.", "/etc/shadow": "Use pytest!", "ok": "fine"}))
    _, ses = workspace(tmp_path / "w")
    job = dream(store, [ses])
    assert job.status == "completed", job.error
    out = tmp_path / "box" / "accepted.json"
    accept(job, to=out)
    assert _all_files(tmp_path / "box") == {"m.json", "accepted.json"}
    keys = set(json.loads(out.read_text()))
    assert keys <= {"../../etc/passwd", "/etc/shadow", "ok"}


def test_json_folder_store_entry_paths_cannot_escape(tmp_path):
    """JSON folder store: an entry whose path points outside is written to a file inside dest."""
    from agentdream.adapters.jsonstore import JSONAdapter

    d = write(tmp_path / "box" / "mem" / "a.json", json.dumps([{"id": "1", "content": "one"}])).parent
    ad = JSONAdapter(d)
    entries = [*ad.load(), MemoryEntry(id="2", content="two", path="../../evil.json")]
    ad.write(entries, tmp_path / "box" / "out")
    assert not (tmp_path / "evil.json").exists() and not (tmp_path / "box" / "evil.json").exists()
    assert {e.content for e in JSONAdapter(tmp_path / "box" / "out").load()} == {"one", "two"}


def test_llm_targets_with_traversal_ids_are_dropped(tmp_path):
    """A model naming '../../etc/passwd' as a target id: op dropped as unknown id, nothing written outside."""
    mem, ses = workspace(tmp_path / "box")
    fake = FakeLLM({"consolidate": {"operations": [
        {"type": "merge", "targets": ["../../etc/passwd", "MEMORY.md#2"], "new_content": "pwned"}]}})
    job = dream(mem, [ses], backend=fake)
    assert any("unknown entry id" in w and "etc/passwd" in w for w in job.warnings)
    assert "pwned" not in json.dumps(job.report)


def test_symlinked_file_outside_store_is_not_read(tmp_path):
    """A symlink in the memory folder pointing to a file outside the store is not loaded or sent to the model,
    and is kept as a symlink (not copied as content) in the output."""
    secret = write(tmp_path / "private" / "id_notes.md", "- TOP SECRET outside content\n")
    mem, ses = workspace(tmp_path / "box")
    os.symlink(secret, mem / "linked.md")
    entries = get_adapter(mem).load()
    assert not any("TOP SECRET" in e.content for e in entries)
    fake = FakeLLM()
    job = dream(mem, [ses], backend=fake)
    assert "TOP SECRET" not in fake.sent
    assert any("symlink" in n and "linked.md" in n for n in job.report["notices"])
    staged = job.out_dir / "store"
    assert (staged / "linked.md").is_symlink() and os.readlink(staged / "linked.md") == str(secret)
    digest = tree_digest(secret)
    accept(job)
    assert (mem / "linked.md").is_symlink() and tree_digest(secret) == digest


def test_symlinked_dir_outside_store_is_not_followed(tmp_path):
    """A symlinked directory pointing outside the store is not traversed."""
    write(tmp_path / "private" / "x.md", "- OUTSIDE DIR NOTE\n")
    mem, _ = workspace(tmp_path / "box")
    os.symlink(tmp_path / "private", mem / "linkdir", target_is_directory=True)
    assert not any("OUTSIDE DIR NOTE" in e.content for e in get_adapter(mem).load())


def test_symlink_inside_store_is_fine(tmp_path):
    """A symlink to another file *inside* the store is read normally."""
    mem, _ = workspace(tmp_path / "box")
    write(mem / "real" / "inner.md", "- inner note\n")
    os.symlink(mem / "real" / "inner.md", mem / "alias.md")
    contents = [e.content for e in get_adapter(mem).load()]
    assert contents.count("inner note") == 2


@pytest.mark.parametrize("bad_id", ["../../etc", "../jobs", "a/../../b", "/etc/passwd", "dream-*", "x?y", "a\x00b",
                                    "..", ".", "", "a/b"])
def test_job_ids_with_traversal_rejected(bad_id, tmp_path):
    """Job ids containing path separators, '..', globs or NUL are rejected before touching the filesystem."""
    with pytest.raises(JobNotFound, match="invalid job id"):
        DreamJob.load(bad_id)


def test_cli_and_mcp_reject_traversal_job_ids(tmp_path, capsys):
    """`agentdream show ../../x` and the MCP accept tool with job '../x' fail cleanly."""
    assert main(["show", "../../etc"]) == 2
    assert "invalid job id" in capsys.readouterr().err
    server = MCPServer(None)
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "accept", "arguments": {"job_id": "../x"}}})
    assert resp["result"]["isError"] and "invalid job id" in resp["result"]["content"][0]["text"]


def test_job_dir_symlink_escape_rejected(tmp_path, isolated_home):
    """A job dir that is a symlink to somewhere outside the jobs dir is refused."""
    jobs = isolated_home / "jobs"
    jobs.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere" / "dream-x"
    write(elsewhere / "job.json", json.dumps({"id": "dream-x", "status": "completed"}))
    os.symlink(elsewhere, jobs / "dream-x", target_is_directory=True)
    with pytest.raises(JobNotFound, match="outside"):
        DreamJob.load("dream-x")


@pytest.mark.parametrize("where", ["same", "inside", "parent-of-store"])
def test_out_dir_overlapping_input_store_refused(tmp_path, where):
    """--out equal to / inside the memory store, or an --out whose store/ would be the input, is refused."""
    mem, ses = workspace(tmp_path)
    digest = tree_digest(mem)
    out = {"same": mem, "inside": mem / "dreams", "parent-of-store": tmp_path / "x"}[where]
    if where == "parent-of-store":
        mem2 = out / "store"
        mem2.parent.mkdir()
        os.rename(mem, mem2)
        mem = mem2
        digest = tree_digest(mem)
    job = dream(mem, [ses], out=str(out))
    assert job.status == "failed" and "overlaps the memory store" in job.error
    assert tree_digest(mem) == digest
