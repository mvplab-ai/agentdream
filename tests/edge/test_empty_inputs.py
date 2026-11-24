"""Empty and missing inputs: stores, store paths, transcripts, sessions."""

from __future__ import annotations

import json

import pytest

from agentdream import DreamJob, accept, dream, get_adapter, make_config
from agentdream.adapters import AdapterError
from agentdream.cli import main
from agentdream.loaders import LoaderError, load_transcripts

from edgekit import tree_digest, workspace, write, write_bytes, write_jsonl


def test_empty_markdown_store_dir_completes(tmp_path):
    """Empty memory folder (no entries): dream completes, 0 entries before, nothing crashes."""
    mem = tmp_path / "mem"
    mem.mkdir()
    _, ses = workspace(tmp_path / "w")
    job = dream(mem, [ses])
    assert job.status == "completed", job.error
    assert job.report["counts"]["entries_before"] == 0
    assert job.report["counts"]["unchanged"] == 0


def test_store_with_only_headings_has_no_entries(tmp_path):
    """Markdown file with headings and blank lines only -> zero entries, dream completes."""
    mem = write(tmp_path / "mem" / "MEMORY.md", "# Memory\n\n## Empty section\n\n\n").parent
    _, ses = workspace(tmp_path / "w")
    assert get_adapter(mem).load() == []
    job = dream(mem, [ses])
    assert job.status == "completed", job.error


def test_empty_json_and_jsonl_stores(tmp_path):
    """0-byte .json and .jsonl stores load as empty and round-trip through a dream."""
    _, ses = workspace(tmp_path / "w")
    for name in ("m.json", "m.jsonl"):
        store = write_bytes(tmp_path / name, b"")
        assert get_adapter(store).load() == []
        job = dream(store, [ses])
        assert job.status == "completed", job.error


def test_missing_store_path_fails_clearly(tmp_path):
    """Memory path that does not exist -> job failed with AdapterError naming the path."""
    _, ses = workspace(tmp_path / "w")
    job = dream(tmp_path / "nope", [ses])
    assert job.status == "failed"
    assert "AdapterError" in job.error and "nope" in job.error


def test_missing_store_path_cli_clean_error(tmp_path, capsys):
    """CLI with a missing memory path exits 1 with one error line, no traceback."""
    _, ses = workspace(tmp_path / "w")
    code = main(["dream", "--memory", str(tmp_path / "nope"), "--sessions", str(ses)])
    err = capsys.readouterr().err
    assert code == 1
    assert "memory store not found" in err and "Traceback" not in err


def test_json_store_path_is_a_directory(tmp_path):
    """Directory named like a file store (mem.json/) forced to json: 0 entries, other files survive accept."""
    d = tmp_path / "mem.json"
    d.mkdir()
    write(d / "notes.md", "- a note\n")
    _, ses = workspace(tmp_path / "w")
    job = dream(d, [ses], memory_format="json")
    assert job.status == "completed", job.error
    assert job.report["counts"]["entries_before"] == 0
    accept(job, backup=False)
    assert (d / "notes.md").read_text() == "- a note\n"  # non-JSON files are not deleted by the swap


def test_sqlite_format_forced_on_a_directory(tmp_path):
    """--memory-format sqlite on a directory -> failed with AdapterError 'not a file'."""
    d = tmp_path / "db"
    d.mkdir()
    _, ses = workspace(tmp_path / "w")
    job = dream(d, [ses], memory_format="sqlite")
    assert job.status == "failed"
    assert "AdapterError" in job.error and "not a file" in job.error


def test_unknown_file_store_type(tmp_path):
    """A memory file of an unknown type (.yaml) -> AdapterError 'could not detect'."""
    f = write(tmp_path / "mem.yaml", "a: b\n")
    with pytest.raises(AdapterError, match="could not detect"):
        get_adapter(f)


def test_accept_refuses_file_vs_dir_target_mismatch(tmp_path):
    """accept --to an existing *file* for a folder store is refused; the file is untouched."""
    mem, ses = workspace(tmp_path)
    job = dream(mem, [ses])
    target = write(tmp_path / "target.md", "precious\n")
    from agentdream import ReviewError

    with pytest.raises(ReviewError, match="is a file"):
        accept(job, to=target, backup=False)
    assert target.read_text() == "precious\n"


def test_accept_refuses_dir_target_for_single_file_store(tmp_path):
    """accept --to an existing *directory* for a single-file store is refused, nothing deleted."""
    mem = write(tmp_path / "MEMORY.md", "- Use pytest.\n- Use pytest!\n")
    _, ses = workspace(tmp_path / "w")
    job = dream(mem, [ses])
    target = tmp_path / "somedir"
    write(target / "keep.txt", "keep")
    from agentdream import ReviewError

    with pytest.raises(ReviewError, match="is a directory"):
        accept(job, to=target, backup=False)
    assert (target / "keep.txt").read_text() == "keep"


def test_zero_byte_transcript_skipped_with_notice(tmp_path):
    """A 0-byte transcript is skipped with a notice; other sessions still load."""
    _, ses = workspace(tmp_path)
    write_bytes(ses / "0-empty.jsonl", b"")
    ts, notices = load_transcripts([ses])
    assert len(ts) == 2
    assert any("skipped empty transcript" in n and "0-empty" in n for n in notices)


def test_whitespace_only_transcript_skipped(tmp_path):
    """A whitespace-only transcript (spaces, tabs, CR/LF) is skipped as empty."""
    _, ses = workspace(tmp_path)
    write_bytes(ses / "ws.txt", b"  \t\r\n\n   \r\n")
    ts, notices = load_transcripts([ses])
    assert len(ts) == 2 and any("ws.txt" in n for n in notices)


def test_tool_noise_only_transcript_skipped(tmp_path):
    """Claude Code session with only thinking, successful tool results and system reminders is skipped."""
    _, ses = workspace(tmp_path)
    rows = [
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x", "content": "ok", "is_error": False}]}},
        {"type": "user", "message": {"role": "user", "content": "<system-reminder>noise</system-reminder>"}},
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta"}},
    ]
    write_jsonl(ses / "noise.jsonl", rows)
    ts, notices = load_transcripts([ses])
    assert [t.id for t in ts] == ["1", "2"]
    assert any("noise.jsonl" in n for n in notices)


def test_all_transcripts_empty_fails_clearly(tmp_path):
    """Every transcript empty -> job failed: 'need at least 1 session', mentions the skipped files."""
    mem, _ = workspace(tmp_path)
    ses = tmp_path / "empty"
    write_bytes(ses / "a.jsonl", b"")
    write_bytes(ses / "b.txt", b"\n\n")
    job = dream(mem, [ses])
    assert job.status == "failed"
    assert "need at least 1 session" in job.error and "2 empty" in job.error


def test_no_sessions_given_fails_clearly(tmp_path):
    """No session sources at all -> job failed with 'need at least 1 session'."""
    mem, _ = workspace(tmp_path)
    job = dream(mem, [])
    assert job.status == "failed" and "need at least 1 session" in job.error


def test_empty_sessions_directory(tmp_path):
    """An existing but empty sessions directory -> LoaderError, not a crash."""
    d = tmp_path / "none"
    d.mkdir()
    with pytest.raises(LoaderError, match="need at least 1"):
        load_transcripts([d], min_sessions=1)


def test_missing_sessions_path(tmp_path):
    """A sessions path that doesn't exist -> LoaderError naming it."""
    with pytest.raises(LoaderError, match="not found"):
        load_transcripts([tmp_path / "missing"])


def test_empty_store_input_untouched_and_job_json_valid(tmp_path):
    """Dream over an empty store leaves it byte-identical and writes a parseable job.json."""
    mem = tmp_path / "mem"
    mem.mkdir()
    _, ses = workspace(tmp_path / "w")
    before = tree_digest(mem, ses)
    job = dream(mem, [ses])
    assert tree_digest(mem, ses) == before
    assert json.loads(job.path.read_text())["status"] == "completed"
    assert DreamJob.load(job.id).status == "completed"
    assert make_config(mem, [ses]).memory == str(mem.absolute())
