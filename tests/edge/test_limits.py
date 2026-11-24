"""Size limits exactly at and just over each boundary: sessions, bytes per transcript, total tokens, entries."""

from __future__ import annotations

import itertools

import pytest

from agentdream import dream, make_config
from agentdream.context import DreamLimitError
from agentdream.pipeline import run_dream

from edgekit import workspace, write, write_jsonl


def _sessions(d, n, prefix="s"):
    for i in range(n):
        write_jsonl(d / f"{prefix}{i:03d}.jsonl",
                    [{"role": "user", "content": f"note number {i} about llamas",
                      "timestamp": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z"}])
    return d


_RUNS = itertools.count()


def _ctx(tmp_path, mem, ses, **kw):
    return run_dream(make_config(mem, [ses], **kw), tmp_path / f"out-{next(_RUNS)}")


def test_exactly_max_sessions_no_notice(tmp_path):
    """Exactly max_sessions (100) sessions: all kept, no truncation notice."""
    mem, _ = workspace(tmp_path)
    ses = _sessions(tmp_path / "many", 100)
    job = dream(mem, [ses])
    assert job.status == "completed", job.error
    assert len(job.report["sessions"]) == 100
    assert not any("max_sessions" in n for n in job.report["notices"])


def test_one_over_max_sessions_keeps_most_recent(tmp_path):
    """101 sessions with the default limit: the oldest is dropped, notice says so."""
    mem, _ = workspace(tmp_path)
    ses = _sessions(tmp_path / "many", 101)
    job = dream(mem, [ses])
    assert job.status == "completed", job.error
    ids = [s["id"] for s in job.report["sessions"]]
    assert len(ids) == 100 and "s000" not in ids and "s100" in ids
    assert any("101 sessions exceed max_sessions=100" in n for n in job.report["notices"])


def test_far_over_max_sessions_with_small_limit(tmp_path):
    """150 sessions, max_sessions=7: keeps the 7 most recent."""
    mem, _ = workspace(tmp_path)
    ses = _sessions(tmp_path / "many", 150)
    job = dream(mem, [ses], max_sessions=7)
    assert [s["id"] for s in job.report["sessions"]] == [f"s{i:03d}" for i in range(143, 150)]


def test_over_max_sessions_fail_mode(tmp_path):
    """on_limit=fail with max_sessions+1 sessions: job failed with DreamLimitError."""
    mem, _ = workspace(tmp_path)
    ses = _sessions(tmp_path / "many", 4)
    assert dream(mem, [ses], max_sessions=4, on_limit="fail").status == "completed"
    job = dream(mem, [ses], max_sessions=3, on_limit="fail")
    assert job.status == "failed" and "DreamLimitError" in job.error and "max_sessions=3" in job.error


def _byte_session(tmp_path, sizes):
    d = tmp_path / "bytes"
    rows = [{"role": "user", "content": "x" * n} for n in sizes]
    write_jsonl(d / "big.jsonl", rows)
    return d


def test_transcript_exactly_at_byte_limit_untouched(tmp_path):
    """A transcript whose text is exactly max_bytes_per_transcript bytes is not truncated."""
    mem, _ = workspace(tmp_path)
    ses = _byte_session(tmp_path, [600, 400])
    ctx = _ctx(tmp_path, mem, ses, max_bytes_per_transcript=1000)
    assert [len(m.content) for m in ctx.transcripts[0].messages] == [600, 400]
    assert not any("max_bytes_per_transcript" in n for n in ctx.notices)


def test_transcript_one_byte_over_drops_oldest_messages(tmp_path):
    """One byte over max_bytes_per_transcript: oldest messages dropped, notice with counts."""
    mem, _ = workspace(tmp_path)
    ses = _byte_session(tmp_path, [600, 400])
    ctx = _ctx(tmp_path, mem, ses, max_bytes_per_transcript=999)
    assert [len(m.content) for m in ctx.transcripts[0].messages] == [400]
    assert any("1000 bytes (> max_bytes_per_transcript=999)" in n and "1 dropped" in n for n in ctx.notices)


def test_single_message_over_byte_limit_is_cut_not_dropped(tmp_path):
    """One message larger than the byte limit is cut to fit (with a marker), not silently emptied."""
    mem, _ = workspace(tmp_path)
    ses = _byte_session(tmp_path, [5000])
    ctx = _ctx(tmp_path, mem, ses, max_bytes_per_transcript=1000)
    msgs = ctx.transcripts[0].messages
    assert len(msgs) == 1
    assert len(msgs[0].content.encode()) <= 1000 and msgs[0].content.endswith("[truncated]")
    assert any("cut to fit" in n for n in ctx.notices)


def test_byte_limit_cut_respects_multibyte_characters(tmp_path):
    """Cutting a message of 4-byte emoji never produces broken UTF-8 or exceeds the limit."""
    mem, _ = workspace(tmp_path)
    d = tmp_path / "emoji"
    write_jsonl(d / "e.jsonl", [{"role": "user", "content": "🦙" * 500}])
    ctx = _ctx(tmp_path, mem, d, max_bytes_per_transcript=101)
    c = ctx.transcripts[0].messages[0].content
    assert len(c.encode("utf-8")) <= 101 and "�" not in c and c.startswith("🦙")


def test_byte_limit_fail_mode_exact_and_over(tmp_path):
    """on_limit=fail: exactly at the byte limit passes, one byte over fails with a clear message."""
    mem, _ = workspace(tmp_path)
    ses = _byte_session(tmp_path, [600, 400])
    assert dream(mem, [ses], max_bytes_per_transcript=1000, on_limit="fail").status == "completed"
    job = dream(mem, [ses], max_bytes_per_transcript=999, on_limit="fail")
    assert job.status == "failed" and "max_bytes_per_transcript=999" in job.error


def test_total_tokens_exactly_at_limit_and_one_over(tmp_path):
    """max_total_tokens == estimate: nothing dropped; estimate - 1: oldest session dropped with notice."""
    mem, _ = workspace(tmp_path)
    ses = _sessions(tmp_path / "t", 3)
    tok = _ctx(tmp_path, mem, ses).stats["estimated_tokens"]
    at = _ctx(tmp_path, mem, ses, max_total_tokens=tok)
    assert len(at.transcripts) == 3 and not any("max_total_tokens" in n for n in at.notices)
    over = _ctx(tmp_path, mem, ses, max_total_tokens=tok - 1)
    assert [t.id for t in over.transcripts] == ["s001", "s002"]
    assert any(f"estimated {tok} tokens exceed max_total_tokens={tok - 1}" in n for n in over.notices)


def test_total_tokens_one_over_fail_mode(tmp_path):
    """on_limit=fail and one token over max_total_tokens -> DreamLimitError."""
    mem, _ = workspace(tmp_path)
    ses = _sessions(tmp_path / "t", 2)
    tok = _ctx(tmp_path, mem, ses).stats["estimated_tokens"]
    with pytest.raises(DreamLimitError, match="max_total_tokens"):
        _ctx(tmp_path, mem, ses, max_total_tokens=tok - 1, on_limit="fail")


def test_memory_alone_over_token_limit_is_a_clear_error(tmp_path):
    """Memory store alone larger than max_total_tokens -> failed, message says the store is too large."""
    mem = write(tmp_path / "mem" / "MEMORY.md", "".join(f"- fact {i} " + "word " * 50 + "\n" for i in range(40))).parent
    ses = _sessions(tmp_path / "t", 1)
    job = dream(mem, [ses], max_total_tokens=500)
    assert job.status == "failed" and "memory store alone is too large" in job.error


def test_max_entries_exactly_at_and_one_over(tmp_path):
    """max_entries == entry count passes; one fewer fails clearly (entries are never truncated)."""
    mem, ses = workspace(tmp_path)  # 6 entries
    assert dream(mem, [ses], max_entries=6).status == "completed"
    job = dream(mem, [ses], max_entries=5)
    assert job.status == "failed"
    assert "6 entries exceed max_entries=5" in job.error and "never truncated" in job.error


def test_limits_validated(tmp_path):
    """Nonsense limits (0 / negative) are rejected up front with ValueError, not silently misbehaving."""
    mem, ses = workspace(tmp_path)
    for bad in ({"max_sessions": 0}, {"max_bytes_per_transcript": -1}, {"max_total_tokens": 0},
                {"max_entries": 0}, {"on_limit": "explode"}):
        job = dream(mem, [ses], **bad)
        assert job.status == "failed" and "ValueError" in job.error, bad
