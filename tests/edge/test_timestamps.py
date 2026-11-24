"""Duplicate, out-of-order, missing, naive/aware-mixed and unparsable timestamps."""

from __future__ import annotations

import json

import pytest

from agentdream import dream, get_adapter
from agentdream.loaders import load_transcripts
from agentdream.util import parse_timestamp

from edgekit import workspace, write, write_jsonl


@pytest.mark.parametrize("value", ["2026-13-45", "2026-02-30T10:00:00Z", "yesterday", "not a date", "2026-01-01T25:61",
                                   "0000-00-00", "\x00", "9" * 400, "2026-01-01T10:00:00+25:00x", "NaN", "inf"])
def test_parse_timestamp_never_raises(value):
    """parse_timestamp returns a float or None for garbage, impossible dates and overflow; it never raises."""
    out = parse_timestamp(value)
    assert out is None or isinstance(out, float)


def test_parse_timestamp_naive_is_utc_and_aware_respected():
    """Naive ISO timestamps are UTC; offsets are honoured, so mixed naive/aware values compare correctly."""
    naive = parse_timestamp("2026-01-01T10:00:00")
    aware = parse_timestamp("2026-01-01T09:00:00-02:00")
    assert naive is not None and aware is not None and aware - naive == 3600
    assert parse_timestamp("2026-01-01T10:00:00Z") == naive
    assert parse_timestamp(1767261600000) == parse_timestamp(1767261600)  # ms epoch


def _session(d, name, ts, text="hello"):
    row = {"role": "user", "content": f"{text} {name}"}
    if ts is not None:
        row["timestamp"] = ts
    return write_jsonl(d / f"{name}.jsonl", [row])


def test_sessions_ordered_by_timestamp_not_filename(tmp_path):
    """Sessions whose filenames are out of chronological order are sorted by first timestamp."""
    d = tmp_path / "s"
    _session(d, "a", "2026-03-01T00:00:00Z")
    _session(d, "b", "2026-01-01T00:00:00Z")
    _session(d, "c", "2026-02-01T00:00:00Z")
    ts, _ = load_transcripts([d])
    assert [t.id for t in ts] == ["b", "c", "a"]


def test_duplicate_session_timestamps_are_stable(tmp_path):
    """Sessions with identical timestamps keep a deterministic (path) order."""
    d = tmp_path / "s"
    for n in ("s2", "s10", "s1"):
        _session(d, n, "2026-01-01T00:00:00Z")
    orders = {tuple(t.id for t in load_transcripts([d])[0]) for _ in range(3)}
    assert len(orders) == 1


def test_mixed_naive_and_aware_session_timestamps(tmp_path):
    """A naive (UTC) and an offset timestamp are ordered by real instant."""
    d = tmp_path / "s"
    _session(d, "later", "2026-01-01T09:00:00-02:00")  # 11:00Z
    _session(d, "earlier", "2026-01-01T10:00:00")       # 10:00Z
    ts, _ = load_transcripts([d])
    assert [t.id for t in ts] == ["earlier", "later"]


def test_missing_timestamps_fall_back_to_natural_path_order(tmp_path):
    """If any session lacks a timestamp, order is natural path order (s2 before s10)."""
    d = tmp_path / "s"
    _session(d, "s10", "2026-01-01T00:00:00Z")
    _session(d, "s2", None)
    _session(d, "s1", "2026-05-01T00:00:00Z")
    assert [t.id for t in load_transcripts([d])[0]] == ["s1", "s2", "s10"]


def test_unparsable_session_timestamp_counts_as_missing(tmp_path):
    """A session with an unparsable timestamp ('yesterday') is treated like a missing one, not sorted last."""
    d = tmp_path / "s"
    _session(d, "s1", "yesterday")
    _session(d, "s2", "2026-01-01T00:00:00Z")
    _session(d, "s3", "2025-01-01T00:00:00Z")
    assert [t.id for t in load_transcripts([d])[0]] == ["s1", "s2", "s3"]


def test_first_parsable_timestamp_is_used(tmp_path):
    """A session whose first message has a junk timestamp but later ones are valid sorts by the first valid one."""
    d = tmp_path / "s"
    write_jsonl(d / "x.jsonl", [{"role": "user", "content": "a", "timestamp": "??"},
                                {"role": "user", "content": "b", "timestamp": "2026-06-01T00:00:00Z"}])
    _session(d, "y", "2026-01-01T00:00:00Z")
    assert [t.id for t in load_transcripts([d])[0]] == ["y", "x"]


def test_weird_timestamp_types_in_transcripts(tmp_path):
    """Timestamps that are dicts, lists, booleans or huge numbers don't crash loading or the dream."""
    d = tmp_path / "s"
    for i, ts in enumerate([{"a": 1}, [1, 2], True, 10 ** 30, -5, "2026-02-30"]):
        write_jsonl(d / f"w{i}.jsonl", [{"role": "user", "content": f"msg {i}", "timestamp": ts}])
    mem, _ = workspace(tmp_path)
    job = dream(mem, [d])
    assert job.status == "completed", job.error


def _replace_ops(job):
    return [o for o in job.operations if o.type == "replace"]


def test_stale_replacement_uses_dates_not_file_order(tmp_path):
    """Newer dated note wins even when it comes first in the file."""
    mem, ses = workspace(tmp_path, memory="- The default branch is main. (2026-02-01)\n"
                                          "- The default branch is master. (2026-01-01)\n")
    (op,) = _replace_ops(dream(mem, [ses]))
    assert op.after == ["The default branch is main. (2026-02-01)"]


def test_stale_replacement_duplicate_dates_later_position_wins(tmp_path):
    """Two conflicting notes with the same date: the later one in the store wins."""
    mem, ses = workspace(tmp_path, memory="- The default branch is master. (2026-01-01)\n"
                                          "- The default branch is main. (2026-01-01)\n")
    (op,) = _replace_ops(dream(mem, [ses]))
    assert op.after == ["The default branch is main. (2026-01-01)"]


def test_stale_replacement_naive_vs_aware_json_store(tmp_path):
    """JSON store: updated_at naive 10:00 vs -02:00 09:00 (= 11:00Z) -> the offset one is newer and wins."""
    store = write(tmp_path / "m.json", json.dumps([
        {"id": "b", "content": "The API server listens on port 9090.", "updated_at": "2026-01-01T09:00:00-02:00"},
        {"id": "a", "content": "The API server listens on port 8080.", "updated_at": "2026-01-01T10:00:00"},
    ]))
    _, ses = workspace(tmp_path / "w")
    (op,) = _replace_ops(dream(store, [ses]))
    assert op.after == ["The API server listens on port 9090."]


def test_unparsable_entry_timestamps_fall_back_to_order(tmp_path):
    """Entries with garbage updated_at fall back to store order (later wins) instead of crashing."""
    store = write(tmp_path / "m.json", json.dumps([
        {"id": "a", "content": "The API server listens on port 8080.", "updated_at": "2026-13-45"},
        {"id": "b", "content": "The API server listens on port 9090.", "updated_at": "soon"},
    ]))
    _, ses = workspace(tmp_path / "w")
    job = dream(store, [ses])
    assert job.status == "completed", job.error
    (op,) = _replace_ops(job)
    assert op.after == ["The API server listens on port 9090."]
    assert get_adapter(store).load()[0].updated_at == "2026-13-45"  # input untouched
