"""Contradiction chains (A -> not-A -> A), rewording, and conflicts within vs across sessions."""

from __future__ import annotations

from agentdream import dream, get_adapter

from edgekit import write, write_jsonl

USE = "Use Poetry for dependency management."
DONT = "Don't use Poetry for dependency management."


def _run(tmp_path, memory, sessions):
    mem = write(tmp_path / "mem" / "MEMORY.md", memory).parent
    for name, rows in sessions.items():
        write_jsonl(tmp_path / "s" / name, rows)
    job = dream(mem, [tmp_path / "s"])
    assert job.status == "completed", job.error
    return job, [e.content for e in get_adapter(job.store_path).load()]


def _row(text, ts, role="user"):
    return {"role": role, "content": text, "timestamp": ts}


def _temporal(job):
    return [o for o in job.operations if o.type == "contradiction" and o.kind == "temporal"]


def test_memory_chain_a_not_a_a_final_is_a_with_full_history(tmp_path):
    """Memory notes A (Jan) -> not-A (Feb) -> A (Mar): final value is A; history lists all three in order."""
    job, final = _run(tmp_path, f"- {USE} (2026-01-01)\n- {DONT} (2026-02-01)\n- {USE} (2026-03-01)\n",
                      {"x.jsonl": [_row("unrelated chatter", "2026-04-01T00:00:00Z")]})
    (op,) = _temporal(job)
    assert op.after[0].startswith("Use Poetry")
    assert [h[:10] for h in op.history] == ["2026-01-01", "2026-02-01", "2026-03-01"]
    assert "Don't use Poetry" in op.history[1]
    assert len(final) == 1 and final[0].startswith("Use Poetry")


def test_memory_chain_out_of_file_order(tmp_path):
    """The same A / not-A / A chain written in shuffled file order still resolves by date to A."""
    job, final = _run(tmp_path, f"- {USE} (2026-03-01)\n- {USE} (2026-01-01)\n- {DONT} (2026-02-01)\n",
                      {"x.jsonl": [_row("unrelated chatter", "2026-04-01T00:00:00Z")]})
    (op,) = _temporal(job)
    assert [h[:10] for h in op.history] == ["2026-01-01", "2026-02-01", "2026-03-01"]
    assert final[0].startswith("Use Poetry")


def test_memory_chain_not_a_a_not_a(tmp_path):
    """The mirror chain not-A -> A -> not-A ends at not-A."""
    job, _final = _run(tmp_path, f"- {DONT} (2026-01-01)\n- {USE} (2026-02-01)\n- {DONT} (2026-03-01)\n",
                      {"x.jsonl": [_row("unrelated chatter", "2026-04-01T00:00:00Z")]})
    (op,) = _temporal(job)
    assert op.after[0].startswith("Don't use Poetry")
    assert len(op.history) == 3


def test_session_chain_across_sessions(tmp_path):
    """Memory not-A, then sessions A / not-A / A: final A, history has the memory note and all three sessions."""
    job, final = _run(tmp_path, f"- {DONT} (2026-01-01)\n- Llamas are calm.\n", {
        "1.jsonl": [_row("Let's use Poetry for dependency management.", "2026-02-01T00:00:00Z")],
        "2.jsonl": [_row("Don't use Poetry for dependency management, it broke CI.", "2026-03-01T00:00:00Z")],
        "3.jsonl": [_row("Use Poetry for dependency management again.", "2026-04-01T00:00:00Z")],
    })
    (op,) = _temporal(job)
    assert op.after[0].startswith("Use Poetry")
    assert len(op.history) == 4
    assert [h.split(": ", 1)[0].rsplit("/", 1)[-1] for h in op.history[1:]] == ["1.jsonl:1", "2.jsonl:1", "3.jsonl:1"]
    assert {e.citation.rsplit("/", 1)[-1] for e in op.evidence} >= {"1.jsonl:1", "2.jsonl:1", "3.jsonl:1"}
    assert final[0].startswith("Use Poetry")


def test_session_chain_ending_where_memory_started_is_unchanged(tmp_path):
    """Memory A, sessions not-A then A again: the latest stance equals the note, so nothing is rewritten."""
    job, final = _run(tmp_path, f"- {USE} (2026-01-01)\n", {
        "1.jsonl": [_row("Don't use Poetry for dependency management.", "2026-02-01T00:00:00Z")],
        "2.jsonl": [_row("Use Poetry for dependency management again.", "2026-03-01T00:00:00Z")],
    })
    assert not [o for o in _temporal(job) if o.phase == "verify"]
    assert final[0] == f"{USE} (2026-01-01)"


def test_conflict_within_one_session(tmp_path):
    """A reversal inside a single session (message 1 A, message 2 not-A) resolves to the later message."""
    job, _final = _run(tmp_path, "- Use yarn for installing packages.\n- Llamas are calm.\n", {
        "1.jsonl": [_row("Use yarn for installing packages.", "2026-02-01T00:00:00Z"),
                    _row("Actually, don't use yarn for installing packages.", "2026-02-01T00:05:00Z")],
    })
    (op,) = _temporal(job)
    assert op.after[0].startswith("Don't use yarn")
    assert op.history[-1].split(": ", 1)[0].endswith("1.jsonl:2")


def test_same_subject_different_wording(tmp_path):
    """'Use yarn for installing packages' vs later 'Never use yarn to install packages' is one temporal conflict."""
    job, final = _run(tmp_path, "- Use yarn for installing packages. (2026-01-01)\n"
                                "- Never use yarn to install packages. (2026-02-01)\n",
                      {"x.jsonl": [_row("unrelated chatter", "2026-04-01T00:00:00Z")]})
    (op,) = _temporal(job)
    assert len(op.targets) == 2
    assert any(w in op.after[0] for w in ("Don't", "Never", "Avoid"))
    assert len(final) == 1


def test_transcript_only_chain_records_every_flip(tmp_path):
    """No memory note: sessions A -> not-A -> A produce one temporal insight whose history has three steps."""
    job, _final = _run(tmp_path, "- Llamas are calm.\n", {
        "1.jsonl": [_row("Let's use tabs for indentation in this repo.", "2026-02-01T00:00:00Z")],
        "2.jsonl": [_row("Don't use tabs for indentation in this repo.", "2026-03-01T00:00:00Z")],
        "3.jsonl": [_row("Use tabs for indentation in this repo after all.", "2026-04-01T00:00:00Z")],
    })
    ops = [o for o in _temporal(job) if o.phase == "insights"]
    assert len(ops) == 1
    assert len(ops[0].history) == 3 and ops[0].after[0].startswith("Use tabs")


def test_newer_note_beats_older_transcript_evidence(tmp_path):
    """A memory note newer than the contradicting transcript is flagged, not rewritten."""
    job, final = _run(tmp_path, f"- {DONT} (2026-06-01)\n", {
        "1.jsonl": [_row("Let's use Poetry for dependency management.", "2026-02-01T00:00:00Z")],
    })
    assert final[0] == f"{DONT} (2026-06-01)"
    flags = [o for o in job.operations if o.type == "contradiction" and not o.changes_store]
    assert flags and flags[0].kind == "temporal"
