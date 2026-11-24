"""Duplicate merge, stale replacement and temporal contradictions."""

import json
from pathlib import Path

from agentdream import dream

from conftest import final_contents, ops_of, write, write_jsonl


def _one_session(tmp_path, text="User: hello there\n"):
    return write(tmp_path / "s" / "chat.txt", text)


def test_duplicate_merge(tmp_path):
    mem = write(tmp_path / "mem" / "MEMORY.md",
                "## Testing\n- Run the test suite with `pytest -q` before committing.\n"
                "- Unrelated: the logo is blue.\n"
                "- run the tests with pytest -q before every commit\n"
                "- Run the test suite with `pytest -q` before committing!\n")
    job = dream(mem.parent, [_one_session(tmp_path)])
    merges = ops_of(job, "merge")
    assert len(merges) == 1
    m = merges[0]
    assert sorted(m.targets) == ["MEMORY.md#1", "MEMORY.md#3", "MEMORY.md#4"]
    assert len(m.remove) == 2
    contents = final_contents(job)
    assert sum("pytest" in c for c in contents) == 1
    assert "Unrelated: the logo is blue." in contents
    assert job.report["counts"]["merged"] == 1
    assert job.report["counts"]["unchanged"] == 1


def test_values_differing_are_not_duplicates(tmp_path):
    mem = write(tmp_path / "mem" / "a.md",
                "- Use 4 spaces for indentation in Python.\n- Use 2 spaces for indentation in YAML.\n")
    job = dream(mem.parent, [_one_session(tmp_path)])
    assert ops_of(job, "merge") == []


def test_merge_across_files_and_json(tmp_path):
    store = tmp_path / "mem.json"
    store.write_text(json.dumps([
        {"id": "a", "content": "The staging database is Postgres 16.", "updated_at": "2026-01-01"},
        {"id": "b", "content": "Staging database: Postgres 16", "updated_at": "2026-02-01"},
    ]))
    job = dream(store, [_one_session(tmp_path)])
    (m,) = ops_of(job, "merge")
    assert m.remove == ["a"]  # newest survives
    data = json.loads(Path(job.store_path).read_text())
    assert [d["id"] for d in data] == ["b"]


def test_stale_replacement_by_timestamp(tmp_path):
    store = tmp_path / "mem.jsonl"
    write_jsonl(store, [
        {"id": "p1", "content": "The API server listens on port 9090.", "updated_at": "2026-03-01"},
        {"id": "x", "content": "The CI provider is GitHub Actions."},
        {"id": "p0", "content": "The API server listens on port 8080.", "updated_at": "2025-11-20"},
    ])
    job = dream(store, [_one_session(tmp_path)])
    (r,) = ops_of(job, "replace")
    assert r.remove == ["p0"]
    assert r.after == ["The API server listens on port 9090."]
    contents = final_contents(job)
    assert "The API server listens on port 9090." in contents
    assert not any("8080" in c for c in contents)
    assert job.report["sections"]["replaced"] == [r.id]


def test_stale_replacement_by_order_when_no_timestamps(tmp_path):
    mem = write(tmp_path / "m" / "MEMORY.md", "- The default branch is master.\n- The default branch is main.\n")
    job = dream(mem.parent, [_one_session(tmp_path)])
    (r,) = ops_of(job, "replace")
    assert r.remove == ["MEMORY.md#1"]
    assert final_contents(job) == ["The default branch is main."]


def test_temporal_contradiction_in_memory(tmp_path):
    store = tmp_path / "mem.json"
    store.write_text(json.dumps({"entries": [
        {"id": "n1", "content": "Don't use SQLAlchemy for the data layer.", "updated_at": "2026-01-10"},
        {"id": "n2", "content": "Maybe SQLAlchemy is better after all.", "updated_at": "2026-04-02"},
    ]}))
    job = dream(store, [_one_session(tmp_path)])
    cons = ops_of(job, "contradiction")
    assert len(cons) == 1
    c = cons[0]
    assert c.kind == "temporal"
    current = c.after[0]
    assert current.startswith("Use SQLAlchemy for the data layer")
    assert "History" in current and "Don't use SQLAlchemy" in current
    assert c.history and "2026-01-10" in c.history[0]
    entries = json.loads(Path(job.store_path).read_text())["entries"]
    assert len(entries) == 1 and entries[0]["content"] == current
    assert entries[0]["metadata"]["history"][0]["content"].startswith("Don't use SQLAlchemy")
    assert job.report["counts"]["temporal_contradictions"] == 1
    md = job.report_md.split("## Contradictions")[1]
    assert "[temporal]" in md


def test_temporal_contradiction_memory_vs_later_transcript(tmp_path):
    mem = write(tmp_path / "m" / "MEMORY.md", "- Never use pnpm in this repo. (2026-01-02)\n")
    s = tmp_path / "s"
    write_jsonl(s / "late.jsonl", [
        {"role": "user", "content": "I benchmarked again and pnpm is much faster.",
         "timestamp": "2026-05-01T10:00:00Z"},
        {"role": "user", "content": "Let's use pnpm from now on.", "timestamp": "2026-05-01T10:01:00Z"},
    ])
    job = dream(mem.parent, [s])
    (c,) = ops_of(job, "contradiction")
    assert c.kind == "temporal"
    assert c.after[0].lower().startswith("use pnpm")
    assert "Never use pnpm" in c.after[0]  # history keeps the earlier rejection
    assert any(e.citation.endswith("late.jsonl:2") for e in c.evidence)


def test_newer_memory_beats_older_transcript(tmp_path):
    mem = write(tmp_path / "m" / "MEMORY.md", "- Don't use Docker for local dev. (2026-06-01)\n")
    s = tmp_path / "s"
    write_jsonl(s / "old.jsonl", [{"role": "user", "content": "Use Docker for local dev.",
                                   "timestamp": "2026-01-01T00:00:00Z"}])
    job = dream(mem.parent, [s])
    (c,) = ops_of(job, "contradiction")
    assert not c.changes_store  # flag only
    assert final_contents(job) == ["Don't use Docker for local dev. (2026-06-01)"]


def test_temporal_contradiction_transcripts_only(tmp_path):
    mem = write(tmp_path / "m" / "MEMORY.md", "- The project is called Falcon.\n")
    s = tmp_path / "s"
    write(s / "01.txt", "User: Don't use Tailwind for styling, plain CSS is enough.\nAssistant: OK.\n")
    write(s / "02.txt",
          "User: Hmm, maybe Tailwind is better for this after all. Use Tailwind.\nAssistant: Switching.\n")
    job = dream(mem.parent, [s])
    cons = [c for c in ops_of(job, "contradiction") if c.kind == "temporal"]
    assert len(cons) == 1
    c = cons[0]
    assert c.after[0].startswith("Use Tailwind for styling")
    assert "Don't use Tailwind" in c.after[0]
    assert {e.citation.rsplit("/", 1)[-1] for e in c.evidence} == {"01.txt:1", "02.txt:1"}
    assert any("Use Tailwind for styling" in x for x in final_contents(job))
