"""New insights from transcripts: recurring mistakes, repeated preferences, workflows."""

from agentdream import dream

from conftest import final_contents, ops_of, write, write_jsonl


def _mem(tmp_path, text="- The project is called Heron.\n"):
    return write(tmp_path / "m" / "MEMORY.md", text).parent


def test_recurring_mistake_across_sessions(tmp_path):
    s = tmp_path / "s"
    write_jsonl(s / "a.jsonl", [
        {"role": "assistant", "content": "Running the migration script."},
        {"role": "tool", "content": "Error: forgot to activate the virtualenv before running migrations"},
    ])
    write_jsonl(s / "b.jsonl", [
        {"role": "assistant", "content": "Oops, I forgot to activate the virtualenv again before the migrations."},
    ])
    write_jsonl(s / "c.jsonl", [{"role": "user", "content": "Unrelated: please update the README."}])
    job = dream(_mem(tmp_path), [s])
    ins = [o for o in ops_of(job, "insight") if "Recurring mistake" in o.after[0]]
    assert len(ins) == 1
    op = ins[0]
    assert "virtualenv" in op.after[0]
    assert "2 sessions" in op.after[0]
    srcs = {e.citation.rsplit("/", 1)[-1].split(":")[0] for e in op.evidence}
    assert srcs == {"a.jsonl", "b.jsonl"}
    assert any("virtualenv" in c for c in final_contents(job))


def test_repeated_preference_insight(tmp_path):
    s = tmp_path / "s"
    write(s / "1.txt", "User: Please always use type hints in Python code.\nAssistant: Will do.\n")
    write(s / "2.txt", "User: Remember to keep it short.\nAssistant: Sure.\n")
    write(s / "3.txt", "User: Always use type hints, remember?\nAssistant: Yes.\n")
    job = dream(_mem(tmp_path), [s])
    prefs = [o for o in ops_of(job, "insight") if o.after[0].startswith("Preference")]
    assert len(prefs) == 1
    assert "type hints" in prefs[0].after[0]
    assert "2 sessions" in prefs[0].after[0]
    assert len(prefs[0].evidence) == 2


def test_single_mention_is_not_an_insight(tmp_path):
    s = tmp_path / "s"
    write(s / "1.txt", "User: Please always use type hints.\n")
    write(s / "2.txt", "User: What's the weather?\n")
    job = dream(_mem(tmp_path), [s])
    assert ops_of(job, "insight") == []


def test_converged_workflow(tmp_path):
    s = tmp_path / "s"
    for i in range(3):
        write_jsonl(s / f"{i}.jsonl", [
            {"role": "assistant", "content": f"Editing module {i}."},
            {"role": "assistant", "content": "Ran `ruff check .` then `pytest -q`, all green."},
            {"role": "assistant", "content": "Now `git push`."},
        ])
    job = dream(_mem(tmp_path), [s])
    flows = [o for o in ops_of(job, "insight") if "workflow" in o.after[0].lower()]
    assert flows
    assert "`pytest -q`" in flows[0].after[0] and "3 sessions" in flows[0].after[0]


def test_insight_already_in_memory_is_skipped(tmp_path):
    s = tmp_path / "s"
    write(s / "1.txt", "User: Please always use type hints in Python code.\n")
    write(s / "2.txt", "User: Always use type hints in Python code.\n")
    job = dream(_mem(tmp_path, "- Always use type hints in Python code.\n"), [s])
    assert [o for o in ops_of(job, "insight")] == []


def test_insights_land_in_new_file_for_markdown(tmp_path):
    s = tmp_path / "s"
    write(s / "1.txt", "User: Always write commit messages in English.\n")
    write(s / "2.txt", "User: Please always write commit messages in English!\n")
    job = dream(_mem(tmp_path), [s])
    from pathlib import Path

    text = (Path(job.store_path) / "dream-insights.md").read_text()
    assert "commit messages in English" in text
    # original file copied byte-for-byte because nothing in it changed
    assert (Path(job.store_path) / "MEMORY.md").read_bytes() == (tmp_path / "m" / "MEMORY.md").read_bytes()
