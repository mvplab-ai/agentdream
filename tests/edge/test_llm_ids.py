"""Entry ids returned by small models (seen live with qwen2.5:1.5b via Ollama on examples/): prompt format,
id normalisation (id= prefixes, quotes, brackets, comma-joined lists) and unique-match resolution."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from agentdream import dream, get_adapter
from agentdream.engine.llm import SYSTEM_PROMPT, normalize_entry_id, render_memory, resolve_targets

from edgekit import FakeLLM, write

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def examples(tmp_path):
    mem = tmp_path / "memory"
    shutil.copytree(EXAMPLES / "memory", mem)
    ses = tmp_path / "sessions"
    shutil.copytree(EXAMPLES / "sessions", ses)
    return mem, ses


def _citation(prompt: str, suffix: str) -> str:
    m = re.search(r"\[([^\]\s]*" + re.escape(suffix) + r")\]", prompt)
    assert m, f"no citation ending in {suffix} in prompt"
    return m.group(1)


def test_prompt_renders_ids_in_brackets(examples):
    """Prompts show entries as '[MEMORY.md#1] text' (no id='...' that small models copy back verbatim)."""
    mem, ses = examples
    fake = FakeLLM()
    dream(mem, [ses], backend=fake)
    consolidate = next(p for _s, p in fake.calls if "Task: CONSOLIDATE" in p)
    assert "[MEMORY.md#1] The PokéCenter in Viridian City is north of the gym." in consolidate
    assert "id='" not in consolidate and "id=" not in consolidate.split("## Memory store", 1)[1]
    assert "verbatim" in SYSTEM_PROMPT and "[MEMORY.md#1]" in SYSTEM_PROMPT


def test_render_memory_metadata_line():
    """Section / date / order metadata is still rendered, on its own line, after the bracketed id."""
    from agentdream.models import MemoryEntry

    e = MemoryEntry(id="notes/a.md#2", content="Line one\nline two", path="notes/a.md", key="Tools",
                    updated_at="2026-01-02", metadata={"_order": 4})
    out = render_memory([e])
    assert out.splitlines()[0] == "[notes/a.md#2] Line one"
    assert "    line two" in out
    assert "section: Tools" in out and "updated: 2026-01-02" in out


def test_qwen_style_ids_from_live_run_are_resolved(examples):
    """Exact strings returned live by qwen2.5:1.5b ("id='MEMORY.md#1'", "id='id='MEMORY.md#6'") resolve; ops apply."""
    mem, ses = examples

    def respond(phase, prompt):
        if phase == "consolidate":
            return {"operations": [
                {"type": "merge", "targets": ["id='MEMORY.md#4'", "id='id='MEMORY.md#5'"],
                 "new_content": "Run the tests with `pytest -q` before every commit."},
                {"type": "replace", "targets": ["id='notes/project.md#1'", "id='id='MEMORY.md#6'"],
                 "new_content": "The API server listens on port 9090."},
            ]}
        if phase == "verify":
            cit = _citation(prompt, "session-03.jsonl:2")
            return {"operations": [{"type": "correct", "targets": ["id='MEMORY.md#1'"],
                                    "new_content": "The PokéCenter in Viridian City is south of the gym.",
                                    "evidence": [{"citation": cit, "quote": "It is just south of the gym"}]}]}
        return {"operations": []}

    job = dream(mem, [ses], backend=FakeLLM(respond))
    assert job.status == "completed", job.error
    llm = {o.type: o for o in job.operations if o.origin.startswith("fake")}
    assert llm["merge"].targets == ["MEMORY.md#4", "MEMORY.md#5"]
    assert llm["replace"].targets == ["notes/project.md#1", "MEMORY.md#6"]
    assert llm["correct"].targets == ["MEMORY.md#1"]
    assert not [w for w in job.warnings if "unknown entry id" in w], job.warnings
    contents = [e.content for e in get_adapter(job.store_path).load()]
    assert "The PokéCenter in Viridian City is south of the gym." in contents


@pytest.mark.parametrize("raw", [
    "MEMORY.md#6", " MEMORY.md#6 ", "'MEMORY.md#6'", '"MEMORY.md#6"', "`MEMORY.md#6`", "[MEMORY.md#6]",
    "'[MEMORY.md#6]'", "(MEMORY.md#6)", "id=MEMORY.md#6", "id='MEMORY.md#6'", "id='id='MEMORY.md#6'",
    "id: MEMORY.md#6", "ID=\"MEMORY.md#6\"", "id = 'MEMORY.md#6',", "\"id='MEMORY.md#6'\"", "<MEMORY.md#6>",
    "[MEMORY.md#6] The API server listens on port 8080.", "entry MEMORY.md#6", "memory.md#6",
])
def test_normalize_and_resolve_variants(raw):
    """id= prefixes (repeated), quotes, backticks, brackets, trailing punctuation, copied text, case all resolve."""
    known = ["MEMORY.md#5", "MEMORY.md#6", "MEMORY.md#60", "notes/project.md#1"]
    resolved, unresolved = resolve_targets(raw, known)
    assert (resolved, unresolved) == (["MEMORY.md#6"], [])


def test_normalize_entry_id_strips_layers():
    """normalize_entry_id peels nested id=/quote layers until stable."""
    assert normalize_entry_id("id='id='id=\"MEMORY.md#6\"'") == "MEMORY.md#6"
    assert normalize_entry_id("  [ 'x' ]  ") == "x"


@pytest.mark.parametrize("raw", [
    "MEMORY.md#4, MEMORY.md#5", "MEMORY.md#4,MEMORY.md#5", ["MEMORY.md#4, MEMORY.md#5"],
    "id='MEMORY.md#4', id='MEMORY.md#5'", "[MEMORY.md#4, MEMORY.md#5]", "['MEMORY.md#4', 'MEMORY.md#5']",
    "MEMORY.md#4; MEMORY.md#5", "MEMORY.md#4 and MEMORY.md#5",
])
def test_comma_joined_target_lists(raw):
    """A list given as one comma/semicolon-joined string (or a stringified list) is split into ids."""
    known = ["MEMORY.md#4", "MEMORY.md#5", "MEMORY.md#6"]
    assert resolve_targets(raw, known) == (["MEMORY.md#4", "MEMORY.md#5"], [])


def test_ids_containing_commas_resolve_exactly():
    """A real id that contains a comma (JSON stores allow any id) is matched whole, not split."""
    known = ["a,b", "a", "b"]
    assert resolve_targets(["a,b"], known) == (["a,b"], [])
    assert resolve_targets("a,b", known) == (["a,b"], [])


def test_basename_match_only_when_unique():
    """'project.md#1' resolves to 'notes/project.md#1' when unique; ambiguous basenames stay unresolved."""
    assert resolve_targets(["project.md#1"], ["notes/project.md#1", "MEMORY.md#1"]) == (["notes/project.md#1"], [])
    amb = ["a/x.md#1", "b/x.md#1"]
    assert resolve_targets(["x.md#1"], amb) == ([], ["x.md#1"])


def test_case_insensitive_only_when_unique():
    """Case-insensitive matching is used only when exactly one id matches."""
    assert resolve_targets(["readme.md#1"], ["README.md#1", "notes.md#1"]) == (["README.md#1"], [])
    assert resolve_targets(["readme.md#1"], ["README.md#1", "Readme.md#1"]) == ([], ["readme.md#1"])


def test_duplicate_targets_after_normalisation_are_collapsed():
    """The same id given twice in different spellings counts once (a merge needs distinct ids)."""
    assert resolve_targets(["MEMORY.md#6", "id='MEMORY.md#6'", "MEMORY.md#5"],
                           ["MEMORY.md#5", "MEMORY.md#6"]) == (["MEMORY.md#6", "MEMORY.md#5"], [])


def test_warning_only_for_truly_unresolvable_ids(examples):
    """A reply mixing a mangled-but-resolvable id and a made-up id warns about the made-up one only."""
    mem, ses = examples

    def respond(phase, prompt):
        if phase == "consolidate":
            return {"operations": [
                {"type": "merge", "targets": ["id='MEMORY.md#4'", "MEMORY.md#99"], "new_content": "x"},
                {"type": "merge", "targets": "id='MEMORY.md#4', id='id='MEMORY.md#5'",
                 "new_content": "Run the tests with `pytest -q` before every commit."},
            ]}
        return {"operations": []}

    job = dream(mem, [ses], backend=FakeLLM(respond))
    unknown = [w for w in job.warnings if "unknown entry id" in w]
    assert len(unknown) == 1 and "MEMORY.md#99" in unknown[0] and "MEMORY.md#4" not in unknown[0]
    assert [o.targets for o in job.operations if o.type == "merge" and o.origin.startswith("fake")] == \
        [["MEMORY.md#4", "MEMORY.md#5"]]


def test_numeric_and_null_targets(tmp_path):
    """Targets given as numbers, null or nested lists don't crash; unresolvable ones are reported once."""
    store = write(tmp_path / "m.json", json.dumps([{"id": "1", "content": "Use pytest."},
                                                   {"id": "2", "content": "Use pytest!"}]))
    write(tmp_path / "s" / "1.jsonl", json.dumps({"role": "user", "content": "hi"}) + "\n")

    def respond(phase, prompt):
        if phase == "consolidate":
            return {"operations": [{"type": "merge", "targets": [1, 2], "new_content": "Use pytest."},
                                   {"type": "merge", "targets": None}, {"type": "merge", "targets": [[1], [None]]}]}
        return {"operations": []}

    job = dream(store, [tmp_path / "s"], backend=FakeLLM(respond))
    assert job.status == "completed", job.error
    assert [o.targets for o in job.operations if o.type == "merge" and o.origin.startswith("fake")] == [["1", "2"]]
