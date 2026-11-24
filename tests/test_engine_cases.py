"""Deterministic-engine behaviours: same-named subjects, generic value corrections, determinism,
instruction handling, preference wording, and LLM proposal validation edge cases."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agentdream import LLMBackend, dream
from agentdream.context import DreamConfig, DreamContext
from agentdream.engine.llm import (
    Call,
    build_prompt,
    convert_operations,
    extract_operations,
    parse_json_loose,
    plan_calls,
    render_memory,
    render_transcripts,
)
from agentdream.models import MemoryEntry, Message, Transcript
from agentdream.text import parse_claim, strip_markup

from conftest import EXAMPLES, final_contents, ops_of, write, write_jsonl

ROOT = Path(__file__).resolve().parent.parent


def _ws(tmp_path, notes, sessions):
    mem = write(tmp_path / "m" / "MEMORY.md", "".join(f"- {n}\n" for n in notes)).parent
    for name, rows in sessions.items():
        write_jsonl(tmp_path / "s" / name, [{"role": "assistant", "content": r} for r in rows])
    return mem, tmp_path / "s"


# ------------------------------------------------------------------ regression: same-named subjects


def test_same_named_places_in_different_cities_are_not_merged_or_cross_corrected(tmp_path):
    mem, s = _ws(tmp_path, ["The PokéCenter in Viridian City is north of the gym.",
                            "The PokéCenter in Pewter City is south of the museum."], {
        "1.jsonl": ["Arrived in Pewter City.", "Found the PokéCenter! It is south of the museum.",
                    "Travelled back to Viridian City.", "Walked north of the gym. No PokéCenter here.",
                    "Still can't find the PokéCenter north of the gym.",
                    "Found the PokéCenter in Viridian City! It is east of the gym."],
        "2.jsonl": ["Back in Viridian City again.", "Healed at the PokéCenter east of the gym."]})
    job = dream(mem, [s])
    assert ops_of(job, "replace") == [] and ops_of(job, "merge") == []  # two different PokéCenters
    (corr,) = ops_of(job, "correct")
    assert corr.targets == ["MEMORY.md#1"] and "east of the gym" in corr.after[0]
    assert all("museum" not in e.quote for e in corr.evidence)  # Pewter evidence didn't leak
    assert sorted(final_contents(job)) == ["The PokéCenter in Pewter City is south of the museum.",
                                           "The PokéCenter in Viridian City is east of the gym."]


def test_same_subject_still_detected_without_distinct_qualifiers():
    a = parse_claim("The PokéCenter in Viridian City is north of the gym.")
    b = parse_claim("The PokéCenter in Viridian is south of the gym.")
    c = parse_claim("The PokéCenter in Pewter City is south of the museum.")
    assert a.same_subject(b) and not a.same_subject(c) and not c.same_subject(a)


# ------------------------------------------------------------------ regression: generic value corrections


def test_generic_value_correction_with_labelled_sentences(tmp_path):
    mem, s = _ws(tmp_path, ["The staging database is called orders_v1."], {
        "1.jsonl": ["Checked the config: the staging database is called orders_v2 now."],
        "2.jsonl": ["Confirmed: the staging database is called orders_v2."]})
    job = dream(mem, [s])
    (corr,) = ops_of(job, "correct")
    assert corr.after == ["The staging database is called orders_v2."]  # snake_case kept, no "now"
    assert len(corr.evidence) == 2


def test_generic_value_kept_when_transcripts_confirm_the_old_value(tmp_path):
    mem, s = _ws(tmp_path, ["The staging database is called orders_v1."], {
        "1.jsonl": ["The staging database is called orders_v2."],
        "2.jsonl": ["Confirmed: the staging database is called orders_v1."],
        "3.jsonl": ["Verified again: the staging database is called orders_v1."]})
    job = dream(mem, [s])
    assert ops_of(job, "correct") == []


def test_label_prefix_parsing_keeps_key_value_facts():
    assert parse_claim("Confirmed: the API port is 9090").subject == parse_claim("the API port is 9090").subject
    kv = parse_claim("Port: 8080")
    assert kv is not None and kv.value == "8080"


def test_strip_markup_keeps_identifiers():
    assert strip_markup("- **Use** `orders_v2` and _really_ my_var__x") == "Use orders_v2 and really my_var__x"
    assert strip_markup("1. [x] done *now*") == "done now"


# ------------------------------------------------------------------ determinism


def test_output_is_identical_across_hash_seeds():
    outs = set()
    for seed in ("0", "123"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, str(EXAMPLES / "run_example.py")], capture_output=True, text=True,
                           env=env, cwd=ROOT, timeout=120)
        assert r.returncode == 0, r.stderr
        outs.add(re.sub(r"dream-2026\S+|Created: \S+|/\S+/jobs/", "X", r.stdout))
    assert len(outs) == 1


# ------------------------------------------------------------------ instructions


def test_instructions_skip_and_only_phases(tmp_path):
    job = dream(EXAMPLES / "memory", [EXAMPLES / "sessions"], instructions="please skip insights this time")
    assert ops_of(job, "insight") == [] and ops_of(job, "correct")
    assert any("phase 'insights' skipped" in n for n in job.report["notices"])
    job = dream(EXAMPLES / "memory", [EXAMPLES / "sessions"], instructions="only verify")
    assert {o.phase for o in job.operations} == {"verify"}
    job = dream(EXAMPLES / "memory", [EXAMPLES / "sessions"], instructions="focus on the PokéCenter navigation")
    assert any("match the instruction focus" in n for n in job.report["notices"])


def test_preference_reversal_wording_when_the_old_note_cannot_be_flipped(tmp_path):
    mem = write(tmp_path / "m" / "MEMORY.md", "- Ditch Poetry for builds. (2026-01-01)\n"
                                              "- Let's use Poetry for builds again. (2026-03-01)\n").parent
    s = write_jsonl(tmp_path / "s" / "1.jsonl", [{"role": "user", "content": "hello"}]).parent
    job = dream(mem, [s])
    (c,) = [o for o in ops_of(job, "contradiction") if o.kind == "temporal"]
    assert c.after[0].startswith("Use Poetry for builds again") and "History:" in c.after[0]


# ------------------------------------------------------------------ LLM proposal validation


def _ctx(entries, transcripts):
    ctx = DreamContext(DreamConfig())
    ctx.entries = list(entries)
    ctx.transcripts = list(transcripts)
    return ctx


ENTRIES = [MemoryEntry(id="a", content="alpha note"), MemoryEntry(id="b", content="beta note")]
T = Transcript("s", [Message("user", "alpha is now gamma", None, 1, "/l/s.jsonl", 0),
                     Message("assistant", "ok", None, 2, "/l/s.jsonl", 1)], "/l/s.jsonl", "jsonl")


def test_convert_operations_drops_and_downgrades_bad_proposals():
    ctx = _ctx(ENTRIES, [T])
    ev = [{"citation": "/l/s.jsonl:1", "quote": "alpha is now gamma"}]
    raw = [
        {"type": "teleport", "targets": ["a"]},                                       # unsupported type
        {"type": "correct", "targets": ["a"]},                                         # no new content
        {"type": "correct", "targets": ["a"], "new_content": "gamma", "evidence": ev, "confidence": "high"},
        {"type": "contradiction"},                                                     # nothing to say
        "not even a dict",
    ]
    ops = convert_operations(ctx, "verify", [r for r in raw if isinstance(r, dict)], "fake")
    (op,) = ops
    assert op.type == "correct" and op.confidence == 0.6 and op.evidence[0].quote == "alpha is now gamma"
    assert any("unsupported type 'teleport'" in w for w in ctx.warnings)
    assert any("needs a target and new_content" in w for w in ctx.warnings)
    ins = convert_operations(ctx, "insights", [
        {"type": "insight"},                                                           # no content
        {"type": "insight", "new_content": "x", "evidence": [{"citation": "nope:1", "quote": "q"}]},
        {"type": "insight", "new_content": "Alpha became gamma.",
         "evidence": [{"citation": "s.jsonl:1", "quote": "something else"}, 42]},      # basename + bad quote
    ], "fake")
    (i,) = ins
    assert i.upsert[0].id.startswith("insight-") and i.evidence[0].quote == "alpha is now gamma"  # real text
    assert any("unverifiable evidence" in w for w in ctx.warnings)
    merges = convert_operations(ctx, "consolidate", [{"type": "merge", "targets": ["a"]},
                                                     {"type": "replace", "targets": ["a", "b"], "new_content": "n"}],
                                "fake", Call(0, 2, [ENTRIES[0]], None))
    assert merges == []  # merge needs 2 targets; replace touches 'b', which this chunk didn't show
    assert any("needs more targets" in w for w in ctx.warnings)
    assert any("not shown in this part" in w for w in ctx.warnings)


def test_json_helpers_and_rendering():
    assert parse_json_loose('{"type": "merge"}') == {"type": "merge"}
    assert extract_operations({"type": "merge", "targets": []}) == [{"type": "merge", "targets": []}]
    assert extract_operations({"ops": [{"type": "x"}, 3]}) == [{"type": "x"}]
    for bad in (None, "", "no json at all"):
        with pytest.raises(ValueError):
            parse_json_loose(bad)
    with pytest.raises(ValueError):
        extract_operations(5)
    part = MemoryEntry(id="big", content="x", metadata={"_part": [1, 3], "_order": 0})
    assert "EXCERPT 1/3" in render_memory([part]) and render_memory([]) == "(empty store)"
    ctx = _ctx(ENTRIES, [T])
    assert "[/l/s.jsonl:1] user: alpha is now gamma" in render_transcripts(ctx)
    assert "## Session transcripts" in build_prompt(ctx, "verify")
    assert "## Session" not in build_prompt(ctx, "consolidate")


def test_plan_calls_mid_sized_store_rides_along_with_every_transcript_chunk():
    entries = [MemoryEntry(id=f"e{i}", content=f"note number {i} about widgets") for i in range(10)]
    msgs = [Message("user", f"message {k} " + "word " * 40, None, k + 1, "/l/t.jsonl", k) for k in range(60)]
    ctx = _ctx(entries, [Transcript("t", msgs, "/l/t.jsonl", "jsonl")])
    ctx.config.max_chunk_tokens = 2500
    calls, notices = plan_calls(ctx, "insights")
    assert len(calls) > 1 and all(len(c.entries) == 10 for c in calls) and notices == []


def test_plan_calls_clips_huge_messages_and_skips_irrelevant_chunks():
    entries = [MemoryEntry(id=f"e{i}", content=f"The zeta{i} service is on port {9000 + i}.") for i in range(300)]
    huge = Message("tool", "x" * 50_000, None, 1, "/l/t.jsonl", 0)
    other = [Message("user", f"unrelated chatter {k} " * 20, None, k + 2, "/l/t.jsonl", k + 1) for k in range(40)]
    ctx = _ctx(entries, [Transcript("t", [huge, *other], "/l/t.jsonl", "jsonl")])
    ctx.config.max_chunk_tokens = 4000
    calls, notices = plan_calls(ctx, "verify")
    assert calls == [] and any("no transcript part mentions" in n for n in notices)


class OneShot(LLMBackend):
    name = "oneshot"

    def complete(self, system, prompt, json_schema=None):
        if "Task: VERIFY" in prompt:
            return json.dumps({"operations": [
                {"type": "contradiction", "kind": "conflict", "targets": ["MEMORY.md#1", "MEMORY.md#2"]}]})
        return "{}"


def test_llm_flag_with_two_targets_never_deletes(tmp_path):
    mem, s = _ws(tmp_path, ["Alpha is one.", "Beta is two."], {"1.jsonl": ["nothing much"]})
    job = dream(mem, [s], backend=OneShot())
    (flag,) = [o for o in job.operations if o.origin == "oneshot"]
    assert not flag.changes_store and len(final_contents(job)) == 2
