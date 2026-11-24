"""Chunking, LLM map-reduce over chunks, blocking exactness and large-input performance."""

import itertools
import json
import random
import re
import time

import pytest

from agentdream import LLMBackend, dream
from agentdream.blocking import TokenIndex, cluster_order, similar_pairs
from agentdream.chunking import (
    SESSION_HEADER_TOKENS,
    ChunkingError,
    chunk_memory,
    chunk_transcripts,
    entry_cost,
    message_cost,
    pack,
    split_entry,
    split_message,
    split_text,
)
from agentdream.context import DreamConfig, DreamContext
from agentdream.engine.llm import Call, convert_operations, plan_calls, reduce_operations
from agentdream.models import MemoryEntry, Message, Operation, Transcript
from agentdream.text import jaccard
from agentdream.util import estimate_tokens

from conftest import final_contents, ops_of, write, write_jsonl
from synth import make_large_workspace

# ------------------------------------------------------------------ split_text / pack


def test_split_text_is_lossless_and_prefers_boundaries():
    para = "First paragraph sentence one. Sentence two is here.\n\nSecond paragraph is a bit longer than one."
    pieces = split_text(para, 16)  # 64 chars max
    assert "".join(pieces) == para
    assert all(len(p) <= 64 for p in pieces)
    assert pieces[0].endswith("\n\n")  # cut after the blank line, not mid-sentence
    long_word = "x" * 300
    pieces = split_text(long_word, 10)
    assert "".join(pieces) == long_word and all(len(p) == 40 for p in pieces[:-1])
    assert split_text("short", 100) == ["short"]
    with pytest.raises(ChunkingError):
        split_text("abc", 0)


def test_split_text_random_roundtrip():
    rng = random.Random(3)
    for _ in range(200):
        alphabet = ["word ", "end. ", "\n", "\n\n", "longerword", "é", " "]
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 300)))
        budget = rng.randint(1, 40)
        pieces = split_text(text, budget)
        assert "".join(pieces) == text
        assert all(len(p) <= budget * 4 for p in pieces)


def test_pack_respects_budget_and_order():
    items = list(range(20))
    chunks = pack(items, lambda i: 10, 35)
    assert [c.items for c in chunks][:2] == [[0, 1, 2], [3, 4, 5]]
    assert [i for c in chunks for i in c.items] == items
    assert all(c.tokens <= 35 for c in chunks)
    with pytest.raises(ChunkingError):
        pack([1], lambda i: 50, 10)
    with pytest.raises(ChunkingError):
        pack([1], lambda i: 1, 0)
    assert pack([], lambda i: 1, 10) == []


def test_pack_overlap_stays_within_group():
    items = [("a", i) for i in range(6)] + [("b", i) for i in range(6)]
    chunks = pack(items, lambda it: 10, 40, overlap_tokens=10, same_group=lambda x, y: x[0] == y[0])
    new_items = [it for c in chunks for it in c.items[c.overlap:]]
    assert new_items == items  # every item exactly once as "new"
    for prev, cur in itertools.pairwise(chunks):
        carried = cur.items[:cur.overlap]
        assert carried == prev.items[len(prev.items) - len(carried):]
        assert all(c[0] == cur.items[cur.overlap][0] for c in carried)  # never across groups
    assert any(c.overlap for c in chunks)


# ------------------------------------------------------------------ memory / transcripts


def _entries(n, size=40):
    return [MemoryEntry(id=f"e{i}", content=f"note {i} " + "w" * size) for i in range(n)]


def test_chunk_memory_never_splits_entries_that_fit():
    ents = _entries(30)
    budget = 3 * entry_cost(ents[0]) + 1
    chunks = chunk_memory(ents, budget)
    assert [e.id for c in chunks for e in c.items] == [e.id for e in ents]
    assert all(len(c.items) == 3 and c.tokens <= budget for c in chunks)
    assert all("_part" not in e.metadata for c in chunks for e in c.items)


def test_oversized_entry_becomes_excerpts():
    big = MemoryEntry(id="big", content="Sentence number one is here. " * 200, path="MEMORY.md")
    parts = split_entry(big, 100)
    assert len(parts) > 1 and all(p.id == "big" for p in parts)
    assert [p.metadata["_part"] for p in parts] == [[k, len(parts)] for k in range(1, len(parts) + 1)]
    assert "".join(p.content for p in parts) == big.content
    assert all(entry_cost(p) <= 100 for p in parts)
    chunks = chunk_memory([MemoryEntry(id="s", content="small"), big], 100)
    assert all(c.tokens <= 100 for c in chunks)


def _transcript(tid, n, size=30, path=None):
    path = path or f"/logs/{tid}.jsonl"
    msgs = [Message("user", f"{tid} message {i} " + "z" * size, None, i + 1, path, i) for i in range(n)]
    return Transcript(tid, msgs, path, "jsonl")


def test_chunk_transcripts_overlap_and_coverage():
    ts = [_transcript("s1", 12), _transcript("s2", 12)]
    budget = 200
    chunks = chunk_transcripts(ts, budget, overlap_tokens=40)
    assert len(chunks) > 2
    assert all(c.tokens <= budget for c in chunks)
    seen = []
    for c in chunks:
        for sl in c.items:
            seen += [(sl.transcript.id, m.line_no) for m in sl.messages[sl.overlap:]]
    assert seen == [(t.id, m.line_no) for t in ts for m in t.messages]  # each message once, in order
    overl = [sl for c in chunks for sl in c.items if sl.overlap]
    assert overl, "consecutive chunks of one session should overlap"
    for c in chunks:
        # overlap only ever belongs to the first slice and never spans sessions
        assert all(sl.overlap == 0 for sl in c.items[1:])


def test_oversized_message_split_keeps_citation():
    m = Message("assistant", "Long log line. " * 400, None, 7, "/x/s.jsonl", 3)
    parts = split_message(m, 120)
    assert len(parts) > 1 and all(p.citation == "/x/s.jsonl:7" for p in parts)
    assert parts[0].content.startswith("(part 1/")
    assert all(message_cost(p) <= 120 + 4 for p in parts)
    chunks = chunk_transcripts([Transcript("s", [m], "/x/s.jsonl", "jsonl")], 160)
    msgs = [x for c in chunks for sl in c.items for x in sl.messages]
    assert len(msgs) == len(split_message(m, 160 - SESSION_HEADER_TOKENS)) > 1
    assert all(c.tokens <= 160 for c in chunks) and all(x.citation == m.citation for x in msgs)
    with pytest.raises(ChunkingError):
        chunk_transcripts([Transcript("s", [m])], 20)


# ------------------------------------------------------------------ blocking exactness


def test_similar_pairs_is_exact_against_brute_force():
    rng = random.Random(11)
    vocab = [f"t{i}" for i in range(25)]
    sets = [frozenset(rng.sample(vocab, rng.randint(0, 8))) for _ in range(250)]
    for th in (0.3, 0.5, 0.75, 1.0):
        truth = {(i, j) for i, j in itertools.combinations(range(len(sets)), 2)
                 if sets[i] and sets[j] and jaccard(sets[i], sets[j]) >= th}
        cand = similar_pairs(sets, th)
        assert truth <= cand
        assert all(i < j for i, j in cand)
    with pytest.raises(ValueError):
        similar_pairs(sets, 0)


def test_token_index_queries_and_cluster_order():
    idx = TokenIndex([{"a", "b"}, {"b", "c"}, {"c"}, set()])
    assert idx.with_all({"b"}) == {0, 1} and idx.with_all({"b", "c"}) == {1} and idx.with_all(set()) == set()
    assert idx.with_any({"a", "c"}) == {0, 1, 2}
    assert idx.overlap_counts({"b", "c"}) == {0: 1, 1: 2, 2: 1}
    assert cluster_order(5, [(0, 3), (1, 4)]) == [0, 3, 1, 4, 2]


# ------------------------------------------------------------------ LLM map-reduce

POKEMON_NOTE = "The PokéCenter in Viridian City is north of the gym."


class PartialSight(LLMBackend):
    """A fake model that only knows what is in its prompt: it proposes a correction for any
    memory id it sees together with a message saying the PokéCenter is south, and a merge for
    duplicate notes it sees side by side."""

    name = "partial"

    def __init__(self, fail_parts=(), **kw):
        super().__init__(model="sight-1", **kw)
        self.prompts = []
        self.fail_parts = set(fail_parts)

    def complete(self, system, prompt, json_schema=None):
        self.prompts.append(prompt)
        part = re.search(r"Input part (\d+) of (\d+)", prompt)
        if part and int(part.group(1)) in self.fail_parts and "Task: VERIFY" in prompt:
            raise RuntimeError("simulated outage")
        ids = dict(re.findall(r"^- id='([^']+)'.*\n  (.*)$", prompt, re.M))
        cites = re.findall(r"^\[([^\]]+)\](?: @\S+)? \w+: (.*)$", prompt, re.M)
        ops = []
        if "Task: CONSOLIDATE" in prompt:
            by_text = {}
            for i, text in ids.items():
                by_text.setdefault(text.strip().lower(), []).append(i)
            for group in by_text.values():
                if len(group) > 1:
                    ops.append({"type": "merge", "targets": group, "new_content": ids[group[0]],
                                "summary": "dupes", "confidence": 0.8})
        if "Task: VERIFY" in prompt:
            south = [(c, q) for c, q in cites if "south of the gym" in q]
            for i, text in ids.items():
                if "north of the gym" in text and south:
                    ops.append({"type": "correct", "targets": [i], "new_content": text.replace("north", "south"),
                                "confidence": 0.9, "evidence": [{"citation": c, "quote": q} for c, q in south]})
        return json.dumps({"operations": ops})


def _pokemon_workspace(tmp_path, filler_entries=60, filler_msgs=40):
    filler = (f"Filler note {i}: the {i}th widget uses gear size {i * 3}." for i in range(filler_entries))
    lines = [POKEMON_NOTE, *filler]
    lines.insert(filler_entries // 2, "Always run the linter before committing.")
    lines.append("Always run the linter before committing.")
    mem = write(tmp_path / "mem" / "MEMORY.md", "".join(f"- {ln}\n" for ln in lines)).parent
    s = tmp_path / "s"
    rows = [{"role": "assistant", "content": f"Filler step {i}: looked at widget {i} and moved on."}
            for i in range(filler_msgs)]
    rows[5] = {"role": "assistant", "content": "Walked north of the gym. No PokéCenter here."}
    rows[filler_msgs - 3] = {"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."}
    write_jsonl(s / "run1.jsonl", rows)
    write_jsonl(s / "run2.jsonl", [{"role": "assistant", "content": f"Another filler {i}."} for i in range(30)]
                + [{"role": "assistant", "content": "Confirmed again: the PokéCenter is south of the gym."}])
    return mem, s


def test_map_reduce_with_partial_sight_backend(tmp_path):
    mem, s = _pokemon_workspace(tmp_path)
    be = PartialSight()
    job = dream(mem, [s], backend=be, max_chunk_tokens=1500, chunk_overlap=60)
    assert job.status == "completed", job.error
    assert len(be.prompts) > 3, "input should have been split into several calls"
    # every individual prompt saw only part of the transcripts
    assert all(not ("Walked north" in p and "Confirmed again" in p) for p in be.prompts if "Task: VERIFY" in p)
    (corr,) = ops_of(job, "correct")
    assert corr.origin == "partial (sight-1)"
    cites = {e.citation.rsplit("/", 1)[-1] for e in corr.evidence}
    assert {"run1.jsonl:38", "run2.jsonl:31"} <= cites, cites  # evidence from different chunks unioned
    (merge,) = ops_of(job, "merge")
    assert len(merge.targets) == 2
    assert POKEMON_NOTE.replace("north", "south") in final_contents(job)
    assert any("map-reduce" in n for n in job.report["notices"])


def test_prompts_fit_the_budget(tmp_path):
    from agentdream.engine.llm import SYSTEM_PROMPT

    mem, s = _pokemon_workspace(tmp_path, filler_entries=120, filler_msgs=120)
    be = PartialSight()
    job = dream(mem, [s], backend=be, max_chunk_tokens=2000)
    assert job.status == "completed", job.error
    assert max(estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(p) for p in be.prompts) <= 2000


def test_chunked_and_unchunked_runs_give_same_op_ids(tmp_path):
    mem, s = _pokemon_workspace(tmp_path)
    small = dream(mem, [s], backend=PartialSight(), max_chunk_tokens=1500)
    whole = dream(mem, [s], backend=PartialSight(), max_chunk_tokens=200_000)
    assert sorted(o.id for o in small.operations if o.origin != "deterministic") == \
        sorted(o.id for o in whole.operations)
    again = dream(mem, [s], backend=PartialSight(), max_chunk_tokens=1500)
    assert [o.id for o in again.operations] == [o.id for o in small.operations]


def test_failed_chunk_is_filled_in_deterministically(tmp_path):
    mem, s = _pokemon_workspace(tmp_path)
    be = PartialSight(fail_parts={1})
    job = dream(mem, [s], backend=be, max_chunk_tokens=1500)
    assert job.status == "completed", job.error
    assert any("part 1/" in w and "failed" in w for w in job.warnings)
    assert any("filled in" in w for w in job.warnings)
    assert POKEMON_NOTE.replace("north", "south") in final_contents(job)


def test_max_chunks_fail_and_truncate(tmp_path):
    mem, s = _pokemon_workspace(tmp_path)
    job = dream(mem, [s], backend=PartialSight(), max_chunk_tokens=1500, max_chunks=2, on_limit="fail")
    assert job.status == "failed" and "max_chunks=2" in job.error
    be = PartialSight()
    job = dream(mem, [s], backend=be, max_chunk_tokens=1500, max_chunks=2)
    assert job.status == "completed", job.error
    assert any("max_chunks=2" in n and "deterministic" in n for n in job.report["notices"])
    assert sum("Task: VERIFY" in p for p in be.prompts) <= 2


def test_too_small_chunk_budget_is_a_clear_error(tmp_path):
    mem, s = _pokemon_workspace(tmp_path, 5, 10)
    job = dream(mem, [s], backend=PartialSight(), max_chunk_tokens=300)
    assert job.status == "failed" and "max_chunk_tokens=300 is too small" in job.error


def _ctx(entries, transcripts=(), **cfg):
    ctx = DreamContext(DreamConfig(**cfg))
    ctx.entries = list(entries)
    ctx.transcripts = list(transcripts)
    return ctx


def test_reduce_dedupes_unifies_merges_and_resolves_conflicts():
    ents = [MemoryEntry(id=x, content=f"note {x}") for x in "abcd"]
    t = _transcript("s", 4, path="/l/s.jsonl")
    ctx = _ctx(ents, [t])
    mk = lambda raw, idx=0: convert_operations(ctx, raw.pop("_phase"), [raw], "fake", Call(idx, 3, ents, None))  # noqa: E731
    ev = lambda n: [{"citation": f"/l/s.jsonl:{n}", "quote": f"s message {n - 1}"}]  # noqa: E731
    c1 = mk({"_phase": "verify", "type": "correct", "targets": ["a"], "new_content": "A fixed", "evidence": ev(1)})
    c1b = mk({"_phase": "verify", "type": "correct", "targets": ["a"], "new_content": "A fixed", "evidence": ev(2)}, 1)
    c2 = mk({"_phase": "verify", "type": "correct", "targets": ["a"], "new_content": "A other", "evidence": ev(3)}, 2)
    out = reduce_operations(ctx, [c1, c1b, c2])
    (win,) = out
    assert win.after == ["A fixed"] and len(win.evidence) == 2  # duplicates unioned -> most evidence wins
    assert any("A other" in h and "/l/s.jsonl:3" in h for h in win.history)
    assert any("conflict" in w for w in ctx.warnings)

    m1 = mk({"_phase": "consolidate", "type": "merge", "targets": ["a", "b"], "new_content": "AB"})
    m2 = mk({"_phase": "consolidate", "type": "merge", "targets": ["b", "c"], "new_content": "AB",
             "confidence": 0.9}, 1)
    flag = mk({"_phase": "consolidate", "type": "contradiction", "targets": ["d", "a"], "kind": "conflict"}, 2)
    out = reduce_operations(ctx, [m1, m2, flag])
    merge = next(o for o in out if o.type == "merge")
    assert merge.targets == ["a", "b", "c"] and merge.remove == ["b", "c"]
    assert merge.id == Operation.make_id("merge", ["a", "b", "c"], ["AB"])
    f = next(o for o in out if o.type == "contradiction")
    assert not f.changes_store, "a contradiction flag without new content must not delete entries"
    assert reduce_operations(ctx, [[], []]) == []


def test_excerpt_entries_are_read_only(tmp_path):
    big = "The deploy checklist: " + "step text. " * 600 + "The PokéCenter is north of the gym."
    ents = [MemoryEntry(id="big", content=big)]
    ctx = _ctx(ents, [_transcript("s", 2, path="/l/s.jsonl")], max_chunk_tokens=1200)
    calls, _ = plan_calls(ctx, "consolidate")
    assert len(calls) > 1 and all(c.excerpt_ids() == {"big"} for c in calls)
    ops = convert_operations(ctx, "verify", [{"type": "correct", "targets": ["big"], "new_content": "short!",
                                              "evidence": [{"citation": "/l/s.jsonl:1", "quote": "s message 0"}]}],
                             "fake", calls[0])
    (op,) = ops
    assert op.type == "contradiction" and not op.changes_store
    assert any("excerpt" in w for w in ctx.warnings)


def test_large_store_pairs_transcript_chunks_with_relevant_entries():
    ents = [MemoryEntry(id=f"e{i}", content=f"The {w} service runs on port {1000 + i}.")
            for i, w in enumerate(f"alpha{k}" for k in range(400))]
    msgs = [Message("assistant", f"The alpha{k} service runs on port {5000 + k} now.", None, k + 1, "/l/a.jsonl", k)
            for k in range(0, 400, 40)]
    ctx = _ctx(ents, [Transcript("a", msgs, "/l/a.jsonl", "jsonl")], max_chunk_tokens=3000)
    calls, _ = plan_calls(ctx, "verify")
    assert calls
    for c in calls:
        mentioned = {m.content.split()[1] for sl in c.slices for m in sl.messages}
        assert {e.content.split()[1] for e in c.entries} <= mentioned  # only relevant entries ride along


# ------------------------------------------------------------------ performance


def test_deterministic_engine_scales_to_large_inputs(tmp_path):
    mem, ses = make_large_workspace(tmp_path, n_entries=3000, n_sessions=200)
    t0 = time.perf_counter()
    job = dream(mem, [ses], max_sessions=200, max_total_tokens=10**8, jobs_dir=tmp_path / "jobs")
    elapsed = time.perf_counter() - t0
    assert job.status == "completed", job.error
    assert job.counts["entries_before"] == 3000 and job.counts["merged"] > 0 and job.counts["corrected"] > 0
    # ~3-4 s on a laptop; an O(n^2) engine takes minutes. Generous bound for slow CI runners.
    assert elapsed < 30, f"deterministic dream on 3000 entries + 200 sessions took {elapsed:.1f}s"
