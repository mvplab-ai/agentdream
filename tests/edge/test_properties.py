"""Property-based tests (hypothesis): redaction, loaders, dreams, chunking, accept/discard round trips."""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from agentdream import accept, discard, dream, get_adapter, make_config
from agentdream.adapters import AdapterError
from agentdream.chunking import (
    SESSION_HEADER_TOKENS,
    chunk_memory,
    chunk_transcripts,
    entry_cost,
    message_cost,
    split_entry,
    split_message,
)
from agentdream.loaders import LoaderError, load_transcript
from agentdream.models import MemoryEntry, Message, Transcript
from agentdream.pipeline import run_dream
from agentdream.redact import Redactor
from agentdream.review import ReviewError, restore

from edgekit import tree_digest, write, write_jsonl

FAST = settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                         HealthCheck.too_slow])
SLOW = settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                         HealthCheck.too_slow])

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_body = st.text(alphabet=ALNUM, min_size=40, max_size=48)
_upper = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=16, max_size=16)
# passwords: letters and digits both present, so they are never "just a number" or a prose word
_pw = st.tuples(st.text(alphabet="abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ", min_size=6, max_size=10),
                st.text(alphabet="23456789", min_size=3, max_size=5)).map(lambda t: t[0] + t[1])


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


# (rendered secret, substring that must never survive)
SECRETS = st.one_of(
    _body.map(lambda b: ("sk-ant-api03-" + b, b)),
    _body.map(lambda b: ("ghp_" + b[:36], b[:36])),
    _upper.map(lambda b: ("AKIA" + b, "AKIA" + b)),
    _pw.map(lambda p: ("password=" + p, p)),
    _pw.map(lambda p: (f"postgres://admin:{p}@db.example.com:5432/app", p)),
    _pw.map(lambda p: (f"https://api.example.com/x?token={p}&a=1", p)),
    _pw.map(lambda p: (_b64("api_key=" + p), _b64("api_key=" + p))),
    _body.map(lambda b: ('{"client_secret": "' + b + '"}', b)),
)
CONTEXT = st.text(st.characters(blacklist_categories=("Cs",)), max_size=60)


@FAST
@given(before=CONTEXT, secret=SECRETS, after=CONTEXT)
def test_redaction_never_leaks_planted_secrets(before, secret, after):
    """Whatever text surrounds it (separated by whitespace), a planted secret never survives redaction."""
    rendered, core = secret
    assume(core not in before and core not in after)  # the context must not contain the secret by chance
    out = Redactor().redact(f"{before} {rendered} {after}")
    assert core not in out


@FAST
@given(text=st.one_of(CONTEXT, st.tuples(CONTEXT, SECRETS, CONTEXT).map(lambda t: f"{t[0]} {t[1][0]}\n{t[2]}")))
def test_redaction_is_idempotent(text):
    """redact(redact(x)) == redact(x), and the second pass counts nothing."""
    red = Redactor(emails=True)
    once = red.redact(text)
    red.report.counts.clear()
    assert red.redact(once) == once
    assert red.report.total == 0


EXTS = [".jsonl", ".json", ".txt", ".log", ".md"]
FORMATS = [None, "jsonl", "claude-code", "openai", "text"]
_json = st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=20),
                     lambda c: st.lists(c, max_size=4) | st.dictionaries(
                         st.sampled_from(["role", "content", "type", "message", "text", "timestamp", "tool_calls",
                                          "input", "is_error", "messages", "name"]), c, max_size=5),
                     max_leaves=12)


def _check_load(data: bytes, ext: str):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"t{ext}"
        p.write_bytes(data)
        for fmt in FORMATS:
            try:
                t = load_transcript(p, fmt)
            except LoaderError:
                continue
            assert all(isinstance(m.content, str) and m.content for m in t.messages)
            assert all(m.line_no is None or m.line_no >= 1 for m in t.messages)


@FAST
@given(data=st.binary(max_size=600), ext=st.sampled_from(EXTS))
def test_loaders_never_crash_on_arbitrary_bytes(data, ext):
    """Any bytes, any extension, any forced format: a Transcript or LoaderError, never another exception."""
    _check_load(data, ext)


@FAST
@given(rows=st.lists(_json, max_size=6), ext=st.sampled_from(EXTS), as_doc=st.booleans())
def test_loaders_never_crash_on_arbitrary_json_shapes(rows, ext, as_doc):
    """JSON / JSONL with arbitrary shapes (wrong types everywhere) loads or raises LoaderError."""
    data = json.dumps({"messages": rows} if as_doc else rows) if as_doc or ext == ".json" else \
        "\n".join(json.dumps(r) for r in rows)
    _check_load(data.encode(), ext)


@FAST
@given(data=st.binary(max_size=400), ext=st.sampled_from([".md", ".json", ".jsonl", ".txt"]))
def test_memory_adapters_never_crash_on_arbitrary_bytes(data, ext):
    """Memory stores with arbitrary bytes: entries or AdapterError, never another exception."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"m{ext}"
        p.write_bytes(data)
        try:
            entries = get_adapter(p).load()
        except AdapterError:
            return
        assert all(isinstance(e.content, str) for e in entries)


PHRASES = ["Use pytest for tests.", "Use pytest for tests!", "Don't use Poetry for dependency management.",
           "Use Poetry for dependency management.", "The API server listens on port 8080.",
           "The API server listens on port 9090.", "The PokéCenter is north of the gym.", "Llamas are calm.",
           "Deploys go through the release branch.", "שלום עולם", "🦙 emoji note"]
_note = st.one_of(st.sampled_from(PHRASES),
                  st.tuples(st.sampled_from(PHRASES), st.sampled_from(["", " (2026-01-01)", " (2026-03-05)"]))
                  .map("".join),
                  st.text(st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=40))
_line = st.sampled_from(["Found it: the PokéCenter is south of the gym.", "Let's use Poetry for dependency management.",
                         "Error: forgot to run pytest again.", "The API server listens on port 9090.", "hello"])


def _added_by(ops):
    return sum(1 for o in ops if o.type == "insight" or (o.type == "contradiction" and not o.targets and o.upsert))


@SLOW
@given(notes=st.lists(_note, max_size=10), lines=st.lists(_line, min_size=1, max_size=6))
def test_dream_never_mutates_input_and_bounds_output(notes, lines):
    """Random memories: input bytes unchanged, output entries <= input + new insight notes, every id accounted for."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        mem = write(root / "mem" / "MEMORY.md", "".join(f"- {n.strip() or 'x'}\n" for n in notes)).parent
        ses = write_jsonl(root / "s" / "1.jsonl", [{"role": "user", "content": ln} for ln in lines]).parent
        digest = tree_digest(mem, ses)
        ctx = run_dream(make_config(mem, [ses]), root / "out")
        assert tree_digest(mem, ses) == digest
        before = {e.id for e in ctx.original_entries}
        assert len(ctx.final_entries) <= len(before) + _added_by(ctx.ops)
        new_ids = {u.id for o in ctx.ops for u in o.upsert}
        assert {e.id for e in ctx.final_entries} <= before | new_ids
        assert len(get_adapter(ctx.store_path).load()) >= 0


_msgs = st.lists(st.tuples(st.sampled_from(["user", "assistant", "tool"]),
                           st.text(min_size=1, max_size=900)), min_size=0, max_size=25)


@FAST
@given(sessions=st.lists(_msgs, min_size=1, max_size=4), budget=st.integers(120, 1500),
       overlap=st.integers(0, 600))
def test_chunking_covers_every_message_once_and_respects_budget(sessions, budget, overlap):
    """Transcript chunks: each within budget; non-overlap parts concatenate to exactly the input (in order);
    overlapped messages repeat the tail of the previous chunk of the same session, within the overlap cap."""
    ts = [Transcript(f"t{i}", [Message(r, c, None, j + 1, f"s{i}.jsonl", j) for j, (r, c) in enumerate(ms)],
                     f"s{i}.jsonl") for i, ms in enumerate(sessions)]
    chunks = chunk_transcripts(ts, budget, overlap)
    expected = [(p.citation, p.content) for t in ts for m in t.messages
                for p in split_message(m, budget - SESSION_HEADER_TOKENS)]
    got, prev = [], None
    for ch in chunks:
        cost = sum(message_cost(m) for sl in ch.items for m in sl.messages) + SESSION_HEADER_TOKENS * len(ch.items)
        assert cost <= budget
        assert ch.items[0].overlap == ch.overlap and all(sl.overlap == 0 for sl in ch.items[1:])
        flat = [(sl.transcript.id, m) for sl in ch.items for m in sl.messages]
        if ch.overlap:
            assert prev is not None
            assert [m.citation for _t, m in flat[:ch.overlap]] == [m.citation for _t, m in prev[-ch.overlap:]]
            assert sum(message_cost(m) for _t, m in flat[:ch.overlap]) <= min(overlap, budget // 4)
        got += [(m.citation, m.content) for _t, m in flat[ch.overlap:]]
        prev = flat
    assert got == expected


@FAST
@given(contents=st.lists(st.text(min_size=0, max_size=1200), max_size=20), budget=st.integers(60, 900))
def test_memory_chunking_covers_entries_and_respects_budget(contents, budget):
    """Memory chunks: every entry (or each excerpt of an oversized one) appears exactly once, within budget."""
    entries = [MemoryEntry(id=f"m#{i}", content=c, path="m.md") for i, c in enumerate(contents)]
    chunks = chunk_memory(entries, budget)
    assert all(sum(entry_cost(e) for e in ch.items) <= budget for ch in chunks)
    flat = [(e.id, e.content) for ch in chunks for e in ch.items]
    assert flat == [(p.id, p.content) for e in entries for p in split_entry(e, budget)]
    for e in entries:
        assert "".join(c for i, c in flat if i == e.id) == e.content


@SLOW
@given(notes=st.lists(st.sampled_from(PHRASES), min_size=1, max_size=8))
def test_accept_discard_restore_round_trip(notes):
    """accept --to reproduces the staged store; accept over the original + restore returns the exact bytes;
    discard removes the staged store and blocks accept."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        mem = write(root / "mem" / "MEMORY.md", "".join(f"- {n}\n" for n in notes)).parent
        ses = write_jsonl(root / "s" / "1.jsonl", [{"role": "user", "content": "Use pytest for tests!"}]).parent
        jobs = str(root / "jobs")
        digest = tree_digest(mem)
        job = dream(mem, [ses], jobs_dir=jobs)
        assume(job.status == "completed")
        staged = [e.content for e in get_adapter(job.store_path).load()]
        accept(job, to=root / "copy")
        assert [e.content for e in get_adapter(root / "copy").load()] == staged
        assert tree_digest(mem) == digest

        job2 = dream(mem, [ses], jobs_dir=jobs)
        accept(job2)
        assert [e.content for e in get_adapter(mem).load()] == staged
        restore(job2.id, jobs_dir=jobs)
        assert tree_digest(mem) == digest

        job3 = dream(mem, [ses], jobs_dir=jobs)
        discard(job3)
        assert not (job3.out_dir / "store").exists()
        with pytest.raises(ReviewError):
            accept(job3)
        assert tree_digest(mem) == digest
