"""Malformed, partial and truncated JSON / JSONL in transcripts and memory stores; mixed formats."""

from __future__ import annotations

import json

import pytest

from agentdream import dream, get_adapter
from agentdream.adapters import AdapterError
from agentdream.loaders import LoaderError, detect_format, load_transcript, load_transcripts

from edgekit import workspace, write, write_bytes, write_jsonl

ROWS = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"}]


def _jsonl_text(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


def test_jsonl_last_line_truncated(tmp_path):
    """JSONL whose last line was cut off mid-write: earlier lines load, notice reports the skipped line."""
    text = _jsonl_text(ROWS[:2]) + '{"role": "user", "content": "thi'
    p = write(tmp_path / "s" / "cut.jsonl", text)
    t = load_transcript(p)
    assert [m.content for m in t.messages] == ["first", "second"]
    assert t.metadata["skipped_lines"] == [3]
    _, notices = load_transcripts([p])
    assert any("1 malformed line" in n and "cut.jsonl" in n and "3" in n for n in notices)


def test_jsonl_with_utf8_bom(tmp_path):
    """A UTF-8 BOM before the first JSONL line doesn't lose the first message or change detection."""
    p = write_bytes(tmp_path / "s" / "bom.jsonl", b"\xef\xbb\xbf" + _jsonl_text(ROWS).encode())
    t = load_transcript(p)
    assert t.format == "jsonl"
    assert [m.content for m in t.messages] == ["first", "second", "third"]
    assert [m.line_no for m in t.messages] == [1, 2, 3]


def test_single_line_jsonl_with_bom(tmp_path):
    """A one-line JSONL file with a BOM is still detected as JSONL."""
    p = write_bytes(tmp_path / "s" / "one.jsonl", b"\xef\xbb\xbf" + json.dumps(ROWS[0]).encode() + b"\n")
    assert load_transcript(p).format == "jsonl"


def test_openai_json_with_bom(tmp_path):
    """OpenAI-style messages JSON with a BOM is detected and parsed."""
    p = write_bytes(tmp_path / "s" / "o.json", b"\xef\xbb\xbf" + json.dumps({"messages": ROWS}).encode())
    t = load_transcript(p)
    assert t.format == "openai" and len(t.messages) == 3


def test_jsonl_trailing_commas_and_non_object_lines(tmp_path):
    """JSONL lines with trailing commas, arrays, numbers, strings or null are skipped and counted, not fatal."""
    v = [json.dumps({"role": "user", "content": f"v{i}"}) for i in range(5)]
    lines = [v[0], '{"role": "user", "content": "comma",}', "[1, 2]", v[1], "42", v[2], '"str"', v[3], "null", v[4]]
    p = write(tmp_path / "s" / "mixed.jsonl", "\n".join(lines) + "\n")
    t = load_transcript(p)
    assert t.format == "jsonl"
    assert [(m.content, m.line_no) for m in t.messages] == [("v0", 1), ("v1", 4), ("v2", 6), ("v3", 8), ("v4", 10)]
    assert t.metadata["skipped_lines"] == [2, 3, 5, 7, 9]


def test_claude_code_lines_with_wrong_shapes(tmp_path):
    """Claude Code lines whose message/content/input have unexpected types are skipped, not a crash."""
    rows = [
        {"type": "user", "message": "just a string"},
        {"type": "assistant", "message": {"role": "assistant", "content": 42}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Bash", "input": ["not", "a", "dict"]}, "bare string block", None]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
    ]
    p = write_jsonl(tmp_path / "s" / "cc.jsonl", rows)
    t = load_transcript(p, "claude-code")
    assert [m.content for m in t.messages][-1] == "ok"
    assert any(m.content.startswith("[tool] Bash") for m in t.messages)


def test_openai_shapes_wrong_types(tmp_path):
    """OpenAI messages with non-list tool_calls, non-dict items, or non-string roles don't crash."""
    data = {"messages": [{"role": "assistant", "content": None, "tool_calls": "oops"}, "junk", 7,
                         {"role": ["weird"], "content": "hello"}, {"role": "user", "content": [{"text": "hi"}]}]}
    p = write(tmp_path / "s" / "o.json", json.dumps(data))
    t = load_transcript(p)
    assert [m.content for m in t.messages][-2:] == ["hello", "hi"]


def test_truncated_openai_json_falls_back_or_errors_cleanly(tmp_path):
    """A truncated OpenAI JSON file: auto-detect falls back to text; forcing 'openai' raises LoaderError."""
    p = write(tmp_path / "s" / "o.json", json.dumps({"messages": ROWS})[:-10])
    t = load_transcript(p)
    assert t.format == "text"
    with pytest.raises(LoaderError, match=r"o\.json"):
        load_transcript(p, "openai")


def test_forced_format_on_wrong_content(tmp_path):
    """Forcing jsonl/claude-code on plain text yields no messages (skipped), never an exception."""
    p = write(tmp_path / "s" / "plain.txt", "User: hi\nAssistant: hello\n")
    for fmt in ("jsonl", "claude-code"):
        assert load_transcript(p, fmt).messages == []
    with pytest.raises(LoaderError, match="unknown transcript format"):
        load_transcript(p, "yaml")


def test_deeply_nested_json_does_not_crash_loader(tmp_path):
    """Pathologically nested JSON ([[[[...) doesn't raise RecursionError out of the loader."""
    p = write(tmp_path / "s" / "deep.json", "[" * 100_000 + "]" * 100_000)
    t = load_transcript(p)  # detected as text (or empty), but no crash
    assert t.format in ("text", "openai")
    with pytest.raises(LoaderError):
        load_transcript(p, "openai")


def test_mixed_formats_in_one_directory(tmp_path):
    """One directory holding Claude Code JSONL, generic JSONL, OpenAI JSON and plain text: each detected."""
    d = tmp_path / "mixed"
    write_jsonl(d / "a-claude.jsonl", [{"type": "user", "message": {"role": "user", "content": "cc hello"},
                                        "timestamp": "2026-01-01T00:00:00Z"}])
    write_jsonl(d / "b-generic.jsonl", [{"role": "user", "content": "generic", "timestamp": "2026-01-02T00:00:00Z"}])
    write(d / "c-openai.json", json.dumps([{"role": "user", "content": "openai", "timestamp": "2026-01-03T00:00:00Z"}]))
    write(d / "d-text.txt", "User: plain text\nAssistant: reply\n")
    write(d / "e-raw.log", "2026-01-05 10:00 started\n2026-01-05 10:01 done\n")
    ts, _ = load_transcripts([d])
    fmts = {t.source_path.rsplit("/", 1)[-1]: t.format for t in ts}
    assert fmts == {"a-claude.jsonl": "claude-code", "b-generic.jsonl": "jsonl", "c-openai.json": "openai",
                    "d-text.txt": "text", "e-raw.log": "text"}
    mem, _ = workspace(tmp_path)
    assert dream(mem, [d]).status == "completed"


def test_jsonl_memory_store_truncated_last_line_is_an_error(tmp_path):
    """A JSONL *memory* store with a cut-off last line fails with file:line (never drops the entry silently)."""
    store = write(tmp_path / "m.jsonl", '{"id": "a", "content": "one"}\n{"id": "b", "content": "tw')
    with pytest.raises(AdapterError, match=r"m\.jsonl:2"):
        get_adapter(store).load()
    _, ses = workspace(tmp_path / "w")
    job = dream(store, [ses])
    assert job.status == "failed" and "m.jsonl:2" in job.error


def test_json_memory_store_trailing_comma_is_an_error(tmp_path):
    """A JSON memory store with a trailing comma -> AdapterError with line/column, not a bare JSONDecodeError."""
    store = write(tmp_path / "m.json", '[\n  {"id": "a", "content": "one"},\n]\n')
    with pytest.raises(AdapterError, match=r"m\.json: invalid JSON at line 2 column \d+: .*trailing comma"):
        get_adapter(store).load()


def test_json_memory_store_with_bom(tmp_path):
    """A JSON / JSONL memory store saved with a UTF-8 BOM loads normally."""
    for name, body in (("m.json", '[{"id": "a", "content": "one"}]'), ("m.jsonl", '{"id": "a", "content": "one"}\n')):
        store = write_bytes(tmp_path / name, b"\xef\xbb\xbf" + body.encode())
        assert [e.content for e in get_adapter(store).load()] == ["one"]


def test_jsonl_memory_store_non_object_line(tmp_path):
    """A number on a JSONL memory line -> AdapterError naming file:line."""
    store = write(tmp_path / "m.jsonl", '{"id": "a", "content": "one"}\n42\n')
    with pytest.raises(AdapterError, match=r"m\.jsonl:2.*not an object"):
        get_adapter(store).load()


def test_json_memory_store_scalar_top_level(tmp_path):
    """A JSON store whose top level is a number or string -> AdapterError 'unsupported JSON top-level'."""
    store = write(tmp_path / "m.json", "42")
    with pytest.raises(AdapterError, match="unsupported JSON top-level"):
        get_adapter(store).load()


def test_detect_format_ignores_bom_and_leading_blank_lines():
    """detect_format sees through a leading BOM and blank lines."""
    text = "﻿\n\n" + _jsonl_text(ROWS)
    assert detect_format(text) == "jsonl"
