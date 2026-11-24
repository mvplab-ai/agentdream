"""Transcript loaders: every format, auto-detection, sources, ordering, citations."""

import json

import pytest

from agentdream import load_transcript, load_transcripts
from agentdream.loaders import LoaderError, detect_format, expand_sources

from conftest import EXAMPLES, write, write_jsonl


def test_generic_jsonl(tmp_path):
    p = write_jsonl(tmp_path / "a.jsonl", [
        {"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00Z"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": ""},
        {"speaker": "human", "text": "bye"},
    ])
    t = load_transcript(p)
    assert t.format == "jsonl"
    assert [(m.role, m.content, m.line_no) for m in t.messages] == [
        ("user", "hi", 1), ("assistant", "hello", 2), ("user", "bye", 4)]
    assert t.messages[0].citation == f"{p}:1"
    assert t.messages[0].timestamp == "2026-01-01T00:00:00Z"


def test_claude_code_session(tmp_path):
    rows = [
        {"type": "summary", "summary": "x"},
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "<command-name>/clear</command-name>"}},
        {"type": "user", "timestamp": "2026-02-01T10:00:00Z", "message": {"role": "user", "content": "Fix the bug"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "secret thoughts"},
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "lots of passing output"}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "is_error": True,
             "content": [{"type": "text", "text": "ModuleNotFoundError: x"}]}]}},
        {"type": "user", "message": {"role": "user", "content": "<local-command-stdout>noise</local-command-stdout>"}},
    ]
    p = write_jsonl(tmp_path / "abc123.jsonl", rows)
    t = load_transcript(p)
    assert t.format == "claude-code"
    got = [(m.role, m.content, m.line_no) for m in t.messages]
    assert got == [("user", "Fix the bug", 3), ("tool", "[tool] Bash: pytest -q", 4), ("assistant", "Looking.", 4),
                   ("tool", "[tool error] ModuleNotFoundError: x", 6)]
    assert all("secret thoughts" not in m.content and "passing output" not in m.content for m in t.messages)


@pytest.mark.parametrize("wrapped", [True, False])
def test_openai_messages(tmp_path, wrapped):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "search"}}]},
            {"role": "assistant", "content": "a"}]
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"messages": msgs} if wrapped else msgs, indent=2))
    t = load_transcript(p)
    assert t.format == "openai"
    assert [(m.role, m.content, m.index) for m in t.messages] == [
        ("system", "sys", 0), ("user", "q", 1), ("assistant", "[tool] search", 2), ("assistant", "a", 3)]
    lines = p.read_text().splitlines()
    for m in t.messages:
        assert '"role"' in lines[m.line_no - 1]


def test_plain_text_turns_and_raw(tmp_path):
    p = write(tmp_path / "t.txt", "User: hello\nthere\nAssistant: hi!\n\nHuman: again\nAI: ok\n")
    t = load_transcript(p)
    assert t.format == "text"
    assert [(m.role, m.content, m.line_no) for m in t.messages] == [
        ("user", "hello\nthere", 1), ("assistant", "hi!", 3), ("user", "again", 5), ("assistant", "ok", 6)]
    raw = load_transcript(write(tmp_path / "game.log", "[2026-01-01 10:00] moved north\n\nopened menu\n"))
    assert [(m.role, m.line_no) for m in raw.messages] == [("log", 1), ("log", 3)]
    assert raw.messages[0].timestamp == "2026-01-01 10:00"


def test_detect_format():
    assert detect_format('[{"role": "user", "content": "x"}]') == "openai"
    assert detect_format('{"messages": []}') == "openai"
    assert detect_format('{"role": "user", "content": "x"}\n{"role": "assistant", "content": "y"}\n') == "jsonl"
    assert detect_format('{"type": "user", "message": {"role": "user", "content": "x"}}\n') == "claude-code"
    assert detect_format("User: hi\nAssistant: yo") == "text"
    assert detect_format("just some log") == "text"


def test_sources_dirs_globs_and_ordering(tmp_path):
    d = tmp_path / "logs"
    write(d / "s10.txt", "User: ten\n")
    write(d / "s2.txt", "User: two\n")
    write(d / "nested" / "s3.jsonl", '{"role": "user", "content": "three"}\n')
    write(d / ".hidden" / "x.txt", "User: hidden\n")
    write(d / "image.png", "nope")
    files = expand_sources([d])
    assert sorted(f.name for f in files) == ["s10.txt", "s2.txt", "s3.jsonl"]
    ts, _ = load_transcripts([str(d / "*.txt")])
    assert [t.id for t in ts] == ["s2", "s10"]  # natural order when no timestamps
    with pytest.raises(LoaderError):
        expand_sources([tmp_path / "missing"])


def test_timestamp_ordering_and_max_sessions(tmp_path):
    write_jsonl(tmp_path / "b.jsonl", [{"role": "user", "content": "later", "timestamp": "2026-02-01T00:00:00Z"}])
    write_jsonl(tmp_path / "a.jsonl", [{"role": "user", "content": "latest", "timestamp": "2026-03-01T00:00:00Z"}])
    write_jsonl(tmp_path / "c.jsonl", [{"role": "user", "content": "earliest", "timestamp": "2026-01-01T00:00:00Z"}])
    ts, notes = load_transcripts([tmp_path], max_sessions=2)
    assert [t.messages[0].content for t in ts] == ["later", "latest"]
    assert notes and "most recent 2" in notes[0]
    with pytest.raises(LoaderError):
        load_transcripts([tmp_path], min_sessions=5)


def test_examples_all_formats_autodetected():
    ts, _ = load_transcripts([EXAMPLES / "sessions"])
    assert {t.format for t in ts} == {"jsonl", "claude-code", "openai"}
