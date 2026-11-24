"""Windows CRLF, old-Mac CR and mixed line endings in markdown memory and transcripts; citation line numbers."""

from __future__ import annotations

import json
import re

from agentdream import accept, dream, get_adapter
from agentdream.loaders import load_transcript

from edgekit import write_bytes

ROWS = [{"role": "assistant", "content": "Walked north of the gym. No PokéCenter here."},
        {"role": "assistant", "content": "Still no PokéCenter north of the gym."},
        {"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."}]


def _jsonl(rows, eols):
    return "".join(json.dumps(r, ensure_ascii=False) + eol for r, eol in zip(rows, eols, strict=True)).encode()


def _line_of(path, needle):
    """1-based line (universal newlines) of the first line containing needle."""
    lines = re.split(r"\r\n|\r|\n", path.read_bytes().decode("utf-8"))
    return next(i for i, ln in enumerate(lines, 1) if needle in ln)


def test_crlf_jsonl_line_numbers_and_content(tmp_path):
    """CRLF JSONL: messages keep 1-based line numbers and contain no stray \\r."""
    p = write_bytes(tmp_path / "s" / "w.jsonl", _jsonl(ROWS, ["\r\n"] * 3))
    t = load_transcript(p)
    assert [m.line_no for m in t.messages] == [1, 2, 3]
    assert not any("\r" in m.content for m in t.messages)


def test_mixed_and_cr_only_jsonl_line_numbers(tmp_path):
    """JSONL mixing LF, CRLF and a lone CR (old Mac) counts lines like an editor does."""
    p = write_bytes(tmp_path / "s" / "m.jsonl", b"\n" + _jsonl(ROWS, ["\r\n", "\r", "\n"]))
    t = load_transcript(p)
    assert [m.line_no for m in t.messages] == [2, 3, 4]


def test_crlf_text_transcript_line_numbers(tmp_path):
    """Plain-text User:/Assistant: transcript with CRLF: turn line numbers are right, content has no \\r."""
    p = write_bytes(tmp_path / "s" / "t.txt", b"User: hi\r\n\r\nAssistant: hello\r\nmore\r\nUser: bye\r\n")
    t = load_transcript(p)
    assert [(m.role, m.line_no) for m in t.messages] == [("user", 1), ("assistant", 3), ("user", 5)]
    assert t.messages[1].content == "hello\nmore"


def test_crlf_pretty_openai_json_line_numbers(tmp_path):
    """Pretty-printed OpenAI JSON with CRLF: each message cites the line of its "role" key."""
    text = json.dumps({"messages": ROWS}, indent=2, ensure_ascii=False).replace("\n", "\r\n")
    p = write_bytes(tmp_path / "s" / "o.json", text.encode())
    t = load_transcript(p)
    raw = p.read_bytes().decode().split("\r\n")
    for m in t.messages:
        assert '"role"' in raw[m.line_no - 1]


def test_citations_correct_for_crlf_sessions(tmp_path):
    """Pokémon correction over CRLF/mixed sessions cites file:line that really holds the quoted text."""
    mem = write_bytes(tmp_path / "mem" / "MEMORY.md",
                      b"# Notes\r\n\r\n- The Pok\xc3\xa9Center in Viridian City is north of the gym.\r\n").parent
    ses = tmp_path / "s"
    p = write_bytes(ses / "1.jsonl", b"\r\n" + _jsonl(ROWS, ["\r\n", "\n", "\r\n"]))
    job = dream(mem, [ses])
    corr = [o for o in job.operations if o.type == "correct"]
    assert corr, [o.summary for o in job.operations]
    for ev in corr[0].evidence:
        path, line = ev.citation.rsplit(":", 1)
        assert path == str(p)
        assert int(line) == _line_of(p, json.dumps(ev.quote, ensure_ascii=False)[1:30])


def test_crlf_markdown_entries_have_no_cr_and_right_source_lines(tmp_path):
    """CRLF markdown: entry content has no \\r and source points at the right line."""
    mem = write_bytes(tmp_path / "mem" / "MEMORY.md",
                      b"# Title\r\n\r\n- first note\r\n  continued\r\n- second note\r\n\r\nA paragraph.\r\n").parent
    es = get_adapter(mem).load()
    assert [e.content for e in es] == ["first note\ncontinued", "second note", "A paragraph."]
    assert [e.source for e in es] == ["MEMORY.md:3", "MEMORY.md:5", "MEMORY.md:7"]


def test_crlf_markdown_rewrite_keeps_crlf(tmp_path):
    """A CRLF markdown file rewritten by a merge keeps CRLF on every line (no mixed endings introduced)."""
    raw = b"# Tooling\r\n\r\n- Use pytest for tests.\r\n- Use pytest for tests!\r\n- Llamas are great.\r\n"
    mem = write_bytes(tmp_path / "mem" / "MEMORY.md", raw).parent
    ses = tmp_path / "s"
    write_bytes(ses / "1.jsonl", _jsonl(ROWS[:1], ["\n"]))
    job = dream(mem, [ses])
    assert any(o.type == "merge" for o in job.operations)
    out = tmp_path / "out"
    accept(job, to=out)
    data = (out / "MEMORY.md").read_bytes()
    assert data != raw
    assert b"\n" not in data.replace(b"\r\n", b"")  # every newline is CRLF
    assert b"Llamas are great." in data


def test_untouched_crlf_file_is_byte_identical(tmp_path):
    """An untouched CRLF file in a folder store is copied byte for byte."""
    raw = b"- Only note here.\r\n"
    mem = write_bytes(tmp_path / "mem" / "keep.md", raw).parent
    write_bytes(mem / "MEMORY.md", b"- Use pytest.\n- Use pytest!\n")
    ses = tmp_path / "s"
    write_bytes(ses / "1.jsonl", _jsonl(ROWS[:1], ["\n"]))
    job = dream(mem, [ses])
    out = tmp_path / "out"
    accept(job, to=out)
    assert (out / "keep.md").read_bytes() == raw


def test_mixed_line_endings_markdown_uses_majority(tmp_path):
    """A markdown file with mostly CRLF and one LF line is rewritten with CRLF only."""
    raw = b"- Use pytest for tests.\r\n- Use pytest for tests!\n- Llamas are great.\r\n- Alpacas too.\r\n"
    mem = write_bytes(tmp_path / "mem" / "MEMORY.md", raw).parent
    assert [e.content for e in get_adapter(mem).load()][1] == "Use pytest for tests!"
    ses = tmp_path / "s"
    write_bytes(ses / "1.jsonl", _jsonl(ROWS[:1], ["\n"]))
    job = dream(mem, [ses])
    out = tmp_path / "out"
    accept(job, to=out)
    data = (out / "MEMORY.md").read_bytes()
    assert b"\n" not in data.replace(b"\r\n", b"")


def test_crlf_front_matter(tmp_path):
    """Claude Code style front-matter file with CRLF: metadata values have no \\r, body is the entry."""
    raw = b"---\r\nname: prefs\r\nupdated: 2026-01-02\r\n---\r\nUser likes short answers.\r\n"
    mem = write_bytes(tmp_path / "mem" / "prefs.md", raw).parent
    (e,) = get_adapter(mem).load()
    assert e.content == "User likes short answers."
    assert e.metadata["name"] == "prefs" and e.updated_at == "2026-01-02"
