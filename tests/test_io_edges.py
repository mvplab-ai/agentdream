"""Edge cases of transcript loaders and memory adapters (shapes, malformed input, layout preservation)."""

import contextlib
import json
import sqlite3

import pytest

from agentdream import dream, get_adapter, load_transcript, load_transcripts
from agentdream.adapters import AdapterError, JSONAdapter, MarkdownAdapter, SQLiteAdapter
from agentdream.loaders import LoaderError, _text_of, detect_format, expand_sources
from agentdream.models import MemoryEntry

from conftest import hash_tree, write, write_jsonl

# ------------------------------------------------------------------ loaders


def test_text_of_shapes():
    assert _text_of(None) == "" and _text_of("x") == "x" and _text_of(5) == "5"
    assert _text_of(["a", {"type": "text", "text": "b"}, {"type": "image"}, {"text": "c"}]) == "a\nb\nc"
    assert _text_of({"text": "d"}) == "d"


def test_claude_code_parser_skips_noise(tmp_path):
    rows = [
        "not json",
        json.dumps([1, 2]),
        json.dumps({"type": "summary", "summary": "x"}),
        json.dumps({"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta"}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": "<command-name>/clear</command-name>"}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": 42}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            "stray string", {"type": "thinking", "thinking": "hmm"},
            {"type": "tool_use", "name": "Edit", "input": {"old": "a", "new": "b"}},
            {"type": "tool_use", "name": "Noop"},
            {"type": "text", "text": "Done."}]}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "is_error": True, "content": "Traceback: boom"},
            {"type": "tool_result", "content": "fine output"}]}}),
    ]
    p = write(tmp_path / "s.jsonl", "\n".join(rows) + "\n\n")
    t = load_transcript(p)
    assert t.format == "claude-code"
    texts = [m.content for m in t.messages]
    assert texts == ['[tool] Edit: {"old": "a", "new": "b"}', "[tool] Noop: ", "Done.", "[tool error] Traceback: boom"]
    assert [m.line_no for m in t.messages] == [7, 7, 7, 8]


def test_jsonl_and_openai_parsers(tmp_path):
    p = write(tmp_path / "chat.jsonl", "garbage\n" + json.dumps([1]) + "\n"
              + json.dumps({"speaker": "Human", "text": "hi", "ts": 5}) + "\n"
              + json.dumps({"role": "assistant", "content": "  "}) + "\n")
    t = load_transcript(p)
    assert t.format == "jsonl" and [(m.role, m.content, m.timestamp) for m in t.messages] == [("user", "hi", "5")]
    o = write(tmp_path / "o.json", json.dumps({"messages": [
        {"role": "user", "content": "q"}, "junk", {"role": "assistant", "content": None,
                                                   "tool_calls": [{"function": {"name": "search"}}]},
        {"role": "tool", "content": "z" * 1000}, {"role": "assistant", "content": ""}]}, indent=1))
    t = load_transcript(o)
    assert t.format == "openai" and [m.content[:12] for m in t.messages] == ["q", "[tool] searc", "zzzzzzzzzzzz"]
    assert t.messages[2].content.endswith("...") and len(t.messages[2].content) == 403
    bad = write(tmp_path / "bad.json", json.dumps({"messages": "nope"}))
    with pytest.raises(LoaderError, match="no messages list"):
        load_transcript(bad, "openai")
    with pytest.raises(LoaderError, match="unknown transcript format"):
        load_transcript(o, "carrier-pigeon")
    assert detect_format("") == "text" and detect_format("{broken") == "text"


def test_expand_sources_and_ordering(tmp_path):
    write(tmp_path / "logs" / "a" / "1.txt", "User: a\n")
    write(tmp_path / "logs" / "b" / "1.txt", "User: b\n")
    write(tmp_path / "logs" / "b" / ".hidden.txt", "User: hidden\n")
    write(tmp_path / "logs" / "empty.txt", "\n\n")
    found = expand_sources(str(tmp_path / "logs" / "*"))
    assert sorted(p.name for p in found) == ["1.txt", "1.txt", "empty.txt"]
    assert len(expand_sources([tmp_path / "logs", tmp_path / "logs" / "a" / "1.txt"])) == 3  # deduplicated
    with pytest.raises(LoaderError, match="not found"):
        expand_sources(tmp_path / "missing")
    ts, notices = load_transcripts([tmp_path / "logs"])
    assert [t.id for t in ts] == ["1", "1-2"]  # unique ids for same-named files
    assert any("skipped empty transcript" in n for n in notices)
    ts, notices = load_transcripts([tmp_path / "logs"], max_sessions=1)
    assert len(ts) == 1 and any("keeping the most recent 1" in n for n in notices)
    with pytest.raises(LoaderError, match="need at least 3"):
        load_transcripts([tmp_path / "logs"], min_sessions=3)


# ------------------------------------------------------------------ adapter registry


def test_get_adapter_formats_and_detection(tmp_path):
    md = write(tmp_path / "a.md", "- x\n")
    assert isinstance(get_adapter(md, "md"), MarkdownAdapter)
    js = write(tmp_path / "a.jsonl", json.dumps({"id": "1", "content": "x"}) + "\n")
    assert isinstance(get_adapter(js, "jsonl"), JSONAdapter)
    with pytest.raises(AdapterError, match="unknown memory format"):
        get_adapter(md, "papyrus")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert isinstance(get_adapter(empty), MarkdownAdapter)  # empty folder -> new markdown store
    weird = write(tmp_path / "blob.bin", "\x00\x01")
    with pytest.raises(AdapterError, match="could not detect"):
        get_adapter(weird)
    with pytest.raises(AdapterError, match="not found"):
        get_adapter(tmp_path / "nope.md")


def test_stancelog_prefix_without_package(monkeypatch, tmp_path):
    monkeypatch.setattr("agentdream.adapters.stancelog.available", lambda: False)
    with pytest.raises(AdapterError, match="needs the 'stancelog' package"):
        get_adapter(f"stancelog:{tmp_path / 'b.db'}")


# ------------------------------------------------------------------ JSON adapter


def test_json_adapter_shapes_and_errors(tmp_path):
    strings = write(tmp_path / "s.json", json.dumps(["plain note", "another"]))
    es = JSONAdapter(strings).load()
    assert [e.content for e in es] == ["plain note", "another"]
    out = JSONAdapter(strings).write(es[:1], tmp_path / "out" / "s.json")
    assert json.loads(out.read_text()) == ["plain note"]
    for body, msg in [([1], "not an object"), ([{"id": "a"}], "no content field"), (7, "unsupported")]:
        p = write(tmp_path / f"bad{msg[:3]}.json", json.dumps(body))
        with pytest.raises(AdapterError, match=msg):
            JSONAdapter(p).load()
    rich = write(tmp_path / "r.json", json.dumps([{"id": "a", "value": {"nested": 1}, "key": "k",
                                                    "created": "2026-01-01", "source": "manual", "tag": "t"}]))
    (e,) = JSONAdapter(rich).load()
    assert e.content == '{"nested": 1}' and e.key == "k" and e.created_at == "2026-01-01"
    back = json.loads(JSONAdapter(rich).write([e], tmp_path / "r2.json").read_text())
    assert back == [{"id": "a", "value": '{"nested": 1}', "key": "k", "created_at": "2026-01-01",
                     "source": "manual", "tag": "t"}]
    with pytest.raises(AdapterError, match="not found"):
        JSONAdapter(tmp_path / "gone.json").load()


def test_json_folder_store_rewrites_cleanly(tmp_path):
    root = tmp_path / "store"
    write_jsonl(root / "a.jsonl", [{"id": "1", "content": "Use pytest."}, {"id": "2", "content": "Use pytest!"}])
    write(root / "b.json", json.dumps({"k": "v"}))
    assert JSONAdapter.detect(root)
    ad = JSONAdapter(root)
    entries = ad.load()
    dest = tmp_path / "out"
    write(dest / "old.json", "{}")          # existing destination is replaced
    write(tmp_path / "out.tmp-write" / "x", "")  # and so is a stale temp dir
    ad.write([*entries, MemoryEntry(id="n", content="new note")], dest)
    assert sorted(p.name for p in dest.iterdir()) == ["a.jsonl", "b.json"]
    assert "new note" in (dest / "a.jsonl").read_text()


# ------------------------------------------------------------------ markdown adapter


def test_markdown_bullets_continuations_and_rendering(tmp_path):
    text = ("# Notes\n\n"
            "- First bullet\n  continued line\n\n  second paragraph of the bullet\n"
            "lazy continuation\n"
            "- Second bullet\n"
            "  - nested child\n"
            "1. Numbered item\n\n"
            "A free paragraph\nspanning lines.\n\n"
            "Another paragraph to drop.\n\n"
            "## Later\n\n- tail item\n")
    root = write(tmp_path / "mem" / "notes.md", text).parent
    ad = MarkdownAdapter(root)
    es = ad.load()
    assert es[0].content == "First bullet\ncontinued line\n\nsecond paragraph of the bullet\nlazy continuation"
    assert es[1].content == "Second bullet\n- nested child"
    by_content = {e.content.split("\n")[0]: e for e in es}
    keep = [e for e in es if not e.content.startswith("Another paragraph")]
    keep[0] = MemoryEntry(**{**keep[0].to_dict(), "content": "First bullet, rewritten"})
    keep.append(MemoryEntry(id="new-1", content="Brand new note", key="Notes > Brand New"))
    keep.append(MemoryEntry(id="new-2", content="Section note", key="Notes > Later"))
    out = ad.write(keep, tmp_path / "out")
    rendered = (out / "notes.md").read_text()
    assert "- First bullet, rewritten\n" in rendered and "Another paragraph" not in rendered
    # an entry without a path goes to dream-insights.md under its section; one with a known section joins it
    assert "## Notes > Brand New\n\n- Brand new note" in (out / "dream-insights.md").read_text()
    assert rendered.index("- Section note") > rendered.index("- tail item")
    assert "A free paragraph\nspanning lines." in rendered
    assert "Numbered item" in by_content and "1. Numbered item" in rendered


def test_markdown_single_file_and_front_matter(tmp_path):
    fm = write(tmp_path / "mem" / "pref.md", "---\nname: terse\nupdated: 2026-03-01\n---\n\nBe terse.\n")
    ad = MarkdownAdapter(fm.parent)
    (e,) = ad.load()
    assert e.key == "terse" and e.updated_at == "2026-03-01"
    out = ad.write([e, MemoryEntry(id="x", content="Extra note", path="pref.md")], tmp_path / "out")
    assert (out / "pref.md").read_text() == "---\nname: terse\nupdated: 2026-03-01\n---\n\nBe terse.\n\n- Extra note\n"
    out2 = ad.write([], tmp_path / "out2")
    assert not (out2 / "pref.md").exists()  # front-matter file whose entry was removed
    empty_fm = write(tmp_path / "mem2" / "e.md", "---\nname: x\n---\n")
    assert MarkdownAdapter(empty_fm.parent).load() == []
    single = write(tmp_path / "CLAUDE.md", "- one\n- two\n")
    sad = MarkdownAdapter(single)
    out3 = sad.write(sad.load()[:1], tmp_path / "staged" / "CLAUDE.md")
    assert out3.read_text() == "- one\n"
    with pytest.raises(AdapterError, match="refusing to write over the input"):
        sad.write([], single)
    with pytest.raises(AdapterError, match="not found"):
        MarkdownAdapter(tmp_path / "missing").load()
    assert not MarkdownAdapter.detect(tmp_path / "nothing-here")


def test_markdown_folder_keeps_other_files_and_hidden_dirs_ignored(tmp_path):
    root = tmp_path / "mem"
    write(root / "a.md", "- Use pytest.\n- Use pytest!\n")
    write(root / "img" / "logo.png", "PNG")
    write(root / ".git" / "config", "[core]")
    before = hash_tree(root)
    s = write_jsonl(tmp_path / "s" / "1.jsonl", [{"role": "user", "content": "hi"}]).parent
    job = dream(root, [s])
    store = job.store_path
    assert job.status == "completed" and hash_tree(root) == before
    from pathlib import Path

    assert (Path(store) / "img" / "logo.png").read_text() == "PNG" and not (Path(store) / ".git").exists()


# ------------------------------------------------------------------ SQLite adapter


def test_sqlite_edges(tmp_path):
    db = tmp_path / "plain.data"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE notes (id TEXT, text TEXT, metadata TEXT, extra TEXT)")
    con.execute("INSERT INTO notes VALUES ('1', 'Use pytest.', 'not json', 'keep-me')")
    con.commit()
    con.close()
    assert SQLiteAdapter.detect(db)  # by magic bytes, not extension
    (e,) = SQLiteAdapter(db).load()
    assert e.metadata["raw_metadata"] == "not json" and e.metadata["_extra_columns"] == {"extra": "keep-me"}
    dest = tmp_path / "out.db"
    write(tmp_path / "out.db.tmp", "stale")
    SQLiteAdapter(db).write([e], dest)
    with contextlib.closing(sqlite3.connect(dest)) as con2:
        row = con2.execute("SELECT id, text, extra FROM notes").fetchone()
    assert row == ("1", "Use pytest.", "keep-me")
    other = tmp_path / "other.db"
    con = sqlite3.connect(other)
    con.execute("CREATE TABLE t (a TEXT)")
    con.commit()
    con.close()
    with pytest.raises(AdapterError, match="no table with id"):
        SQLiteAdapter(other).load()
    with pytest.raises(AdapterError, match="not found"):
        SQLiteAdapter(tmp_path / "nope.db").load()
    assert not SQLiteAdapter.detect(tmp_path)
