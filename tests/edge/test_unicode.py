"""Unicode, emoji, RTL, combining and zero-width characters, NUL bytes and invalid UTF-8."""

from __future__ import annotations

import json
import unicodedata

from agentdream import accept, dream, get_adapter
from agentdream.loaders import load_transcript

from edgekit import tree_digest, workspace, write, write_bytes, write_jsonl

RTL_MEMORY = (
    "- المستخدم يفضل الردود القصيرة.\n"          # Arabic
    "- המשתמש מעדיף תשובות קצרות.\n"             # Hebrew
    "- Mixed: the user مرحبا said שלום 🦙.\n"
)


def test_rtl_and_emoji_round_trip_unchanged(tmp_path):
    """Arabic/Hebrew/emoji notes survive a dream + accept byte-for-byte when untouched."""
    mem, ses = workspace(tmp_path, memory=RTL_MEMORY)
    before = (mem / "MEMORY.md").read_bytes()
    job = dream(mem, [ses])
    assert job.status == "completed", job.error
    out = tmp_path / "accepted"
    accept(job, to=out)
    assert (out / "MEMORY.md").read_bytes() == before
    contents = [e.content for e in get_adapter(out).load()]
    assert "המשתמש מעדיף תשובות קצרות." in contents and "Mixed: the user مرحبا said שלום 🦙." in contents


def test_rtl_and_emoji_in_transcripts_are_cited(tmp_path):
    """Transcript messages in Arabic/Hebrew/emoji keep correct line numbers and appear in the report."""
    mem, _ = workspace(tmp_path)
    ses = tmp_path / "rtl"
    write_jsonl(ses / "a.jsonl", [{"role": "user", "content": "שלום 👋"},
                                  {"role": "assistant", "content": "مرحبا 🌍"}])
    t = load_transcript(ses / "a.jsonl")
    assert [(m.content, m.line_no) for m in t.messages] == [("שלום 👋", 1), ("مرحبا 🌍", 2)]
    job = dream(mem, [ses])
    assert job.status == "completed"
    json.dumps(job.report, ensure_ascii=False)  # serialisable


def test_nfc_and_nfd_duplicates_merge(tmp_path):
    """'Café' written precomposed (NFC) and decomposed (NFD) is recognised as a duplicate."""
    nfc = unicodedata.normalize("NFC", "Meet at the Café Zürich on Fridays.")
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    mem, ses = workspace(tmp_path, memory=f"- {nfc}\n- {nfd}\n- Llamas are great.\n")
    job = dream(mem, [ses])
    merges = [o for o in job.operations if o.type == "merge"]
    assert len(merges) == 1 and len(merges[0].targets) == 2


def test_zero_width_characters_do_not_hide_duplicates(tmp_path):
    """Notes that differ only by zero-width spaces / joiners / BOM / soft hyphens are merged."""
    a = "Deploy the billing service with helm."
    b = "Deploy the bil\u200bling ser\xadvice with he\u200dlm.\ufeff"
    mem, ses = workspace(tmp_path, memory=f"- {a}\n- {b}\n- Llamas are great.\n")
    job = dream(mem, [ses])
    merges = [o for o in job.operations if o.type == "merge"]
    assert len(merges) == 1, [o.summary for o in job.operations]


def test_combining_marks_and_zalgo_do_not_crash(tmp_path):
    """Heavy stacks of combining marks in memory and transcripts are handled (no crash, input intact)."""
    zalgo = "H" + "́̂̃̄" * 50 + "ello wor" + "̶" * 30 + "ld"
    mem, _ = workspace(tmp_path, memory=f"- {zalgo}\n- plain note\n")
    ses = tmp_path / "z"
    write_jsonl(ses / "z.jsonl", [{"role": "user", "content": zalgo}])
    before = tree_digest(mem)
    job = dream(mem, [ses])
    assert job.status == "completed", job.error
    assert tree_digest(mem) == before


def test_nul_bytes_in_markdown_memory(tmp_path):
    """NUL bytes inside a markdown note are kept (not truncated) and the dream completes."""
    mem = write_bytes(tmp_path / "mem" / "MEMORY.md", b"- before\x00after note\n- second note\n").parent
    _, ses = workspace(tmp_path / "w")
    entries = get_adapter(mem).load()
    assert entries[0].content == "before\x00after note"
    job = dream(mem, [ses])
    assert job.status == "completed", job.error


def test_nul_bytes_in_json_memory_and_transcripts(tmp_path):
    """NUL characters (\\u0000) in JSON memory values and transcript content round-trip."""
    store = tmp_path / "m.json"
    store.write_text(json.dumps([{"id": "a", "content": "nul\u0000inside"}, {"id": "b", "content": "ok"}]))
    ses = tmp_path / "s"
    write_jsonl(ses / "1.jsonl", [{"role": "user", "content": "x\u0000y"}])
    assert load_transcript(ses / "1.jsonl").messages[0].content == "x\x00y"
    job = dream(store, [ses])
    assert job.status == "completed", job.error
    assert [e.content for e in get_adapter(job.store_path).load()] == ["nul\x00inside", "ok"]


def test_invalid_utf8_in_markdown_memory(tmp_path):
    """Invalid UTF-8 bytes in a markdown store: loads with U+FFFD, completes, untouched files keep their bytes."""
    raw = b"- caf\xe9 au lait\n- \xff\xfe broken\n"
    mem = write_bytes(tmp_path / "mem" / "MEMORY.md", raw).parent
    write(mem / "other.md", "- Use pytest.\n- Use pytest!\n")
    _, ses = workspace(tmp_path / "w")
    assert "�" in get_adapter(mem).load()[0].content
    job = dream(mem, [ses])
    assert job.status == "completed", job.error
    out = tmp_path / "out"
    accept(job, to=out)
    assert (out / "MEMORY.md").read_bytes() == raw  # untouched file: copied byte for byte
    assert any("invalid UTF-8" in n and "MEMORY.md" in n for n in job.report["notices"])


def test_invalid_utf8_in_transcripts(tmp_path):
    """Non-UTF-8 bytes in a JSONL transcript decode with replacement; valid lines are unaffected."""
    p = write_bytes(tmp_path / "s" / "bad.jsonl",
                    b'{"role": "user", "content": "caf\xe9"}\n{"role": "assistant", "content": "fine"}\n')
    t = load_transcript(p)
    assert [m.content for m in t.messages] == ["caf�", "fine"]
    assert [m.line_no for m in t.messages] == [1, 2]


def test_latin1_transcript_text_file(tmp_path):
    """A Latin-1 encoded plain-text transcript loads (with replacement chars) instead of crashing."""
    p = write_bytes(tmp_path / "s" / "old.txt", "User: déjà vu\nAssistant: naïve\n".encode("latin-1"))
    t = load_transcript(p)
    assert [m.role for m in t.messages] == ["user", "assistant"]


def test_invalid_utf8_json_store_fails_clearly(tmp_path):
    """A JSON memory store with invalid UTF-8 -> failed with AdapterError naming the file, not a raw traceback."""
    store = write_bytes(tmp_path / "m.json", b'[{"id": "a", "content": "caf\xe9"}]')
    _, ses = workspace(tmp_path / "w")
    job = dream(store, [ses])
    assert job.status == "failed"
    assert "AdapterError" in job.error and "m.json" in job.error and "UTF-8" in job.error


def test_unicode_line_separators_inside_jsonl_strings(tmp_path):
    """U+2028 / U+2029 / U+0085 / form feed inside JSON strings don't split JSONL lines or shift line numbers."""
    p = tmp_path / "s" / "ls.jsonl"
    p.parent.mkdir(parents=True)
    rows = [{"role": "user", "content": "one\u2028two"}, {"role": "user", "content": "three\u2029four\x85five\x0c"},
            {"role": "assistant", "content": "last"}]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    t = load_transcript(p)
    assert [m.line_no for m in t.messages] == [1, 2, 3]
    assert t.messages[0].content == "one\u2028two"


def test_unicode_line_separator_in_markdown_is_preserved(tmp_path):
    """A U+2028 inside a markdown note stays inside the note and survives a rewrite of that file."""
    mem = write(tmp_path / "mem" / "MEMORY.md", "- line\u2028sep note\n- Use pytest.\n- Use pytest!\n").parent
    entries = get_adapter(mem).load()
    assert entries[0].content == "line\u2028sep note" and len(entries) == 3
    _, ses = workspace(tmp_path / "w")
    job = dream(mem, [ses])
    assert any(o.type == "merge" for o in job.operations)
    out = tmp_path / "out"
    accept(job, to=out)
    assert "line\u2028sep note" in (out / "MEMORY.md").read_text(encoding="utf-8")
