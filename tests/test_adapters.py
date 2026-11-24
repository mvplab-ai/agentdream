"""Memory store adapters: detection, load, write round-trip, registry."""

import json
import sqlite3

import pytest

from agentdream import MemoryEntry, available_adapters, get_adapter, register_adapter
from agentdream.adapters import (
    AdapterError,
    JSONAdapter,
    MarkdownAdapter,
    MemoryAdapter,
    SQLiteAdapter,
    create_sqlite_store,
)

from conftest import write, write_jsonl

MD = """# Memory

Intro paragraph that is
two lines long.

## Tools

- Use pytest for tests.
  Continuation line.
- Use ruff (2026-02-01).
  - nested detail

1. First numbered
2. Second numbered

```bash
make deploy
```
"""


def test_markdown_granularity_and_roundtrip(tmp_path):
    root = tmp_path / "mem"
    write(root / "MEMORY.md", MD)
    write(root / "feedback_x.md", "---\nname: terse\ndescription: be terse\ntype: feedback\n"
          "updated: 2026-03-01\n---\n\n"
                                  "Keep answers short.\n\n**Why:** user asked.\n")
    write(root / "img.png", "binary-ish")
    ad = get_adapter(root)
    assert isinstance(ad, MarkdownAdapter)
    entries = ad.load()
    by_id = {e.id: e for e in entries}
    assert by_id["MEMORY.md#1"].content == "Intro paragraph that is\ntwo lines long."
    assert by_id["MEMORY.md#2"].content == "Use pytest for tests.\nContinuation line."
    assert by_id["MEMORY.md#2"].key == "Memory > Tools"
    assert by_id["MEMORY.md#3"].content.startswith("Use ruff") and "nested detail" in by_id["MEMORY.md#3"].content
    assert by_id["MEMORY.md#3"].updated_at == "2026-02-01"
    assert by_id["MEMORY.md#4"].content == "First numbered"
    assert by_id["MEMORY.md#6"].content.startswith("```bash")
    fm = by_id["feedback_x.md"]
    assert fm.content.startswith("Keep answers short.") and fm.metadata["type"] == "feedback"
    assert fm.updated_at == "2026-03-01"

    # identity write: byte-identical files
    out = tmp_path / "out"
    ad.write(entries, out)
    assert (out / "MEMORY.md").read_text() == MD
    assert (out / "feedback_x.md").read_text() == (root / "feedback_x.md").read_text()
    assert (out / "img.png").exists()

    # edit / remove / add
    entries2 = [e.copy() for e in entries if e.id != "MEMORY.md#4"]
    for e in entries2:
        if e.id == "MEMORY.md#2":
            e.content = "Use pytest -q for tests."
        if e.id == "feedback_x.md":
            e.content = "Keep answers very short."
    entries2.append(MemoryEntry("new1", "Use uv for envs.", key="Memory > Tools"))
    entries2.append(MemoryEntry("new2", "A brand new insight."))
    out2 = tmp_path / "out2"
    ad.write(entries2, out2)
    text = (out2 / "MEMORY.md").read_text()
    assert "- Use pytest -q for tests." in text and "Continuation line." not in text
    assert "First numbered" not in text and "2. Second numbered" in text
    assert text.index("Use uv for envs.") < text.index("Second numbered")
    assert "---\nname: terse" in (out2 / "feedback_x.md").read_text()
    assert "Keep answers very short." in (out2 / "feedback_x.md").read_text()
    assert "A brand new insight." in (out2 / "dream-insights.md").read_text()
    reloaded = {e.content for e in MarkdownAdapter(out2).load()}
    assert {"Use pytest -q for tests.", "Use uv for envs.", "A brand new insight."} <= reloaded


def test_markdown_single_file(tmp_path):
    f = write(tmp_path / "CLAUDE.md", "# Rules\n\n- One.\n- Two.\n")
    ad = get_adapter(f)
    assert ad.single_file
    es = ad.load()
    es.append(MemoryEntry("n", "Three."))
    out = ad.output_path(tmp_path / "store")
    ad.write(es, out)
    assert out.name == "CLAUDE.md"
    assert out.read_text().count("- ") == 3


@pytest.mark.parametrize("shape", ["list", "wrapped", "map", "jsonl"])
def test_json_roundtrip(tmp_path, shape):
    if shape == "jsonl":
        p = write_jsonl(tmp_path / "m.jsonl", [{"id": "a", "text": "Alpha.", "tags": ["x"]},
                                         {"id": "b", "text": "Beta."}])
    else:
        data = {"list": [{"id": "a", "content": "Alpha.", "updated_at": "2026-01-01", "tags": ["x"]},
                         {"id": "b", "content": "Beta."}],
                "wrapped": {"version": 2, "memories": [{"id": "a", "content": "Alpha."},
                                                   {"id": "b", "content": "Beta."}]},
                "map": {"a": "Alpha.", "b": "Beta."}}[shape]
        p = tmp_path / "m.json"
        p.write_text(json.dumps(data))
    ad = get_adapter(p)
    assert isinstance(ad, JSONAdapter)
    es = ad.load()
    assert [e.content for e in es] == ["Alpha.", "Beta."]
    es[1].content = "Beta two."
    es.append(MemoryEntry("c", "Gamma."))
    out = tmp_path / "out" / p.name
    ad.write(es, out)
    again = JSONAdapter(out).load()
    assert [(e.id, e.content) for e in again] == [("a", "Alpha."), ("b", "Beta two."), ("c", "Gamma.")]
    raw = out.read_text()
    if shape == "wrapped":
        assert json.loads(raw)["version"] == 2
    if shape in ("list", "jsonl"):
        assert '"tags"' in raw  # unknown fields preserved
    if shape == "map":
        assert json.loads(raw) == {"a": "Alpha.", "b": "Beta two.", "c": "Gamma."}


def test_json_folder(tmp_path):
    d = tmp_path / "j"
    write_jsonl(d / "a.jsonl", [{"id": "1", "content": "One."}])
    write(d / "sub" / "b.json", json.dumps([{"id": "2", "content": "Two."}]))
    ad = get_adapter(d)
    assert isinstance(ad, JSONAdapter) and not ad.single_file
    es = ad.load()
    assert {e.path for e in es} == {"a.jsonl", "sub/b.json"}
    ad.write([*es, MemoryEntry("3", "Three.")], tmp_path / "o")
    assert {e.content for e in JSONAdapter(tmp_path / "o").load()} == {"One.", "Two.", "Three."}


def test_sqlite_roundtrip(tmp_path):
    db = create_sqlite_store(tmp_path / "m.sqlite", [MemoryEntry("1", "One.", metadata={"tag": "a"}),
                                                     MemoryEntry("2", "Two.", updated_at="2026-01-01")])
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE other (x TEXT)")
    con.execute("INSERT INTO other VALUES ('keep me')")
    con.commit()
    con.close()
    ad = get_adapter(db)
    assert isinstance(ad, SQLiteAdapter)
    es = ad.load()
    assert [(e.id, e.content) for e in es] == [("1", "One."), ("2", "Two.")]
    assert es[0].metadata["tag"] == "a" and es[1].updated_at == "2026-01-01"
    es = [es[1], MemoryEntry("3", "Three.")]
    out = tmp_path / "o" / "m.sqlite"
    ad.write(es, out)
    again = SQLiteAdapter(out).load()
    assert [(e.id, e.content) for e in again] == [("2", "Two."), ("3", "Three.")]
    con = sqlite3.connect(out)
    assert con.execute("SELECT x FROM other").fetchall() == [("keep me",)]
    con.close()


def test_sqlite_custom_columns(tmp_path):
    db = tmp_path / "custom.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT, author TEXT)")
    con.execute("INSERT INTO notes (text, author) VALUES ('Hello.', 'me')")
    con.commit()
    con.close()
    es = get_adapter(db).load()
    assert es[0].content == "Hello." and es[0].metadata["_extra_columns"] == {"author": "me"}
    get_adapter(db).write(es, tmp_path / "o.db")
    con = sqlite3.connect(tmp_path / "o.db")
    assert con.execute("SELECT text, author FROM notes").fetchall() == [("Hello.", "me")]
    con.close()


def test_registry_and_errors(tmp_path):
    assert {"markdown", "json", "sqlite", "stancelog"} <= set(available_adapters())

    class Upper(MemoryAdapter):
        name = "upper"
        single_file = True

        @classmethod
        def detect(cls, path):
            return path.suffix == ".upper"

        def load(self):
            return [MemoryEntry(str(i), ln) for i, ln in enumerate(self.path.read_text().splitlines())]

        def write(self, entries, dest):
            dest.write_text("\n".join(e.content.upper() for e in entries))
            return dest

    register_adapter("upper", Upper)
    f = write(tmp_path / "x.upper", "a\nb")
    assert isinstance(get_adapter(f), Upper)
    assert isinstance(get_adapter(f, format="upper"), Upper)
    with pytest.raises(AdapterError):
        get_adapter(tmp_path / "missing")
    with pytest.raises(AdapterError):
        get_adapter(f, format="nope")


def test_stancelog_adapter_optional(tmp_path):
    from agentdream.adapters.stancelog import available

    if available():  # pragma: no cover - depends on environment
        pytest.skip("stancelog installed; covered by its own integration")
    with pytest.raises(AdapterError, match="stancelog"):
        get_adapter(f"stancelog:{tmp_path}/log")
