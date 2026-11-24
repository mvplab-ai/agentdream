"""The stancelog adapter against the real stancelog library (skipped when not installed)."""

from pathlib import Path

import pytest

stancelog = pytest.importorskip("stancelog")

from agentdream import accept, dream, get_adapter  # noqa: E402
from agentdream.adapters import AdapterError  # noqa: E402
from agentdream.adapters.stancelog import StancelogAdapter  # noqa: E402

from conftest import hash_tree, ops_of, write  # noqa: E402


def make_db(path, rows):
    """rows: (text, valid_from) remembered in order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with stancelog.Memory(str(path)) as m:
        for text, vf in rows:
            m.remember(text, valid_from=vf)
    return path


def load(path):
    return StancelogAdapter(f"stancelog:{path}").load()


def session(tmp_path):
    return write(tmp_path / "s" / "chat.txt", "User: hello there\n")


BASIC = [
    ("Don't do A", "2026-01-10"),
    ("Maybe A is better", "2026-04-02"),
    ("The CI provider is GitHub Actions.", "2026-02-01"),
]


def test_get_adapter_by_prefix(tmp_path):
    db = make_db(tmp_path / "mem" / "beliefs.db", BASIC)
    ad = get_adapter(f"stancelog:{db}")
    assert isinstance(ad, StancelogAdapter)
    assert ad.path == db and ad.single_file


def test_load_one_entry_per_current_belief(tmp_path):
    db = make_db(tmp_path / "mem" / "beliefs.db", BASIC)
    entries = load(db)
    by_key = {e.key: e for e in entries}
    assert set(by_key) == {"a", "ci provider"}  # "don't do A" was superseded inside stancelog

    a = by_key["a"]
    assert a.content == "Maybe A is better"
    assert a.source == "stancelog"
    md = a.metadata
    assert md["subject"] == "a" and md["stance"] == "positive" and md["polarity"] == 1
    assert 0 < md["confidence"] <= 1
    assert md["valid_from"].startswith("2026-04-02") and a.updated_at == md["valid_from"]
    assert a.created_at.startswith("2026-01-10")  # first stance on the subject
    (old,) = md["stancelog_history"]
    assert old["content"] == "Don't do A" and old["status"] == "superseded"
    assert old["polarity"] == -1 and old["valid_to"].startswith("2026-04-02")

    ci = by_key["ci provider"]
    assert ci.content == "The CI provider is GitHub Actions."  # full text, not the bare value
    assert ci.metadata["value"] == "GitHub Actions" and ci.metadata["stance"] == "neutral"
    assert "stancelog_history" not in ci.metadata
    # oldest recorded first
    assert [e.key for e in entries] == ["a", "ci provider"]


def test_load_missing_db(tmp_path):
    with pytest.raises(AdapterError, match="not found"):
        load(tmp_path / "nope.db")


def test_input_db_unchanged_by_load_and_dream(tmp_path):
    db = make_db(tmp_path / "mem" / "beliefs.db", BASIC)
    before = hash_tree(db.parent)
    load(db)
    job = dream(f"stancelog:{db}", [session(tmp_path)])
    assert job.status == "completed", job.error
    assert hash_tree(db.parent) == before  # no new WAL/lock/journal files, same bytes
    assert job.input_hash


def test_dream_a_bad_then_a_good_linked_subject(tmp_path):
    """stancelog already linked the two stances: the dream must keep only the current one."""
    db = make_db(tmp_path / "mem" / "beliefs.db", BASIC)
    job = dream(f"stancelog:{db}", [session(tmp_path)])
    assert job.status == "completed", job.error
    out = load(job.store_path)
    contents = [e.content for e in out]
    assert "Maybe A is better" in contents
    assert not any(c.startswith("Don't do A") for c in contents)
    (a,) = [e for e in out if e.key == "a"]
    assert a.metadata["polarity"] == 1
    with stancelog.Memory(job.store_path) as m:  # a real stancelog db
        assert m.current("A").stance == "positive"


def test_dream_a_bad_then_a_good_unlinked_subjects(tmp_path):
    """stancelog kept two subjects; agentdream resolves the temporal contradiction."""
    db = make_db(tmp_path / "mem" / "beliefs.db", [
        ("Don't use SQLAlchemy for the data layer.", "2026-01-10"),
        ("Maybe SQLAlchemy is better after all.", "2026-04-02"),
        ("The CI provider is GitHub Actions.", "2026-02-01"),
    ])
    assert len(load(db)) == 3
    job = dream(f"stancelog:{db}", [session(tmp_path)])
    assert job.status == "completed", job.error
    (c,) = ops_of(job, "contradiction")
    assert c.kind == "temporal"
    out = load(job.store_path)
    sql = [e for e in out if "SQLAlchemy" in e.content]
    assert len(sql) == 1
    current = sql[0]
    assert current.content.startswith("Use SQLAlchemy for the data layer")
    assert "Don't use SQLAlchemy" in current.content  # history kept in the text
    assert current.metadata["polarity"] == 1  # newest stance wins, not re-extracted as neutral
    assert current.metadata["valid_from"].startswith("2026-04-02")
    assert "The CI provider is GitHub Actions." in [e.content for e in out]


def test_write_round_trip(tmp_path):
    db = make_db(tmp_path / "mem" / "beliefs.db", BASIC + [("Always run pytest before committing.", "2026-03-01")])
    ad = StancelogAdapter(f"stancelog:{db}")
    entries = ad.load()
    dest = tmp_path / "new" / "beliefs.db"
    assert ad.write(entries, f"stancelog:{dest}") == dest
    assert sorted(p.name for p in dest.parent.iterdir()) == ["beliefs.db"]  # no journal/lock left behind
    again = load(dest)

    def view(es):
        return [(e.key, e.content, e.metadata["polarity"], e.metadata["confidence"],
                 e.metadata["valid_from"], e.metadata["value"]) for e in es]

    assert view(again) == view(entries)


def test_write_refuses_input_and_existing_dest(tmp_path):
    db = make_db(tmp_path / "mem" / "beliefs.db", BASIC)
    ad = StancelogAdapter(f"stancelog:{db}")
    entries = ad.load()
    before = hash_tree(db.parent)
    with pytest.raises(AdapterError, match="refusing"):
        ad.write(entries, db)
    other = make_db(tmp_path / "other.db", [("x is fine", "2026-01-01")])
    with pytest.raises(AdapterError, match="exists"):
        ad.write(entries, other)
    assert hash_tree(db.parent) == before


def test_write_new_entry_without_subject(tmp_path):
    db = make_db(tmp_path / "mem" / "beliefs.db", BASIC)
    ad = StancelogAdapter(f"stancelog:{db}")
    from agentdream.models import MemoryEntry

    entries = ad.load() + [MemoryEntry(id="ins-1", content="Prefer small, focused pull requests.")]
    dest = ad.write(entries, tmp_path / "new.db")
    out = load(dest)
    assert len(out) == 3
    (new,) = [e for e in out if e.content == "Prefer small, focused pull requests."]
    assert new.metadata["source"] == "agentdream"
    assert new.metadata["stancelog_metadata"]["agentdream_id"] == "ins-1"


def test_accept_swaps_db_and_drops_old_journal(tmp_path):
    db = make_db(tmp_path / "mem" / "beliefs.db", [
        ("Don't use SQLAlchemy for the data layer.", "2026-01-10"),
        ("Maybe SQLAlchemy is better after all.", "2026-04-02"),
    ])
    journal = db.with_name(db.name + ".journal.jsonl")
    assert journal.exists()
    job = dream(f"stancelog:{db}", [session(tmp_path)])
    res = accept(job)
    assert res.target == str(db)
    assert not journal.exists()  # would otherwise resurrect the old beliefs on reopen
    backup = Path(res.backup)
    assert backup.exists() and backup.with_name(journal.name).exists()
    with stancelog.Memory(str(db)) as m:  # plain stancelog open, journal enabled
        cur = m.all_current()
        assert m.recovery is None
    assert len(cur) == 1
    assert cur[0].text.startswith("Use SQLAlchemy for the data layer") and cur[0].polarity == 1
