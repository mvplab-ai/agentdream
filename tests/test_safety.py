"""Input is never mutated (every adapter), redaction, size limits."""

import json
import sqlite3

import pytest

from agentdream import MemoryEntry, Redactor, accept, dream, redact
from agentdream.adapters import create_sqlite_store

from conftest import hash_tree, ops_of, write, write_jsonl

POKEMON_NOTE = "The PokéCenter in Viridian City is north of the gym."


def _sessions(tmp_path):
    s = tmp_path / "sessions"
    write_jsonl(s / "1.jsonl", [{"role": "assistant", "content": "Walked north of the gym. No PokéCenter here."},
                                {"role": "assistant", "content": "Still no PokéCenter north of the gym."}])
    write_jsonl(s / "2.jsonl", [{"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."},
                                {"role": "user", "content": "my api key is sk-ant-abcdefghijklmnop1234567890"}])
    return s


def _stores(tmp_path):
    md = write(tmp_path / "md" / "MEMORY.md", f"- {POKEMON_NOTE}\n- Use pytest.\n- Use pytest!\n")
    write(tmp_path / "md" / "notes" / "x.md", "---\nname: x\ntype: project\n---\nThe codename is Owl.\n")
    js = tmp_path / "mem.json"
    js.write_text(json.dumps([{"id": "1", "content": POKEMON_NOTE}, {"id": "2", "content": "Use pytest."},
                              {"id": "3", "content": "Use pytest!"}]))
    jl = write_jsonl(tmp_path / "mem.jsonl", [{"id": "1", "text": POKEMON_NOTE}, {"id": "2", "text": "Use pytest."}])
    jdir = tmp_path / "jdir"
    write_jsonl(jdir / "a.jsonl", [{"id": "1", "content": POKEMON_NOTE}])
    write(jdir / "b.json", json.dumps({"codename": "Owl", "tool": "Use pytest."}))
    db = create_sqlite_store(tmp_path / "mem.db", [MemoryEntry("1", POKEMON_NOTE), MemoryEntry("2", "Use pytest."),
                                                   MemoryEntry("3", "Use pytest!")])
    return {"markdown": md.parent, "json": js, "jsonl": jl, "json-dir": jdir, "sqlite": db}


@pytest.mark.parametrize("kind", ["markdown", "json", "jsonl", "json-dir", "sqlite"])
def test_input_never_mutated(tmp_path, kind):
    store = _stores(tmp_path)[kind]
    sessions = _sessions(tmp_path)
    before = hash_tree(store, sessions)
    job = dream(store, [sessions])
    assert job.status == "completed", job.error
    assert ops_of(job, "correct"), "the dream should have produced a correction"
    assert hash_tree(store, sessions) == before
    # partial accept into another location also leaves the input alone
    accept(job, to=tmp_path / f"accepted-{kind}" / store.name)
    assert hash_tree(store, sessions) == before


def test_redaction_patterns():
    text = ("key sk-ant-api03-abcdefghijklmnopqrstuvwxyz and sk-proj-ABCDEFGHIJKLMNOPQRSTUV123 "
            "aws AKIAIOSFODNN7EXAMPLE secret aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY "
            "gh ghp_abcdefghijklmnopqrstuvwxyz0123456789 password: hunter22 token=abc123xyz "
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456 "
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY----- "
            "postgres://user:s3cretpw@db.example.com/x mail me at dev@example.com")
    out = redact(text)
    for secret in ("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUV123", "AKIAIOSFODNN7EXAMPLE",
                   "wJalrXUtnFEMI", "ghp_", "hunter22", "abc123xyz", "MIIEowIBAAKCAQEA", "s3cretpw"):
        assert secret not in out, secret
    assert "dev@example.com" in out  # emails are opt-in
    assert "dev@example.com" not in Redactor(emails=True).redact(text)
    assert "[REDACTED:aws_access_key]" in out and "[REDACTED:private_key]" in out
    assert "password: [REDACTED" in out  # keeps the key name for context
    assert redact("The password is stored in 1Password.") == "The password is stored in 1Password."


def test_secrets_redacted_before_backend_and_in_reports(tmp_path):
    from agentdream.backends import LLMBackend

    seen = []

    class Spy(LLMBackend):
        name = "spy"

        def complete(self, system, prompt, json_schema=None):
            seen.append(prompt)
            return '{"operations": []}'

    mem = write(tmp_path / "m" / "MEMORY.md", "- Deploy token=supersecretvalue123 lives in vault.\n- Use pytest.\n")
    job = dream(mem.parent, [_sessions(tmp_path)], backend=Spy())
    assert seen and all("sk-ant-abcdefghijklmnop" not in p and "supersecretvalue123" not in p for p in seen)
    assert any("[REDACTED:" in p for p in seen)
    assert "sk-ant-abcdefghijklmnop" not in job.report_md
    assert "supersecretvalue123" not in json.dumps(job.report)
    assert job.report["redactions"]
    # untouched entries keep their original text in the new store (secrets are not destroyed)
    assert "Deploy token=supersecretvalue123 lives in vault." in (tmp_path / "m" / "MEMORY.md").read_text()
    from conftest import final_contents

    assert "Deploy token=supersecretvalue123 lives in vault." in final_contents(job)


def test_size_limit_truncate_and_fail(tmp_path):
    mem = write(tmp_path / "m" / "MEMORY.md", "- Note one.\n")
    s = tmp_path / "s"
    for i in range(5):
        write(s / f"{i:02d}.txt", "".join(f"User: message {i}-{k} " + "x" * 50 + "\n" for k in range(40)))
    job = dream(mem.parent, [s], max_sessions=3, max_bytes_per_transcript=1000)
    assert job.status == "completed"
    rep = job.report
    assert len(rep["sessions"]) == 3
    assert [x["id"] for x in rep["sessions"]] == ["02", "03", "04"]  # most recent kept
    assert all(x["messages"] < 40 for x in rep["sessions"])
    assert any("max_sessions" in n for n in rep["notices"])
    assert any("truncated" in n for n in rep["notices"])
    assert "max_sessions" in job.report_md

    failed = dream(mem.parent, [s], max_sessions=3, on_limit="fail")
    assert failed.status == "failed" and "max_sessions" in failed.error

    tok = dream(mem.parent, [s], max_total_tokens=1500)
    assert tok.status == "completed"
    assert any("max_total_tokens" in n for n in tok.report["notices"])
    tok_fail = dream(mem.parent, [s], max_total_tokens=1500, on_limit="fail")
    assert tok_fail.status == "failed" and "max_total_tokens" in tok_fail.error


def test_session_count_bounds(tmp_path):
    mem = write(tmp_path / "m" / "MEMORY.md", "- Note.\n")
    job = dream(mem.parent, [])
    assert job.status == "failed" and "at least 1" in job.error
    ok = dream(mem.parent, [], min_sessions=0)
    assert ok.status == "completed"


def test_dest_equal_to_input_is_refused(tmp_path):
    from agentdream import get_adapter
    from agentdream.adapters import AdapterError

    mem = write(tmp_path / "m" / "MEMORY.md", "- Note.\n").parent
    ad = get_adapter(mem)
    with pytest.raises(AdapterError):
        ad.write(ad.load(), mem)


def test_sqlite_input_opened_read_only(tmp_path):
    db = create_sqlite_store(tmp_path / "m.db", [MemoryEntry("1", "Use pytest.")])
    before = hash_tree(db)
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=DELETE")
    con.close()
    before = hash_tree(db)
    write(tmp_path / "s.txt", "User: hi\n")
    dream(db, [tmp_path / "s.txt"])
    assert hash_tree(db) == before
    assert not (tmp_path / "m.db-journal").exists() and not (tmp_path / "m.db-wal").exists()
