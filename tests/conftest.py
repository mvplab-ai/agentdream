import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own job store so nothing leaks into the repo."""
    home = tmp_path / ".agentdream-home"
    monkeypatch.setenv("AGENTDREAM_HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


def hash_tree(*paths) -> dict:
    """sha256 of every file (path -> digest) under the given paths."""
    out = {}
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for f in files:
            out[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def ops_of(job, type_=None):
    ops = job.operations
    return [o for o in ops if type_ is None or o.type == type_]


def final_contents(job):
    from agentdream import get_adapter

    return [e.content for e in get_adapter(job.store_path).load()]
