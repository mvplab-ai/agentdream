"""Live verification against REAL model backends. Never runs in CI.

Enable with environment variables:
  AGENTDREAM_LIVE=1                      -> `claude` CLI backend (uses your Claude Code login)
  AGENTDREAM_OLLAMA_MODEL=qwen2.5:1.5b   -> OpenAI-compatible backend against a local Ollama
  (AGENTDREAM_OLLAMA_URL, default http://localhost:11434/v1)
  ANTHROPIC_API_KEY=...                  -> Anthropic Messages API backend

Every prompt actually sent to the model is captured by a recording wrapper so the
tests can assert that planted secrets never leave the machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import pytest

from agentdream import dream
from agentdream.backends import get_backend
from agentdream.backends.base import LLMBackend

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _ollama_ok() -> bool:
    if not os.environ.get("AGENTDREAM_OLLAMA_MODEL"):
        return False
    url = os.environ.get("AGENTDREAM_OLLAMA_URL", "http://localhost:11434/v1").rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=3):
            return True
    except Exception:
        return False


BACKENDS = []
if os.environ.get("AGENTDREAM_LIVE") == "1" and shutil.which("claude"):
    BACKENDS.append(pytest.param(("claude-cli", None, None, True), id="claude-cli"))
if _ollama_ok():
    BACKENDS.append(pytest.param(("openai", os.environ["AGENTDREAM_OLLAMA_MODEL"],
                                  os.environ.get("AGENTDREAM_OLLAMA_URL", "http://localhost:11434/v1"), False),
                                 id="ollama"))
if os.environ.get("ANTHROPIC_API_KEY"):
    BACKENDS.append(pytest.param(("anthropic", None, None, True), id="anthropic"))
if not BACKENDS:
    BACKENDS.append(pytest.param(None, id="no-live-backend",
                                 marks=pytest.mark.skip(reason="no live backend configured")))


class Recorder(LLMBackend):
    """Delegates to a real backend and keeps every prompt it sent."""

    def __init__(self, inner: LLMBackend):
        super().__init__(model=inner.model)
        self.inner = inner
        self.name = inner.name
        self.sent: list[str] = []
        self.calls = 0

    def complete(self, system, prompt, json_schema=None):
        self.calls += 1
        self.sent.append(system + "\n" + prompt)
        return self.inner.complete(system, prompt, json_schema)

    def describe(self):
        return self.inner.describe()


def _backend(spec):
    name, model, base_url, _strong = spec
    kw = {"base_url": base_url} if base_url else {}
    return Recorder(get_backend(name, model=model, **kw))


def _hash_tree(p: Path) -> dict:
    return {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(p.rglob("*")) if f.is_file()}


def _contents(job) -> str:
    from agentdream import get_adapter
    return "\n".join(e.content for e in get_adapter(job.store_path).load())


def _model_ops(job, rec) -> list:
    return [o for o in job.report.get("operations", []) if str(o.get("origin", "")).startswith(rec.name)]


@pytest.fixture(params=BACKENDS)
def spec(request):
    return request.param


def test_live_full_examples(spec, tmp_path):
    """All bundled examples: memory folder + 5 sessions in 3 transcript formats."""
    mem = tmp_path / "memory"
    shutil.copytree(EXAMPLES / "memory", mem)
    before = _hash_tree(mem)
    rec = _backend(spec)
    job = dream(mem, [EXAMPLES / "sessions"], backend=rec, jobs_dir=tmp_path / "jobs")
    assert job.status == "completed", job.error
    assert rec.calls >= 1
    assert _hash_tree(mem) == before
    assert _model_ops(job, rec), job.report.get("warnings")
    if spec[3]:  # strong models must get the Pokémon correction right
        assert "south of the gym" in _contents(job).lower()
    (tmp_path / "report.md").write_text((job.out_dir / "report.md").read_text())


@pytest.mark.parametrize("session", sorted(p.name for p in (EXAMPLES / "sessions").iterdir()))
def test_live_each_transcript_format(spec, session, tmp_path):
    """Each bundled session on its own (generic JSONL, Claude Code JSONL, OpenAI JSON)."""
    mem = tmp_path / "memory"
    shutil.copytree(EXAMPLES / "memory", mem)
    before = _hash_tree(mem)
    rec = _backend(spec)
    job = dream(mem, [EXAMPLES / "sessions" / session], backend=rec, jobs_dir=tmp_path / "jobs")
    assert job.status == "completed", job.error
    assert rec.calls >= 1
    assert _hash_tree(mem) == before


def test_live_temporal_a_case(spec, tmp_path):
    """'Don't use A' early, 'use A' later -> latest stance wins, history kept."""
    mem = tmp_path / "m"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("- Don't use Poetry for dependency management; stick with pip.\n")
    s = tmp_path / "s"
    s.mkdir()
    (s / "1.jsonl").write_text("\n".join(json.dumps(m) for m in [
        {"role": "user", "content": "Actually, maybe Poetry is better. Let's use Poetry for dependency management from now on.",
         "timestamp": "2026-11-20T10:00:00Z"},
        {"role": "assistant", "content": "Switched the project to Poetry.", "timestamp": "2026-11-20T10:01:00Z"},
    ]) + "\n")
    rec = _backend(spec)
    job = dream(mem, [s], backend=rec, jobs_dir=tmp_path / "jobs")
    assert job.status == "completed", job.error
    if spec[3]:
        text = _contents(job).lower()
        assert "poetry" in text
        c = job.report["counts"]
        assert c.get("contradictions", 0) + c.get("replaced", 0) + c.get("corrected", 0) >= 1, c


def test_live_secrets_never_sent(spec, tmp_path):
    """Planted secrets in memory and transcripts must never reach the real model."""
    secrets = ["sk-ant-api03-" + "Q" * 40 + "x7Yz", "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
               "AKIAIOSFODNN7EXAMPLE", "hunter2-SuperSecret-Pass!"]
    mem = tmp_path / "m"
    mem.mkdir()
    (mem / "MEMORY.md").write_text(f"- Deploy key is {secrets[0]}\n- The db password=\"{secrets[3]}\"\n")
    s = tmp_path / "s"
    s.mkdir()
    (s / "1.jsonl").write_text("\n".join(json.dumps(m) for m in [
        {"role": "user", "content": f"use token {secrets[1]} for github"},
        {"role": "assistant", "content": json.dumps({"aws": {"key": secrets[2]}})},
    ]) + "\n")
    rec = _backend(spec)
    job = dream(mem, [s], backend=rec, jobs_dir=tmp_path / "jobs")
    assert job.status == "completed", job.error
    blob = "\n".join(rec.sent)
    for sec in secrets:
        assert sec not in blob
    for f in job.out_dir.rglob("report.*"):
        for sec in secrets:
            assert sec not in f.read_text()


def test_live_chunked_map_reduce(spec, tmp_path):
    """Force tiny chunks so the live model sees the input in several map calls, then reduce."""
    mem = tmp_path / "memory"
    shutil.copytree(EXAMPLES / "memory", mem)
    rec = _backend(spec)
    job = dream(mem, [EXAMPLES / "sessions"], backend=rec, jobs_dir=tmp_path / "jobs",
                max_chunk_tokens=1800, chunk_overlap=50)
    assert job.status == "completed", job.error
    baseline_phases = 3
    assert rec.calls > baseline_phases, f"expected multiple chunked calls, got {rec.calls}"


def test_live_real_claude_code_session(spec, tmp_path):
    """A real Claude Code session file from ~/.claude/projects, if one exists on this machine."""
    from agentdream.loaders import load_transcript
    files = [f for f in Path.home().glob(".claude/projects/*/*.jsonl") if f.stat().st_size < 10_000_000]
    scored = sorted(((len(load_transcript(f).messages), f) for f in files), reverse=True)
    if not scored or scored[0][0] < 4:
        pytest.skip("no substantial Claude Code session files on this machine")
    files = [scored[0][1]]
    mem = tmp_path / "memory"
    shutil.copytree(EXAMPLES / "memory", mem)
    rec = _backend(spec)
    job = dream(mem, [files[0]], backend=rec, jobs_dir=tmp_path / "jobs")
    assert job.status == "completed", job.error
    assert job.report["sessions"], job.report
    assert rec.calls >= 1
