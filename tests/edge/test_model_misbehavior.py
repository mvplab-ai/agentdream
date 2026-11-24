"""Model misbehaviour via fake backends: junk, empty, bad JSON/schema, unknown ids, fake citations,
timeouts, HTTP 429 with Retry-After (bounded retries, no real sleeping), HTTP 500, claude-cli failures."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request

import pytest

from agentdream import dream
from agentdream.backends import AnthropicBackend, BackendError, ClaudeCLIBackend, OpenAIBackend
from agentdream.engine import llm as llm_engine

from edgekit import FakeHTTPResponse, FakeLLM, http_error, workspace


@pytest.fixture
def sleeps(monkeypatch):
    """Record retry back-off waits instead of sleeping."""
    waits: list[float] = []
    monkeypatch.setattr(llm_engine, "retry_sleep", waits.append)
    return waits


def _fallback_warned(job, phase=None):
    return [w for w in job.warnings if "fell back to deterministic" in w and (phase is None or w.startswith(phase))]


def _has_det_correction(job):
    return any(o.type == "correct" and o.origin == "deterministic" for o in job.operations)


@pytest.mark.parametrize("output", [
    "I'm sorry, I can't help with that.", "", "   \n ", "{\"operations\": [{\"type\": \"merge\", \"targets\": [}",
    "```json\nnot json\n```", "[" * 50_000, "null", "42", "\"just a string\"",
])
def test_junk_or_empty_output_falls_back(tmp_path, output, sleeps):
    """Prose, empty, whitespace, broken / deeply nested JSON or JSON scalars -> deterministic fallback + warning."""
    mem, ses = workspace(tmp_path)
    job = dream(mem, [ses], backend=FakeLLM(lambda phase, prompt: output))
    assert job.status == "completed", job.error
    assert len(_fallback_warned(job)) == 3
    assert _has_det_correction(job)
    assert sleeps == []  # parse failures are not retried


@pytest.mark.parametrize("payload", [
    {"result": "ok"}, {"operations": "none"}, {"operations": {"type": "merge"}}, [1, 2, 3], ["merge", None],
])
def test_wrong_schema_falls_back(tmp_path, payload):
    """JSON without a usable operations list (wrong key, wrong type, list of non-objects) -> fallback + warning."""
    mem, ses = workspace(tmp_path)
    job = dream(mem, [ses], backend=FakeLLM(lambda phase, prompt: payload))
    assert job.status == "completed"
    assert len(_fallback_warned(job)) == 3


def test_wrong_field_types_inside_operations(tmp_path):
    """Ops with int targets, string evidence, word confidence, dict content: bad parts dropped with warnings."""
    mem, ses = workspace(tmp_path)

    def respond(phase, prompt):
        if phase != "consolidate":
            return {"operations": []}
        return {"operations": [
            {"type": "merge", "targets": 5},
            {"type": "merge", "targets": {"a": 1}},
            {"type": "replace", "targets": ["MEMORY.md#4", "MEMORY.md#5"], "new_content": {"x": 1},
             "confidence": "high", "evidence": "see logs"},
            "not an object",
            {"type": "merge", "targets": ["MEMORY.md#2", "MEMORY.md#3"], "new_content": "Use pytest for tests.",
             "confidence": None, "evidence": {"citation": "nope", "quote": "x"}},
        ]}

    job = dream(mem, [ses], backend=FakeLLM(respond))
    assert job.status == "completed", job.error
    llm_ops = [o for o in job.operations if o.origin.startswith("fake")]
    assert {o.type for o in llm_ops} == {"replace", "merge"}
    assert any("non-object" in w for w in job.warnings)
    assert any("unknown entry id" in w and "5" in w for w in job.warnings)


def test_unknown_ids_dropped_valid_ops_kept(tmp_path):
    """Ops naming ids that don't exist are dropped with a warning; valid ops in the same reply are applied."""
    mem, ses = workspace(tmp_path)

    def respond(phase, prompt):
        if phase != "consolidate":
            return {"operations": []}
        return {"operations": [
            {"type": "merge", "targets": ["MEMORY.md#99", "MEMORY.md#2"], "new_content": "x"},
            {"type": "merge", "targets": ["MEMORY.md#2", "MEMORY.md#3"], "new_content": "Use pytest for tests."},
        ]}

    job = dream(mem, [ses], backend=FakeLLM(respond))
    assert any("unknown entry id" in w and "MEMORY.md#99" in w for w in job.warnings)
    assert [o.targets for o in job.operations if o.type == "merge"] == [["MEMORY.md#2", "MEMORY.md#3"]]
    assert not _fallback_warned(job)


def test_hallucinated_citations(tmp_path):
    """A correction citing a non-existent file:line is downgraded to a flag; an insight citing one is dropped."""
    mem, ses = workspace(tmp_path)

    def respond(phase, prompt):
        if phase == "verify":
            return {"operations": [{"type": "correct", "targets": ["MEMORY.md#1"],
                                    "new_content": "The PokéCenter is east of the gym.",
                                    "evidence": [{"citation": "sessions/9.jsonl:42", "quote": "east!"}]}]}
        if phase == "insights":
            return {"operations": [{"type": "insight", "new_content": "Agent loves llamas.",
                                    "evidence": [{"citation": "made/up.jsonl:1", "quote": "llamas"}]}]}
        return {"operations": []}

    job = dream(mem, [ses], backend=FakeLLM(respond))
    llm_ops = [o for o in job.operations if o.origin.startswith("fake")]
    assert [(o.type, o.kind, o.changes_store) for o in llm_ops] == [("contradiction", "evidence", False)]
    assert any("unverifiable evidence" in w for w in job.warnings)
    assert any("insight" in w and "without verifiable evidence" in w for w in job.warnings)
    assert "east of the gym" not in " ".join(e for o in job.operations for e in o.after)


def test_timeout_is_retried_with_backoff_then_falls_back(tmp_path, sleeps):
    """TimeoutError from the backend: retried llm_retries times with exponential back-off, then fallback."""
    mem, ses = workspace(tmp_path)
    fake = FakeLLM(lambda phase, prompt: TimeoutError("read timed out"))
    t0 = time.monotonic()
    job = dream(mem, [ses], backend=fake, llm_retries=2, llm_retry_backoff=1.5)
    assert time.monotonic() - t0 < 5
    assert len(fake.calls) == 3 * 3  # 3 phases x (1 + 2 retries)
    assert sleeps == [1.5, 3.0] * 3
    assert len(_fallback_warned(job)) == 3 and _has_det_correction(job)
    assert any("retry 1/2" in w for w in job.warnings)


def test_retries_disabled(tmp_path, sleeps):
    """llm_retries=0: one attempt per call, no sleeping."""
    mem, ses = workspace(tmp_path)
    fake = FakeLLM(lambda phase, prompt: TimeoutError("slow"))
    dream(mem, [ses], backend=fake, llm_retries=0)
    assert len(fake.calls) == 3 and sleeps == []


def test_transient_error_then_success_uses_model_output(tmp_path, sleeps):
    """One timeout then a valid reply: the model's operations are used, no fallback."""
    mem, ses = workspace(tmp_path)
    state = {"n": 0}

    def respond(phase, prompt):
        state["n"] += 1
        if state["n"] == 1:
            return TimeoutError("blip")
        if phase == "consolidate":
            return {"operations": [{"type": "merge", "targets": ["MEMORY.md#2", "MEMORY.md#3"],
                                    "new_content": "Use pytest for tests."}]}
        return {"operations": []}

    job = dream(mem, [ses], backend=FakeLLM(respond))
    assert sleeps == [2.0]
    assert not _fallback_warned(job, "consolidate")
    assert any(o.type == "merge" and o.origin.startswith("fake") for o in job.operations)


def _openai_urlopen(responses, seen):
    def fake(req, timeout=None):
        seen.append(req)
        r = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(r, Exception):
            raise r
        return FakeHTTPResponse(r)
    return fake


def test_http_429_retry_after_honoured_then_fallback(tmp_path, monkeypatch, sleeps):
    """OpenAI-compatible HTTP 429 with Retry-After: 7 -> waits 7s (fake) per retry, bounded, then fallback."""
    mem, ses = workspace(tmp_path)
    url = "http://localhost:11434/v1/chat/completions"
    seen: list = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        _openai_urlopen([http_error(url, 429, "slow down", {"Retry-After": "7"})], seen))
    job = dream(mem, [ses], backend="openai", backend_options={"base_url": "http://localhost:11434/v1"},
                llm_retries=2)
    assert len(seen) == 9
    assert sleeps == [7.0, 7.0] * 3
    assert len(_fallback_warned(job)) == 3
    assert any("HTTP 429" in w for w in job.warnings)


def test_http_429_retry_after_capped(tmp_path, monkeypatch, sleeps):
    """A huge Retry-After (3600s) is capped at llm_retry_max_wait."""
    mem, ses = workspace(tmp_path)
    url = "http://localhost:1/v1/chat/completions"
    monkeypatch.setattr(urllib.request, "urlopen",
                        _openai_urlopen([http_error(url, 429, "", {"Retry-After": "3600"})], []))
    dream(mem, [ses], backend="openai", backend_options={"base_url": "http://localhost:1/v1"}, llm_retries=1,
          llm_retry_max_wait=12)
    assert sleeps == [12.0] * 3


def test_http_429_then_success(tmp_path, monkeypatch, sleeps):
    """429 once, then a valid completion: the model's answer is used."""
    mem, ses = workspace(tmp_path)
    url = "http://localhost:1/v1/chat/completions"
    ok = {"choices": [{"message": {"content": json.dumps({"operations": []})}}]}
    monkeypatch.setattr(urllib.request, "urlopen",
                        _openai_urlopen([http_error(url, 429, "", {"Retry-After": "1"}), ok], []))
    job = dream(mem, [ses], backend="openai", backend_options={"base_url": "http://localhost:1/v1"})
    assert sleeps == [1.0]
    assert not _fallback_warned(job)


def test_retry_after_http_date(monkeypatch):
    """Retry-After given as an HTTP date is converted to seconds (and never negative)."""
    url = "http://x/v1/chat/completions"
    monkeypatch.setattr(urllib.request, "urlopen", _openai_urlopen(
        [http_error(url, 429, "", {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})], []))
    with pytest.raises(BackendError) as ei:
        OpenAIBackend(base_url="http://x/v1").complete("s", "p")
    assert ei.value.status == 429 and ei.value.retryable and ei.value.retry_after == 0.0


@pytest.mark.parametrize("backend_cls,kwargs,url", [
    (AnthropicBackend, {"api_key": "test-key"}, "https://api.anthropic.com/v1/messages"),
    (OpenAIBackend, {"api_key": "test-key"}, "https://api.openai.com/v1/chat/completions"),
])
def test_http_500_retried_then_fallback(tmp_path, monkeypatch, sleeps, backend_cls, kwargs, url):
    """HTTP 500 from the Anthropic / OpenAI API: retried with back-off, then deterministic fallback with warning."""
    mem, ses = workspace(tmp_path)
    seen: list = []
    monkeypatch.setattr(urllib.request, "urlopen", _openai_urlopen([http_error(url, 500, "internal error")], seen))
    job = dream(mem, [ses], backend=backend_cls(**kwargs), llm_retries=1)
    assert len(seen) == 6 and sleeps == [2.0] * 3
    assert len(_fallback_warned(job)) == 3 and any("HTTP 500" in w for w in job.warnings)
    assert _has_det_correction(job)


def test_http_400_is_not_retried(tmp_path, monkeypatch, sleeps):
    """A client error (HTTP 400) fails fast: no retries, straight to fallback."""
    mem, ses = workspace(tmp_path)
    url = "https://api.anthropic.com/v1/messages"
    seen: list = []
    monkeypatch.setattr(urllib.request, "urlopen", _openai_urlopen([http_error(url, 400, "bad request")], seen))
    job = dream(mem, [ses], backend=AnthropicBackend(api_key="k"))
    assert len(seen) == 3 and sleeps == []
    assert len(_fallback_warned(job)) == 3


def test_urlopen_socket_timeout_is_retryable(monkeypatch):
    """A socket timeout inside urlopen surfaces as a retryable BackendError."""
    def slow(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", slow)
    with pytest.raises(BackendError) as ei:
        OpenAIBackend(base_url="http://x/v1").complete("s", "p")
    assert ei.value.retryable and "timed out" in str(ei.value)


def _fake_cli(monkeypatch, returncode=0, stdout="", stderr=""):
    calls = []

    def run(cmd, input=None, capture_output=None, text=None, timeout=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("shutil.which", lambda exe: "/usr/bin/claude")
    return calls


def test_claude_cli_nonzero_exit_falls_back(tmp_path, monkeypatch, sleeps):
    """claude-cli exiting 1 -> BackendError (not retried) -> fallback, warning quotes the exit code."""
    mem, ses = workspace(tmp_path)
    calls = _fake_cli(monkeypatch, returncode=1, stderr="Error: not logged in")
    job = dream(mem, [ses], backend=ClaudeCLIBackend())
    assert len(calls) == 3 and sleeps == []
    assert any("exited 1" in w and "not logged in" in w for w in job.warnings)
    assert _has_det_correction(job)


def test_claude_cli_non_json_stdout(tmp_path, monkeypatch):
    """claude-cli printing plain prose (not the JSON envelope) -> treated as model text -> fallback."""
    mem, ses = workspace(tmp_path)
    _fake_cli(monkeypatch, stdout="Sure! I'd be happy to help consolidate your memory.")
    job = dream(mem, [ses], backend=ClaudeCLIBackend())
    assert len(_fallback_warned(job)) == 3


def test_claude_cli_error_envelope(tmp_path, monkeypatch):
    """claude-cli JSON envelope with is_error=true -> fallback with the reported error."""
    mem, ses = workspace(tmp_path)
    _fake_cli(monkeypatch, stdout=json.dumps({"type": "result", "is_error": True, "result": "rate limited"}))
    job = dream(mem, [ses], backend=ClaudeCLIBackend())
    assert any("rate limited" in w for w in job.warnings)


def test_claude_cli_timeout_not_retried(tmp_path, monkeypatch, sleeps):
    """claude-cli subprocess timeout -> BackendError, not retried (each attempt can take minutes)."""
    mem, ses = workspace(tmp_path)

    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("shutil.which", lambda exe: "/usr/bin/claude")
    job = dream(mem, [ses], backend=ClaudeCLIBackend(timeout=1))
    assert sleeps == [] and any("timed out" in w for w in job.warnings)


def test_backend_returning_non_string(tmp_path):
    """A backend returning None or bytes instead of str is handled as unusable output (fallback)."""
    mem, ses = workspace(tmp_path)

    class Weird(FakeLLM):
        def complete(self, system, prompt, json_schema=None):
            return None if "CONSOLIDATE" in prompt else b'{"operations": []}'

    job = dream(mem, [ses], backend=Weird())
    assert job.status == "completed" and len(_fallback_warned(job)) >= 1
