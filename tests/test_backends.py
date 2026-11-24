"""LLM backends with fakes: JSON parsing/repair, validation, fallback, HTTP + subprocess mocks."""

import io
import json
import subprocess

import pytest

from agentdream import LLMBackend, dream, get_backend, register_backend
from agentdream.backends import AnthropicBackend, BackendError, ClaudeCLIBackend, OpenAIBackend
from agentdream.engine.llm import SYSTEM_PROMPT, extract_operations, parse_json_loose

from conftest import final_contents, ops_of, write, write_jsonl


def _setup(tmp_path):
    mem = write(tmp_path / "mem" / "MEMORY.md",
                "- The PokéCenter in Viridian City is north of the gym.\n- Use pytest.\n- Use pytest!\n").parent
    s = tmp_path / "s"
    log = write_jsonl(s / "1.jsonl", [{"role": "assistant", "content": "No PokéCenter north of the gym."},
                                      {"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."}])
    return mem, s, log


class Scripted(LLMBackend):
    """Returns canned responses per phase (detected from the prompt)."""

    name = "scripted"

    def __init__(self, responses, **kw):
        super().__init__(model="fake-1", **kw)
        self.responses = responses
        self.prompts = []

    def complete(self, system, prompt, json_schema=None):
        self.prompts.append((system, prompt, json_schema))
        for phase, resp in self.responses.items():
            if f"Task: {phase.upper()}" in prompt or (phase == "insights" and "NEW INSIGHTS" in prompt):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return '{"operations": []}'


def test_parse_json_loose_repairs():
    assert parse_json_loose('{"operations": []}') == {"operations": []}
    assert parse_json_loose('Sure! Here you go:\n```json\n{"operations": [{"type": "insight",}],}\n```') == \
        {"operations": [{"type": "insight"}]}
    assert parse_json_loose("prefix {'operations': [], 'ok': True} suffix") == {"operations": [], "ok": True}
    assert parse_json_loose('{"operations": [{"type": "merge", "targets": ["a"') == \
        {"operations": [{"type": "merge", "targets": ["a"]}]}  # truncated output closed
    assert extract_operations([{"type": "merge"}]) == [{"type": "merge"}]
    with pytest.raises(ValueError):
        parse_json_loose("no json here at all")
    with pytest.raises(ValueError):
        extract_operations({"foo": 1})


def test_llm_operations_validated_and_applied(tmp_path):
    mem, s, log = _setup(tmp_path)
    cite = f"{log}:2"
    be = Scripted({
        "consolidate": json.dumps({"operations": [
            {"type": "merge", "targets": ["MEMORY.md#2", "MEMORY.md#3"], "new_content": "Use pytest.",
             "summary": "dupes", "confidence": 0.9},
            {"type": "merge", "targets": ["MEMORY.md#2", "ghost-id"], "new_content": "x"},  # unknown id -> dropped
        ]}),
        "verify": "```json\n" + json.dumps({"operations": [
            {"type": "correct", "targets": ["MEMORY.md#1"],
             "new_content": "The PokéCenter in Viridian City is south of the gym.",
             "evidence": [{"citation": cite, "quote": "It is south of the gym."}], "reason": "logs"},
        ]}) + "\n```",
        "insights": json.dumps({"operations": [
            {"type": "insight", "new_content": "Check the town map before walking.",
             "evidence": [{"citation": "made-up.jsonl:9", "quote": "x"}]},  # invalid evidence -> dropped
            {"type": "insight", "new_content": "Searching north of the gym was a recurring dead end.",
             "evidence": [{"citation": "1.jsonl:1", "quote": "No PokéCenter north of the gym."}]},
        ]}),
    })
    job = dream(mem, [s], backend=be, instructions="focus on navigation")
    assert job.status == "completed", job.error
    assert len(be.prompts) == 3
    system, prompt, schema = be.prompts[1]
    assert system == SYSTEM_PROMPT and schema["required"] == ["operations"]
    assert f"[{cite}]" in prompt and "focus on navigation" in prompt
    (m,) = ops_of(job, "merge")
    assert m.origin == "scripted (fake-1)" and m.remove == ["MEMORY.md#3"]
    (c,) = ops_of(job, "correct")
    assert c.evidence[0].citation == cite and c.evidence[0].line_no == 2
    ins = ops_of(job, "insight")
    assert [i.after[0] for i in ins] == ["Searching north of the gym was a recurring dead end."]
    assert any("unknown entry id" in w for w in job.warnings)
    assert any("without verifiable evidence" in w for w in job.warnings)
    assert "The PokéCenter in Viridian City is south of the gym." in final_contents(job)
    assert job.report["backend"] == "scripted (fake-1)"


def test_correction_without_evidence_downgraded(tmp_path):
    mem, s, _ = _setup(tmp_path)
    be = Scripted({"verify": json.dumps({"operations": [
        {"type": "correct", "targets": ["MEMORY.md#1"], "new_content": "It is on the moon.", "evidence": []}]})})
    job = dream(mem, [s], backend=be)
    assert ops_of(job, "correct") == []
    (flag,) = [o for o in ops_of(job, "contradiction") if o.kind == "evidence"]
    assert not flag.changes_store
    assert "It is on the moon." not in final_contents(job)


def test_fallback_to_deterministic_on_garbage(tmp_path):
    mem, s, _ = _setup(tmp_path)
    be = Scripted({"consolidate": "I think everything looks fine!", "verify": BackendError("HTTP 500"),
                   "insights": "{not: valid json"})
    job = dream(mem, [s], backend=be)
    assert job.status == "completed"
    assert sum("fell back to deterministic" in w for w in job.warnings) == 3
    assert ops_of(job, "merge") and ops_of(job, "correct")  # heuristics did the work
    assert all(o.origin == "deterministic" for o in job.operations)
    assert "fell back" in job.report_md


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_anthropic_backend_http(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return FakeResp(json.dumps({"content": [{"type": "text", "text": '{"operations": []}'}],
                                    "stop_reason": "end_turn"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    be = get_backend("anthropic", model="claude-test", api_key="k")
    assert be.complete("sys", "hello") == '{"operations": []}'
    req = calls[0]
    assert req.full_url == "https://api.anthropic.com/v1/messages"
    assert req.headers["X-api-key"] == "k" and req.headers["Anthropic-version"] == "2023-06-01"
    body = json.loads(req.data)
    assert body["model"] == "claude-test" and body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    with pytest.raises(BackendError):
        AnthropicBackend(api_key=None).complete("s", "p")


def test_openai_compatible_backend_http(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return FakeResp(json.dumps({"choices": [{"message": {"content": '{"operations": []}'}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    be = get_backend("ollama", model="llama3.1")
    assert isinstance(be, OpenAIBackend)
    assert be.complete("sys", "p", {"type": "object"}) == '{"operations": []}'
    req = calls[0]
    assert req.full_url == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in req.headers  # local server, no key needed
    body = json.loads(req.data)
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["response_format"] == {"type": "json_object"}
    be2 = OpenAIBackend(api_key="sk-x", base_url="https://example.test/v1/")
    be2.complete("s", "p")
    assert calls[1].full_url == "https://example.test/v1/chat/completions"
    assert calls[1].headers["Authorization"] == "Bearer sk-x"
    with pytest.raises(BackendError):
        OpenAIBackend().complete("s", "p")  # api.openai.com without a key


def test_http_errors_surface_as_backend_error(monkeypatch):
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(BackendError, match="could not reach"):
        OpenAIBackend(base_url="http://localhost:1").complete("s", "p")


def test_claude_cli_backend_subprocess(monkeypatch):
    seen = {}

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        seen["cmd"], seen["input"] = cmd, input
        out = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                          "result": '```json\n{"operations": []}\n```'})
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr("agentdream.backends.builtin.shutil.which", lambda x: "/usr/bin/claude")
    monkeypatch.setattr("agentdream.backends.builtin.subprocess.run", fake_run)
    be = get_backend("claude-cli", model="sonnet")
    assert isinstance(be, ClaudeCLIBackend)
    out = be.complete("SYSTEM", "PROMPT")
    assert parse_json_loose(out) == {"operations": []}
    assert seen["cmd"] == ["claude", "-p", "--output-format", "json", "--model", "sonnet"]
    assert "SYSTEM" in seen["input"] and "PROMPT" in seen["input"]

    monkeypatch.setattr("agentdream.backends.builtin.subprocess.run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not logged in"))
    with pytest.raises(BackendError, match="not logged in"):
        be.complete("s", "p")
    monkeypatch.setattr("agentdream.backends.builtin.shutil.which", lambda x: None)
    with pytest.raises(BackendError, match="not found"):
        ClaudeCLIBackend(executable="definitely-not-claude").complete("s", "p")


def test_claude_cli_missing_falls_back(tmp_path, monkeypatch):
    mem, s, _ = _setup(tmp_path)
    monkeypatch.setattr("agentdream.backends.builtin.shutil.which", lambda x: None)
    job = dream(mem, [s], backend="claude-cli")
    assert job.status == "completed" and ops_of(job, "correct")
    assert any("not found" in w for w in job.warnings)


def test_registry():
    class Mine(LLMBackend):
        name = "mine"

        def complete(self, system, prompt, json_schema=None):
            return "{}"

    register_backend("mine", Mine)
    assert isinstance(get_backend("mine"), Mine)
    assert get_backend("deterministic").is_llm is False
    with pytest.raises(BackendError):
        get_backend("nope")
