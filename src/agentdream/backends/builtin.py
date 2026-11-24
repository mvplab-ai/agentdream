"""Built-in backends: deterministic, anthropic, openai(-compatible), claude-cli."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from .base import BackendError, LLMBackend, http_post_json


class DeterministicBackend(LLMBackend):
    """No model at all: the pipeline runs rule-based heuristics."""

    name = "deterministic"
    is_llm = False

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        return json.dumps({"operations": []})


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API over urllib (needs ANTHROPIC_API_KEY)."""

    name = "anthropic"
    DEFAULT_MODEL = "claude-sonnet-5-5"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 max_tokens: int = 8192, timeout: float = 300.0, **options: Any):
        super().__init__(model or os.environ.get("AGENTDREAM_ANTHROPIC_MODEL") or self.DEFAULT_MODEL, **options)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        if not self.api_key:
            raise BackendError("ANTHROPIC_API_KEY is not set")
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": [{"role": "user", "content": prompt}]}
        resp = http_post_json(f"{self.base_url}/v1/messages",
                              {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body, self.timeout)
        parts = [b.get("text", "") for b in resp.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]
        if not parts:
            raise BackendError(f"anthropic: no text in response (stop_reason={resp.get('stop_reason')})")
        return "".join(parts)


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible chat completions (OpenAI, Ollama, LM Studio, vLLM, llama.cpp server...)."""

    name = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 300.0, json_mode: bool | None = None, temperature: float = 0.0, **options: Any):
        super().__init__(model or os.environ.get("AGENTDREAM_OPENAI_MODEL") or self.DEFAULT_MODEL, **options)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.json_mode = json_mode
        self.temperature = temperature

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        headers = {}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        elif "api.openai.com" in self.base_url:
            raise BackendError("OPENAI_API_KEY is not set (or pass base_url for a local server)")
        body: dict[str, Any] = {"model": self.model, "temperature": self.temperature,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": prompt}]}
        if self.json_mode or (self.json_mode is None and json_schema is not None):
            body["response_format"] = {"type": "json_object"}
        resp = http_post_json(f"{self.base_url}/chat/completions", headers, body, self.timeout)
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"openai: unexpected response shape: {str(resp)[:200]}") from exc


class ClaudeCLIBackend(LLMBackend):
    """Headless Claude Code: ``claude -p --output-format json`` (uses your Claude Code login)."""

    name = "claude-cli"

    def __init__(self, model: str | None = None, executable: str | None = None, timeout: float = 600.0,
                 extra_args: list[str] | None = None, **options: Any):
        super().__init__(model, **options)
        self.executable = executable or os.environ.get("AGENTDREAM_CLAUDE_BIN") or "claude"
        self.timeout = timeout
        self.extra_args = list(extra_args or [])

    def command(self) -> list[str]:
        cmd = [self.executable, "-p", "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        return cmd + self.extra_args

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        if shutil.which(self.executable) is None and not os.path.exists(self.executable):
            raise BackendError(f"'{self.executable}' not found on PATH")
        full = f"{system}\n\n---\n\n{prompt}"
        try:
            proc = subprocess.run(self.command(), input=full, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"claude-cli timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise BackendError(f"claude-cli exited {proc.returncode}: {(proc.stderr or proc.stdout)[:400]}")
        out = proc.stdout.strip()
        try:
            data = json.loads(out)
        except ValueError:
            return out  # plain text output
        if isinstance(data, list):  # stream-ish array of events: take the final result
            data = next((d for d in reversed(data) if isinstance(d, dict) and d.get("type") == "result"), {})
        if isinstance(data, dict):
            if data.get("is_error"):
                raise BackendError(f"claude-cli reported an error: {str(data.get('result'))[:300]}")
            if "result" in data:
                return str(data["result"])
        return out
