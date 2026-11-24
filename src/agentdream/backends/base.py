"""LLM backend interface + stdlib HTTP helper."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class BackendError(Exception):
    pass


class LLMBackend:
    """``complete(system, prompt, json_schema=None) -> str``.

    ``is_llm`` is False for the deterministic backend, which makes the pipeline
    use the built-in heuristics instead of prompting.
    """

    name = "base"
    is_llm = True

    def __init__(self, model: str | None = None, **options: Any):
        self.model = model
        self.options = options

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> str:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}" + (f" ({self.model})" if self.model else "")


def http_post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float = 300.0) -> dict:
    """POST JSON with urllib and return the decoded JSON response."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500] if exc.fp else ""
        raise BackendError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BackendError(f"could not reach {url}: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise BackendError(f"non-JSON response from {url}: {raw[:200]}") from exc
