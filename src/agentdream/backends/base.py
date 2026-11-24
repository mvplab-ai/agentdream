"""LLM backend interface + stdlib HTTP helper."""

from __future__ import annotations

import email.utils
import json
import time
import urllib.error
import urllib.request
from typing import Any

#: HTTP statuses worth retrying (request timeout, rate limit, server errors, Anthropic "overloaded")
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class BackendError(Exception):
    """A backend call failed. ``retryable`` marks transient failures (the LLM engine retries those
    with back-off); ``retry_after`` is the server's requested wait in seconds, if it sent one."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.retryable = retryable


class RateLimitError(BackendError):
    """HTTP 429."""


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """``Retry-After`` header -> seconds (delta-seconds or an HTTP date; never negative)."""
    if not value:
        return None
    v = value.strip()
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(v)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:  # pragma: no cover - older Pythons returned None for junk
        return None
    return max(0.0, dt.timestamp() - (time.time() if now is None else now))


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
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:500] if exc.fp else ""
        except OSError:  # pragma: no cover - body unreadable
            detail = ""
        retry_after = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
        cls = RateLimitError if exc.code == 429 else BackendError
        raise cls(f"HTTP {exc.code} from {url}: {detail}", status=exc.code, retry_after=retry_after,
                  retryable=exc.code in RETRYABLE_STATUS) from exc
    except urllib.error.URLError as exc:
        timed_out = isinstance(exc.reason, TimeoutError)
        raise BackendError(f"could not reach {url}: {'timed out' if timed_out else exc.reason}",
                           retryable=timed_out) from exc
    except TimeoutError as exc:  # socket timeout while reading the response
        raise BackendError(f"request to {url} timed out after {timeout}s", retryable=True) from exc
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise BackendError(f"non-JSON response from {url}: {raw[:200]}") from exc
