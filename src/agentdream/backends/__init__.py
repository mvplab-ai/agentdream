"""LLM backend registry (third parties: ``register_backend`` or the
``agentdream.backends`` entry-point group)."""

from __future__ import annotations

from typing import Any

from .base import BackendError, LLMBackend, RateLimitError, http_post_json
from .builtin import AnthropicBackend, ClaudeCLIBackend, DeterministicBackend, OpenAIBackend

_REGISTRY: dict[str, type[LLMBackend]] = {
    "deterministic": DeterministicBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "claude-cli": ClaudeCLIBackend,
}
_ALIASES = {"none": "deterministic", "rules": "deterministic", "claude": "claude-cli", "claude-code": "claude-cli",
            "ollama": "openai", "lmstudio": "openai", "vllm": "openai", "openai-compatible": "openai"}
_EP_LOADED = False


def register_backend(name: str, cls: type[LLMBackend]) -> None:
    _REGISTRY[name] = cls


def available_backends() -> list[str]:
    _load_eps()
    return list(_REGISTRY)


def _load_eps() -> None:
    global _EP_LOADED
    if _EP_LOADED:
        return
    _EP_LOADED = True
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="agentdream.backends"):
            try:
                register_backend(ep.name, ep.load())
            except Exception:  # pragma: no cover
                continue
    except Exception:  # pragma: no cover
        pass


def get_backend(spec: str | LLMBackend | None = "deterministic", model: str | None = None,
                **options: Any) -> LLMBackend:
    if isinstance(spec, LLMBackend):
        return spec
    _load_eps()
    name = (spec or "deterministic").lower()
    name = _ALIASES.get(name, name)
    if name == "openai" and (spec or "").lower() == "ollama" and "base_url" not in options:
        options["base_url"] = "http://localhost:11434/v1"
    if name == "openai" and (spec or "").lower() == "lmstudio" and "base_url" not in options:
        options["base_url"] = "http://localhost:1234/v1"
    cls = _REGISTRY.get(name)
    if cls is None:
        raise BackendError(f"unknown backend {spec!r}; available: {', '.join(_REGISTRY)}")
    return cls(model=model, **options)


__all__ = [
    "AnthropicBackend",
    "BackendError",
    "ClaudeCLIBackend",
    "DeterministicBackend",
    "LLMBackend",
    "OpenAIBackend",
    "RateLimitError",
    "available_backends",
    "get_backend",
    "http_post_json",
    "register_backend",
]
