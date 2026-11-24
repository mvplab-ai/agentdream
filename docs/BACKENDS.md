# Backends

A backend turns a prompt into text: `complete(system, prompt, json_schema=None) -> str`. agentdream builds
the prompts, parses and validates the JSON, and falls back to its deterministic heuristics when a model
is unavailable. All built-in backends use only the standard library (urllib / subprocess) — no SDKs.

| Name (`--backend`) | Class | Default model | Auth / endpoint |
|---|---|---|---|
| `deterministic` (aliases `none`, `rules`) | `DeterministicBackend` | — | none, no network |
| `anthropic` | `AnthropicBackend` | `claude-sonnet-5-5` (or `$AGENTDREAM_ANTHROPIC_MODEL`) | `ANTHROPIC_API_KEY`; `ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`) |
| `openai` (aliases `openai-compatible`, `vllm`) | `OpenAIBackend` | `gpt-4o-mini` (or `$AGENTDREAM_OPENAI_MODEL`) | `OPENAI_API_KEY` for api.openai.com; `--base-url` / `OPENAI_BASE_URL` |
| `ollama` | `OpenAIBackend` | set `--model` | `http://localhost:11434/v1`, no key |
| `lmstudio` | `OpenAIBackend` | set `--model` | `http://localhost:1234/v1`, no key |
| `claude-cli` (aliases `claude`, `claude-code`) | `ClaudeCLIBackend` | your Claude Code default (or `--model`) | your Claude Code login; `$AGENTDREAM_CLAUDE_BIN` to pick the binary |

## Deterministic

The default. No model, no network, reproducible output; see
[ARCHITECTURE.md](ARCHITECTURE.md#engines-deterministic-and-llm) for what the heuristics do. Good for
clear duplicates, stale values with timestamps, location/number/state corrections backed by transcript
evidence, and repeated preferences / mistakes / command workflows. It does not catch paraphrases.

## Anthropic

```bash
export ANTHROPIC_API_KEY=...            # never put the key in a config file you commit
agentdream dream -m ./memory -s ./logs --backend anthropic
agentdream dream -m ./memory -s ./logs --backend anthropic --model claude-opus-5-5 --max-chunk-tokens 150000
```

Uses the Messages API (`POST /v1/messages`, `anthropic-version: 2023-06-01`) with `max_tokens=8192`.
Options via `[backend_options]` or the Python API: `api_key`, `base_url`, `max_tokens`, `timeout`.

## OpenAI and compatible servers (Ollama, LM Studio, vLLM, llama.cpp)

```bash
agentdream dream -m ./memory -s ./logs --backend openai --model gpt-4o-mini                       # OPENAI_API_KEY
agentdream dream -m ./memory -s ./logs --backend ollama --model llama3.1:8b --max-chunk-tokens 6000
agentdream dream -m ./memory -s ./logs --backend lmstudio --model qwen2.5-14b-instruct
agentdream dream -m ./memory -s ./logs --backend openai --base-url http://localhost:8000/v1 --model my-model
```

Chat Completions (`POST {base_url}/chat/completions`) with `temperature=0` and
`response_format={"type": "json_object"}` (disable with `json_mode = false` in `[backend_options]` for
servers that reject it). Options: `api_key`, `base_url`, `timeout`, `json_mode`, `temperature`.

**Local models**: set `--max-chunk-tokens` a little below the model's context window (Ollama's default
context is small; raise `num_ctx` in the model file or use a model with a larger window). The prompt
scaffolding (system prompt, task, headers) takes roughly 700–750 tokens; a budget that leaves less than
256 tokens for content is rejected with an error naming the minimum.

## Claude Code CLI

```bash
claude --version                         # installed and logged in
agentdream dream -m ./memory -s ./logs --backend claude-cli
agentdream dream -m ./memory -s ./logs --backend claude-cli --model sonnet
```

Runs `claude -p --output-format json [--model M]` with the system prompt and task on stdin, then reads the
`result` field. No API key needed; usage counts against your Claude Code plan. Options: `executable`,
`timeout` (default 600 s), `extra_args`.

## Choosing a model

* Start with `deterministic` — it is free and shows what is obviously wrong.
* For semantic duplicates, subtle corrections and richer insights, use a strong instruction-following
  model that returns valid JSON reliably. The Anthropic default is `claude-sonnet-5-5`; larger models help
  most on big, messy stores.
* Bigger `--max-chunk-tokens` means fewer calls and more context per call (better cross-references);
  smaller means cheaper calls that fit small context windows.

## Python API and custom backends

<!-- test: run -->
```python
from agentdream import dream, get_backend

be = get_backend("ollama", model="llama3.1:8b")        # an OpenAIBackend pointed at localhost:11434
print(be.describe(), be.base_url)                        # openai (llama3.1:8b) http://localhost:11434/v1
job = dream("examples/memory", ["examples/sessions"], backend="deterministic", max_chunk_tokens=8000)
print(job.status)
```

Write your own by subclassing `LLMBackend`; return the model's raw text from `complete`. Set
`is_llm = False` only for rule-based backends that should use the built-in heuristics.

<!-- test: run -->
```python
import json

from agentdream import LLMBackend, dream, register_backend


class NothingToChange(LLMBackend):
    name = "nothing"

    def complete(self, system, prompt, json_schema=None):
        # call your model here; `json_schema` describes the expected {"operations": [...]} object
        return json.dumps({"operations": []})


register_backend("nothing", NothingToChange)           # now usable as --backend nothing from Python
job = dream("examples/memory", ["examples/sessions"], backend=NothingToChange(model="v1"))
print(job.report["backend"], job.counts["operations"])  # nothing (v1) 0
```

Third-party packages can expose backends through the `agentdream.backends` entry-point group:

```toml
[project.entry-points."agentdream.backends"]
mymodel = "my_package.backend:MyBackend"
```

## Failure handling

Transient failures are retried first: timeouts and HTTP 408 / 429 / 5xx (including Anthropic's 529)
get `llm_retries` more attempts (default 2) with exponential back-off starting at `llm_retry_backoff`
seconds (default 2), honouring a `Retry-After` header, every wait capped at `llm_retry_max_wait` (60s).
Other client errors (400, 401, ...) and claude-cli failures (non-zero exit, timeout) are not retried.
A custom backend opts in by raising `BackendError(..., retryable=True, retry_after=seconds)` (or a
`TimeoutError`).

If a call still fails or returns nothing parseable, the phase falls back to the deterministic
heuristics and the report says so under *Warnings*. In chunked runs only failed parts are filled in.
Hallucinated ids/citations are dropped, corrections without verifiable evidence are downgraded to flags.

Entries are shown to the model as `[MEMORY.md#1] text`, and the system prompt asks for the bracket
contents verbatim. Ids the model mangles anyway - `id='MEMORY.md#1'`, `id='id='MEMORY.md#6'`, quotes,
backticks, brackets, a whole list as one `"a, b"` string - are normalised and resolved to a known id
only when the match is unique (exact, then case-insensitive, then by file basename); a warning is
recorded only for ids that stay unresolvable.
