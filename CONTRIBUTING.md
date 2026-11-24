# Contributing to agentdream

Thanks for helping! agentdream aims to stay small, dependable and dependency-free.

## Ground rules

- **The core stays stdlib-only.** No required runtime dependencies. HTTP goes through `urllib`, MCP is
  hand-rolled JSON-RPC. Optional integrations must import lazily and degrade gracefully.
- **Never mutate input.** Every adapter writes a *new* store. If you add an adapter, add it to the
  parametrized hash-before/after test in `tests/test_safety.py`.
- **Heuristics must be general.** Deterministic rules should be phrased in terms of language patterns
  (polarity, value slots, confirmation/failure cues), never in terms of specific test strings. New rules
  need a test with *different* wording than the motivating example (see `test_correction_is_general_not_hardcoded`).
- **Every change must be explainable.** Operations carry a reason, and corrections and insights carry
  transcript evidence (`file:line` + quote).

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"                 # pytest, pytest-cov, ruff, mypy
.venv/bin/pip install -e ".[stancelog]"           # optional: run the stancelog adapter tests too
.venv/bin/python -m pytest -q                      # ~10 s
.venv/bin/python -m pytest -q --cov=agentdream --cov-report=term-missing   # CI enforces fail_under
.venv/bin/ruff check src tests examples
.venv/bin/mypy                                     # config in pyproject.toml
```

Docs are tested too: `tests/test_docs.py` runs every Python block marked `<!-- test: run -->` in the
README and `docs/`, parses every `agentdream ...` command line against the real CLI, and lints the Mermaid
diagrams. Regenerate the README's before/after sample with `python examples/run_example.py` if you change
the heuristics.

The live smoke test (`tests/test_live.py`) runs only with `AGENTDREAM_LIVE=1` and a logged-in `claude` CLI.

## Layout

```
See the module map in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#module-map). In short:

```
src/agentdream/
  models.py          MemoryEntry, Message, Transcript, Evidence, Operation, apply_operations
  text.py            normalisation, stemming, claim extraction, value slots (memoised)
  blocking.py        exact candidate generation (prefix filtering, inverted indexes)
  chunking.py        token-budgeted chunks for LLM map-reduce
  adapters/          markdown, json/jsonl, sqlite, stancelog (lazy) + registry
  loaders.py         transcript formats + auto-detection
  redact.py          secret redaction (patterns + entropy)
  context.py         DreamConfig, DreamContext
  engine/            deterministic heuristics; LLM prompts, planning, validation, reduce, fallback
  backends/          deterministic, anthropic, openai(-compatible), claude-cli + registry
  pipeline.py        phases (load, redact, limits, consolidate, verify, insights, write)
  report.py          report.md / report.json / diff
  jobs.py            DreamJob (status, persistence, cancel)
  review.py          accept (partial), discard
  scheduler.py       idle / interval / every-N-sessions triggers
  claude_hook.py     Claude Code hook + settings.json installer
  config.py          config files, env vars, precedence (_toml.py: TOML loading)
  mcp_server.py      stdio MCP server
  cli.py             `agentdream` command
```

## Extending

- **Adapter:** subclass `agentdream.adapters.MemoryAdapter` (`detect`, `load`, `write(entries, dest)`) and
  `register_adapter("name", Cls)`, or expose it via the `agentdream.adapters` entry-point group.
- **Backend:** subclass `agentdream.LLMBackend` (`complete(system, prompt, json_schema=None) -> str`) and
  `register_backend`, or use the `agentdream.backends` entry-point group.
- **Transcript format:** `agentdream.register_loader("name", parse_fn)`. `parse_fn(path, text)` returns
  `list[Message]` with `line_no` set.
- **Phase:** `Pipeline.default().replace("insights", my_fn)` / `.insert_after("verify", "lint", fn)`.

## Pull requests

- Keep PRs focused; include tests; run `python -m pytest -q`, `ruff check` and `mypy` locally (CI runs
  Python 3.10–3.13 on Linux plus macOS, and the lint job).
- Describe behaviour changes in the README if they are user-visible.
- Be kind in reviews.
