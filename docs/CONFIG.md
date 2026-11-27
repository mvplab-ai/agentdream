# Configuration

Options are resolved in this order (first match wins):

1. **CLI flags** (or arguments to MCP tools / the hook command)
2. **Environment variables** `AGENTDREAM_*`
3. **Project config file**: the nearest `.agentdream.toml`, or `pyproject.toml` with a `[tool.agentdream]`
   table, found by walking up from the current directory (`--config PATH` or `AGENTDREAM_CONFIG` picks a
   file explicitly)
4. **User config file**: `$XDG_CONFIG_HOME/agentdream/config.toml` (default `~/.config/agentdream/config.toml`)
5. **Defaults** (below)

```bash
agentdream config            # every effective value and its source: [cli], [env], a file path, or [default]
agentdream config --json
```

Relative paths in a config file are resolved against the directory of that file; `~` is expanded.
Unknown keys are reported as warnings (on stderr), wrong types as errors that name the file and key.
On Python 3.10 config files are read with `tomli` if installed, otherwise with a built-in parser for the
TOML subset used here (tables, strings, numbers, booleans, arrays, inline tables, comments).

## Reference

```toml
# .agentdream.toml  (or the same keys under [tool.agentdream] in pyproject.toml)
backend = "deterministic"     # deterministic | anthropic | openai | ollama | lmstudio | claude-cli
model = "claude-sonnet-5-5"   # model name for LLM backends
memory = "memory"             # store: markdown folder/file, .json/.jsonl, SQLite, or "stancelog:path"
sessions = ["logs", "~/.claude/projects/-home-me-myrepo/*.jsonl"]   # files, folders, globs
instructions = "focus on deployment notes"
memory_format = "markdown"    # force a store format (otherwise auto-detected)
transcript_format = "jsonl"   # force a transcript format: claude-code | jsonl | openai | text
jobs_dir = ".agentdream/jobs" # where job state lives (default: $AGENTDREAM_HOME/jobs or ./.agentdream/jobs)
out = "dream-output"          # staging dir for the new store + report (default: inside the job dir)

[limits]
max_sessions = 100            # keep the most recent N sessions
min_sessions = 1
max_bytes_per_transcript = 2000000
max_total_tokens = 400000     # estimated tokens of memory + transcripts
max_entries = 20000           # memory entries; never truncated - over the limit is always an error
on_limit = "truncate"         # or "fail"
max_chunk_tokens = 60000      # LLM backends: token budget per call
chunk_overlap = 400           # transcript tokens repeated between consecutive chunks of a session
max_chunks = 64               # max model calls per phase
llm_retries = 2               # retries for timeouts / HTTP 408, 429, 5xx (0 disables)
llm_retry_backoff = 2.0       # seconds; doubles per retry unless the server sends Retry-After
llm_retry_max_wait = 60.0     # cap on any single wait (also caps Retry-After)
insight_min_sessions = 2      # deterministic insights: sessions a pattern must appear in
max_insights = 12

[redaction]
enabled = true
emails = false
entropy = true                # redact long random-looking tokens
entropy_threshold = 4.0       # bits per character
entropy_min_length = 24
cards = true                  # Luhn-valid card numbers
patterns = ["ACME-[0-9]{6}"]  # extra regexes

[backend_options]             # passed to the backend constructor
base_url = "http://localhost:11434/v1"
timeout = 300

[schedule]                    # defaults for `agentdream schedule`
every = "6h"
idle = "30m"
new_sessions = 5
poll = "60s"
state_file = ".agentdream/scheduler-state.json"

[hook]                        # defaults for the Claude Code hook
every = 1
apply = false
memory = "CLAUDE.md"
apply_types = ["merge", "replace"]   # with apply: only these operation types are applied
apply_min_confidence = 0.75          # with apply: only at or above this confidence; the rest stays staged
```

## Environment variables

| Variable | Meaning |
|---|---|
| `AGENTDREAM_<FIELD>` | Any top-level, `[limits]` or `[redaction]` option by its internal name: `AGENTDREAM_BACKEND`, `AGENTDREAM_MODEL`, `AGENTDREAM_MEMORY`, `AGENTDREAM_SESSIONS` (`os.pathsep`-separated or a JSON array), `AGENTDREAM_MAX_SESSIONS`, `AGENTDREAM_MAX_CHUNK_TOKENS`, `AGENTDREAM_ON_LIMIT`, `AGENTDREAM_REDACT` (0/1), `AGENTDREAM_REDACT_EMAILS`, `AGENTDREAM_REDACT_ENTROPY`, `AGENTDREAM_ENTROPY_THRESHOLD`, `AGENTDREAM_REDACT_CARDS`, `AGENTDREAM_REDACT_PATTERNS` (JSON array) ... |
| `AGENTDREAM_BASE_URL` | `backend_options.base_url` |
| `AGENTDREAM_BACKEND_OPTIONS` | JSON object merged into `backend_options` |
| `AGENTDREAM_HOOK_EVERY`, `AGENTDREAM_HOOK_APPLY`, `AGENTDREAM_HOOK_MEMORY`, `AGENTDREAM_HOOK_APPLY_TYPES` (comma-separated or a JSON array), `AGENTDREAM_HOOK_APPLY_MIN_CONFIDENCE` | Hook settings |
| `AGENTDREAM_CONFIG` | Use this config file instead of searching |
| `AGENTDREAM_HOME` | Default jobs dir is `$AGENTDREAM_HOME/jobs` |
| `AGENTDREAM_HOOK_SYNC=1` | Run hook dreams in the foreground (debugging) |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `AGENTDREAM_ANTHROPIC_MODEL` | Anthropic backend |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AGENTDREAM_OPENAI_MODEL` | OpenAI-compatible backend |
| `AGENTDREAM_CLAUDE_BIN` | `claude` executable for the claude-cli backend |

Boolean values accept `1/0`, `true/false`, `yes/no`, `on/off`.

## Example: precedence

<!-- test: run -->
```python
import os
from pathlib import Path

from agentdream.config import resolve

Path(".agentdream.toml").write_text('backend = "ollama"\nmodel = "llama3.1:8b"\n[limits]\nmax_sessions = 20\n')
os.environ["AGENTDREAM_MODEL"] = "qwen2.5:14b"           # env beats the file
s = resolve({"max_sessions": 5})                          # CLI-level values beat both
print(s.dream["backend"], s.dream["model"], s.dream["max_sessions"])   # ollama qwen2.5:14b 5
print(s.sources["dream.backend"].endswith(".agentdream.toml"), s.sources["dream.model"])  # True env
del os.environ["AGENTDREAM_MODEL"]
```

The Python `dream()` function does not read config files; it uses exactly the arguments you pass. Use
`agentdream.config.resolve()` (as above) if you want the same precedence in your own code.
