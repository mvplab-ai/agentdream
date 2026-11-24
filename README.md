# agentdream

**Local-first "dreaming" (sleep-time memory consolidation) for any agent and any model.**

An agent's memory drifts: duplicates pile up, old values go stale, preferences get reversed, and
sometimes a note is just *wrong*, and the agent keeps acting on it. A **dream** is an offline job that
reads the memory store plus recent session transcripts and writes a **new, reorganized store** for you to
review:

- **Merged:** duplicate and near-duplicate notes collapsed into one
- **Replaced (stale):** older values superseded by newer ones
- **Corrected:** notes that transcripts prove wrong get fixed, **with `file:line` citations and quotes**
- **New insights:** recurring mistakes, repeated preferences, converged workflows
- **Contradictions:** including *temporal* ones ("don't use A" → later "use A"); the latest wins and the history is kept
- **Unchanged** count, plus a unified **diff** of old vs new store

The input store is **never modified**. Dreams are dry runs by default; you `accept` (all or some
operations, with an automatic backup) or `discard`.

```
pip install agentdream          # zero runtime dependencies (stdlib only)
agentdream dream --memory ./memory --sessions ./logs
agentdream show <job-id>
agentdream accept <job-id> --only correct-2ec6281c,merge-a91fbf3a
```

## Credit and how this differs

The idea comes from two sources:

- **Anthropic's "Dreams"** for Claude Managed Agents (research preview): an async job that takes a memory
  store and up to 100 session transcripts and produces a reorganized store (duplicates merged, stale
  entries replaced, notes verified against what actually happened, new insights surfaced). Anthropic's
  example is *Claude Plays Pokémon*: a wrong note about where the PokéCenter was got corrected from the
  logs, which got the agent out of a loop.
  See [Anthropic's announcement](https://www.anthropic.com/news) and the
  [Claude docs](https://docs.anthropic.com/).
- **Sleep-time compute** (Letta / UC Berkeley): *"Sleep-time Compute: Beyond Inference Scaling at Test-time"*,
  [arXiv:2504.13171](https://arxiv.org/abs/2504.13171). Use idle time to pre-process context so test-time
  inference is cheaper and better.

agentdream is an **independent, original, open-source implementation** of the idea (MIT). It does not use
Anthropic's code. The differences:

| | agentdream |
|---|---|
| Agents | any agent: Claude Code, Cursor, custom agents, game bots, plain chat logs |
| Models | any model, including local ones (Ollama, LM Studio, vLLM), the Anthropic API, OpenAI, your Claude Code login, **or no model at all** (deterministic heuristics) |
| Where | runs locally; your memory and transcripts stay on your machine unless you pick a hosted backend |
| Lock-in | none: Markdown folders, JSON/JSONL, SQLite, pluggable adapters |

## Quickstart

```python
from agentdream import dream

job = dream(memory="./mem", sessions=["./logs"], backend="deterministic")
print(job.status)                 # completed
print(job.report["counts"])       # {'merged': 1, 'replaced': 1, 'corrected': 1, ...}
print(job.report_md)              # human-readable report
print(job.store_path)             # the NEW store (staged)

from agentdream import accept, discard
accept(job, only=["correct-2ec6281c"])   # partial accept; dependencies are pulled in automatically
```

Async, with status that survives restarts (`.agentdream/jobs/<id>/job.json`, or `$AGENTDREAM_HOME/jobs`):

```python
from agentdream import dream, DreamJob

job = dream("./mem", ["./logs"], wait=False)    # pending -> running -> completed | failed | canceled
job.status, job.phase, job.progress
job.cancel()
DreamJob.load(job.id).status                    # from any process
```

Try it on the bundled example, which reproduces the Pokémon case and the "A rejected, then A preferred" case:

```bash
python examples/run_example.py
```

### CLI

```
agentdream dream   --memory PATH --sessions FILES/DIRS/GLOBS [--backend deterministic|anthropic|openai|ollama|claude-cli]
                   [--model M] [--base-url URL] [--instructions "focus on ..."] [--out DIR] [--apply]
                   [--max-sessions 100] [--max-bytes-per-transcript N] [--max-total-tokens N] [--on-limit truncate|fail]
                   [--no-redact] [--redact-emails]
agentdream status  [JOB] [--json]
agentdream list
agentdream show    JOB [--json | --diff]
agentdream accept  JOB [--only op1,op3 | --exclude op2] [--to DIR] [--no-backup] [--force]
agentdream discard JOB
agentdream cancel  JOB
agentdream schedule --memory ... --sessions ... (--every 6h | --idle 30m | --new-sessions 5) [--once] [--apply]
agentdream serve-mcp
```

`dream` is a **dry run**: it writes `<out>/store/` (the new store), `<out>/report.md` and
`<out>/report.json`, and applies nothing. `--apply` accepts everything afterwards (the backup is still made).

## Example report (generated from `examples/`)

Excerpt of `python examples/run_example.py` (deterministic backend, no model):

| Merged | Replaced | Corrected | New insights | Contradictions (temporal) | Unchanged |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 3 | 2 (2) | 3 |

```markdown
## Corrected (1)

### `correct-2ec6281c` — Corrected from transcript evidence: "The PokéCenter in Viridian City is south of the gym. (2026-03-03)"

- Entries: `MEMORY.md#1`
- Before: The PokéCenter in Viridian City is north of the gym. (2026-01-03)
- After: The PokéCenter in Viridian City is south of the gym. (2026-03-03)
- Why: Transcripts show 'south', not 'north' (3 supporting, 4 refuting observation(s)).
- Evidence:
  - `examples/sessions/session-01.jsonl:3` (refutes, assistant): "Walked north of the gym for 20 steps. No PokéCenter here, only tall grass."
  - `examples/sessions/session-02.jsonl:2` (refutes, assistant): "Walked north of the gym again. No PokéCenter here, this is a dead end."
  - `examples/sessions/session-03.jsonl:2` (supports, assistant): "It is just south of the gym, next to the Poké Mart."
  - `examples/sessions/session-03.jsonl:3` (supports, assistant): "Healed the whole team at the PokéCenter south of the gym."

## Contradictions (2)

### `contradiction-c4b66adf` [temporal] — Preference reversed in later sessions: now "Use Poetry for dependency management."

- Before: Don't use Poetry for dependency management — it was too slow on CI. (2026-01-05)
- After: Use Poetry for dependency management. (History: earlier said "Don't use Poetry for dependency
  management — it was too slow on CI" (2026-01-05); reversed later (examples/sessions/session-04-claude-code.jsonl:2).)

## New insights (3)

- Preference (stated in 2 sessions): Always use type hints in Python code.
- Recurring mistake (3 occurrences across 2 sessions): "Still can't find the PokéCenter north of the gym" — check before repeating.
- Converged workflow (used in 2 sessions): `pytest -q` → `git commit -m`.
```

```diff
-- The PokéCenter in Viridian City is north of the gym. (2026-01-03)
+- The PokéCenter in Viridian City is south of the gym. (2026-03-03)
-- Don't use Poetry for dependency management — it was too slow on CI. (2026-01-05)
-- Run the test suite with `pytest -q` before committing.
+- Use Poetry for dependency management. (History: ...)
 - Run the tests with `pytest -q` before every commit.
-- The API server listens on port 8080. (2025-12-01)
```

## Integrations

### Claude Code

```bash
claude mcp add agentdream -- agentdream serve-mcp
```

Then ask Claude to dream over `~/.claude/projects/<project>/memory` with that project's `*.jsonl`
sessions. Claude Code session files are auto-detected. See
[`examples/integrations/claude-code.md`](examples/integrations/claude-code.md). To use your Claude Code login as
the model: `--backend claude-cli` (runs `claude -p --output-format json`, no API key needed).

### Cursor

`.cursor/mcp.json` ([example](examples/integrations/cursor-mcp.json)):

```json
{ "mcpServers": { "agentdream": { "command": "agentdream", "args": ["serve-mcp"] } } }
```

MCP tools: `dream`, `dream_status`, `accept`, `discard`, `list_dreams`, `show_report`. The server is
stdlib JSON-RPC 2.0 over stdio with no SDK.

### Scheduling

```bash
agentdream schedule --memory ./mem --sessions "./logs/*.jsonl" --idle 30m        # after 30 min of no new activity
agentdream schedule --memory ./mem --sessions ./logs --every 6h                  # interval
agentdream schedule --memory ./mem --sessions ./logs --new-sessions 5 --once     # every 5 new sessions; single check
```

The scheduler remembers which sessions it already dreamed over (`.agentdream/scheduler-state.json`)
and dreams only over new ones. Results are still staged for review unless you pass `--apply`.

## Inputs

**Memory stores** (auto-detected; force with `--memory-format`):

| Store | Granularity |
|---|---|
| Markdown / text folder or single file (`MEMORY.md`, `CLAUDE.md`, `notes/*.md`) | A file with front matter (`---` … `---`, like Claude Code memory files) is **one entry**. Otherwise **one entry per top-level bullet / numbered item** (continuation and nested lines included) and one per free paragraph or code block. Headings become the entry `key`. Ids: `path#n` / `path`. Writing re-renders only changed entries, copies untouched files byte for byte, and puts new insights into the matching section or into `dream-insights.md`. |
| JSON / JSONL (file or folder) | a list, `{"entries"/"memories": [...]}`, a flat `{"key": "value"}` map, or JSONL lines. Text in `content`/`text`/`memory`/`value`/`note`. Unknown fields round-trip. |
| SQLite | a table (`memories`/`memory`/`entries`/`notes`, or `table=`) with `id` + a text column. The input is opened read-only, and the output is a backup-API copy with the table rewritten, so other tables survive. |
| stancelog | optional, `stancelog:<path>`, imported lazily; unavailable when the package isn't installed. |

Third-party adapters use `agentdream.register_adapter(name, cls)` or the `agentdream.adapters`
entry-point group. Backends work the same way: `register_backend` or `agentdream.backends`.

**Transcripts** (files, directories, globs; 1..100 sessions by default; format auto-detected per file):
generic JSONL (`{"role","content"}`), Claude Code sessions (`type: user|assistant`, `message.content` str or
blocks; thinking and successful tool output dropped, tool calls kept as one line, failed tool results kept),
OpenAI-style messages JSON (list or `{"messages": [...]}`), plain text (`User:` / `Assistant:` turns, or a raw
log with one message per line). Every message keeps a stable citation, `file:line` (or `file#index`).

## Design notes

Pipeline phases, each a plain `fn(ctx)` that you can replace via `Pipeline.default().replace("insights", fn)`
or `insert_after(...)`:

1. **load:** adapter + transcript loaders
2. **redact:** API keys (Anthropic/OpenAI/GitHub/Slack/Google), AWS keys, JWTs, bearer tokens, private-key
   blocks, URL credentials, `password=`/`token:` assignments, and optionally emails. This runs **before**
   any analysis or LLM call. Reports and diffs only contain redacted text. Untouched entries keep their
   original bytes in the output store.
3. **limits:** max sessions, max bytes per transcript (keeps the most recent messages), max total tokens
   (drops the oldest sessions). Either truncate with a notice in the report or `--on-limit fail`.
4. **consolidate:** near-duplicate detection (normalized text, stemmed token sets, sequence ratio, while
   refusing to merge notes whose numbers, directions or values differ); claim extraction (subject, value,
   polarity) to detect stale facts (newest by timestamp, else store order) and preference reversals.
5. **verify:** each memory claim is checked against transcript sentences that mention its subject.
   Value slots (directions, numbers, on/off states) or generic `X is Y` values are compared. Evidence is
   weighted: confirmations (*found, actually, turns out…*) count double, failures and negations (*no X here,
   can't find*) refute, and plans or beliefs (*heading north, my notes say*) are **not** evidence. A
   correction needs a supported alternative that beats the old value, and it cites every observation used.
   When an entity appears in several places (e.g. PokéCenters in several cities), the most recently
   mentioned qualifier decides which note a sentence is about.
6. **insights:** frequency counting across sessions: failure-marked statements clustered by content
   (recurring mistakes), preference statements repeated in ≥2 sessions, preference flips across sessions
   (temporal contradictions), and command sequences recurring across sessions (workflows). Insights that
   duplicate existing memory are skipped.
7. **write:** the new store via the adapter, `report.json`, `report.md`, and the diff.

Every change is an `Operation` with a stable id (a hash of type, targets and result) and `depends_on`
links, so partial accepts stay consistent. `--only` pulls in dependencies, and `--exclude` drops dependents.

**LLM backends** implement `complete(system, prompt, json_schema=None) -> str`. Each analysis phase sends
the memory (with ids) and transcripts (each line prefixed with its citation), and asks for a JSON list of
operations. Output is repaired (code fences, trailing commas, Python literals, truncated JSON) and
**validated**: unknown ids are dropped, citations must exist, and quotes must match. A correction without
verifiable evidence is downgraded to a flag. An insight without evidence is dropped. If the call or the
parse fails, the phase falls back to the deterministic heuristics and a warning is recorded in the report.

**Safety:** dry run by default; the input is never written to (adapters refuse to write over their
input, SQLite is opened `mode=ro`, and the tests hash every input file before and after for every
adapter); writes are atomic (temp file or directory plus rename); `accept` refuses if the store changed
since the dream (unless `--force`) and backs up the original into the job directory first.

## Limitations (honest)

- The **deterministic engine is English-centric pattern matching**, not language understanding. It
  handles common phrasings ("X is north of Y", "don't use A", "maybe A is better", "always do B",
  "port 8080 → 9090") and is deliberately conservative: when evidence is weak it does nothing or raises a
  flag-only contradiction. Paraphrases, coreference beyond "it/that", sarcasm, and multi-hop reasoning need
  an LLM backend.
- Recency for memory-vs-memory conflicts comes from explicit timestamps (front matter, JSON fields,
  inline ISO dates). Without them it falls back to store order, which may not be chronological.
  Transcripts are treated as newer than memory unless timestamps say otherwise.
- The LLM backends send the whole (redacted, size-limited) context in one prompt per phase. There is no
  chunking or map-reduce for very large inputs yet, so use the size limits.
- Redaction is regex-based. It catches common secret formats, not every secret.
- The scheduler is a foreground polling loop (run it under systemd, launchd, cron or tmux). Cron
  *expressions* are not supported, only intervals, idle time and session counts.
- The stancelog adapter is duck-typed against stancelog's API and was not tested against a real install.
- Markdown granularity is bullet/paragraph based, so prose-heavy notes become coarse entries.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Tests: `python -m pytest -q` (fast, no network; the live
`claude-cli` smoke test runs only with `AGENTDREAM_LIVE=1`).

## License

MIT © 2026 systemagents-ai
