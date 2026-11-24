# agentdream — spec (from Deep Mehta, 2026-11-24)

Open-source, local-first "dreaming" (sleep-time memory consolidation) for ANY agent and ANY model.
Inspired by Anthropic's Claude Managed Agents "Dreams" (research preview) and the Letta/UC Berkeley
"sleep-time compute" paper (arXiv 2504.13171). ORIGINAL implementation; do not copy Anthropic code.

## The idea (Anthropic Dreams, paraphrased)
An async job takes a memory store + 1..100 session transcripts and produces a NEW, reorganized memory
store: duplicates merged, stale/contradicted entries replaced with the latest value, notes verified
against what actually happened in transcripts (e.g. Claude Plays Pokémon: a wrong note about the
PokéCenter location was corrected from the logs, which got the agent out of a loop), new insights
surfaced (recurring mistakes, converged workflows, preferences). Input store is NEVER modified.
User reviews output, then accepts or discards. Job statuses: pending, running, completed, failed, canceled.
Optional free-text "instructions" to focus the dream.

## Requirements
- Package name `agentdream` (import `agentdream`), MIT, Python >=3.10, pyproject (hatchling), pip-installable.
  Core has ZERO required runtime deps (stdlib only). Optional extras: `anthropic`, `openai`, `mcp` only if truly needed
  (prefer stdlib urllib HTTP clients so no SDK is required; the MCP server should be stdlib JSON-RPC over stdio).
- Data model: MemoryEntry(id, content, path/key, metadata, created/updated timestamps, source).
  Transcript(id, messages[Message(role, content, timestamp?, line_no, source_path)]). Every message keeps a stable
  citation (file + line number / message index) so evidence can be cited.
- Memory store adapters (interface: load() -> list[MemoryEntry], write(entries, dest) creates a NEW store, never touching input):
  * markdown/text folder (like Claude Code memory dir / CLAUDE.md, MEMORY.md, notes/*.md; one entry per file or per bullet/section — be sensible, document it)
  * JSON / JSONL folder or file
  * SQLite (table of entries)
  * optional stancelog adapter (https://github.com/systemagents-ai/stancelog, local copy at /workspace/agent-memory) — import lazily; skip if not installed
  * registry so third parties add adapters (entry points or register function).
- Transcript loaders: generic JSONL chat logs ({"role","content"} per line), Claude Code session files
  (~/.claude/projects/<proj>/*.jsonl — lines with type user/assistant, message.content being str or list of blocks;
  skip tool noise sensibly), OpenAI-style messages JSON (list or {"messages": [...]}), plain text ("User: ... / Assistant: ..." or raw).
  Auto-detect format. Accept files, dirs, globs. Limit to 1..100 sessions (configurable).
- Dream pipeline (phases; each phase pluggable):
  1. load + redact secrets (API keys, tokens, passwords, AWS keys, private keys, emails optional) BEFORE anything goes to an LLM
  2. enforce size limits (max sessions, max bytes per transcript, max total tokens estimate) — truncate with notice or fail clearly
  3. consolidate: merge duplicates/near-duplicates; detect contradictions; replace stale with latest; temporal contradictions
     ("A was rejected earlier, latest A preferred" — keep history note, current value = latest)
  4. verify: check memory notes against transcripts; if transcript evidence contradicts a note, correct it and CITE evidence (file:line + quote)
  5. insights: surface new patterns (recurring mistakes, repeated preferences, workflows) from transcripts with evidence
  6. write NEW output store (separate path) + report
- Output: separate output store dir + `report.md` (human-readable) + `report.json` (machine): sections Merged, Replaced (stale),
  Corrected (with evidence citations), New insights, Contradictions (incl. temporal), Unchanged count, plus a unified diff
  of old vs new store. Each change is an Operation with stable id so the user can partially accept.
- Review: `accept` (apply all to a target: write output into a destination, default creating a backup of the original
  first — still never silently mutating), `discard`, partial accept (`--only op1,op3` / `--exclude`). Dry-run is the default
  for `dream` (produces output+report in a staging dir; nothing applied).
- LLM backends (interface: complete(system, prompt, json_schema?) -> str), pluggable:
  * `deterministic` (no LLM; heuristic rules: normalized-text dup detection, negation/polarity + subject extraction for
    contradictions, timestamp/order for staleness, transcript-evidence matching for verification, frequency counting for insights).
    Must be good enough to pass all tests reliably.
  * `anthropic` (Messages API via urllib, ANTHROPIC_API_KEY, model configurable)
  * `openai` (OpenAI-compatible chat completions via urllib, base_url configurable -> works with Ollama/LM Studio/vLLM)
  * `claude-cli` (runs `claude -p --output-format json` headless; no API key; uses user's Claude Code login)
  LLM backends get a well-designed prompt asking for structured JSON operations; validate/repair JSON; fall back to
  deterministic on parse failure, recording a warning. Optional `instructions` focus prompt.
- Run modes:
  * Python API: `from agentdream import dream; job = dream(memory="./mem", sessions=["./logs"], backend="deterministic"); job.report`
    plus async `DreamJob` with status (pending/running/completed/failed/canceled), cancel, progress, persisted job state
    (jobs dir, e.g. .agentdream/jobs/<id>/job.json) so status survives process restarts.
  * CLI `agentdream`: `dream --memory ./mem --sessions ./logs [--backend ...] [--model ...] [--instructions ...] [--out DIR] [--apply]`,
    `status [JOB]`, `list`, `show JOB` (print report), `accept JOB [--only ...] [--to DIR]`, `discard JOB`, `schedule ...`, `serve-mcp`.
  * MCP stdio server with tools `dream`, `dream_status`, `accept` (also `discard`, `list_dreams`, `show_report`). Provide Claude Code
    (`claude mcp add ...`) and Cursor (.cursor/mcp.json) config examples.
  * Scheduler: trigger by idle time (no new transcript activity for N minutes), cron-like interval (every N minutes/hours; simple
    cron expression support optional), and every N new sessions. `agentdream schedule --every 6h` / `--idle 30m` / `--sessions 5`
    running a foreground loop (with --once for testing). Keeps state of which sessions were already dreamed over.
- Safety: dry-run default; size limits; secrets redaction; input never mutated (tests hash input before/after); atomic writes.
- Tests (pytest), all must pass in deterministic mode, fast, no network:
  * Pokémon case: memory note "The PokéCenter in Viridian City is north of the gym" (or similar); transcripts show the agent
    repeatedly walking north and failing, then finding it is south -> dream corrects the note to the transcript-proven value with
    evidence citing transcript file+line.
  * duplicate merge
  * stale replacement (older value superseded by later one)
  * temporal contradiction: "don't use A" early, later "maybe A is better / use A" -> current = A preferred, history notes earlier rejection, listed under Contradictions as temporal
  * input untouched (hash every file before/after, for every adapter)
  * insights (recurring mistake/preference across multiple sessions)
  * each loader format + auto-detect, each adapter round-trip, redaction, size limits, job status lifecycle + persistence + cancel,
    accept/discard/partial accept, CLI end-to-end (subprocess or main(argv)), MCP server JSON-RPC (initialize, tools/list, tools/call),
    scheduler triggers (--once, idle, every-N-sessions with fake clock), LLM backend JSON parsing with a fake backend (mock subprocess/HTTP).
  * one live smoke test with the `claude-cli` backend, marked `@pytest.mark.live` and skipped unless `AGENTDREAM_LIVE=1` and `claude` on PATH.
- Docs: README (what dreaming is, credit Anthropic's Dreams + Letta/Berkeley sleep-time compute with links, how this differs:
  open source, any agent, any model incl. local, local-first, no platform lock-in; quickstart; Claude Code + Cursor integration;
  example report excerpt generated from the examples/ dir; design notes; limitations honest). CONTRIBUTING.md, LICENSE (MIT, copyright 2026 systemagents-ai),
  examples/ (a sample memory folder + sample transcripts reproducing Pokémon + A cases), GitHub Actions CI running pytest on 3.10-3.13 (ubuntu, plus macos for one version optional).
