# Edge cases

The edge-case suite lives in [`tests/edge/`](../tests/edge/): one file per area
(`test_<area>.py`), every test named for what it checks and carrying a one-line docstring.
Where a case exposed a bug, the code was fixed; no assertion was loosened to make a case
pass. All of it runs with `pytest` (no network, no real sleeping, no real model), and
`pytest -m edge` runs only this suite.

Regenerate this page (it runs the suite) with:

```bash
python scripts/edge_case_table.py --write docs/EDGE_CASES.md
```

## Documented behaviour, in short

| Situation | Behaviour |
|---|---|
| Empty store / store with only headings | Dream completes with 0 entries before. |
| Missing store, wrong kind (SQLite path is a directory), unknown file type | Job `failed` with `AdapterError` naming the path. |
| `accept --to` an existing file for a folder store (or a directory for a file store) | Refused; nothing is replaced. |
| 0-byte / whitespace-only / tool-noise-only transcript | Skipped with a notice; all empty -> `failed` ("need at least 1 session ... N empty skipped"). |
| More than `max_sessions` (default 100) | Most recent kept with a notice (`on_limit="fail"`: error). Exactly at the limit: no notice. |
| Transcript over `max_bytes_per_transcript` | Oldest messages dropped; a single message larger than the limit is cut to fit (UTF-8 safe) and marked `…[truncated]`. |
| Over `max_total_tokens` | Oldest sessions, then oldest messages dropped; memory alone too large -> clear error. |
| More than `max_entries` (default 20000) memory entries | Always an error: entries are never truncated (that would delete notes on accept). |
| BOM, CRLF, lone CR, mixed endings | Handled; lines count on CRLF/LF/CR only (not U+2028 etc.), so citations match an editor. Rewritten markdown keeps its line ending and BOM. |
| Invalid UTF-8 | Transcripts / markdown: U+FFFD + notice; untouched files copied byte for byte. JSON stores: `AdapterError` naming the file. |
| Malformed JSONL transcript lines (truncated last line, trailing commas, non-objects) | Skipped and reported (`metadata["skipped_lines"]` + notice). |
| Malformed JSON / JSONL *memory* store | `AdapterError` with `file:line` - never silently dropped. |
| Unparsable / impossible timestamps | Treated as missing (`parse_timestamp` never raises); naive timestamps are UTC. |
| Contradiction chains (A -> not-A -> A) | Latest stance wins; history lists every step, including notes merged away as duplicates. |
| Secrets in nested JSON, URLs, base64 blobs, wrapped lines, file names, front matter, headings | Redacted before the model, and in report.md / report.json / job.json / error.log / hook.log. |
| Model returns junk / empty / wrong schema | Deterministic fallback with a warning. |
| Model returns mangled ids (`id='MEMORY.md#1'`, quotes, brackets, comma-joined lists) | Normalised and resolved when the match is unique; warning only for truly unknown ids. |
| Timeout, HTTP 408/429/5xx | Retried `llm_retries` times (default 2) with back-off (`Retry-After` honoured, capped by `llm_retry_max_wait`), then fallback. 4xx and claude-cli failures are not retried. |
| Crash or SIGKILL mid-dream | Input unchanged; job `failed` (on the next load for a killed worker); partial output never served as a report or accepted. |
| Crash mid-accept | The original is put back (or never moved); `agentdream restore JOB` puts the backup back after a completed accept. |
| Concurrent dreams | All complete with distinct ids; concurrent accepts onto one store are serialised and the loser is refused (store changed). |
| Store changed since the dream | `accept` refuses; `--force` overrides. |
| `--only ''`, unknown / ambiguous ids, selection that leaves nothing | Refused with a clear message; duplicates are applied once. |
| Read-only output / jobs / target dirs | One-line error, no traceback, input untouched. |
| Hostile paths (`../`, absolute, `~`, drive letters), symlinks out of the store, `--out` overlapping the store, job ids with `../` | Writes stay inside the destination; outside symlinks are not read (kept as links); overlap and bad ids are refused. |
| Scheduler thresholds | Fire at exactly the threshold; clock going backwards resets the interval; corrupt state file -> fresh start. |
