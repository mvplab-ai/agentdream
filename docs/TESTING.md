# Testing

How the agentdream test suite is organised, what runs in CI, what was checked against a real
model, and the edge-case pass/fail table. Documented behaviour for each situation (empty store,
kills mid-accept, and so on) is in [EDGE_CASES.md](EDGE_CASES.md). Architecture and safety
enforcement: [ARCHITECTURE.md](ARCHITECTURE.md).

## How to run

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install "git+https://github.com/systemagents-ai/stancelog"   # optional: run the stancelog adapter tests too
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -m edge     # only tests/edge
.venv/bin/python scripts/edge_case_table.py --write docs/EDGE_CASES.md   # refresh this table too
.venv/bin/python -m pytest -q --cov=agentdream --cov-report=term-missing
.venv/bin/ruff check src tests examples
.venv/bin/mypy
```

`tests/test_docs.py` also runs: every Python block marked `<!-- test: run -->` in the README and
`docs/`, every documented `agentdream ...` command against the real parser, relative links and
anchors, and a structural lint of the Mermaid diagrams.

Live tests are skipped unless you opt in (they are never run in CI):

```bash
AGENTDREAM_LIVE=1 pytest tests/test_live.py tests/live -q          # claude CLI (must be logged in)
AGENTDREAM_OLLAMA_MODEL=qwen2.5:1.5b pytest tests/live -q          # local Ollama
ANTHROPIC_API_KEY=... pytest tests/live -q                         # Anthropic Messages API
```

## The suites

| Suite | Where | When it runs | What it hits |
|---|---|---|---|
| Unit / integration | `tests/test_*.py` (except `test_live.py`) | CI and default `pytest` | Adapters, loaders, engines, pipeline, jobs, review, CLI, MCP, hook, scheduler, docs, examples. No network, no real model. |
| Edge cases | `tests/edge/` (`pytest -m edge`) | CI and default `pytest` | Hostile / empty / huge / malformed inputs, crashes, concurrency, permissions, redaction, model misbehaviour. Fake backends, fake clocks; no network. Kill-recovery tests do SIGKILL a real subprocess. |
| Property-based | `tests/edge/test_properties.py` (hypothesis) | CI and default `pytest` | Redaction never leaks and is idempotent; loaders and adapters don't crash on arbitrary bytes; dreams never mutate input; chunking covers every item within budget; accept/restore round-trips. |
| stancelog adapter | `tests/test_stancelog_adapter.py` | CI (stancelog is installed there) and locally if the extra is installed | The real stancelog library against a temp database. |
| Live backends | `tests/test_live.py`, `tests/live/` | Only with the env vars above; **never CI** | Real `claude-cli` / Ollama / Anthropic. A recording wrapper asserts planted secrets never leave the machine. |

CI (`.github/workflows/ci.yml`) runs the default pytest with coverage (`fail_under` 94) on Ubuntu
3.10–3.13 (required) and macOS 3.10 and 3.13 (best-effort: those jobs use `continue-on-error` because
macOS runners are often unavailable and would otherwise keep the badge red), a ruff + mypy lint job
(including mypy under 3.10 semantics), the example runner, and a CLI smoke test. Live tests are
collected and skipped.

## Live vs mocked

This is an honest record of what was actually executed against a real model, versus what is
covered only by fakes.

### Mocked (default suite; this is what CI runs)

These use fake LLM backends, mocked `urllib` / `subprocess`, and a swapped-out `sleep`. They do
**not** call a model or the network.

* Model junk, empty replies, wrong schemas, unknown ids, hallucinated citations.
* Timeouts, HTTP 408 / 429 (including `Retry-After`, including a huge value that must be capped)
  and 5xx, then deterministic fallback; `llm_retries=0`; a transient error followed by success.
* claude-cli non-zero exit and non-JSON stdout; Anthropic / OpenAI HTTP errors (mocked urllib).
* The exact id strings a live Ollama run of `qwen2.5:1.5b` once returned
  (`id='MEMORY.md#1'`, `id='id='MEMORY.md#6'`), plus quotes, brackets and comma-joined lists.
* Secrets in nested JSON, URLs, base64 blobs, wrapped lines, file names and metadata: nothing
  secret reaches the fake backend or report / job / log files.
* Crash and SIGKILL mid-dream and mid-accept; concurrent dreams (threads and processes);
  stale accept; partial accept with invalid ids; read-only dirs; path traversal; scheduler
  boundaries; hypothesis properties.

The mid-dream kill test is a hybrid: the worker is a real Python subprocess that is sent
`SIGKILL`; the model it would have called is still the deterministic backend.

### Live (opt-in)

`tests/live/test_live_backends.py` runs the same cases against every backend you have configured
(claude-cli, a local Ollama, and/or Anthropic):

* All bundled examples (memory folder + five sessions in three transcript formats).
* Each transcript format on its own.
* A temporal A-then-not-A case ("don't use Poetry" in memory, "use Poetry" later in the transcript).
* Planted secrets must not appear in any prompt actually sent to the model, or in the reports.
* A chunked map-reduce run (`max_chunk_tokens=1800`) so the model sees the input in several calls.
* A real Claude Code session file from `~/.claude/projects`, if one exists on the machine.

`tests/test_live.py` is a smaller claude-cli smoke test: it asserts the model produces a
correction ("PokéCenter is south of the gym") and that the input file is untouched.

**What has been run for real**

* **Ollama (`qwen2.5:1.5b`, OpenAI-compatible backend), live.** This is how the mangled-id bug was
  found: the model copied `id='MEMORY.md#1'` out of the old prompt format and the validator dropped
  almost every operation. That is fixed (prompts now render `[MEMORY.md#1] text`; ids are
  normalised and resolved only on a unique match) and regression-tested with a fake backend. The
  live Ollama suite was not re-run in the final hardening pass (no Ollama server on the machine
  that finished the pass).
* **claude-cli, live smoke** (`tests/test_live.py`): re-run on 24 Nov 2026 (America/Los_Angeles)
  against a logged-in Claude Code CLI (claude 2.1.331). It passed in 12s: the model produced the
  "south of the gym" correction and the input file was untouched. The full `tests/live/` matrix
  against claude-cli was not re-run in that pass (it is several dreams and is not part of CI).
* **Anthropic Messages API (`ANTHROPIC_API_KEY`).** Error paths and retry behaviour are tested
  with mocked urllib. A full live run of `tests/live/` against the Anthropic API was not repeated
  in the final hardening pass (no API key in that environment).

If you re-run a live backend, refresh this paragraph if the result changes.

## Coverage

The default suite, with stancelog installed, measures about **96%** line coverage of
`src/agentdream` (CI enforces `fail_under = 94` in `pyproject.toml`). A handful of branches stay
uncovered on purpose: `__main__`, some rare adapter / hook / CLI error paths, and stancelog
failure modes that need a broken on-disk database.

## Remaining limits

* Deterministic mode is English pattern matching: it misses paraphrases a strong model would
  catch and does not understand other languages (unicode is preserved, not interpreted).
* Live model quality still depends on the model. Small local models can return shallow or
  oddly formatted operations; they are validated, but there may be few left.
* Token counts are a ~4-characters-per-token estimate, not the model's tokenizer.
* Windows is not in CI (`msvcrt` locking is used there; permission tests skip as root / on
  Windows).
* Anthropic and Ollama live suites need a key or a running server; they are not run in CI.

## Edge-case results

Regenerate this table (it re-runs `tests/edge`) with
`python scripts/edge_case_table.py --write docs/EDGE_CASES.md`. The same command refreshes the
copy below.

<!-- edge-case-table:start -->
**213 edge-case tests (325 test runs with parametrization) in 17 areas: 325 passed, 0 failed, 0 skipped.**

| Area | Case | Status |
|---|---|---|
| Empty / missing inputs | Empty memory folder (no entries): dream completes, 0 entries before, nothing crashes. <br>`test_empty_markdown_store_dir_completes` | ✅ pass |
| Empty / missing inputs | Markdown file with headings and blank lines only -> zero entries, dream completes. <br>`test_store_with_only_headings_has_no_entries` | ✅ pass |
| Empty / missing inputs | 0-byte .json and .jsonl stores load as empty and round-trip through a dream. <br>`test_empty_json_and_jsonl_stores` | ✅ pass |
| Empty / missing inputs | Memory path that does not exist -> job failed with AdapterError naming the path. <br>`test_missing_store_path_fails_clearly` | ✅ pass |
| Empty / missing inputs | CLI with a missing memory path exits 1 with one error line, no traceback. <br>`test_missing_store_path_cli_clean_error` | ✅ pass |
| Empty / missing inputs | Directory named like a file store (mem.json/) forced to json: 0 entries, other files survive accept. <br>`test_json_store_path_is_a_directory` | ✅ pass |
| Empty / missing inputs | --memory-format sqlite on a directory -> failed with AdapterError 'not a file'. <br>`test_sqlite_format_forced_on_a_directory` | ✅ pass |
| Empty / missing inputs | A memory file of an unknown type (.yaml) -> AdapterError 'could not detect'. <br>`test_unknown_file_store_type` | ✅ pass |
| Empty / missing inputs | accept --to an existing *file* for a folder store is refused; the file is untouched. <br>`test_accept_refuses_file_vs_dir_target_mismatch` | ✅ pass |
| Empty / missing inputs | accept --to an existing *directory* for a single-file store is refused, nothing deleted. <br>`test_accept_refuses_dir_target_for_single_file_store` | ✅ pass |
| Empty / missing inputs | A 0-byte transcript is skipped with a notice; other sessions still load. <br>`test_zero_byte_transcript_skipped_with_notice` | ✅ pass |
| Empty / missing inputs | A whitespace-only transcript (spaces, tabs, CR/LF) is skipped as empty. <br>`test_whitespace_only_transcript_skipped` | ✅ pass |
| Empty / missing inputs | Claude Code session with only thinking, successful tool results and system reminders is skipped. <br>`test_tool_noise_only_transcript_skipped` | ✅ pass |
| Empty / missing inputs | Every transcript empty -> job failed: 'need at least 1 session', mentions the skipped files. <br>`test_all_transcripts_empty_fails_clearly` | ✅ pass |
| Empty / missing inputs | No session sources at all -> job failed with 'need at least 1 session'. <br>`test_no_sessions_given_fails_clearly` | ✅ pass |
| Empty / missing inputs | An existing but empty sessions directory -> LoaderError, not a crash. <br>`test_empty_sessions_directory` | ✅ pass |
| Empty / missing inputs | A sessions path that doesn't exist -> LoaderError naming it. <br>`test_missing_sessions_path` | ✅ pass |
| Empty / missing inputs | Dream over an empty store leaves it byte-identical and writes a parseable job.json. <br>`test_empty_store_input_untouched_and_job_json_valid` | ✅ pass |
| Size limits | Exactly max_sessions (100) sessions: all kept, no truncation notice. <br>`test_exactly_max_sessions_no_notice` | ✅ pass |
| Size limits | 101 sessions with the default limit: the oldest is dropped, notice says so. <br>`test_one_over_max_sessions_keeps_most_recent` | ✅ pass |
| Size limits | 150 sessions, max_sessions=7: keeps the 7 most recent. <br>`test_far_over_max_sessions_with_small_limit` | ✅ pass |
| Size limits | on_limit=fail with max_sessions+1 sessions: job failed with DreamLimitError. <br>`test_over_max_sessions_fail_mode` | ✅ pass |
| Size limits | A transcript whose text is exactly max_bytes_per_transcript bytes is not truncated. <br>`test_transcript_exactly_at_byte_limit_untouched` | ✅ pass |
| Size limits | One byte over max_bytes_per_transcript: oldest messages dropped, notice with counts. <br>`test_transcript_one_byte_over_drops_oldest_messages` | ✅ pass |
| Size limits | One message larger than the byte limit is cut to fit (with a marker), not silently emptied. <br>`test_single_message_over_byte_limit_is_cut_not_dropped` | ✅ pass |
| Size limits | Cutting a message of 4-byte emoji never produces broken UTF-8 or exceeds the limit. <br>`test_byte_limit_cut_respects_multibyte_characters` | ✅ pass |
| Size limits | on_limit=fail: exactly at the byte limit passes, one byte over fails with a clear message. <br>`test_byte_limit_fail_mode_exact_and_over` | ✅ pass |
| Size limits | max_total_tokens == estimate: nothing dropped; estimate - 1: oldest session dropped with notice. <br>`test_total_tokens_exactly_at_limit_and_one_over` | ✅ pass |
| Size limits | on_limit=fail and one token over max_total_tokens -> DreamLimitError. <br>`test_total_tokens_one_over_fail_mode` | ✅ pass |
| Size limits | Memory store alone larger than max_total_tokens -> failed, message says the store is too large. <br>`test_memory_alone_over_token_limit_is_a_clear_error` | ✅ pass |
| Size limits | max_entries == entry count passes; one fewer fails clearly (entries are never truncated). <br>`test_max_entries_exactly_at_and_one_over` | ✅ pass |
| Size limits | Nonsense limits (0 / negative) are rejected up front with ValueError, not silently misbehaving. <br>`test_limits_validated` | ✅ pass |
| Unicode & encodings | Arabic/Hebrew/emoji notes survive a dream + accept byte-for-byte when untouched. <br>`test_rtl_and_emoji_round_trip_unchanged` | ✅ pass |
| Unicode & encodings | Transcript messages in Arabic/Hebrew/emoji keep correct line numbers and appear in the report. <br>`test_rtl_and_emoji_in_transcripts_are_cited` | ✅ pass |
| Unicode & encodings | 'Café' written precomposed (NFC) and decomposed (NFD) is recognised as a duplicate. <br>`test_nfc_and_nfd_duplicates_merge` | ✅ pass |
| Unicode & encodings | Notes that differ only by zero-width spaces / joiners / BOM / soft hyphens are merged. <br>`test_zero_width_characters_do_not_hide_duplicates` | ✅ pass |
| Unicode & encodings | Heavy stacks of combining marks in memory and transcripts are handled (no crash, input intact). <br>`test_combining_marks_and_zalgo_do_not_crash` | ✅ pass |
| Unicode & encodings | NUL bytes inside a markdown note are kept (not truncated) and the dream completes. <br>`test_nul_bytes_in_markdown_memory` | ✅ pass |
| Unicode & encodings | NUL characters (\u0000) in JSON memory values and transcript content round-trip. <br>`test_nul_bytes_in_json_memory_and_transcripts` | ✅ pass |
| Unicode & encodings | Invalid UTF-8 bytes in a markdown store: loads with U+FFFD, completes, untouched files keep their bytes. <br>`test_invalid_utf8_in_markdown_memory` | ✅ pass |
| Unicode & encodings | Non-UTF-8 bytes in a JSONL transcript decode with replacement; valid lines are unaffected. <br>`test_invalid_utf8_in_transcripts` | ✅ pass |
| Unicode & encodings | A Latin-1 encoded plain-text transcript loads (with replacement chars) instead of crashing. <br>`test_latin1_transcript_text_file` | ✅ pass |
| Unicode & encodings | A JSON memory store with invalid UTF-8 -> failed with AdapterError naming the file, not a raw traceback. <br>`test_invalid_utf8_json_store_fails_clearly` | ✅ pass |
| Unicode & encodings | U+2028 / U+2029 / U+0085 / form feed inside JSON strings don't split JSONL lines or shift line numbers. <br>`test_unicode_line_separators_inside_jsonl_strings` | ✅ pass |
| Unicode & encodings | A U+2028 inside a markdown note stays inside the note and survives a rewrite of that file. <br>`test_unicode_line_separator_in_markdown_is_preserved` | ✅ pass |
| Malformed JSON / JSONL | JSONL whose last line was cut off mid-write: earlier lines load, notice reports the skipped line. <br>`test_jsonl_last_line_truncated` | ✅ pass |
| Malformed JSON / JSONL | A UTF-8 BOM before the first JSONL line doesn't lose the first message or change detection. <br>`test_jsonl_with_utf8_bom` | ✅ pass |
| Malformed JSON / JSONL | A one-line JSONL file with a BOM is still detected as JSONL. <br>`test_single_line_jsonl_with_bom` | ✅ pass |
| Malformed JSON / JSONL | OpenAI-style messages JSON with a BOM is detected and parsed. <br>`test_openai_json_with_bom` | ✅ pass |
| Malformed JSON / JSONL | JSONL lines with trailing commas, arrays, numbers, strings or null are skipped and counted, not fatal. <br>`test_jsonl_trailing_commas_and_non_object_lines` | ✅ pass |
| Malformed JSON / JSONL | Claude Code lines whose message/content/input have unexpected types are skipped, not a crash. <br>`test_claude_code_lines_with_wrong_shapes` | ✅ pass |
| Malformed JSON / JSONL | OpenAI messages with non-list tool_calls, non-dict items, or non-string roles don't crash. <br>`test_openai_shapes_wrong_types` | ✅ pass |
| Malformed JSON / JSONL | A truncated OpenAI JSON file: auto-detect falls back to text; forcing 'openai' raises LoaderError. <br>`test_truncated_openai_json_falls_back_or_errors_cleanly` | ✅ pass |
| Malformed JSON / JSONL | Forcing jsonl/claude-code on plain text yields no messages (skipped), never an exception. <br>`test_forced_format_on_wrong_content` | ✅ pass |
| Malformed JSON / JSONL | Pathologically nested JSON ([[[[...) doesn't raise RecursionError out of the loader. <br>`test_deeply_nested_json_does_not_crash_loader` | ✅ pass |
| Malformed JSON / JSONL | One directory holding Claude Code JSONL, generic JSONL, OpenAI JSON and plain text: each detected. <br>`test_mixed_formats_in_one_directory` | ✅ pass |
| Malformed JSON / JSONL | A JSONL *memory* store with a cut-off last line fails with file:line (never drops the entry silently). <br>`test_jsonl_memory_store_truncated_last_line_is_an_error` | ✅ pass |
| Malformed JSON / JSONL | A JSON memory store with a trailing comma -> AdapterError with line/column, not a bare JSONDecodeError. <br>`test_json_memory_store_trailing_comma_is_an_error` | ✅ pass |
| Malformed JSON / JSONL | A JSON / JSONL memory store saved with a UTF-8 BOM loads normally. <br>`test_json_memory_store_with_bom` | ✅ pass |
| Malformed JSON / JSONL | A number on a JSONL memory line -> AdapterError naming file:line. <br>`test_jsonl_memory_store_non_object_line` | ✅ pass |
| Malformed JSON / JSONL | A JSON store whose top level is a number or string -> AdapterError 'unsupported JSON top-level'. <br>`test_json_memory_store_scalar_top_level` | ✅ pass |
| Malformed JSON / JSONL | detect_format sees through a leading BOM and blank lines. <br>`test_detect_format_ignores_bom_and_leading_blank_lines` | ✅ pass |
| Line endings | CRLF JSONL: messages keep 1-based line numbers and contain no stray \r. <br>`test_crlf_jsonl_line_numbers_and_content` | ✅ pass |
| Line endings | JSONL mixing LF, CRLF and a lone CR (old Mac) counts lines like an editor does. <br>`test_mixed_and_cr_only_jsonl_line_numbers` | ✅ pass |
| Line endings | Plain-text User:/Assistant: transcript with CRLF: turn line numbers are right, content has no \r. <br>`test_crlf_text_transcript_line_numbers` | ✅ pass |
| Line endings | Pretty-printed OpenAI JSON with CRLF: each message cites the line of its "role" key. <br>`test_crlf_pretty_openai_json_line_numbers` | ✅ pass |
| Line endings | Pokémon correction over CRLF/mixed sessions cites file:line that really holds the quoted text. <br>`test_citations_correct_for_crlf_sessions` | ✅ pass |
| Line endings | CRLF markdown: entry content has no \r and source points at the right line. <br>`test_crlf_markdown_entries_have_no_cr_and_right_source_lines` | ✅ pass |
| Line endings | A CRLF markdown file rewritten by a merge keeps CRLF on every line (no mixed endings introduced). <br>`test_crlf_markdown_rewrite_keeps_crlf` | ✅ pass |
| Line endings | An untouched CRLF file in a folder store is copied byte for byte. <br>`test_untouched_crlf_file_is_byte_identical` | ✅ pass |
| Line endings | A markdown file with mostly CRLF and one LF line is rewritten with CRLF only. <br>`test_mixed_line_endings_markdown_uses_majority` | ✅ pass |
| Line endings | Claude Code style front-matter file with CRLF: metadata values have no \r, body is the entry. <br>`test_crlf_front_matter` | ✅ pass |
| Timestamps | parse_timestamp returns a float or None for garbage, impossible dates and overflow; it never raises. <br>`test_parse_timestamp_never_raises` | ✅ pass (11/11) |
| Timestamps | Naive ISO timestamps are UTC; offsets are honoured, so mixed naive/aware values compare correctly. <br>`test_parse_timestamp_naive_is_utc_and_aware_respected` | ✅ pass |
| Timestamps | Sessions whose filenames are out of chronological order are sorted by first timestamp. <br>`test_sessions_ordered_by_timestamp_not_filename` | ✅ pass |
| Timestamps | Sessions with identical timestamps keep a deterministic (path) order. <br>`test_duplicate_session_timestamps_are_stable` | ✅ pass |
| Timestamps | A naive (UTC) and an offset timestamp are ordered by real instant. <br>`test_mixed_naive_and_aware_session_timestamps` | ✅ pass |
| Timestamps | If any session lacks a timestamp, order is natural path order (s2 before s10). <br>`test_missing_timestamps_fall_back_to_natural_path_order` | ✅ pass |
| Timestamps | A session with an unparsable timestamp ('yesterday') is treated like a missing one, not sorted last. <br>`test_unparsable_session_timestamp_counts_as_missing` | ✅ pass |
| Timestamps | A session whose first message has a junk timestamp but later ones are valid sorts by the first valid one. <br>`test_first_parsable_timestamp_is_used` | ✅ pass |
| Timestamps | Timestamps that are dicts, lists, booleans or huge numbers don't crash loading or the dream. <br>`test_weird_timestamp_types_in_transcripts` | ✅ pass |
| Timestamps | Newer dated note wins even when it comes first in the file. <br>`test_stale_replacement_uses_dates_not_file_order` | ✅ pass |
| Timestamps | Two conflicting notes with the same date: the later one in the store wins. <br>`test_stale_replacement_duplicate_dates_later_position_wins` | ✅ pass |
| Timestamps | JSON store: updated_at naive 10:00 vs -02:00 09:00 (= 11:00Z) -> the offset one is newer and wins. <br>`test_stale_replacement_naive_vs_aware_json_store` | ✅ pass |
| Timestamps | Entries with garbage updated_at fall back to store order (later wins) instead of crashing. <br>`test_unparsable_entry_timestamps_fall_back_to_order` | ✅ pass |
| Contradictions | Memory notes A (Jan) -> not-A (Feb) -> A (Mar): final value is A; history lists all three in order. <br>`test_memory_chain_a_not_a_a_final_is_a_with_full_history` | ✅ pass |
| Contradictions | The same A / not-A / A chain written in shuffled file order still resolves by date to A. <br>`test_memory_chain_out_of_file_order` | ✅ pass |
| Contradictions | The mirror chain not-A -> A -> not-A ends at not-A. <br>`test_memory_chain_not_a_a_not_a` | ✅ pass |
| Contradictions | Memory not-A, then sessions A / not-A / A: final A, history has the memory note and all three sessions. <br>`test_session_chain_across_sessions` | ✅ pass |
| Contradictions | Memory A, sessions not-A then A again: the latest stance equals the note, so nothing is rewritten. <br>`test_session_chain_ending_where_memory_started_is_unchanged` | ✅ pass |
| Contradictions | A reversal inside a single session (message 1 A, message 2 not-A) resolves to the later message. <br>`test_conflict_within_one_session` | ✅ pass |
| Contradictions | 'Use yarn for installing packages' vs later 'Never use yarn to install packages' is one temporal conflict. <br>`test_same_subject_different_wording` | ✅ pass |
| Contradictions | No memory note: sessions A -> not-A -> A produce one temporal insight whose history has three steps. <br>`test_transcript_only_chain_records_every_flip` | ✅ pass |
| Contradictions | A memory note newer than the contradicting transcript is flagged, not rewritten. <br>`test_newer_note_beats_older_transcript_evidence` | ✅ pass |
| Secrets | Planted secrets (nested JSON, URLs, base64, wrapped lines, filenames, metadata) never hit the model. <br>`test_nothing_secret_reaches_the_llm_backend` | ✅ pass |
| Secrets | With the deterministic engine, report.md / report.json / job.json contain none of the planted secrets. <br>`test_nothing_secret_in_reports_job_json` | ✅ pass |
| Secrets | A model merging entries whose metadata/keys hold secrets: report.json operations stay clean. <br>`test_secrets_not_in_reports_with_llm_echoing_metadata` | ✅ pass |
| Secrets | JSON map store with secrets in nested values and entry metadata: not sent to the model, not reported. <br>`test_nested_json_memory_values_and_metadata` | ✅ pass |
| Secrets | A phase that crashes with a secret in its message: job.json error and error.log are redacted. <br>`test_failed_job_error_and_error_log_are_redacted` | ✅ pass |
| Secrets | HookLog lines never contain secrets (hook errors may quote config or paths). <br>`test_hook_log_redacts` | ✅ pass |
| Secrets | Base64 blobs that decode to key=value secrets (or known key formats) are redacted as base64_secret. <br>`test_base64_encoded_secrets_are_redacted` | ✅ pass (5/5) |
| Secrets | Base64 of ordinary prose (no secret inside) is not redacted. <br>`test_base64_of_harmless_text_is_kept` | ✅ pass |
| Secrets | A key wrapped onto the next line has both halves redacted (prefix match + continuation line). <br>`test_secret_split_across_lines_both_halves_redacted` | ✅ pass |
| Secrets | The next line after a redacted key is only swallowed when it is a lone token, not a sentence. <br>`test_continuation_heuristic_leaves_prose_alone` | ✅ pass |
| Secrets | Escaped JSON inside a string ({\"api_key\": \"...\"}) is redacted. <br>`test_double_encoded_json_secret_redacted` | ✅ pass |
| Secrets | user:pass@ credentials and ?token= / ?access_token= / ?api_key= query values are redacted. <br>`test_url_credentials_and_query_tokens` | ✅ pass (6/6) |
| Secrets | Redacting already-redacted text changes nothing and counts nothing new. <br>`test_redaction_is_idempotent_on_planted_text` | ✅ pass |
| Model misbehaviour | Prose, empty, whitespace, broken / deeply nested JSON or JSON scalars -> deterministic fallback + warning. <br>`test_junk_or_empty_output_falls_back` | ✅ pass (9/9) |
| Model misbehaviour | JSON without a usable operations list (wrong key, wrong type, list of non-objects) -> fallback + warning. <br>`test_wrong_schema_falls_back` | ✅ pass (5/5) |
| Model misbehaviour | Ops with int targets, string evidence, word confidence, dict content: bad parts dropped with warnings. <br>`test_wrong_field_types_inside_operations` | ✅ pass |
| Model misbehaviour | Ops naming ids that don't exist are dropped with a warning; valid ops in the same reply are applied. <br>`test_unknown_ids_dropped_valid_ops_kept` | ✅ pass |
| Model misbehaviour | A correction citing a non-existent file:line is downgraded to a flag; an insight citing one is dropped. <br>`test_hallucinated_citations` | ✅ pass |
| Model misbehaviour | TimeoutError from the backend: retried llm_retries times with exponential back-off, then fallback. <br>`test_timeout_is_retried_with_backoff_then_falls_back` | ✅ pass |
| Model misbehaviour | llm_retries=0: one attempt per call, no sleeping. <br>`test_retries_disabled` | ✅ pass |
| Model misbehaviour | One timeout then a valid reply: the model's operations are used, no fallback. <br>`test_transient_error_then_success_uses_model_output` | ✅ pass |
| Model misbehaviour | OpenAI-compatible HTTP 429 with Retry-After: 7 -> waits 7s (fake) per retry, bounded, then fallback. <br>`test_http_429_retry_after_honoured_then_fallback` | ✅ pass |
| Model misbehaviour | A huge Retry-After (3600s) is capped at llm_retry_max_wait. <br>`test_http_429_retry_after_capped` | ✅ pass |
| Model misbehaviour | 429 once, then a valid completion: the model's answer is used. <br>`test_http_429_then_success` | ✅ pass |
| Model misbehaviour | Retry-After given as an HTTP date is converted to seconds (and never negative). <br>`test_retry_after_http_date` | ✅ pass |
| Model misbehaviour | HTTP 500 from the Anthropic / OpenAI API: retried with back-off, then deterministic fallback with warning. <br>`test_http_500_retried_then_fallback` | ✅ pass (2/2) |
| Model misbehaviour | A client error (HTTP 400) fails fast: no retries, straight to fallback. <br>`test_http_400_is_not_retried` | ✅ pass |
| Model misbehaviour | A socket timeout inside urlopen surfaces as a retryable BackendError. <br>`test_urlopen_socket_timeout_is_retryable` | ✅ pass |
| Model misbehaviour | claude-cli exiting 1 -> BackendError (not retried) -> fallback, warning quotes the exit code. <br>`test_claude_cli_nonzero_exit_falls_back` | ✅ pass |
| Model misbehaviour | claude-cli printing plain prose (not the JSON envelope) -> treated as model text -> fallback. <br>`test_claude_cli_non_json_stdout` | ✅ pass |
| Model misbehaviour | claude-cli JSON envelope with is_error=true -> fallback with the reported error. <br>`test_claude_cli_error_envelope` | ✅ pass |
| Model misbehaviour | claude-cli subprocess timeout -> BackendError, not retried (each attempt can take minutes). <br>`test_claude_cli_timeout_not_retried` | ✅ pass |
| Model misbehaviour | A backend returning None or bytes instead of str is handled as unusable output (fallback). <br>`test_backend_returning_non_string` | ✅ pass |
| LLM entry ids | Prompts show entries as '[MEMORY.md#1] text' (no id='...' that small models copy back verbatim). <br>`test_prompt_renders_ids_in_brackets` | ✅ pass |
| LLM entry ids | Section / date / order metadata is still rendered, on its own line, after the bracketed id. <br>`test_render_memory_metadata_line` | ✅ pass |
| LLM entry ids | Exact strings returned live by qwen2.5:1.5b ("id='MEMORY.md#1'", "id='id='MEMORY.md#6'") resolve; ops apply. <br>`test_qwen_style_ids_from_live_run_are_resolved` | ✅ pass |
| LLM entry ids | id= prefixes (repeated), quotes, backticks, brackets, trailing punctuation, copied text, case all resolve. <br>`test_normalize_and_resolve_variants` | ✅ pass (19/19) |
| LLM entry ids | normalize_entry_id peels nested id=/quote layers until stable. <br>`test_normalize_entry_id_strips_layers` | ✅ pass |
| LLM entry ids | A list given as one comma/semicolon-joined string (or a stringified list) is split into ids. <br>`test_comma_joined_target_lists` | ✅ pass (8/8) |
| LLM entry ids | A real id that contains a comma (JSON stores allow any id) is matched whole, not split. <br>`test_ids_containing_commas_resolve_exactly` | ✅ pass |
| LLM entry ids | 'project.md#1' resolves to 'notes/project.md#1' when unique; ambiguous basenames stay unresolved. <br>`test_basename_match_only_when_unique` | ✅ pass |
| LLM entry ids | Case-insensitive matching is used only when exactly one id matches. <br>`test_case_insensitive_only_when_unique` | ✅ pass |
| LLM entry ids | The same id given twice in different spellings counts once (a merge needs distinct ids). <br>`test_duplicate_targets_after_normalisation_are_collapsed` | ✅ pass |
| LLM entry ids | A reply mixing a mangled-but-resolvable id and a made-up id warns about the made-up one only. <br>`test_warning_only_for_truly_unresolvable_ids` | ✅ pass |
| LLM entry ids | Targets given as numbers, null or nested lists don't crash; unresolvable ones are reported once. <br>`test_numeric_and_null_targets` | ✅ pass |
| Crash / kill recovery | An exception before/after each pipeline phase: job failed, input hash unchanged, no report served. <br>`test_exception_at_each_phase` | ✅ pass (14/14) |
| Crash / kill recovery | Crash after the staged store is written but while writing report.md: job failed, store not presented. <br>`test_crash_between_store_and_report_write` | ✅ pass |
| Crash / kill recovery | `agentdream show` on a failed job prints one clean error line (exit 2). <br>`test_show_cli_refuses_failed_job` | ✅ pass |
| Crash / kill recovery | A real worker process SIGKILLed mid-dream (mid-analysis, or after write but before 'completed'): <br>`test_sigkill_mid_dream` | ✅ pass (2/2) |
| Crash / kill recovery | Exception inside the adapter write (after the backup exists): original intact, clear error, no temp leftovers. <br>`test_crash_during_write_after_backup` | ✅ pass |
| Crash / kill recovery | Exception while swapping the new store in (old store already moved aside): the original is put back. <br>`test_crash_during_swap_restores_original` | ✅ pass |
| Crash / kill recovery | Same as above for a single-file (JSON) store swapped with os.replace. <br>`test_crash_during_swap_single_file_store` | ✅ pass |
| Crash / kill recovery | After accept, `agentdream restore JOB` restores the backup byte-for-byte; the dream can be re-accepted. <br>`test_restore_command_puts_backup_back` | ✅ pass |
| Crash / kill recovery | restore(job, to=DIR) writes the backup to another location instead of the accepted target. <br>`test_restore_python_api_and_to` | ✅ pass |
| Crash / kill recovery | restore on a job accepted with --no-backup (or never accepted) fails with a clear message. <br>`test_restore_without_backup_is_a_clean_error` | ✅ pass |
| Crash / kill recovery | If the backup directory was deleted, restore says so instead of crashing. <br>`test_restore_with_missing_backup_files` | ✅ pass |
| Crash / kill recovery | A 'pending' job.json whose creator died is reported failed on the next load. <br>`test_job_never_started_then_process_died` | ✅ pass |
| Concurrency | Six threads dreaming over the same store at once: all complete, distinct ids, input intact. <br>`test_threads_dream_same_store` | ✅ pass |
| Concurrency | Several dream_async jobs started back to back all complete with consistent reports. <br>`test_async_jobs_same_store` | ✅ pass |
| Concurrency | Three CLI processes dreaming over the same store and jobs dir at the same time: all succeed. <br>`test_processes_dream_same_store` | ✅ pass |
| Concurrency | Two jobs created in the same second with the same random suffix still get distinct ids and dirs. <br>`test_job_id_collision_is_avoided` | ✅ pass |
| Concurrency | Two different dreams accepted onto the same store at once: exactly one applies, the other is refused <br>`test_concurrent_accepts_one_wins_other_refuses` | ✅ pass |
| Accept / partial accept | Store edited / file added / file deleted after the dream: accept refuses with a clear --force hint. <br>`test_stale_accept_refused_by_default` | ✅ pass (3/3) |
| Accept / partial accept | --force applies a stale dream anyway (with a backup). <br>`test_stale_accept_force_overrides` | ✅ pass |
| Accept / partial accept | --only with an unknown id: ReviewError naming it; store untouched, review stays pending. <br>`test_unknown_op_id_rejected_nothing_written` | ✅ pass |
| Accept / partial accept | Ids with path separators, globs or control characters are rejected like unknown ids. <br>`test_invalid_op_id_strings` | ✅ pass |
| Accept / partial accept | A prefix matching several ops is refused and the message lists the matches; a unique prefix works. <br>`test_ambiguous_prefix_lists_candidates` | ✅ pass |
| Accept / partial accept | --only a,a,a applies a once; no error, no duplicate in accepted_ops. <br>`test_duplicate_ids_applied_once` | ✅ pass |
| Accept / partial accept | --only '' (or only=[]) is refused instead of silently applying everything. <br>`test_empty_only_list_is_an_error` | ✅ pass |
| Accept / partial accept | --only A,B --exclude B applies exactly A. <br>`test_only_and_exclude_together` | ✅ pass |
| Accept / partial accept | --only A --exclude A leaves nothing to apply: refused, store untouched. <br>`test_only_and_exclude_cancel_out` | ✅ pass |
| Accept / partial accept | Excluding every operation is refused with a hint to use discard. <br>`test_exclude_everything_is_refused` | ✅ pass |
| Accept / partial accept | Selecting only flag operations (no store change) is fine: the store is rewritten identically. <br>`test_flag_only_selection_is_allowed` | ✅ pass |
| Accept / partial accept | Accepting twice, discarding after accept and accepting after discard are all refused. <br>`test_accept_twice_discard_after_accept` | ✅ pass |
| Accept / partial accept | A discarded dream cannot be accepted. <br>`test_accept_after_discard` | ✅ pass |
| Accept / partial accept | A truncated operations.json / snapshot.json gives a clean ReviewError, not a JSON traceback. <br>`test_corrupt_operations_file` | ✅ pass |
| Accept / partial accept | A corrupt job.json: `status JOB` fails cleanly; `list` skips it. <br>`test_corrupt_job_json` | ✅ pass |
| Permissions | --out under a read-only directory: job failed with 'Permission denied', CLI exit 1, input untouched. <br>`test_read_only_output_dir` | ✅ pass |
| Permissions | Jobs dir whose parent is read-only: one-line error (exit 2), no traceback, input untouched. <br>`test_read_only_jobs_dir` | ✅ pass |
| Permissions | If the job dir turns read-only while the dream runs, the API raises a clean OSError; input untouched. <br>`test_jobs_dir_becomes_read_only_mid_run` | ✅ pass |
| Permissions | accept --to inside a read-only directory: exit 2, clean message, review stays pending. <br>`test_accept_into_read_only_target_dir` | ✅ pass |
| Permissions | accept over the original store when its parent dir is read-only (can't swap): original intact. <br>`test_accept_over_store_in_read_only_parent` | ✅ pass |
| Permissions | A memory file with mode 000: job failed with a clear error naming the file. <br>`test_unreadable_memory_file` | ✅ pass |
| Permissions | An unreadable transcript: job failed with LoaderError naming the file (not a bare traceback). <br>`test_unreadable_session_file` | ✅ pass |
| Permissions | A read-only memory store can still be dreamed over (input is only read). <br>`test_read_only_store_dir_is_fine_for_dreaming` | ✅ pass |
| Permissions | `schedule --once` with a state file in a read-only dir: clean error (exit 2), no traceback. <br>`test_scheduler_state_in_read_only_dir` | ✅ pass |
| Path traversal | Entries whose path is ../.., absolute, Windows-style or '..' are written inside dest, never outside. <br>`test_markdown_write_keeps_hostile_paths_inside_dest` | ✅ pass (12/12) |
| Path traversal | A JSON map store whose keys / ids look like ../../etc/passwd: kept as data, all writes inside dest. <br>`test_json_store_with_traversal_ids_and_keys` | ✅ pass |
| Path traversal | JSON folder store: an entry whose path points outside is written to a file inside dest. <br>`test_json_folder_store_entry_paths_cannot_escape` | ✅ pass |
| Path traversal | A model naming '../../etc/passwd' as a target id: op dropped as unknown id, nothing written outside. <br>`test_llm_targets_with_traversal_ids_are_dropped` | ✅ pass |
| Path traversal | A symlink in the memory folder pointing to a file outside the store is not loaded or sent to the model, <br>`test_symlinked_file_outside_store_is_not_read` | ✅ pass |
| Path traversal | A symlinked directory pointing outside the store is not traversed. <br>`test_symlinked_dir_outside_store_is_not_followed` | ✅ pass |
| Path traversal | A symlink to another file *inside* the store is read normally. <br>`test_symlink_inside_store_is_fine` | ✅ pass |
| Path traversal | Job ids containing path separators, '..', globs or NUL are rejected before touching the filesystem. <br>`test_job_ids_with_traversal_rejected` | ✅ pass (11/11) |
| Path traversal | `agentdream show ../../x` and the MCP accept tool with job '../x' fail cleanly. <br>`test_cli_and_mcp_reject_traversal_job_ids` | ✅ pass |
| Path traversal | A job dir that is a symlink to somewhere outside the jobs dir is refused. <br>`test_job_dir_symlink_escape_rejected` | ✅ pass |
| Path traversal | --out equal to / inside the memory store, or an --out whose store/ would be the input, is refused. <br>`test_out_dir_overlapping_input_store_refused` | ✅ pass (3/3) |
| Scheduler | --idle 30m fires at exactly 30 min of inactivity (and after), not at 29m59s. <br>`test_idle_threshold_boundary` | ✅ pass (4/4) |
| Scheduler | Idle time is measured from the most recently modified pending session. <br>`test_idle_uses_latest_of_several_sessions` | ✅ pass |
| Scheduler | --new-sessions 3 with N-1 / N / N+1 new sessions. <br>`test_every_n_sessions_boundary` | ✅ pass (3/3) |
| Scheduler | After a dream, the counter restarts: sessions already dreamed over don't count again. <br>`test_every_n_sessions_counts_only_new_ones` | ✅ pass |
| Scheduler | --every 1h fires at exactly one hour after the last run (with pending sessions), not a second earlier. <br>`test_interval_boundary` | ✅ pass (3/3) |
| Scheduler | With only_new (default), an elapsed interval and no new sessions does nothing. <br>`test_interval_without_new_sessions_does_not_fire` | ✅ pass |
| Scheduler | Clock jumps back a day: the interval restarts from 'now' instead of stalling for a day. <br>`test_clock_going_backwards_resets_interval_baseline` | ✅ pass |
| Scheduler | A session modified 'in the future' (clock went backwards) is not considered idle; no crash. <br>`test_session_mtime_in_the_future_is_not_idle` | ✅ pass |
| Scheduler | A corrupt / truncated / wrong-typed state file doesn't crash the scheduler; it is rewritten valid. <br>`test_corrupt_state_file_starts_fresh` | ✅ pass (10/10) |
| Scheduler | A restarted scheduler (same state file) doesn't re-dream sessions already handled. <br>`test_state_survives_restart` | ✅ pass |
| Scheduler | A failed dream doesn't mark its sessions as dreamed; the next check retries them. <br>`test_failed_dream_keeps_sessions_pending` | ✅ pass |
| Scheduler | If the sessions directory is deleted while scheduling, checks return no trigger instead of raising. <br>`test_sessions_dir_disappears` | ✅ pass |
| Scheduler | Nonsense triggers (--every 0, --idle -5m, --new-sessions 0) are rejected with ValueError. <br>`test_negative_or_zero_triggers_rejected` | ✅ pass |
| Property-based (hypothesis) | Whatever text surrounds it (separated by whitespace), a planted secret never survives redaction. <br>`test_redaction_never_leaks_planted_secrets` | ✅ pass |
| Property-based (hypothesis) | redact(redact(x)) == redact(x), and the second pass counts nothing. <br>`test_redaction_is_idempotent` | ✅ pass |
| Property-based (hypothesis) | Any bytes, any extension, any forced format: a Transcript or LoaderError, never another exception. <br>`test_loaders_never_crash_on_arbitrary_bytes` | ✅ pass |
| Property-based (hypothesis) | JSON / JSONL with arbitrary shapes (wrong types everywhere) loads or raises LoaderError. <br>`test_loaders_never_crash_on_arbitrary_json_shapes` | ✅ pass |
| Property-based (hypothesis) | Memory stores with arbitrary bytes: entries or AdapterError, never another exception. <br>`test_memory_adapters_never_crash_on_arbitrary_bytes` | ✅ pass |
| Property-based (hypothesis) | Random memories: input bytes unchanged, output entries <= input + new insight notes, every id accounted for. <br>`test_dream_never_mutates_input_and_bounds_output` | ✅ pass |
| Property-based (hypothesis) | Transcript chunks: each within budget; non-overlap parts concatenate to exactly the input (in order); <br>`test_chunking_covers_every_message_once_and_respects_budget` | ✅ pass |
| Property-based (hypothesis) | Memory chunks: every entry (or each excerpt of an oversized one) appears exactly once, within budget. <br>`test_memory_chunking_covers_entries_and_respects_budget` | ✅ pass |
| Property-based (hypothesis) | accept --to reproduces the staged store; accept over the original + restore returns the exact bytes; <br>`test_accept_discard_restore_round_trip` | ✅ pass |
<!-- edge-case-table:end -->
