# Does a dream help?

Two measurements, both scored by code and neither judged by a model:

1. **[Claude Code on real tasks](#claude-code-on-real-tasks)**: the same coding tasks run through the
   actual harness with the memory file before and after a dream.
2. **[The memory store itself](#the-memory-store-itself)**: a labelled store where every defect is
   known in advance, scored before and after.

## Claude Code on real tasks

```bash
python evals/claude_code/run.py --models sonnet,haiku,opus --repeats 3
```

The script generates a small Python project and a `CLAUDE.md` that has drifted in eight ways, plus
five earlier sessions, in Claude Code's own transcript format, in which the agent runs into each
problem and finds the truth:

| The drifted note | The truth, as the sessions show it |
|---|---|
| "Runtime settings live in `config/settings.json`" | That file is a leftover nothing reads; settings are in `orders/defaults.py` |
| "Test fixtures live in `tests/data/`" | There is no such directory; they are in `tests/fixtures/` |
| "Run the test suite with `make test`" (twice, in two wordings) | The Makefile was removed; it is `python -m unittest discover -s tests` |
| "Don't use dataclasses for new models" | The user reversed this: use dataclasses from now on |
| "Don't use the logging module" | The user reversed this too |
| "The staging database host is `db-staging-1.internal`" | Decommissioned; it is `db-staging-3.internal`. The code does not contain it |
| "The API server listens on port 8080" next to a newer note saying 9090 | 9090 |
| "CI fails below 80% coverage" next to a newer note saying 90% | 90% |

Two more notes are correct and stay that way, and two of the ten tasks depend only on those. They are
controls: a dream that damaged good notes would fail them.

agentdream dreams over that memory and those sessions and everything it proposes is accepted. Then
Claude Code runs headless (`claude -p`) on ten tasks in fresh copies of the project. The copies are
identical except for `CLAUDE.md`. Each run is checked by code: the function exists and returns the right
value, the tests pass, the setting the program actually reads has changed, the script names the right
host.

### Results

Three runs per task per memory file, 90 runs per model, 270 in all (Claude Code 2.1.332):

| Tasks passed, out of 30 | Drifted notes | After a rule-based dream (no model) | After a model dream |
|---|---|---|---|
| `claude-haiku-5-5` | 21 | 27 | 30 |
| `claude-sonnet-5-5` | 18 | 24 | 30 |
| `claude-opus-5-5` | 20 | 27 | 30 |
| **All three, out of 90** | **59** | **78** | **90** |

Per task, runs passed out of 9 (three models, three runs each):

| Task | What had drifted | Drifted | Rules | Model |
|---|---|---|---|---|
| Add an `Address` model | a preference the user reversed | 0 | 9 | 9 |
| Add a `cancel` function that records a message | a preference the user reversed | 0 | 9 | 9 |
| Print the staging database DSN | a value that changed, not in the code | 0 | 0 | 9 |
| Raise the request timeout | a note naming a file nothing reads | 5 | 6 | 9 |
| Add `total_with_tax`, run the tests | a command that no longer exists | 9 | 9 | 9 |
| Add a test fixture | a directory that does not exist | 9 | 9 | 9 |
| Write a health check | an old port next to the new one | 9 | 9 | 9 |
| Add a coverage threshold | an old value next to the new one | 9 | 9 | 9 |
| Add a `discount` function | control: the notes are right | 9 | 9 | 9 |
| Add an `item_count` function | control: the notes are right | 9 | 9 | 9 |

Pass and fail were the same in all three runs of 89 of the 90 cells. The exception is Opus on the
timeout task with the drifted notes, which passed twice and failed once.

Turns and cost did not move in a consistent direction:

| Mean turns per task | Drifted | Rules | Model |
|---|---|---|---|
| `claude-haiku-5-5` | 8.2 | 7.6 | 7.9 |
| `claude-sonnet-5-5` | 3.7 | 3.5 | 4.2 |
| `claude-opus-5-5` | 5.9 | 6.2 | 6.3 |

### What the runs show

* **Notes that state a preference or an outside fact were followed every time, by every model.** With
  the drifted notes, all 27 runs of the three tasks that depend on a reversed preference or on the
  decommissioned host failed. Nothing in the repository contradicts such a note, so no amount of care
  catches it. This is where a dream changes the outcome: 0 of 27 before, 27 of 27 after a model dream.
* **Notes about the code are a different story.** A wrong command, a missing directory, an old port
  beside the new one: every model checked the repository and recovered, in every run. Those notes
  cost a few turns at most and no failures.
* **The wrong-file note split the models.** Sonnet changed nothing the program reads in all three
  runs and stopped after three turns. Haiku and Opus looked further and mostly found the real setting
  (3 of 3 and 2 of 3).
* **The free, offline dream recovered 19 of the 31 failures,** 18 of them by flipping the two reversed
  preferences.
  It cannot read "that host was decommissioned" or "nothing reads that file" as a correction, so those
  need a model dream.
* **A dream did no harm.** Both controls passed in all 54 runs across the three memory files, and no
  task that passed with the drifted notes failed after either dream.
* **It did not make the agent faster or cheaper.** The gain is in tasks done correctly, not in turns.
  The model dream also added two insights, one telling the agent to verify notes before relying on
  them, and on Sonnet its runs were slightly longer.

The rule-based dream also left the 80% and 90% coverage notes side by side. All nine runs with that
file still picked 90%, so it cost nothing here, but it is a miss.

### Limits

* **Ten tasks, written to run into the drift.** This shows which kinds of wrong note cost a task and
  that a dream removes the cost. It does not estimate how often your notes are wrong, and so not how
  much a dream would help on your project.
* **One small project, one author.** The project, the notes, the sessions and the checks were written
  by the same hand as the tool. The sessions state each truth plainly; real transcripts are messier
  and a dream will recover less from them.
* **Three runs per cell.** Enough to see that results are stable, not enough for fine differences in
  turn counts.
* **Sandboxed runs.** User-level settings, plugins and MCP servers were switched off
  (`--setting-sources project --strict-mcp-config`) and shell access was limited to python, make and
  read-only commands, so some attempted commands were refused: 12 across Sonnet's 90 runs, 38 across
  Opus's and 98 across Haiku's, in all three memory files. That adds turns where it happened.
* **Pass means the check passed.** The checks are narrow by design (does the model use a dataclass,
  does the script name the right host). They do not grade code quality.

The raw rows, the three memory files and the operations each dream proposed are in
[`evals/claude_code/results.json`](../evals/claude_code/results.json). The 270 runs cost $15.95:
$0.45 on Haiku, $4.53 on Sonnet and $10.97 on Opus.

## The memory store itself

`agentdream benchmark` scores a dream on a store where every defect is known in advance, so the result
is scored by code, not judged by a person or a model.

```bash
agentdream benchmark                            # deterministic backend: no model, no network, ~0.1 s
agentdream benchmark --backend claude-cli       # the same workload through a model
agentdream benchmark --seed 3 --facts 12 --json # another draw, larger, machine-readable
agentdream benchmark --keep /tmp/bench          # keep the generated store, sessions and report to read
```

* [What is measured](#what-is-measured)
* [Results: deterministic backend](#results-deterministic-backend)
* [Results: a model backend](#results-a-model-backend)
* [A model reading the notes](#a-model-reading-the-notes)
* [What the benchmark found in agentdream itself](#what-the-benchmark-found-in-agentdream-itself)
* [What this does not tell you](#what-this-does-not-tell-you)

## What is measured

The workload is a markdown memory folder of 130 project notes and 50 session transcripts, generated
from a seed. It holds 80 facts in seven categories, ten of each (twenty controls), plus ten unrelated
notes:

| Category | What is in the store | What a good dream does |
|---|---|---|
| `duplicate` | The same note twice, word for word, in two files | Keeps one |
| `near_duplicate` | The same note twice with small wording differences | Keeps one |
| `stale` | Two dated notes giving different values | Keeps the newer value |
| `wrong` | A note the transcripts refute: the agent acts on it and fails twice, then finds the truth | Corrects it |
| `reversal` | A "don't use X" note that the user reverses in a later session | Flips it, keeping the history |
| `paraphrase` | One fact stated in two unrelated wordings | Keeps one (hard without a model) |
| `control` | Correct notes, half of them confirmed by a transcript | Leaves them alone |
| *unrelated notes* | Ten notes of identical shape about ten different things | Leaves every one alone |

Each fact has a probe that reads a store and returns the values it asserts for that fact. Two scores
come out of it:

* **Answered**: the store asserts exactly one value for the fact, and it is the true one. This is what
  an agent reading the store would get. A fact with two different values in the store is not answered,
  and neither is one whose only value is wrong.
* **Clean**: answered, and stated in a single note. Duplicates are answered but not clean.

The other rows are store size, the number of redundant notes, and how many of the unrelated notes
survive verbatim, which is the check that a dream did not destroy anything.

Each run is scored three times: the store as it was, the store after applying **every** proposed
operation, and the store after applying only what the hook's `--safe` policy applies on its own
(merges and replacements at confidence >= 0.75; see [CLAUDE_CODE.md](CLAUDE_CODE.md)).

## Results: deterministic backend

Version 0.1.1,
`agentdream benchmark`, seed 7, Python 3.14 on an Apple-silicon laptop, 0.12 s:

| Metric | Before | After (all operations) | After (safe auto-apply) |
|---|---|---|---|
| Facts answered (one value, the true one) | 50/80 (62%) | 80/80 (100%) | 60/80 (75%) |
| Facts clean (answered, in a single note) | 20/80 (25%) | 70/80 (88%) | 50/80 (62%) |
| Redundant notes | 40 | 10 | 10 |
| Notes in the store | 130 | 101 | 100 |
| Store size (estimated tokens) | 2049 | 2090 | 1597 |
| Unrelated notes kept verbatim | 10/10 | 10/10 | 10/10 |

| Category | Facts | Clean before | Clean after (all) | Clean after (safe) |
|---|---|---|---|---|
| duplicate | 10 | 0 | 10 | 10 |
| near_duplicate | 10 | 0 | 10 | 10 |
| stale | 10 | 0 | 10 | 10 |
| wrong | 10 | 0 | 10 | 0 |
| reversal | 10 | 0 | 10 | 0 |
| paraphrase | 10 | 0 | 0 | 0 |
| control | 20 | 20 | 20 | 20 |

Seeds 1 to 5 give the same answered and clean counts (80 and 70 with all operations, 60 and 50 with
the safe policy), with all controls and all unrelated notes intact in every run.

Reading it:

* **Applying everything** takes the share of facts an agent would read correctly from 62% to 100%.
  It did not damage a control or an unrelated note.
* **The safe policy** gets the duplicates and the stale values, which is every defect that can be fixed
  from the notes alone, and leaves the twenty transcript-based rewrites staged. That is the intended
  trade: it reaches 75% answered with no rewrite that rests on a reading of a transcript.
* **Paraphrases are not found.** The rule-based engine matches on shared words, so two unrelated
  wordings of one fact stay as two notes. This is what a model backend is for.
* **The store does not get smaller when everything is applied** (2049 to 2090 estimated tokens),
  although it loses 29 notes. A reversed note keeps a one-line history of what it said before and the
  transcript line that reversed it, and here that line is a long temporary path. Under the safe policy,
  which applies no reversals, the store shrinks by 22%.

## Results: a model backend

The same workload and seed through `--backend claude-cli`, which runs the default model of the local
Claude Code login. One run, 143 s, no fallback to the rule-based engine:

| Metric | Before | After (all operations) | After (safe auto-apply) |
|---|---|---|---|
| Facts answered (one value, the true one) | 50/80 (62%) | 80/80 (100%) | 60/80 (75%) |
| Facts clean (answered, in a single note) | 20/80 (25%) | 80/80 (100%) | 60/80 (75%) |
| Redundant notes | 40 | 0 | 0 |
| Notes in the store | 130 | 92 | 90 |
| Store size (estimated tokens) | 2049 | 1757 | 1448 |
| Unrelated notes kept verbatim | 10/10 | 10/10 | 10/10 |

Side by side, with every operation applied:

| | Deterministic | claude-cli |
|---|---|---|
| Facts answered | 80/80 | 80/80 |
| Facts clean | 70/80 | 80/80 |
| Paraphrased duplicates merged | 0/10 | 10/10 |
| Controls and unrelated notes intact | 30/30 | 30/30 |
| Operations proposed | 51 | 62 |
| Store size after (estimated tokens) | 2090 | 1757 |
| Time | 0.12 s | 143 s |
| Cost | none | model calls on your login |

The model's whole advantage on this workload is the paraphrases: it merged all ten pairs the rules
missed, which also makes the store 14% smaller than it started. On every other category the two agree.
It took about a thousand times longer.

This is a single run of one model, and model output varies between runs, so read it as "a capable
model can do this", not as a guaranteed score. Run it yourself with `--backend claude-cli`,
`--backend anthropic`, or `--backend ollama --model <name>`; a small local model will score lower.

## A model reading the notes

The scores above come from probes, which are regular expressions. `--ask` puts a model in the agent's
place instead: it is given the notes and nothing else, and one question per fact ("Which port does the
staging gateway service listen on?", "Should Poetry be used for dependency management?"). Its answers
are checked by code.

```bash
agentdream benchmark --ask claude-cli --repeats 2
```

Rule-based dream, read by the default model of a Claude Code login, two runs per store:

| | Before | After (all operations) | After (safe auto-apply) |
|---|---|---|---|
| Questions answered correctly | 60 / 80 | 80 / 80 | 60 / 80 |
| Notes sent to the model (estimated tokens) | 3697 | 3730 | 3233 |

| Category | Questions | Correct before | After (all) | After (safe) |
|---|---|---|---|---|
| duplicate, near_duplicate, paraphrase | 30 | 30 | 30 | 30 |
| stale | 10 | 10 | 10 | 10 |
| wrong | 10 | 0 | 10 | 0 |
| reversal | 10 | 0 | 10 | 0 |
| control | 20 | 20 | 20 | 20 |

Both runs of every store gave the same counts. Three things follow:

* **A capable model is not confused by duplicates or by two dated values.** It answered all 40 of those
  questions correctly before any dream, picking the newer date itself. The probe-based score counts a
  stale pair as unanswered; for this reader that is too pessimistic.
* **It is misled by every wrong note.** All 20 questions about a refuted note or a reversed preference
  were answered wrongly before the dream and correctly after. The notes are the only source it has,
  and nothing in a wrong note says it is wrong.
* **The safe policy does not improve this reader's answers.** It applies merges and replacements, which
  this reader did not need. What it buys is a smaller store, 13% fewer tokens here, on every session.
  The answers change only when the transcript-based corrections are accepted.

## What the benchmark found in agentdream itself

Writing the benchmark exposed two defects in the deterministic engine of the published 0.1.0. Both are
fixed in 0.1.1, each with a regression test in `tests/test_auto_apply.py`.

**Notes of the same shape about different things were merged.** The duplicate rule accepted any two
notes with enough shared words. Ten notes reading *"The on-call rotation for &lt;team&gt; changes every
Monday"* differ in two words out of nine, so nine of them were folded into one, at confidence 0.85.
The same happened to *"Deploys of the &lt;name&gt; component go through the `stable` branch"* and to
*"Don't use &lt;tool&gt; for test environments"*. On this workload 0.1.0 scored:

| Metric | 0.1.0 as published | With the fix |
|---|---|---|
| Unrelated notes kept verbatim | 2/10 | 10/10 |
| Facts answered, all operations | 69/80 | 80/80 |
| Facts clean, all operations | 66/80 | 70/80 |
| Merge operations proposed | 26 | 20 |

This is the case that matters most for automatic updates, because a merge is exactly what the safe
policy applies without review. The rule now refuses to merge two notes that line up word for word
except for a swap of unrelated words on both sides. Insertions (*"before committing"* /
*"before every commit"*) and spelling variants still merge.

**A correction could overwrite a direction word that was part of a name.** *"The kiosk is north of the
west stairwell"*, corrected from transcripts that say south, became *"south of the south stairwell"*.
A value word that the transcripts state right next to the new value is now treated as part of the
surroundings and kept.

Both were found because the workload contains controls. A benchmark with only defects to fix would
have scored the false merges as a smaller store.

## What this does not tell you

* **The store score is not a measurement of agent performance.** "Answered" means an agent reading the
  store finds one correct value. For what happens when a real agent works from the notes, see
  [Claude Code on real tasks](#claude-code-on-real-tasks) above.
* **The deterministic numbers are an upper bound.** The notes are template-generated plain English in
  the sentence shapes the rule-based engine parses, and the engine was fixed against this workload.
  Expect lower recall on notes written the way people actually write them. The `paraphrase` row, at
  zero, is the honest indicator of what happens once wording varies.
* **Precision is only checked against this store's controls.** Thirty notes that must not change
  survived. That is evidence the two defects above are fixed, not proof that no other false merge
  exists. Review staged operations, and keep the backups an accept makes.
* **Token counts are estimates** (about four characters per token), as everywhere in agentdream.

To score a dream on your own notes, there is no ground truth to compare against, so read the report:
`agentdream dream` on a copy of your store, then `agentdream show <job-id>`, and check whether each
merge really joins two statements of one fact.
