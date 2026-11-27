---
name: dream
description: Consolidate this project's agent memory with agentdream. Merges duplicate notes, replaces stale values, corrects notes that recent session transcripts contradict (with citations), and surfaces recurring patterns. Use when the user asks to dream, consolidate, clean up, deduplicate or fact-check memory, CLAUDE.md or memory notes, or asks what a past dream changed.
---

# Dream: consolidate agent memory

agentdream reads a memory store and recent session transcripts and stages a new store plus a
report. Nothing is changed until a dream is accepted, and every accept makes a backup first.

## 1. Check the tool

Run `agentdream --version`. If it is missing, tell the user to run `pip install agentdream` (or
`uv tool install agentdream`) and stop.

## 2. Find the memory and the sessions

Use what the user named. Otherwise:

* If `.agentdream.toml` exists (or `pyproject.toml` has `[tool.agentdream]`), `agentdream dream`
  needs no arguments; `agentdream config` shows the effective values.
* Sessions for this project are `~/.claude/projects/<project>/*.jsonl`, where `<project>` is the
  absolute project path with every `/` replaced by `-` (`/home/me/repo` becomes `-home-me-repo`).
  Confirm with `ls ~/.claude/projects`.
* Memory, in this order: `~/.claude/projects/<project>/memory/` if it exists, else `CLAUDE.md`,
  else `.claude/CLAUDE.md`.

If there are no session files, say so and stop: a dream needs transcripts to verify notes against.

## 3. Run a dry-run dream

```bash
agentdream dream --memory <memory> --sessions "<sessions-dir>/*.jsonl" --max-sessions 20
```

This prints a job id. It is a dry run: the memory store is not modified. The default backend is
rule-based and offline. Add `--backend claude-cli` only if the user asks for a model-backed dream;
it sends redacted memory and transcript text to the model through their Claude Code login.

## 4. Show the user what it found

```bash
agentdream show <job-id>
```

Summarise the report for the user, grouped by kind, each with its operation id:

* **merge** and **replace**: low risk. Say what was folded together or which value won.
* **correct** and **contradiction**: quote the before and after text and the cited transcript lines.
  Read the cited lines yourself and say whether the evidence really supports the change.
* **insight**: new notes. Say plainly if one looks like noise.

Report any warnings in the report, especially one saying an operation rewrites a note that held a
redacted secret: that operation must be excluded or the secret restored by hand.

## 5. Apply only what the user approves

Never accept without the user's explicit go-ahead in this conversation.

```bash
agentdream accept <job-id>                                   # everything
agentdream accept <job-id> --only <op-id>,<op-id>            # chosen operations
agentdream accept <job-id> --exclude <op-id>                 # everything but
agentdream accept <job-id> --types merge,replace --min-confidence 0.75   # low-risk only
agentdream discard <job-id>                                  # throw the dream away
agentdream restore <job-id>                                  # undo an accept from its backup
```

After an accept, report the number of operations applied and the backup path it printed. If accept
refuses because the store changed since the dream, re-run the dream rather than passing `--force`.

## Jobs created by the hook

If the project has the auto-dream hook installed, its jobs live in `.agentdream/jobs`. Add
`--jobs-dir .agentdream/jobs` to `list`, `show`, `accept`, `discard` and `restore`, and read
`.agentdream/hook.log` to see what the last background dream did.

## Setting up automatic dreams

Only when the user asks for it:

```bash
agentdream install-claude-hook --every 3          # dry runs after every third session
agentdream install-claude-hook --every 3 --safe   # also auto-apply merges and replacements;
                                                  # corrections and insights stay staged for review
agentdream uninstall-claude-hook
```
