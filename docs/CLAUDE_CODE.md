# Claude Code integration

Three independent ways to use agentdream from Claude Code. Use any one, or all three:

1. **`/dream` skill** — type `/dream` (or ask "clean up my memory") and Claude runs a dry-run dream,
   walks you through the report and applies only what you approve. The quickest way to start.
2. **Hook** — a dream runs automatically in the background after sessions end. Opt-in, a dry run by
   default; `--safe` also keeps the memory up to date by applying the low-risk changes on its own.
3. **MCP server** — Claude gets `dream` / `show_report` / `accept` as tools, for clients without skills.

## Setup in two minutes

```bash
pip install agentdream                              # or: uv tool install agentdream
cd ~/code/myrepo
agentdream install-claude-skill                     # adds .claude/skills/dream/SKILL.md
agentdream install-claude-hook --every 3 --safe     # optional: keep memory up to date automatically
```

Then, in Claude Code (restart a running session so it picks the skill up):

```text
/dream
```

Claude finds this project's memory and recent sessions, runs `agentdream dream` (a dry run), and
summarises the report: which duplicates it would merge, which stale values it would replace, and which
notes the transcripts contradict, with the cited lines. Nothing is written until you say so; then it
runs `agentdream accept` for the operations you approved. `agentdream restore <job-id>` undoes it.

Which one should you use?

| You want | Use |
|---|---|
| To clean up memory now and see every change first | `/dream` |
| Memory that stays tidy without you thinking about it | the hook with `--safe`, plus `/dream` now and then to review what it staged |
| Every change reviewed, but the dream already run when you look | the hook without `--safe` (dry runs), then `/dream` or `agentdream show` |
| A client that has MCP but no skills (Cursor, others) | the MCP server |

## Where Claude Code keeps memory and sessions

* Transcripts: `~/.claude/projects/<project>/<session-id>.jsonl`, where `<project>` is derived from the
  project path with separators replaced by `-` (e.g. `/home/me/myrepo` → `-home-me-myrepo`; check
  `ls ~/.claude/projects` for the exact name). The hook doesn't need to know it: Claude Code passes the
  transcript path. agentdream auto-detects the format:
  text blocks are kept, tool calls are reduced to one line (`[tool] Bash: pytest -q`), failed tool results
  are kept (they reveal recurring mistakes), thinking blocks and successful tool output are dropped.
* Memory: `CLAUDE.md` in the repo (or `.claude/CLAUDE.md`), and Claude Code's per-project memory folder
  `~/.claude/projects/<project>/memory/` when present. Markdown files with front matter are one entry each;
  other markdown files are one entry per bullet / paragraph.

Dream over them by hand:

```bash
PROJ=~/.claude/projects/-home-me-myrepo
agentdream dream --memory CLAUDE.md --sessions "$PROJ/*.jsonl" --max-sessions 20
agentdream show <job-id>
agentdream accept <job-id>                  # backs up CLAUDE.md first
agentdream dream --memory "$PROJ/memory" --sessions "$PROJ" --backend claude-cli   # use your Claude login as the model
```

## 1. The `/dream` skill

```bash
agentdream install-claude-skill             # this project: .claude/skills/dream/SKILL.md
agentdream install-claude-skill --user      # every project: ~/.claude/skills/dream/SKILL.md
agentdream uninstall-claude-skill           # remove it (same --user flag)
```

The skill is a markdown file of instructions, not code: it tells Claude how to locate the memory store
and the session files, to run the dream as a dry run, to check the cited transcript lines before
recommending a correction, and never to run `accept` without your go-ahead. Running the installer again
after upgrading agentdream updates the file. A `/dream` skill that is not agentdream's is left alone
unless you pass `--force`; uninstall only ever removes agentdream's own file.

Commit `.claude/skills/dream/` if you want teammates to have it; they need `agentdream` installed too.

Things to ask once it is installed:

* `/dream` — dream over the default memory and the last 20 sessions.
* *"/dream over CLAUDE.md, only merges and stale values"* — the skill passes your scope along.
* *"What did last night's dream change?"* — it reads `.agentdream/hook.log` and the latest job.
* *"Undo the last dream"* — `agentdream restore`.

## 2. MCP server

```bash
claude mcp add agentdream -- agentdream serve-mcp                   # user-wide
claude mcp add --scope project agentdream -- agentdream serve-mcp   # this repo (writes .mcp.json)
claude mcp add agentdream -- python -m agentdream serve-mcp          # if agentdream isn't on PATH
```

Tools:

| Tool | Arguments | Does |
|---|---|---|
| `dream` | `memory`, `sessions`, `backend`, `model`, `instructions`, `out`, `wait` | Runs a dry-run dream; returns status, counts and the operation list. Missing arguments come from `.agentdream.toml`. |
| `dream_status` | `job_id` | Status / phase / progress of a job (use with `wait: false`). |
| `show_report` | `job_id`, `format` (`markdown` / `json`) | The report. |
| `accept` | `job_id`, `only`, `exclude`, `to` | Applies (part of) a dream with a backup. |
| `discard` | `job_id` | Deletes the staged output. |
| `list_dreams` | — | All jobs. |

Example prompt: *"Dream over CLAUDE.md using my last 10 sessions in ~/.claude/projects/-home-me-myrepo,
show me the report, and accept only the corrections."*

## 3. Auto-dream hook

```bash
cd ~/code/myrepo
agentdream install-claude-hook                          # SessionEnd, every session, dry run
agentdream install-claude-hook --every 3                # every third session
agentdream install-claude-hook --memory notes/agent-memory --every 5
agentdream install-claude-hook --local                  # .claude/settings.local.json (just you)
agentdream install-claude-hook --user                   # ~/.claude/settings.json (all projects)
agentdream install-claude-hook --safe                   # auto-apply merges and replacements only (see below)
agentdream install-claude-hook --apply                  # apply everything automatically (backups are kept)
agentdream install-claude-hook --event Stop             # after every response instead of at session end
agentdream uninstall-claude-hook                        # remove it again (same --local / --user flags)
```

This adds an entry like the following to the settings file (other settings and hooks are left as they
are; the previous file is saved as `settings.json.agentdream-backup-<timestamp>`; running the installer
again updates the entry instead of adding a second one):

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {"type": "command", "command": "/usr/bin/python3 -m agentdream hook --every 3", "timeout": 30}
        ]
      }
    ]
  }
}
```

The command uses the absolute path of the Python that ran the installer, so it works even if
`agentdream` isn't on Claude Code's `PATH`.

### What happens at the end of a session

1. Claude Code runs the command and passes JSON on stdin: `session_id`, `transcript_path`, `cwd`,
   `hook_event_name` (and `reason` for SessionEnd).
2. `agentdream hook` records the session in `<project>/.agentdream/hook-state.json` and returns
   immediately (~0.1 s) unless at least `every` sessions are waiting.
3. When enough sessions are waiting it starts a **detached** worker process and exits. The worker:
   * takes `<project>/.agentdream/dream.lock` (if another dream for this project is running it just exits;
     it re-checks for new sessions when done);
   * picks the memory store: `--memory` / `[hook] memory` / `memory` in the config, else
     `~/.claude/projects/<project>/memory/`, else `CLAUDE.md`, else `.claude/CLAUDE.md`;
   * dreams over the waiting sessions with the configured backend into `<project>/.agentdream/jobs/`;
   * **does not apply anything** unless `--apply` (or `[hook] apply = true`) was configured;
   * marks the sessions as dreamed (a failed dream is retried with the next session, up to three times).
4. Everything is logged to `<project>/.agentdream/hook.log`.

Review what it did:

```bash
tail .agentdream/hook.log
agentdream list --jobs-dir .agentdream/jobs
agentdream show <job-id> --jobs-dir .agentdream/jobs
agentdream accept <job-id> --jobs-dir .agentdream/jobs
```

Add `.agentdream/` to your `.gitignore`.

### Keeping memory up to date automatically

By default the hook only stages dreams. There are two ways to let it write:

| Mode | Flag | Applied without you | Left staged for review |
|---|---|---|---|
| Review everything | *(none)* | nothing | every operation |
| Safe auto-update | `--safe` | duplicate merges and stale-value replacements at confidence >= 0.75 | corrections, stance reversals, insights |
| Apply everything | `--apply` | every operation | nothing |

`--safe` is short for `--apply --apply-types merge,replace --apply-min-confidence 0.75`. The split
follows what each kind of operation is based on. A merge or a replacement only reorganises what your
notes already say: the same sentence twice, or two dated values where the newer one wins. A correction
or a reversal rewrites a note on the strength of what a transcript appears to show, and an insight adds
a note nobody wrote, so those wait for a person. You can set your own policy with the two flags, or in
the config file:

```toml
[hook]
every = 3
apply = true
apply_types = ["merge", "replace"]
apply_min_confidence = 0.75
```

What an automatic update does and does not do:

* A backup of the store is taken before every write, in `.agentdream/jobs/<job-id>/backups/`;
  `agentdream restore <job-id> --jobs-dir .agentdream/jobs` puts it back.
* The job is marked `partially_accepted`, and `hook.log` says how many operations were applied and how
  many were left staged, with the command to review them.
* If the memory store changed while the dream ran, nothing is applied (the next dream picks it up).
* If no operation meets the policy, nothing is written and the dream stays pending.
* An operation that depends on a staged one is staged with it, so the store is never left half-updated.

The same policy works by hand on any dream:

```bash
agentdream accept <job-id> --types merge,replace --min-confidence 0.75
```

How much each mode fixes on a labelled store is measured in [BENCHMARK.md](BENCHMARK.md).

### Guarantees

* **Opt-in**: nothing runs until you install the hook; uninstalling removes only agentdream's entry.
* **Dry run by default**: dreams are staged for review; your memory is only changed with `--safe` or `--apply`.
* **Never blocks or fails a session**: the hook catches every error (including bad arguments — exit code 2
  would block a `Stop` hook) and always exits 0; the dream runs detached from Claude Code's process group.
* **No double runs**: a state lock serialises session bookkeeping across concurrent hooks and a dream lock
  allows one dream per project at a time. Locks are OS-level (`flock` / `msvcrt`) and released if a
  process dies.
* **SessionEnd vs Stop**: `SessionEnd` fires once per session and is the recommended event. `Stop` fires
  after every response; sessions are counted once (by `session_id`), and a dream triggered mid-session sees
  the transcript up to that point.

### Configuration

Hook defaults can live in `.agentdream.toml` (see [CONFIG.md](CONFIG.md)); flags in the installed command
take precedence:

```toml
backend = "claude-cli"          # use your Claude Code login for hook dreams

[hook]
every = 3
apply = false
memory = "CLAUDE.md"
```

Environment overrides: `AGENTDREAM_HOOK_EVERY`, `AGENTDREAM_HOOK_APPLY`, `AGENTDREAM_HOOK_MEMORY`,
`AGENTDREAM_HOOK_APPLY_TYPES` (comma-separated), `AGENTDREAM_HOOK_APPLY_MIN_CONFIDENCE`.
`AGENTDREAM_HOOK_SYNC=1` runs the dream in the foreground (for debugging the hook by hand):

```bash
echo '{"session_id": "test", "transcript_path": "'"$HOME"'/.claude/projects/-home-me-myrepo/abc.jsonl", "cwd": "'"$PWD"'"}' \
  | AGENTDREAM_HOOK_SYNC=1 agentdream hook --every 1
```
