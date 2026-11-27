"""``agentdream`` command line.

Option values come from (highest first) CLI flags, ``AGENTDREAM_*`` environment variables,
the project config file (``.agentdream.toml`` / ``[tool.agentdream]``), the user config file
and built-in defaults; see :mod:`agentdream.config`. ``agentdream config`` shows the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--jobs-dir", help="where job state lives (default: [jobs_dir] from config, $AGENTDREAM_HOME/jobs "
                                      "or ./.agentdream/jobs)")
    p.add_argument("--config", help="config file to use instead of the nearest .agentdream.toml / pyproject.toml")


def _dream_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--memory", "-m", help="memory store: markdown folder/file, JSON/JSONL, SQLite, stancelog:<path> "
                                          "(or `memory` in the config file)")
    p.add_argument("--sessions", "-s", nargs="+", default=None, help="transcript files, directories or globs")
    p.add_argument("--backend", "-b", default=None,
                   help="deterministic | anthropic | openai | ollama | claude-cli (default: deterministic)")
    p.add_argument("--model", help="model name for LLM backends")
    p.add_argument("--base-url", help="base URL for the openai-compatible backend (Ollama, LM Studio, vLLM)")
    p.add_argument("--instructions", "-i", help="free-text focus for the dream")
    p.add_argument("--memory-format", help="force memory format (markdown|json|sqlite|stancelog)")
    p.add_argument("--transcript-format", help="force transcript format (claude-code|jsonl|openai|text)")
    p.add_argument("--max-sessions", type=int, default=None, help="default 100")
    p.add_argument("--max-bytes-per-transcript", type=int, default=None, help="default 2000000")
    p.add_argument("--max-total-tokens", type=int, default=None, help="default 400000")
    p.add_argument("--on-limit", choices=["truncate", "fail"], default=None, help="default truncate")
    p.add_argument("--max-chunk-tokens", type=int, default=None,
                   help="LLM backends: token budget per model call; larger inputs are chunked (default 60000)")
    p.add_argument("--chunk-overlap", type=int, default=None,
                   help="tokens of transcript context repeated between consecutive chunks (default 400)")
    p.add_argument("--max-chunks", type=int, default=None, help="max model calls per phase (default 64)")
    p.add_argument("--no-redact", action="store_true", default=None,
                   help="disable secret redaction (not recommended)")
    p.add_argument("--redact-emails", action="store_true", default=None)
    p.add_argument("--no-redact-entropy", action="store_true", default=None,
                   help="don't redact long random-looking tokens (known secret formats are still redacted)")
    p.add_argument("--entropy-threshold", type=float, default=None,
                   help="bits/char for the high-entropy check (default 4.0; lower = more aggressive)")
    p.add_argument("--redact-pattern", action="append", default=None, metavar="REGEX",
                   help="extra regex to redact (repeatable)")
    p.add_argument("--apply", action="store_true", default=None,
                   help="accept all operations right away (backup is still made)")
    _common(p)


def cli_dream_values(a: argparse.Namespace) -> dict[str, Any]:
    """Only the DreamConfig fields given explicitly on the command line."""
    vals: dict[str, Any] = {
        "memory": a.memory, "sessions": a.sessions, "backend": a.backend, "model": a.model,
        "instructions": a.instructions, "memory_format": a.memory_format, "transcript_format": a.transcript_format,
        "max_sessions": a.max_sessions, "max_bytes_per_transcript": a.max_bytes_per_transcript,
        "max_total_tokens": a.max_total_tokens, "on_limit": a.on_limit, "max_chunk_tokens": a.max_chunk_tokens,
        "chunk_overlap": a.chunk_overlap, "max_chunks": a.max_chunks,
        "redact": False if a.no_redact else None, "redact_emails": a.redact_emails,
        "redact_entropy": False if a.no_redact_entropy else None, "entropy_threshold": a.entropy_threshold,
        "redact_patterns": a.redact_pattern, "jobs_dir": a.jobs_dir, "out": getattr(a, "out", None),
    }
    if a.base_url:
        vals["backend_options"] = {"base_url": a.base_url}
    for k in ("memory", "jobs_dir", "out"):
        v = vals.get(k)
        if v and not str(v).startswith("stancelog:"):
            vals[k] = str(Path(os.path.expanduser(str(v))).absolute())
    return {k: v for k, v in vals.items() if v is not None}


def _scope_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--project", dest="scope", action="store_const", const="project",
                   help="this project's .claude/settings.json (default)")
    g.add_argument("--local", dest="scope", action="store_const", const="local",
                   help="this project's .claude/settings.local.json (personal, usually not committed)")
    g.add_argument("--user", dest="scope", action="store_const", const="user", help="~/.claude/settings.json")
    p.add_argument("--settings", help="explicit settings.json path (overrides the scope)")
    p.set_defaults(scope="project")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agentdream", description="Sleep-time memory consolidation for any agent.")
    ap.add_argument("--version", action="version", version=f"agentdream {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dream", help="run a dream (dry run: stages a new store + report)")
    _dream_args(p)
    p.add_argument("--out", "-o", help="output directory for the new store + report")
    p.add_argument("--json", action="store_true", help="print the job as JSON")

    p = sub.add_parser("status", help="show job status (latest job if none given)")
    p.add_argument("job", nargs="?")
    p.add_argument("--json", action="store_true")
    _common(p)

    p = sub.add_parser("list", help="list dream jobs")
    p.add_argument("--json", action="store_true")
    _common(p)

    p = sub.add_parser("show", help="print a job's report")
    p.add_argument("job")
    p.add_argument("--json", action="store_true", help="print report.json instead of report.md")
    p.add_argument("--diff", action="store_true", help="print only the diff")
    _common(p)

    p = sub.add_parser("accept", help="apply a dream (backup of the original first)")
    p.add_argument("job")
    p.add_argument("--only", help="comma-separated operation ids to apply")
    p.add_argument("--exclude", help="comma-separated operation ids to skip")
    p.add_argument("--to", help="write the result here instead of over the original store")
    p.add_argument("--types", help="apply only these operation types, comma-separated "
                                   "(merge, replace, correct, insight, contradiction)")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="apply only operations at or above this confidence (0-1)")
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--force", action="store_true", help="apply even if the store changed since the dream")
    _common(p)

    p = sub.add_parser("discard", help="discard a dream's staged output")
    p.add_argument("job")
    _common(p)

    p = sub.add_parser("restore", help="put back the backup an accept made (undo an accept)")
    p.add_argument("job")
    p.add_argument("--to", help="restore the backup here instead of over the accepted store")
    _common(p)

    p = sub.add_parser("cancel", help="request cancellation of a running dream")
    p.add_argument("job")
    _common(p)

    p = sub.add_parser("schedule", help="dream automatically (idle / interval / every N sessions)")
    _dream_args(p)
    p.add_argument("--every", help="interval, e.g. 30m, 6h, 1d")
    p.add_argument("--idle", help="dream after this much inactivity in new transcripts, e.g. 30m")
    p.add_argument("--new-sessions", "--every-sessions", dest="new_sessions", type=int,
                   help="dream every N new sessions")
    p.add_argument("--poll", default=None, help="how often to check triggers (default 60s)")
    p.add_argument("--once", action="store_true", help="check triggers once and exit")
    p.add_argument("--state-file")

    p = sub.add_parser("serve-mcp", help="run the MCP stdio server")
    _common(p)

    p = sub.add_parser("config", help="show the effective configuration and where each value comes from")
    p.add_argument("--json", action="store_true")
    _common(p)

    def hook_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--memory", "-m", help="memory store (default: [hook] memory, `memory`, then the project's "
                                              "Claude memory: ~/.claude/projects/<p>/memory/, CLAUDE.md)")
        p.add_argument("--every", type=int, default=None, help="dream every N sessions (default 1)")
        p.add_argument("--apply", action="store_true", default=None,
                       help="apply the dream automatically (default: dry run, review with `agentdream show`)")
        p.add_argument("--apply-types", default=None,
                       help="with --apply: only apply these operation types (comma-separated); the rest stays staged")
        p.add_argument("--apply-min-confidence", type=float, default=None,
                       help="with --apply: only apply operations at or above this confidence (0-1)")
        p.add_argument("--backend", "-b", default=None)
        p.add_argument("--model", default=None)
        _common(p)

    p = sub.add_parser("hook", help="Claude Code hook entry point (reads hook JSON on stdin; always exits 0)")
    hook_opts(p)
    p = sub.add_parser("hook-worker", help=argparse.SUPPRESS)
    hook_opts(p)
    p.add_argument("--project", required=True)

    p = sub.add_parser("install-claude-hook", help="dream automatically after Claude Code sessions (opt-in)")
    _scope_args(p)
    p.add_argument("--memory", "-m", help="memory store the hook dreams over (default: the project's Claude memory)")
    p.add_argument("--every", type=int, default=None, help="dream every N sessions (default 1)")
    p.add_argument("--apply", action="store_true",
                   help="apply dreams automatically (a backup is still made). Default: dry run only")
    p.add_argument("--safe", action="store_true",
                   help="auto-apply only the low-risk changes: --apply --apply-types merge,replace "
                        "--apply-min-confidence 0.75; corrections and insights stay staged for review")
    p.add_argument("--apply-types", default=None,
                   help="with --apply: only apply these operation types (comma-separated)")
    p.add_argument("--apply-min-confidence", type=float, default=None,
                   help="with --apply: only apply operations at or above this confidence (0-1)")
    p.add_argument("--event", choices=["SessionEnd", "Stop"], default="SessionEnd",
                   help="SessionEnd (default, once per session) or Stop (after every response)")
    p.add_argument("--backend", "-b", default=None, help="backend for hook dreams (default: from config)")
    p.add_argument("--model", default=None)
    p.add_argument("--config", help="config file the hook should use")

    p = sub.add_parser("uninstall-claude-hook", help="remove the agentdream hook from Claude Code settings")
    _scope_args(p)

    for name, text in (("install-claude-skill", "add the /dream skill to Claude Code (.claude/skills/dream)"),
                       ("uninstall-claude-skill", "remove the /dream skill again")):
        p = sub.add_parser(name, help=text)
        g = p.add_mutually_exclusive_group()
        g.add_argument("--project", dest="scope", action="store_const", const="project",
                       help="this project's .claude/skills (default)")
        g.add_argument("--user", dest="scope", action="store_const", const="user", help="~/.claude/skills")
        p.set_defaults(scope="project")
        if name.startswith("install"):
            p.add_argument("--force", action="store_true", help="replace a /dream skill that is not agentdream's")

    p = sub.add_parser("benchmark", help="score a dream on a labelled synthetic store (before / after)")
    p.add_argument("--backend", "-b", default="deterministic")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None, help="base URL for an OpenAI-compatible backend")
    p.add_argument("--facts", type=int, default=10, help="facts per category (default 10; 7 categories)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--keep", help="write the generated store, sessions and staged output here")
    p.add_argument("--ask", metavar="BACKEND", default=None,
                   help="also ask a model one question per fact, given only the notes, before and after the dream "
                        "(e.g. claude-cli, anthropic, ollama)")
    p.add_argument("--ask-model", default=None, help="model for --ask")
    p.add_argument("--repeats", type=int, default=1, help="repeat each --ask call N times (model answers vary)")
    p.add_argument("--json", action="store_true")
    return ap


def _job_or_latest(job: str | None, jobs_dir: str | None) -> Any:
    from .jobs import DreamJob

    if job:
        return DreamJob.load(job, jobs_dir)
    jobs = DreamJob.list(jobs_dir)
    if not jobs:
        raise SystemExit("no dream jobs yet")
    return jobs[-1]


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["hook"], ["hook-worker"]):
        return _hook_main(argv)
    args = build_parser().parse_args(argv)
    from .adapters import AdapterError
    from .backends import BackendError
    from .config import ConfigError
    from .context import DreamLimitError
    from .jobs import JobNotFound
    from .loaders import LoaderError
    from .review import ReviewError

    try:
        return _run(args)
    except (JobNotFound, ReviewError, ValueError, ConfigError, AdapterError, LoaderError, DreamLimitError,
            BackendError, OSError) as exc:
        # one line, no traceback: these are user-facing conditions (bad input, permissions, disk...)
        msg = str(exc) if not isinstance(exc, OSError) or not exc.filename else \
            f"{exc.strerror or exc}: {exc.filename}"
        print(f"agentdream: error: {msg}".replace("\n", " "), file=sys.stderr)
        return 2


def _types(value: str | None) -> list[str] | None:
    return [x.strip().lower() for x in value.split(",") if x.strip()] if value else None


def _hook_main(argv: list[str]) -> int:
    """Hooks must never fail or block the Claude session (exit code 2 would block a Stop hook),
    so even argument errors end in exit 0 with a log line."""
    from .claude_hook import HookLog, HookOptions, run_hook, run_worker

    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        HookLog(Path.cwd() / ".agentdream" / "hook.log")(f"invalid hook arguments: {argv!r}")
        return 0
    opts = HookOptions(memory=args.memory, every=args.every, apply=args.apply, config=args.config,
                       jobs_dir=args.jobs_dir, backend=args.backend, model=args.model,
                       apply_types=_types(args.apply_types), apply_min_confidence=args.apply_min_confidence)
    try:
        if args.cmd == "hook":
            return run_hook(opts)
        return run_worker(opts, Path(args.project))
    except BaseException:  # pragma: no cover - run_hook/run_worker already catch everything
        return 0


def _settings(args: argparse.Namespace, cli: dict[str, Any] | None = None) -> Any:
    from .config import resolve

    s = resolve(cli if cli is not None else {"jobs_dir": _abs(getattr(args, "jobs_dir", None))},
                config_path=getattr(args, "config", None))
    for w in s.warnings:
        print(f"agentdream: warning: {w}", file=sys.stderr)
    return s


def _abs(p: str | None) -> str | None:
    return str(Path(os.path.expanduser(p)).absolute()) if p else None


def _jobs_dir(args: argparse.Namespace) -> str | None:
    return _settings(args).dream.get("jobs_dir")


def _split_dream(settings: Any) -> tuple[str, list[str], str, dict[str, Any]]:
    opts = dict(settings.dream)
    memory = opts.pop("memory", None)
    if not memory:
        raise ValueError("no memory store: pass --memory or set `memory` in .agentdream.toml")
    sessions = list(opts.pop("sessions", None) or [])
    backend = opts.pop("backend", None) or "deterministic"
    return memory, sessions, backend, opts


def _run(args: argparse.Namespace) -> int:
    from .jobs import DreamJob

    if args.cmd == "dream":
        from .api import dream
        from .review import accept

        memory, sessions, backend, opts = _split_dream(_settings(args, cli_dream_values(args)))
        job = dream(memory, sessions, backend, **opts)
        if args.json:
            print(json.dumps(job.to_dict(), indent=2, default=str))
        else:
            print(f"job {job.id}: {job.status}")
            if job.status == "completed":
                c = job.counts
                print(f"  merged={c.get('merged')} replaced={c.get('replaced')} corrected={c.get('corrected')} "
                      f"insights={c.get('insights')} contradictions={c.get('contradictions')} "
                      f"unchanged={c.get('unchanged')}")
                print(f"  report: {job.out_dir / 'report.md'}")
                print(f"  new store (staged): {job.store_path}")
                for w in job.warnings:
                    print(f"  warning: {w}")
            elif job.error:
                print(f"  error: {job.error}", file=sys.stderr)
        if job.status != "completed":
            return 1
        if args.apply:
            res = accept(job)
            if not args.json:
                print(f"applied {len(res.applied_ops)} operation(s) to {res.target}" +
                      (f" (backup: {res.backup})" if res.backup else ""))
        elif not args.json:
            print(f"  dry run: nothing applied. Next: agentdream accept {job.id}  |  agentdream discard {job.id}")
        return 0

    if args.cmd == "status":
        job = _job_or_latest(args.job, _jobs_dir(args))
        if args.json:
            print(json.dumps(job.to_dict(), indent=2, default=str))
        else:
            print(job.summary_line())
            if job.phase:
                print(f"  phase: {job.phase} ({job.progress:.0%})")
            if job.error:
                print(f"  error: {job.error}")
        return 0

    if args.cmd == "list":
        jobs = DreamJob.list(_jobs_dir(args))
        if args.json:
            print(json.dumps([j.to_dict() for j in jobs], indent=2, default=str))
        elif not jobs:
            print("no dream jobs")
        for j in ([] if args.json else jobs):
            print(j.summary_line())
        return 0

    if args.cmd == "show":
        job = DreamJob.load(args.job, _jobs_dir(args))
        if args.json:
            print(json.dumps(job.report, indent=2, default=str))
        elif args.diff:
            print(job.report.get("diff", ""), end="")
        else:
            print(job.report_md)
        return 0

    if args.cmd == "accept":
        from .review import accept

        res = accept(args.job, only=args.only, exclude=args.exclude, to=args.to, backup=not args.no_backup,
                     force=args.force, jobs_dir=_jobs_dir(args), types=args.types,
                     min_confidence=args.min_confidence)
        for w in res.warnings:
            print(f"note: {w}")
        print(f"applied {len(res.applied_ops)} operation(s) to {res.target}" +
              (f"; skipped {len(res.skipped_ops)}" if res.skipped_ops else "") +
              (f"; backup at {res.backup}" if res.backup else ""))
        return 0

    if args.cmd == "discard":
        from .review import discard

        job = discard(args.job, jobs_dir=_jobs_dir(args))
        print(f"discarded {job.id}")
        return 0

    if args.cmd == "restore":
        from .review import restore

        jd = _jobs_dir(args)
        target = restore(args.job, to=args.to, jobs_dir=jd)
        print(f"restored {target} from the backup of {DreamJob.load(args.job, jd).backup_path}")
        return 0

    if args.cmd == "cancel":
        job = DreamJob.load(args.job, _jobs_dir(args))
        job.cancel()
        print(f"cancel requested for {job.id} (status: {job.status})")
        return 0

    if args.cmd == "schedule":
        from .scheduler import Scheduler
        from .util import parse_duration

        settings = _settings(args, cli_dream_values(args))
        sch = settings.schedule
        every = args.every or sch.get("every")
        idle = args.idle or sch.get("idle")
        new_sessions = args.new_sessions if args.new_sessions is not None else sch.get("new_sessions")
        if not (every or idle or new_sessions):
            print("agentdream: error: give --every, --idle and/or --new-sessions (or a [schedule] table)",
                  file=sys.stderr)
            return 2
        memory, sessions, backend, opts = _split_dream(settings)
        sched = Scheduler(memory, sessions, every=every, idle=idle, new_sessions=new_sessions,
                          state_file=args.state_file or sch.get("state_file"),
                          dream_options={**opts, "backend": backend}, apply=bool(args.apply))
        sched.run_forever(poll=parse_duration(args.poll or sch.get("poll") or "60s"), once=args.once,
                          log=lambda m: print(f"[agentdream schedule] {m}", flush=True))
        return 0

    if args.cmd == "serve-mcp":
        from .mcp_server import MCPServer

        MCPServer(_jobs_dir(args), config_path=args.config).serve()
        return 0

    if args.cmd == "config":
        s = _settings(args)
        eff = s.effective_dream()
        if args.json:
            print(json.dumps({"files": s.files, "dream": eff, "schedule": s.schedule, "hook": s.hook,
                              "sources": s.sources, "warnings": s.warnings}, indent=2, default=str))
            return 0
        print("config files: " + (", ".join(s.files) if s.files else "(none found)"))
        for k in sorted(eff):
            src = s.sources.get(f"dream.{k}", "default")
            print(f"  {k} = {eff[k]!r}  [{src}]")
        for section in ("schedule", "hook"):
            for k, v in sorted(getattr(s, section).items()):
                print(f"  {section}.{k} = {v!r}  [{s.sources.get(f'{section}.{k}', 'default')}]")
        return 0

    if args.cmd == "install-claude-hook":
        from .claude_hook import HookOptions, install
        from .review import ReviewError, parse_types

        apply_types = _types(args.apply_types) or (["merge", "replace"] if args.safe else None)
        min_conf = args.apply_min_confidence if args.apply_min_confidence is not None else (0.75 if args.safe else None)
        applying = bool(args.apply or args.safe)
        if (apply_types or min_conf is not None) and not applying:
            print("agentdream: error: --apply-types / --apply-min-confidence need --apply (or use --safe)",
                  file=sys.stderr)
            return 2
        try:
            parse_types(apply_types)
            if min_conf is not None and not 0.0 <= min_conf <= 1.0:
                raise ReviewError(f"--apply-min-confidence must be between 0 and 1, got {min_conf}")
        except ReviewError as exc:
            print(f"agentdream: error: {exc}", file=sys.stderr)
            return 2
        hook_opts = HookOptions(
            memory=_abs(args.memory) if args.memory and not args.memory.startswith("stancelog:") else args.memory,
            every=args.every, apply=applying or None, config=_abs(args.config), backend=args.backend,
            model=args.model, apply_types=apply_types, apply_min_confidence=min_conf)
        installed = install(args.scope, Path.cwd(), hook_opts, event=args.event, path=args.settings)
        print(installed.message)
        print(f"  command: {installed.command}")
        if not applying:
            mode = "dry run: dreams are staged for review, nothing is applied"
        elif apply_types or min_conf is not None:
            limits = [f"types {', '.join(apply_types)}" if apply_types else "",
                      f"confidence >= {min_conf}" if min_conf is not None else ""]
            mode = ("APPLY automatically only " + " and ".join(x for x in limits if x)
                    + " (backups are kept); everything else is staged for review")
        else:
            mode = "APPLY automatically (backups are kept)"
        print(f"  mode: {mode}; every {args.every or 1} session(s)")
        print("  log: <project>/.agentdream/hook.log   review: agentdream list --jobs-dir <project>/.agentdream/jobs")
        print("  tip: add .agentdream/ to your .gitignore. Remove with: agentdream uninstall-claude-hook"
              + ("" if args.scope == "project" else f" --{args.scope}"))
        return 0

    if args.cmd in ("install-claude-skill", "uninstall-claude-skill"):
        from .claude_hook import install_skill, uninstall_skill

        done = (install_skill(args.scope, Path.cwd(), force=args.force) if args.cmd.startswith("install")
                else uninstall_skill(args.scope, Path.cwd()))
        print(done.message)
        if args.cmd.startswith("install"):
            print("  use it in Claude Code with: /dream   (restart a running session to pick it up)")
        return 0

    if args.cmd == "benchmark":
        from .bench import render, run_benchmark

        if not 1 <= args.facts <= 12:
            print("agentdream: error: --facts must be between 1 and 12", file=sys.stderr)
            return 2
        result = run_benchmark(args.backend, args.model, per_category=args.facts, seed=args.seed,
                               backend_options={"base_url": args.base_url} if args.base_url else None,
                               workdir=_abs(args.keep), reader=args.ask, reader_model=args.ask_model,
                               repeats=args.repeats)
        print(json.dumps(result, indent=2) if args.json else render(result), end="" if not args.json else "\n")
        return 0

    if args.cmd == "uninstall-claude-hook":
        from .claude_hook import uninstall

        removed = uninstall(args.scope, Path.cwd(), path=args.settings)
        print(removed.message)
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
