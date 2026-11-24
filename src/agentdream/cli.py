"""``agentdream`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__


def _common_jobs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--jobs-dir", help="where job state lives (default: $AGENTDREAM_HOME/jobs or ./.agentdream/jobs)")


def _dream_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--memory", "-m", required=True, help="memory store: markdown folder/file, JSON/JSONL, SQLite, "
                                                         "stancelog:<path>")
    p.add_argument("--sessions", "-s", nargs="+", default=[], help="transcript files, directories or globs")
    p.add_argument("--backend", "-b", default="deterministic",
                   help="deterministic | anthropic | openai | ollama | claude-cli (default: deterministic)")
    p.add_argument("--model", help="model name for LLM backends")
    p.add_argument("--base-url", help="base URL for the openai-compatible backend (Ollama, LM Studio, vLLM)")
    p.add_argument("--instructions", "-i", help="free-text focus for the dream")
    p.add_argument("--memory-format", help="force memory format (markdown|json|sqlite|stancelog)")
    p.add_argument("--transcript-format", help="force transcript format (claude-code|jsonl|openai|text)")
    p.add_argument("--max-sessions", type=int, default=100)
    p.add_argument("--max-bytes-per-transcript", type=int, default=2_000_000)
    p.add_argument("--max-total-tokens", type=int, default=400_000)
    p.add_argument("--on-limit", choices=["truncate", "fail"], default="truncate")
    p.add_argument("--max-chunk-tokens", type=int, default=None,
                   help="LLM backends: token budget per model call; larger inputs are chunked (default 60000)")
    p.add_argument("--chunk-overlap", type=int, default=None,
                   help="tokens of transcript context repeated between consecutive chunks (default 400)")
    p.add_argument("--max-chunks", type=int, default=None, help="max model calls per phase (default 64)")
    p.add_argument("--no-redact", action="store_true", help="disable secret redaction (not recommended)")
    p.add_argument("--redact-emails", action="store_true")
    p.add_argument("--no-redact-entropy", action="store_true",
                   help="don't redact long random-looking tokens (known secret formats are still redacted)")
    p.add_argument("--entropy-threshold", type=float, default=None,
                   help="bits/char for the high-entropy check (default 4.0; lower = more aggressive)")
    p.add_argument("--redact-pattern", action="append", default=None, metavar="REGEX",
                   help="extra regex to redact (repeatable)")
    p.add_argument("--apply", action="store_true", help="accept all operations right away (backup is still made)")
    _common_jobs(p)


def _dream_options(a: argparse.Namespace) -> dict:
    opts = {"model": a.model, "instructions": a.instructions, "memory_format": a.memory_format,
            "transcript_format": a.transcript_format, "max_sessions": a.max_sessions,
            "max_bytes_per_transcript": a.max_bytes_per_transcript, "max_total_tokens": a.max_total_tokens,
            "on_limit": a.on_limit, "redact": not a.no_redact, "redact_emails": a.redact_emails,
            "redact_entropy": False if a.no_redact_entropy else None, "entropy_threshold": a.entropy_threshold,
            "redact_patterns": a.redact_pattern,
            "jobs_dir": a.jobs_dir, "max_chunk_tokens": a.max_chunk_tokens, "chunk_overlap": a.chunk_overlap,
            "max_chunks": a.max_chunks}
    if getattr(a, "out", None):
        opts["out"] = a.out
    if a.base_url:
        opts["backend_options"] = {"base_url": a.base_url}
    return {k: v for k, v in opts.items() if v is not None}


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
    _common_jobs(p)

    p = sub.add_parser("list", help="list dream jobs")
    p.add_argument("--json", action="store_true")
    _common_jobs(p)

    p = sub.add_parser("show", help="print a job's report")
    p.add_argument("job")
    p.add_argument("--json", action="store_true", help="print report.json instead of report.md")
    p.add_argument("--diff", action="store_true", help="print only the diff")
    _common_jobs(p)

    p = sub.add_parser("accept", help="apply a dream (backup of the original first)")
    p.add_argument("job")
    p.add_argument("--only", help="comma-separated operation ids to apply")
    p.add_argument("--exclude", help="comma-separated operation ids to skip")
    p.add_argument("--to", help="write the result here instead of over the original store")
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--force", action="store_true", help="apply even if the store changed since the dream")
    _common_jobs(p)

    p = sub.add_parser("discard", help="discard a dream's staged output")
    p.add_argument("job")
    _common_jobs(p)

    p = sub.add_parser("cancel", help="request cancellation of a running dream")
    p.add_argument("job")
    _common_jobs(p)

    p = sub.add_parser("schedule", help="dream automatically (idle / interval / every N sessions)")
    _dream_args(p)
    p.add_argument("--every", help="interval, e.g. 30m, 6h, 1d")
    p.add_argument("--idle", help="dream after this much inactivity in new transcripts, e.g. 30m")
    p.add_argument("--new-sessions", "--every-sessions", dest="new_sessions", type=int,
                   help="dream every N new sessions")
    p.add_argument("--poll", default="60s", help="how often to check triggers (default 60s)")
    p.add_argument("--once", action="store_true", help="check triggers once and exit")
    p.add_argument("--state-file")

    p = sub.add_parser("serve-mcp", help="run the MCP stdio server")
    _common_jobs(p)
    return ap


def _job_or_latest(job: str | None, jobs_dir: str | None):
    from .jobs import DreamJob

    if job:
        return DreamJob.load(job, jobs_dir)
    jobs = DreamJob.list(jobs_dir)
    if not jobs:
        raise SystemExit("no dream jobs yet")
    return jobs[-1]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .jobs import JobNotFound
    from .review import ReviewError

    try:
        return _run(args)
    except (JobNotFound, ReviewError, FileNotFoundError, ValueError) as exc:
        print(f"agentdream: error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    from .jobs import DreamJob

    if args.cmd == "dream":
        from .api import dream
        from .review import accept

        job = dream(args.memory, args.sessions, args.backend, **_dream_options(args))
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
        job = _job_or_latest(args.job, args.jobs_dir)
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
        jobs = DreamJob.list(args.jobs_dir)
        if args.json:
            print(json.dumps([j.to_dict() for j in jobs], indent=2, default=str))
        elif not jobs:
            print("no dream jobs")
        for j in ([] if args.json else jobs):
            print(j.summary_line())
        return 0

    if args.cmd == "show":
        job = DreamJob.load(args.job, args.jobs_dir)
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
                     force=args.force, jobs_dir=args.jobs_dir)
        for w in res.warnings:
            print(f"note: {w}")
        print(f"applied {len(res.applied_ops)} operation(s) to {res.target}" +
              (f"; skipped {len(res.skipped_ops)}" if res.skipped_ops else "") +
              (f"; backup at {res.backup}" if res.backup else ""))
        return 0

    if args.cmd == "discard":
        from .review import discard

        job = discard(args.job, jobs_dir=args.jobs_dir)
        print(f"discarded {job.id}")
        return 0

    if args.cmd == "cancel":
        job = DreamJob.load(args.job, args.jobs_dir)
        job.cancel()
        print(f"cancel requested for {job.id} (status: {job.status})")
        return 0

    if args.cmd == "schedule":
        from .scheduler import Scheduler
        from .util import parse_duration

        if not (args.every or args.idle or args.new_sessions):
            print("agentdream: error: give --every, --idle and/or --new-sessions", file=sys.stderr)
            return 2
        sched = Scheduler(args.memory, args.sessions, every=args.every, idle=args.idle,
                          new_sessions=args.new_sessions, state_file=args.state_file,
                          dream_options={**_dream_options(args), "backend": args.backend}, apply=args.apply)
        sched.run_forever(poll=parse_duration(args.poll), once=args.once,
                          log=lambda m: print(f"[agentdream schedule] {m}", flush=True))
        return 0

    if args.cmd == "serve-mcp":
        from .mcp_server import MCPServer

        MCPServer(args.jobs_dir).serve()
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
