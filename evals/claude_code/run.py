"""Does a dreamed memory file make Claude Code do better work?

    python evals/claude_code/run.py --model sonnet --repeats 3

This runs the real harness. A small Python project is generated together with a ``CLAUDE.md`` that has
drifted the way agent memory drifts: a note pointing at the wrong file, a note with a command that no
longer exists, a preference the user has since reversed, a stale value next to its replacement, and a
duplicate. Earlier sessions, in Claude Code's own transcript format, show the agent tripping over each
of them and finding the truth.

agentdream dreams over that memory and those sessions, and the result is accepted. Then Claude Code is
run headless (``claude -p``) on the same coding tasks in fresh copies of the project. The copies are
identical except for ``CLAUDE.md``:

    before        the drifted notes
    after-rules   after a dream with the rule-based engine (no model)
    after-model   after a dream with a model backend (--dream-backend, default claude-cli)

Each run is checked by code (the function exists, the tests pass, the right file changed) and the
harness's own report gives turns, wall time and cost. Nothing here is judged by a model.

It spends real usage on your Claude login: tasks x arms x repeats headless sessions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentdream import accept, dream

MEMORY = """\
# Project notes

## Layout

- Runtime settings (timeouts, retry limits) live in `config/settings.json`. (2026-01-08)
- Pricing logic is in `orders/pricing.py`.

## Workflow

- Run the test suite with `make test` before committing. (2026-01-10)
- Run the tests with `make test` before every commit.
- Keep functions small and give every public function a docstring.

## Conventions

- Don't use dataclasses for new models — the team prefers plain dicts. (2026-01-05)

## Service facts

- The API server listens on port 8080. (2025-12-01)
- The API server listens on port 9090. (2026-02-10)
"""

FILES = {
    "README.md": "# orders\n\nA small order-handling service. See CLAUDE.md for working notes.\n",
    "orders/__init__.py": "",
    "orders/defaults.py": '"""Runtime settings."""\n\nTIMEOUT_SECONDS = 30\nMAX_RETRIES = 3\n',
    "orders/client.py": (
        '"""HTTP client settings."""\n\nfrom orders import defaults\n\n\n'
        'def request_timeout() -> int:\n    """Seconds to wait for an upstream response."""\n'
        "    return defaults.TIMEOUT_SECONDS\n"),
    "orders/pricing.py": (
        '"""Pricing."""\n\n\ndef subtotal(items: list[dict]) -> float:\n'
        '    """Sum of price * quantity over the items."""\n'
        '    return round(sum(i["price"] * i["quantity"] for i in items), 2)\n'),
    "orders/server.py": (
        '"""API server address."""\n\nHOST = "127.0.0.1"\nPORT = 9090\n\n\n'
        'def address() -> str:\n    """host:port the API server binds to."""\n    return f"{HOST}:{PORT}"\n'),
    "orders/models.py": '"""Data models."""\n',
    "config/settings.json": '{\n  "timeout_seconds": 30,\n  "max_retries": 3\n}\n',
    "tests/__init__.py": "",
    "tests/test_pricing.py": (
        "import unittest\n\nfrom orders.pricing import subtotal\n\n\nclass SubtotalTest(unittest.TestCase):\n"
        "    def test_subtotal(self):\n"
        '        self.assertEqual(subtotal([{"price": 2.5, "quantity": 2}, {"price": 1.0, "quantity": 1}]), 6.0)\n'),
    "tests/test_client.py": (
        "import unittest\n\nfrom orders import defaults\nfrom orders.client import request_timeout\n\n\n"
        "class TimeoutTest(unittest.TestCase):\n    def test_timeout_comes_from_defaults(self):\n"
        "        self.assertEqual(request_timeout(), defaults.TIMEOUT_SECONDS)\n"),
}


def _turn(role: str, text: str, ts: str) -> str:
    return json.dumps({"type": role, "timestamp": ts, "message": {"role": role, "content": text}})


SESSIONS = {
    "session-01.jsonl": [
        ("user", "The timeout change didn't take effect. I still see 30 seconds.", "2026-03-02T09:00:00Z"),
        ("assistant", "My notes say runtime settings live in config/settings.json. I edited config/settings.json "
                      "but the timeout is still 30.", "2026-03-02T09:01:00Z"),
        ("assistant", "Found it. config/settings.json is a leftover file that nothing reads. Runtime settings "
                      "actually live in orders/defaults.py.", "2026-03-02T09:03:00Z"),
        ("user", "Right, config/settings.json is dead. Runtime settings live in orders/defaults.py.",
         "2026-03-02T09:04:00Z"),
    ],
    "session-02.jsonl": [
        ("assistant", "Ran `make test`. It failed: make: *** No rule to make target 'test'. There is no Makefile "
                      "in this repo.", "2026-03-05T14:00:00Z"),
        ("assistant", "Ran `python -m unittest discover -s tests` instead. All tests passed.",
         "2026-03-05T14:01:00Z"),
        ("user", "Yes, we removed the Makefile. Always run the tests with `python -m unittest discover -s tests`.",
         "2026-03-05T14:02:00Z"),
    ],
    "session-03.jsonl": [
        ("assistant", "Ran `make test` again. Failed: No rule to make target 'test'.", "2026-03-09T10:00:00Z"),
        ("assistant", "`python -m unittest discover -s tests` works. All tests passed.", "2026-03-09T10:01:00Z"),
        ("user", "Actually, let's use dataclasses for new models from now on.", "2026-03-09T10:05:00Z"),
        ("assistant", "Understood, switching new models to dataclasses.", "2026-03-09T10:05:30Z"),
        ("assistant", "curl localhost:8080/health failed: connection refused. The API server listens on port "
                      "9090, not 8080.", "2026-03-09T10:09:00Z"),
    ],
}


def _py(project: Path, code: str) -> bool:
    return subprocess.run([sys.executable, "-c", code], cwd=project, capture_output=True, timeout=60).returncode == 0


def _tests_pass(project: Path) -> bool:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=project,
                          capture_output=True, timeout=120).returncode == 0


def check_timeout(project: Path) -> bool:
    return _py(project, "from orders.client import request_timeout; assert request_timeout() == 45")


def check_tax(project: Path) -> bool:
    code = ("from orders.pricing import total_with_tax; "
            "assert total_with_tax([{'price': 10.0, 'quantity': 2}], 0.1) == 22.0")
    return _py(project, code) and _tests_pass(project)


def check_dataclass(project: Path) -> bool:
    code = ("import dataclasses; from orders.models import Address; assert dataclasses.is_dataclass(Address); "
            "a = Address(street='1 Main St', city='Leeds', postcode='LS1 1AA'); assert a.city == 'Leeds'")
    return _py(project, code)


def check_healthcheck(project: Path) -> bool:
    script = project / "scripts" / "healthcheck.py"
    if not script.is_file():
        return False
    text = script.read_text(encoding="utf-8")
    return "8080" not in text and ("9090" in text or "PORT" in text or "address" in text)


@dataclasses.dataclass
class Task:
    id: str
    drift: str  # which kind of bad note this task runs into
    prompt: str
    check: Callable[[Path], bool]


TASKS = [
    Task("timeout", "note points at the wrong file",
         "Raise the request timeout to 45 seconds.", check_timeout),
    Task("tax", "note gives a command that no longer exists",
         "Add a function total_with_tax(items, rate) to orders/pricing.py that returns the subtotal plus tax, "
         "rounded to 2 decimals. Run the test suite to confirm nothing broke.", check_tax),
    Task("model", "note states a preference the user reversed",
         "Add an Address model with street, city and postcode fields to orders/models.py.", check_dataclass),
    Task("healthcheck", "stale value next to its replacement",
         "Add scripts/healthcheck.py: it sends a GET request to /health on the local API server and exits 0 if "
         "the status is 200, 1 otherwise. Use the port the API server listens on.", check_healthcheck),
]

ALLOWED = ["Read", "Edit", "Write", "Glob", "Grep", "Bash(python:*)", "Bash(python3:*)", "Bash(make:*)",
           "Bash(ls:*)", "Bash(cat:*)", "Bash(grep:*)", "Bash(find:*)"]


def write_project(dest: Path, memory: str) -> None:
    for rel, text in FILES.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (dest / "CLAUDE.md").write_text(memory, encoding="utf-8")


def dreamed_memory(work: Path, name: str, backend: str, model: str | None) -> tuple[str, dict]:
    """Dream over the drifted memory with ``backend``, accept everything, return the new CLAUDE.md."""
    root = work / f"dream-{name}"
    (root / "sessions").mkdir(parents=True)
    (root / "CLAUDE.md").write_text(MEMORY, encoding="utf-8")
    for fname, turns in SESSIONS.items():
        (root / "sessions" / fname).write_text("\n".join(_turn(*t) for t in turns) + "\n", encoding="utf-8")
    here = os.getcwd()
    os.chdir(root)  # dream from the project folder, as a user would, so citations are short relative paths
    try:
        job = dream(memory="CLAUDE.md", sessions=["sessions"], backend=backend, model=model, jobs_dir="jobs")
        if job.status != "completed":
            raise RuntimeError(f"dream with {backend} {job.status}: {job.error}")
        if job.operations:
            accept(job, to="CLAUDE.after.md")
            text = Path("CLAUDE.after.md").read_text(encoding="utf-8")
        else:
            text = MEMORY
    finally:
        os.chdir(here)
    ops = [{"id": o.id, "type": o.type, "confidence": o.confidence, "summary": o.summary} for o in job.operations]
    return text, {"backend": backend, "operations": ops, "warnings": list(job.report.get("warnings") or [])}


def run_one(work: Path, arm: str, memory: str, task: Task, rep: int, model: str | None, timeout: int) -> dict:
    project = work / "runs" / f"{arm}-{task.id}-{rep}"
    write_project(project, memory)
    cmd = ["claude", "-p", "--output-format", "json", "--setting-sources", "project", "--strict-mcp-config",
           "--no-session-persistence", "--permission-mode", "acceptEdits", "--allowedTools", *ALLOWED]
    if model:
        cmd += ["--model", model]
    row: dict = {"arm": arm, "task": task.id, "rep": rep}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, input=task.prompt, cwd=project, capture_output=True, text=True, timeout=timeout)
        data = json.loads(proc.stdout)
        usage = data.get("usage") or {}
        row.update(turns=data.get("num_turns"), seconds=round((data.get("duration_ms") or 0) / 1000, 1),
                   cost_usd=round(data.get("total_cost_usd") or 0, 4), output_tokens=usage.get("output_tokens"),
                   denials=len(data.get("permission_denials") or []), error=bool(data.get("is_error")),
                   models=sorted((data.get("modelUsage") or {}).keys()))
    except (subprocess.TimeoutExpired, ValueError) as exc:
        row.update(turns=None, seconds=round(time.time() - t0, 1), cost_usd=None, output_tokens=None, denials=None,
                   error=True, models=[], failure=type(exc).__name__)
    row["passed"] = bool(task.check(project))
    print(f"  {arm:12} {task.id:12} #{rep}  {'pass' if row['passed'] else 'FAIL'}  "
          f"turns={row['turns']} {row['seconds']}s ${row['cost_usd']}", flush=True)
    return row


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(vals), 2) if vals else None


def summarise(rows: list[dict], arms: list[str]) -> dict:
    out: dict = {"overall": {}, "by_task": {}}
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        out["overall"][arm] = {"passed": sum(r["passed"] for r in mine), "runs": len(mine),
                               "turns": _mean(mine, "turns"), "seconds": _mean(mine, "seconds"),
                               "cost_usd": _mean(mine, "cost_usd")}
        for task in TASKS:
            sel = [r for r in mine if r["task"] == task.id]
            out["by_task"].setdefault(task.id, {})[arm] = {
                "passed": sum(r["passed"] for r in sel), "runs": len(sel), "turns": _mean(sel, "turns"),
                "seconds": _mean(sel, "seconds"), "cost_usd": _mean(sel, "cost_usd")}
    return out


def render(result: dict) -> str:
    arms, s = result["arms"], result["summary"]
    lines = [f"Claude Code on {len(TASKS)} tasks, {result['repeats']} run(s) each, model: {result['model']}", "",
             "| Memory file | Tasks passed | Mean turns | Mean seconds | Mean cost |", "|---|---|---|---|---|"]
    for arm in arms:
        o = s["overall"][arm]
        lines.append(f"| {arm} | {o['passed']} / {o['runs']} | {o['turns']} | {o['seconds']} | ${o['cost_usd']} |")
    lines += ["", "| Task | What the drifted note does | " + " | ".join(arms) + " |",
              "|---|---|" + "---|" * len(arms)]
    for task in TASKS:
        cells = []
        for arm in arms:
            t = s["by_task"][task.id][arm]
            cells.append(f"{t['passed']}/{t['runs']} passed, {t['turns']} turns")
        lines.append(f"| {task.id} | {task.drift} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="model for the Claude Code runs (default: your default)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dream-backend", default="claude-cli", help="backend for the after-model arm; 'none' skips it")
    ap.add_argument("--dream-model", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per Claude Code run")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.json"))
    ap.add_argument("--only-dream", action="store_true", help="dream and print the memory files; run no tasks")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="agentdream-harness-"))
    memories = {"before": MEMORY}
    dreams = {}
    memories["after-rules"], dreams["after-rules"] = dreamed_memory(work, "rules", "deterministic", None)
    if args.dream_backend != "none":
        memories["after-model"], dreams["after-model"] = dreamed_memory(work, "model", args.dream_backend,
                                                                        args.dream_model)
    for arm, text in memories.items():
        print(f"===== CLAUDE.md: {arm} =====\n{text}")
    if args.only_dream:
        return 0

    jobs = [(arm, memories[arm], task, rep) for rep in range(1, args.repeats + 1) for task in TASKS
            for arm in memories]
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        rows = list(pool.map(lambda j: run_one(work, j[0], j[1], j[2], j[3], args.model, args.timeout), jobs))
    used = sorted({m for r in rows for m in r.get("models") or []})
    result = {"model": ", ".join(used) or args.model or "default", "repeats": args.repeats,
              "arms": list(memories), "memories": memories, "dreams": dreams, "rows": rows,
              "summary": summarise(rows, list(memories))}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\n" + render(result))
    print(f"results: {args.out}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
