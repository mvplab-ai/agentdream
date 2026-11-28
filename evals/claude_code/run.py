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
- Test fixtures live in `tests/data/`. (2026-01-09)

## Workflow

- Run the test suite with `make test` before committing. (2026-01-10)
- Run the tests with `make test` before every commit.
- Keep functions small and give every public function a docstring.

## Conventions

- Don't use dataclasses for new models — the team prefers plain dicts. (2026-01-05)
- Don't use the logging module — print() is enough for this service. (2026-01-12)
- Round money amounts to 2 decimals with `round()`.

## Service facts

- The API server listens on port 8080. (2025-12-01)
- The API server listens on port 9090. (2026-02-10)
- The staging database host is `db-staging-1.internal`. (2025-11-20)
- CI fails below 80% test coverage. (2025-10-01)
- CI fails below 90% test coverage. (2026-02-01)
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
    "orders/service.py": (
        '"""Order operations."""\n\n\ndef confirm(order_id: str) -> bool:\n'
        '    """Mark an order as confirmed."""\n    return True\n'),
    "pyproject.toml": '[project]\nname = "orders"\nversion = "1.3.0"\nrequires-python = ">=3.10"\n',
    "tests/fixtures/order_basic.json": '{"id": "A-1", "items": [{"price": 2.5, "quantity": 2}]}\n',
    "tests/test_fixtures.py": (
        "import json\nimport unittest\nfrom pathlib import Path\n\nfrom orders.pricing import subtotal\n\n"
        'FIXTURES = Path(__file__).parent / "fixtures"\n\n\nclass FixtureTest(unittest.TestCase):\n'
        "    def test_basic_order(self):\n"
        '        order = json.loads((FIXTURES / "order_basic.json").read_text())\n'
        '        self.assertEqual(subtotal(order["items"]), 5.0)\n'),
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
    "session-04.jsonl": [
        ("assistant", "Looked for the test fixtures in tests/data/. That directory does not exist.",
         "2026-03-12T11:00:00Z"),
        ("assistant", "Found them. Test fixtures live in tests/fixtures/.", "2026-03-12T11:01:00Z"),
        ("user", "Yes, test fixtures live in tests/fixtures/. There is no tests/data.", "2026-03-12T11:02:00Z"),
        ("user", "Actually, let's use the logging module from now on.", "2026-03-12T11:10:00Z"),
        ("assistant", "Understood, new code will use the logging module instead of print().",
         "2026-03-12T11:10:30Z"),
    ],
    "session-05.jsonl": [
        ("assistant", "Connecting to db-staging-1.internal failed: host not found.", "2026-03-15T16:00:00Z"),
        ("user", "db-staging-1 was decommissioned last month. The staging database host is db-staging-3.internal "
                 "now.", "2026-03-15T16:01:00Z"),
        ("assistant", "Connected to db-staging-3.internal. The staging database is reachable.",
         "2026-03-15T16:02:00Z"),
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


def check_logging(project: Path) -> bool:
    src = (project / "orders" / "service.py").read_text(encoding="utf-8")
    return ("logging" in src and "print(" not in src
            and _py(project, "from orders.service import cancel; assert cancel('A-1') is True"))


def check_fixture(project: Path) -> bool:
    new = project / "tests" / "fixtures" / "sample_order.json"
    if not new.is_file() or (project / "tests" / "data").exists():
        return False
    try:
        json.loads(new.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return _tests_pass(project)


def check_staging(project: Path) -> bool:
    script = project / "scripts" / "staging_dsn.py"
    if not script.is_file():
        return False
    text = script.read_text(encoding="utf-8")
    return "db-staging-3.internal" in text and "db-staging-1" not in text


def check_coverage(project: Path) -> bool:
    code = ("import tomllib; d = tomllib.load(open('pyproject.toml', 'rb')); "
            "assert d['tool']['coverage']['report']['fail_under'] == 90")
    return _py(project, code)


def check_discount(project: Path) -> bool:
    code = ("from orders.pricing import discount; "
            "assert discount([{'price': 10.0, 'quantity': 3}], 15) == 25.5; "
            "assert discount([{'price': 0.1, 'quantity': 3}], 10) == 0.27")
    return _py(project, code) and _tests_pass(project)


def check_item_count(project: Path) -> bool:
    code = ("from orders.pricing import item_count; "
            "assert item_count([{'price': 1.0, 'quantity': 2}, {'price': 3.0, 'quantity': 5}]) == 7; "
            "assert (item_count.__doc__ or '').strip()")
    return _py(project, code)


@dataclasses.dataclass
class Task:
    id: str
    drift: str  # which kind of bad note this task runs into ("" for a control: the notes it needs are right)
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
    Task("logging", "note states a preference the user reversed",
         "Add a function cancel(order_id) to orders/service.py. It records a message saying which order was "
         "cancelled and returns True.", check_logging),
    Task("fixture", "note names a directory that does not exist",
         "Add a test fixture sample_order.json (one order with two items) alongside the existing test fixtures, "
         "and a test that loads it and checks its subtotal.", check_fixture),
    Task("staging", "note gives a value that changed; the code does not contain it",
         "Add scripts/staging_dsn.py: it prints the Postgres DSN for the staging database (user orders, "
         "database orders, port 5432).", check_staging),
    Task("coverage", "stale value next to its replacement",
         "Add a [tool.coverage.report] section to pyproject.toml that fails the run below the coverage "
         "threshold CI enforces.", check_coverage),
    Task("discount", "",
         "Add a function discount(items, percent) to orders/pricing.py that returns the subtotal with the "
         "percentage discount taken off.", check_discount),
    Task("item_count", "",
         "Add a function item_count(items) to orders/pricing.py that returns the total quantity across the "
         "items.", check_item_count),
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


def render(result: dict, arms: list[str]) -> str:
    s = result["summary"]
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
        lines.append(f"| {task.id} | {task.drift or 'control: the notes it needs are right'} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="sonnet",
                    help="comma-separated models for the Claude Code runs; 'default' is your default model")
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

    arms = list(memories)
    out: dict = {"arms": arms, "tasks": [{"id": t.id, "drift": t.drift, "prompt": t.prompt} for t in TASKS],
                 "memories": memories, "dreams": dreams, "models": {}}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        model = None if name == "default" else name
        mwork = work / name
        jobs = [(arm, memories[arm], task, rep) for rep in range(1, args.repeats + 1) for task in TASKS
                for arm in arms]
        print(f"----- model: {name} ({len(jobs)} runs) -----", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            rows = list(pool.map(lambda j, m=model, w=mwork: run_one(w, j[0], j[1], j[2], j[3], m, args.timeout),
                                 jobs))
        used = sorted({m for r in rows for m in r.get("models") or []})
        result = {"model": ", ".join(used) or name, "repeats": args.repeats, "rows": rows,
                  "summary": summarise(rows, arms),
                  "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 2)}
        out["models"][name] = result
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")  # keep what has finished
        print("\n" + render(result, arms), flush=True)
    print(f"results: {args.out}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
