#!/usr/bin/env python3
"""Run the edge-case suite (tests/edge) and print a markdown table: area | case | status.

    python scripts/edge_case_table.py                 # print the table (exit code = pytest's)
    python scripts/edge_case_table.py --write docs/EDGE_CASES.md   # regenerate the doc
                                                                 # (also refreshes the table in docs/TESTING.md)

Conventions it relies on (see tests/edge/edgekit.py): one file per area, ``test_<area>.py``;
the first docstring line of each test function is the case description. Parametrized tests
are one row; the status shows how many of their parameter sets passed.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDGE = ROOT / "tests" / "edge"

#: display order and titles; unknown areas are appended alphabetically
AREAS = {
    "empty_inputs": "Empty / missing inputs",
    "limits": "Size limits",
    "unicode": "Unicode & encodings",
    "malformed": "Malformed JSON / JSONL",
    "line_endings": "Line endings",
    "timestamps": "Timestamps",
    "contradictions": "Contradictions",
    "secrets": "Secrets",
    "model_misbehavior": "Model misbehaviour",
    "llm_ids": "LLM entry ids",
    "crash_recovery": "Crash / kill recovery",
    "concurrency": "Concurrency",
    "review_edges": "Accept / partial accept",
    "permissions": "Permissions",
    "path_traversal": "Path traversal",
    "scheduler_edges": "Scheduler",
    "properties": "Property-based (hypothesis)",
}

#: page intro + behaviour summary (markdown); the generated table is appended after it
INTRO_FILE = Path(__file__).with_name("edge_cases_intro.md")


def docstrings() -> dict[tuple[str, str], str]:
    out = {}
    for f in sorted(EDGE.glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                doc = ast.get_docstring(node) or node.name.replace("_", " ")
                out[(f.stem, node.name)] = doc.strip().splitlines()[0]
    return out


def run_suite(junit: Path, extra: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", str(EDGE), "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *extra]
    # pytest's own output goes to stderr so stdout is just the table (`... > table.md` works)
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                          stdout=sys.stderr).returncode


def results(junit: Path) -> dict[tuple[str, str], list[str]]:
    by_test: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case in ET.parse(junit).getroot().iter("testcase"):
        module = case.get("classname", "").split(".")[-1]
        func = case.get("name", "").split("[")[0]
        if case.find("failure") is not None or case.find("error") is not None:
            status = "failed"
        elif case.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        by_test[(module, func)].append(status)
    return by_test


def status_cell(statuses: list[str]) -> str:
    n, ok = len(statuses), statuses.count("passed")
    bad, skipped = statuses.count("failed"), statuses.count("skipped")
    if bad:
        return f"❌ fail ({bad}/{n})"
    if skipped == n:
        return "⏭ skipped"
    suffix = f" ({ok}/{n})" if n > 1 else ""
    return f"✅ pass{suffix}" + (f", {skipped} skipped" if skipped else "")


def table(docs: dict[tuple[str, str], str], res: dict[tuple[str, str], list[str]]) -> str:
    order = list(AREAS) + sorted({m.removeprefix("test_") for m, _ in docs} - set(AREAS))
    rank = {a: i for i, a in enumerate(order)}
    keys = sorted(docs, key=lambda k: (rank[k[0].removeprefix("test_")], list(docs).index(k)))
    lines = ["| Area | Case | Status |", "|---|---|---|"]
    for module, func in keys:
        area = module.removeprefix("test_")
        case = docs[(module, func)].replace("|", "\\|")
        st = status_cell(res.get((module, func), [])) if (module, func) in res else "not run"
        lines.append(f"| {AREAS.get(area, area.replace('_', ' '))} | {case} <br>`{func}` | {st} |")
    total = sum(len(v) for v in res.values())
    passed = sum(v.count("passed") for v in res.values())
    failed = sum(v.count("failed") for v in res.values())
    head = (f"**{len(keys)} edge-case tests ({total} test runs with parametrization) in "
            f"{len({k[0] for k in keys})} areas: {passed} passed, {failed} failed, "
            f"{total - passed - failed} skipped.**")
    return head + "\n\n" + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", metavar="PATH", help="write a full markdown page (intro + table) to PATH")
    ap.add_argument("pytest_args", nargs="*", help="extra pytest arguments (after --)")
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as d:
        junit = Path(d) / "edge.xml"
        code = run_suite(junit, a.pytest_args)
        if not junit.exists():
            print("pytest produced no junit report", file=sys.stderr)
            return code or 1
        md = table(docstrings(), results(junit))
    if a.write:
        intro = INTRO_FILE.read_text(encoding="utf-8")
        Path(a.write).write_text(intro + "\n## Cases\n\n" + md, encoding="utf-8")
        print(f"wrote {a.write}", file=sys.stderr)
        testing = ROOT / "docs" / "TESTING.md"
        if testing.exists():
            refresh_testing(testing, md)
    print(md)
    return code


MARKER_START = "<!-- edge-case-table:start -->"
MARKER_END = "<!-- edge-case-table:end -->"


def refresh_testing(path: Path, md: str) -> None:
    """Replace the marked table section in docs/TESTING.md, if both markers are present."""
    text = path.read_text(encoding="utf-8")
    start, end = text.find(MARKER_START), text.find(MARKER_END)
    if start < 0 or end < 0 or end < start:
        print(f"{path} has no {MARKER_START} ... {MARKER_END} section; left unchanged", file=sys.stderr)
        return
    updated = text[: start + len(MARKER_START)] + "\n" + md.rstrip() + "\n" + text[end:]
    path.write_text(updated, encoding="utf-8")
    print(f"updated table in {path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
