"""report.md / report.json rendering and store diffs."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from .models import MemoryEntry, Operation
from .text import shorten

SECTIONS = [
    ("merged", "Merged", lambda o: o.type == "merge"),
    ("replaced", "Replaced (stale)", lambda o: o.type == "replace"),
    ("corrected", "Corrected", lambda o: o.type == "correct"),
    ("insights", "New insights", lambda o: o.type == "insight"),
    ("contradictions", "Contradictions", lambda o: o.type == "contradiction" or (o.type == "replace" and o.kind)),
]


def render_entries(entries: list[MemoryEntry]) -> list[str]:
    lines: list[str] = []
    cur = object()
    for e in sorted(entries, key=lambda x: (x.path or "~", x.metadata.get("_order", 10 ** 9), x.id)):
        if e.path != cur:
            cur = e.path
            lines.append(f"## {e.path or '(new entries)'}")
        lines.append(f"[{e.id}] " + e.content.replace("\n", "\n    "))
    return [ln + "\n" for ln in "\n".join(lines).split("\n")] if lines else []


def store_diff(input_path: Path | None, output_path: Path | None, before: list[MemoryEntry],
               after: list[MemoryEntry]) -> str:
    """Unified diff. Text-file stores are diffed file by file; others by rendered entries."""
    text_exts = {".md", ".markdown", ".txt", ".json", ".jsonl"}
    if input_path and output_path and input_path.exists() and output_path.exists():
        def files(root: Path) -> dict[str, Path]:
            if root.is_file():
                return {root.name: root} if root.suffix.lower() in text_exts else {}
            return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*"))
                    if p.is_file() and p.suffix.lower() in text_exts
                    and not any(part.startswith(".") for part in p.relative_to(root).parts)}

        fa, fb = files(input_path), files(output_path)
        if fa or fb:
            chunks = []
            for rel in sorted(set(fa) | set(fb)):
                a = fa[rel].read_text(encoding="utf-8", errors="replace").splitlines(True) if rel in fa else []
                b = fb[rel].read_text(encoding="utf-8", errors="replace").splitlines(True) if rel in fb else []
                if a == b:
                    continue
                chunks.extend(_fix_eol(difflib.unified_diff(a, b, fromfile=f"a/{rel}" if rel in fa else "/dev/null",
                                                            tofile=f"b/{rel}" if rel in fb else "/dev/null")))
            return "".join(chunks)
    return "".join(_fix_eol(difflib.unified_diff(render_entries(before), render_entries(after),
                                                 fromfile="a/memory", tofile="b/memory")))


def _fix_eol(lines):
    for ln in lines:
        yield ln if ln.endswith("\n") else ln + "\n\\ No newline at end of file\n"


def build_report(*, job_id: str, config: dict[str, Any], ops: list[Operation], original: list[MemoryEntry],
                 final: list[MemoryEntry], transcripts: list[dict[str, Any]], warnings: list[str],
                 notices: list[str], redactions: dict[str, int], diff: str, backend: str,
                 store_path: str | None, created_at: str) -> dict[str, Any]:
    touched = set()
    for o in ops:
        if o.changes_store:
            touched.update(o.targets)
            touched.update(o.remove)
            touched.update(u.id for u in o.upsert)
    orig_ids = {e.id for e in original}
    sections: dict[str, list[str]] = {}
    for key, _title, pred in SECTIONS:
        sections[key] = [o.id for o in ops if pred(o)]
    contradictions = [o for o in ops if o.id in sections["contradictions"]]
    return {
        "job_id": job_id,
        "created_at": created_at,
        "backend": backend,
        "instructions": config.get("instructions"),
        "memory": config.get("memory"),
        "store_path": store_path,
        "sessions": transcripts,
        "counts": {
            "entries_before": len(original),
            "entries_after": len(final),
            "operations": len(ops),
            "merged": len(sections["merged"]),
            "replaced": len(sections["replaced"]),
            "corrected": len(sections["corrected"]),
            "insights": len(sections["insights"]),
            "contradictions": len(contradictions),
            "temporal_contradictions": sum(1 for o in contradictions if o.kind == "temporal"),
            "unchanged": len(orig_ids - touched),
        },
        "sections": sections,
        "operations": [o.to_dict() for o in ops],
        "warnings": warnings,
        "notices": notices,
        "redactions": redactions,
        "diff": diff,
    }


def _evidence_lines(op: Operation) -> list[str]:
    out = []
    for ev in op.evidence:
        role = f", {ev.role}" if ev.role else ""
        out.append(f"  - `{ev.citation}` ({ev.stance}{role}): \"{shorten(ev.quote, 200)}\"")
    return out


def render_markdown(report: dict[str, Any], ops: list[Operation]) -> str:
    c = report["counts"]
    by_id = {o.id: o for o in ops}
    lines = [
        f"# Dream report — {report['job_id']}",
        "",
        f"- Created: {report['created_at']}",
        f"- Backend: {report['backend']}",
        f"- Memory: `{report.get('memory')}` ({c['entries_before']} entries → {c['entries_after']})",
        f"- Sessions: {len(report['sessions'])}",
    ]
    if report.get("instructions"):
        lines.append(f"- Instructions: {report['instructions']}")
    if report.get("store_path"):
        lines.append(f"- New store (staged, not applied): `{report['store_path']}`")
    lines += [
        "",
        "| Merged | Replaced | Corrected | New insights | Contradictions (temporal) | Unchanged |",
        "|---|---|---|---|---|---|",
        f"| {c['merged']} | {c['replaced']} | {c['corrected']} | {c['insights']} | "
        f"{c['contradictions']} ({c['temporal_contradictions']}) | {c['unchanged']} |",
        "",
        "Review, then `agentdream accept " + report["job_id"] + "` (or `--only OP_ID,...` / `--exclude ...`) or "
        "`agentdream discard " + report["job_id"] + "`.",
        "",
    ]
    for key, title, _pred in SECTIONS:
        ids = report["sections"].get(key, [])
        lines.append(f"## {title} ({len(ids)})")
        lines.append("")
        if not ids:
            lines.append("_None._")
            lines.append("")
            continue
        for oid in ids:
            op = by_id[oid]
            kind = f" [{op.kind}]" if op.kind else ""
            lines.append(f"### `{op.id}`{kind} — {op.summary}")
            lines.append("")
            if op.targets:
                lines.append(f"- Entries: {', '.join('`' + t + '`' for t in op.targets)}")
            for b in op.before:
                lines.append(f"- Before: {shorten(b, 220)}")
            for a in op.after:
                lines.append(f"- After: {shorten(a, 300)}")
            if not op.changes_store:
                lines.append("- (flag only — no change to the store)")
            if op.history:
                lines.append("- History:")
                lines.extend(f"  - {shorten(h, 200)}" for h in op.history)
            if op.reason:
                lines.append(f"- Why: {op.reason}")
            if op.evidence:
                lines.append("- Evidence:")
                lines.extend(_evidence_lines(op))
            if op.depends_on:
                lines.append(f"- Depends on: {', '.join('`' + d + '`' for d in op.depends_on)}")
            lines.append(f"- Confidence: {op.confidence:.2f} · origin: {op.origin}")
            lines.append("")
    lines.append(f"## Unchanged ({c['unchanged']})")
    lines.append("")
    lines.append(f"{c['unchanged']} entr{'y' if c['unchanged'] == 1 else 'ies'} carried over unchanged.")
    lines.append("")
    if report.get("redactions"):
        total = sum(report["redactions"].values())
        kinds = ", ".join(f"{k}×{v}" for k, v in sorted(report["redactions"].items()))
        lines += ["## Redactions", "", f"{total} secret(s) redacted before analysis: {kinds}.", ""]
    if report.get("warnings") or report.get("notices"):
        lines += ["## Warnings & notices", ""]
        lines += [f"- ⚠ {w}" for w in report.get("warnings", [])]
        lines += [f"- {n}" for n in report.get("notices", [])]
        lines.append("")
    lines += ["## Diff", ""]
    if report.get("diff"):
        lines += ["```diff", report["diff"].rstrip("\n"), "```"]
    else:
        lines.append("_No changes._")
    lines.append("")
    return "\n".join(lines)
