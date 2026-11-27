"""A labelled benchmark: does a dream leave the memory store better than it found it?

    agentdream benchmark                      # deterministic backend, no model, no network
    agentdream benchmark --backend claude-cli # the same workload through a model

The workload is synthetic and every defect in it is known, so the result can be scored without a
judge. A store of project notes is generated with seven kinds of entry:

* ``duplicate``        the same note twice, word for word, in two files
* ``near_duplicate``   the same note twice with small wording differences
* ``stale``            two dated notes giving different values; the newer one is true
* ``wrong``            a note the transcripts refute (the agent fails acting on it, then finds the truth)
* ``reversal``         a "don't use X" note the user later reverses in a session
* ``paraphrase``       the same fact in two unrelated wordings (hard for rule-based matching)
* ``control``          correct notes that must come through unchanged

Each fact has a probe that reads a store and returns the values it asserts for that fact. A fact is
**answered** when the store asserts exactly the true value (one value, and the right one) and
**clean** when, in addition, it is stated in a single note. Scores are computed for the store before
the dream, after applying every proposed operation, and after applying only the low-risk ones (the
``--safe`` auto-apply policy: merge and replace at confidence >= 0.75).

What this does not measure: whether an agent then performs better on real tasks. "Answered" is a
proxy for it (an agent reading the store finds one, correct value), not a measurement of it. The notes
are also template-generated English, which suits rule-based matching; treat the deterministic numbers
as an upper bound for stores written this plainly, and see the ``paraphrase`` row for what it misses.
"""

from __future__ import annotations

import json
import random
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .context import DreamConfig
from .models import MemoryEntry, Operation, apply_operations
from .pipeline import run_dream
from .review import policy_split, select_operations
from .util import estimate_tokens

CATEGORIES = ("duplicate", "near_duplicate", "stale", "wrong", "reversal", "paraphrase", "control")
SAFE_TYPES = ("merge", "replace")
SAFE_MIN_CONFIDENCE = 0.75

_ADJ = ("primary", "legacy", "staging", "regional", "internal", "public", "nightly", "shadow", "edge", "batch",
        "mobile", "partner", "audit", "sandbox", "archive")
_NOUNS = ("cache", "router", "parser", "worker", "queue", "bucket", "gateway", "scheduler", "indexer", "crawler",
          "ledger", "mailer", "billing", "search", "uploader", "renderer", "notifier", "exporter", "importer",
          "tracker")
_PLACES = ("lobby", "cafeteria", "loading dock", "server room", "print shop", "bike shed", "reception desk",
           "mail room", "west stairwell", "data vault", "test lab", "supply closet")
_DIRS = ("north", "south", "east", "west")
_OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
_TOOLS = ("Poetry", "Docker", "Terraform", "Redis", "Webpack", "Bazel", "Ansible", "Celery", "Kafka", "Helm",
          "Gradle", "Vagrant")
_TASKS = ("dependency management", "local development", "infrastructure changes", "session storage",
          "asset bundling", "release builds", "server provisioning", "background jobs", "event streaming",
          "cluster deploys", "mobile builds", "test environments")
_HISTORY = re.compile(r"\(History:.*\)\s*$", re.S)


@dataclass
class Fact:
    id: str
    category: str
    truth: str
    probe: Callable[[str], str | None]  # note text (history stripped) -> the value it asserts, or None
    notes: list[str] = field(default_factory=list)  # the notes written to the store for this fact


@dataclass
class Workload:
    memory: Path
    sessions: Path
    facts: list[Fact]
    filler: list[str]  # unrelated notes that must survive verbatim


def _regex_probe(pattern: str) -> Callable[[str], str | None]:
    rx = re.compile(pattern, re.I)

    def probe(text: str) -> str | None:
        m = rx.search(text)
        return m.group(1).lower() if m else None
    return probe


def _stance_probe(tool: str, task: str) -> Callable[[str], str | None]:
    neg = re.compile(rf"\b(?:don'?t|do not|never|avoid|stop)\b[^.]*\b{re.escape(tool)}\b", re.I)
    pos = re.compile(rf"\b{re.escape(tool)}\b", re.I)
    about = re.compile(re.escape(task), re.I)

    def probe(text: str) -> str | None:
        if not about.search(text) or not pos.search(text):
            return None
        return "avoid" if neg.search(text) else "use"
    return probe


def _presence_probe(*needles: str) -> Callable[[str], str | None]:
    def probe(text: str) -> str | None:
        low = text.lower()
        return "stated" if all(n.lower() in low for n in needles) else None
    return probe


def make_workload(root: Path, per_category: int = 10, seed: int = 7) -> Workload:
    """Write a labelled memory folder and transcripts under ``root``."""
    rng = random.Random(seed)
    pairs = [(a, n) for a in _ADJ for n in _NOUNS]
    rng.shuffle(pairs)
    names = iter(f"{a} {n}" for a, n in pairs)
    places = [(p, rng.choice(_DIRS)) for p in _PLACES]
    stances = [(t, k) for t in _TOOLS for k in _TASKS]
    rng.shuffle(stances)
    facts: list[Fact] = []
    notes: list[tuple[str, str]] = []           # (file, bullet)
    sessions: list[list[tuple[str, str]]] = []  # one list of (role, text) per session

    def add(fact: Fact, placed: list[tuple[str, str]]) -> None:
        fact.notes = [text for _, text in placed]
        facts.append(fact)
        notes.extend(placed)

    def port_probe(name: str) -> Callable[[str], str | None]:
        return _regex_probe(rf"\b{re.escape(name)} service\b.*?\bport (\d+)")

    for i in range(per_category):
        # duplicate: word for word, in two files
        name, port = next(names), rng.randint(1100, 9900)
        line = f"The {name} service listens on port {port}."
        add(Fact(f"duplicate-{i}", "duplicate", str(port), port_probe(name)),
            [("services.md", line), ("notes/misc.md", line)])

        # near duplicate: small wording differences
        name = next(names)
        cmd = rng.choice(("pytest -q", "make lint", "ruff check .", "npm test", "cargo test", "go vet ./..."))
        add(Fact(f"near_duplicate-{i}", "near_duplicate", "stated", _presence_probe(name, cmd)),
            [("workflow.md", f"Run `{cmd}` for the {name} module before committing."),
             ("notes/misc.md", f"Run `{cmd}` for the {name} module before every commit.")])

        # stale: two dated values, the newer one is true
        name, old, new = next(names), rng.randint(1100, 5000), rng.randint(5100, 9900)
        add(Fact(f"stale-{i}", "stale", str(new), port_probe(name)),
            [("services.md", f"The {name} service listens on port {old}. (2025-11-0{1 + i % 9})"),
             ("notes/infra.md", f"The {name} service listens on port {new}. (2026-02-1{i % 10})")])

        # wrong: the transcripts refute the note
        name = next(names)
        place, _ = places[i % len(places)]
        wrong = _DIRS[i % 4]
        right = _OPPOSITE[wrong]
        thing = f"{name} kiosk"
        add(Fact(f"wrong-{i}", "wrong", right,
                 _regex_probe(rf"\b{re.escape(thing)}\b.*?\b(north|south|east|west) of the {re.escape(place)}")),
            [("places.md", f"The {thing} is {wrong} of the {place}. (2026-01-0{1 + i % 9})")])
        for _ in range(2):
            sessions.append([
                ("user", f"Please restock the {thing}."),
                ("assistant", f"My notes say the {thing} is {wrong} of the {place}. Heading {wrong}."),
                ("assistant", f"Walked {wrong} of the {place}. No {thing} here, only an empty corridor."),
                ("assistant", f"Still can't find the {thing} {wrong} of the {place}."),
            ])
        sessions.append([
            ("assistant", f"Searching {wrong} failed twice, so this time I will look {right} of the {place}."),
            ("assistant", f"Found the {thing}! It is just {right} of the {place}."),
            ("assistant", f"Restocked the {thing} {right} of the {place}."),
            ("user", "Great, that worked."),
        ])

        # reversal: the user changes their mind in a session
        tool, task = stances[i]
        add(Fact(f"reversal-{i}", "reversal", "use", _stance_probe(tool, task)),
            [("tooling.md", f"Don't use {tool} for {task} — it was too slow on CI. (2026-01-1{i % 10})")])
        sessions.append([
            ("user", f"The new {tool} release fixed the slowness. Actually, let's use {tool} for {task} from now on."),
            ("assistant", f"Sounds good, switching {task} to {tool}."),
        ])

        # paraphrase: one fact, two unrelated wordings
        name = next(names)
        branch = rng.choice(("release", "stable", "production", "deploy"))
        add(Fact(f"paraphrase-{i}", "paraphrase", "stated", _presence_probe(name, f"`{branch}`")),
            [("workflow.md", f"Deploys of the {name} component go through the `{branch}` branch."),
             ("notes/misc.md", f"To ship the {name} component, merge your change into `{branch}` first.")])

        # control: a correct note, and a correct note the transcripts confirm
        name, port = next(names), rng.randint(1100, 9900)
        add(Fact(f"control-{i}a", "control", str(port), port_probe(name)),
            [("services.md", f"The {name} service listens on port {port}.")])
        name = next(names)
        place, direction = places[(i + 5) % len(places)]
        thing = f"{name} kiosk"
        add(Fact(f"control-{i}b", "control", direction,
                 _regex_probe(rf"\b{re.escape(thing)}\b.*?\b(north|south|east|west) of the {re.escape(place)}")),
            [("places.md", f"The {thing} is {direction} of the {place}.")])
        sessions.append([
            ("user", f"Check on the {thing}."),
            ("assistant", f"Found the {thing} {direction} of the {place}, as noted. All good."),
        ])

    filler = [f"The on-call rotation for {next(names)} changes every Monday." for _ in range(per_category)]
    notes.extend(("team.md", f) for f in filler)

    mem, ses = root / "memory", root / "sessions"
    by_file: dict[str, list[str]] = {}
    for rel, text in notes:
        by_file.setdefault(rel, []).append(text)
    for rel, lines in sorted(by_file.items()):
        p = mem / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Notes\n\n" + "".join(f"- {ln}\n" for ln in lines), encoding="utf-8")
    ses.mkdir(parents=True, exist_ok=True)
    rng.shuffle(sessions)
    for n, msgs in enumerate(sessions):
        day, hour = 1 + n % 27, 8 + n % 10
        rows = [{"role": role, "content": text, "timestamp": f"2026-03-{day:02d}T{hour:02d}:{k:02d}:00Z"}
                for k, (role, text) in enumerate(msgs)]
        (ses / f"session-{n:03d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Workload(mem, ses, facts, filler)


# ------------------------------------------------------------------ scoring

def _texts(entries: list[MemoryEntry]) -> list[str]:
    return [_HISTORY.sub("", e.content).strip() for e in entries]


def score(workload: Workload, entries: list[MemoryEntry]) -> dict[str, Any]:
    """Score one store against the workload's ground truth."""
    texts = _texts(entries)
    per_cat: dict[str, dict[str, int]] = {c: {"facts": 0, "answered": 0, "clean": 0} for c in CATEGORIES}
    redundant = 0
    for f in workload.facts:
        values = [v for v in (f.probe(t) for t in texts) if v is not None]
        answered = bool(values) and set(values) == {f.truth}
        row = per_cat[f.category]
        row["facts"] += 1
        row["answered"] += answered
        row["clean"] += answered and len(values) == 1
        redundant += max(0, len(values) - 1)
    whole = {e.content.strip() for e in entries}
    kept = sum(1 for t in workload.filler if t in whole)
    total = len(workload.facts)
    return {
        "notes": len(entries),
        "tokens": sum(estimate_tokens(e.content) for e in entries),
        "redundant_notes": redundant,
        "answered": sum(r["answered"] for r in per_cat.values()),
        "clean": sum(r["clean"] for r in per_cat.values()),
        "facts": total,
        "unrelated_notes_kept": kept,
        "unrelated_notes": len(workload.filler),
        "by_category": per_cat,
    }


def _apply(original: list[MemoryEntry], ops: list[Operation], safe: bool) -> tuple[list[MemoryEntry], int]:
    if safe:
        _, below = policy_split(ops, SAFE_TYPES, SAFE_MIN_CONFIDENCE)
        ops, _ = select_operations(ops, None, below)
    return apply_operations([e.copy() for e in original], ops), len(ops)


def run_benchmark(backend: str = "deterministic", model: str | None = None, per_category: int = 10, seed: int = 7,
                  backend_options: dict[str, Any] | None = None, workdir: str | Path | None = None,
                  **options: Any) -> dict[str, Any]:
    """Generate the workload, dream over it, and score before / after."""
    with tempfile.TemporaryDirectory(prefix="agentdream-bench-") as tmp:
        root = Path(workdir) if workdir else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        wl = make_workload(root, per_category, seed)
        cfg = DreamConfig(memory=str(wl.memory), sessions=[str(wl.sessions)], backend=backend, model=model,
                          backend_options=dict(backend_options or {}), **options)
        t0 = time.perf_counter()
        ctx = run_dream(cfg, root / "out")
        seconds = time.perf_counter() - t0
        original = ctx.original_entries
        after_all, n_all = _apply(original, ctx.ops, safe=False)
        after_safe, n_safe = _apply(original, ctx.ops, safe=True)
        by_type: dict[str, int] = {}
        for o in ctx.ops:
            by_type[o.type] = by_type.get(o.type, 0) + 1
        return {
            "backend": ctx.backend.describe() if ctx.backend else "deterministic",
            "seed": seed, "per_category": per_category,
            "sessions": len(ctx.transcripts), "seconds": round(seconds, 2),
            "operations": {"total": len(ctx.ops), "by_type": by_type, "applied_all": n_all, "applied_safe": n_safe},
            "warnings": list(ctx.warnings),
            "before": score(wl, original),
            "after_all": score(wl, after_all),
            "after_safe": score(wl, after_safe),
        }


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({round(100 * a / b) if b else 0}%)"


def render(result: dict[str, Any]) -> str:
    b, a, s = result["before"], result["after_all"], result["after_safe"]
    ops = result["operations"]
    kinds = ", ".join(f"{n} {t}" for t, n in sorted(ops["by_type"].items())) or "none"
    out = [
        f"agentdream benchmark — backend: {result['backend']}, seed {result['seed']}, "
        f"{b['facts']} facts, {b['notes']} notes, {result['sessions']} sessions, {result['seconds']} s",
        f"operations proposed: {ops['total']} ({kinds}); the safe policy applies {ops['applied_safe']}",
        "",
        "| Metric | Before | After (all operations) | After (safe auto-apply) |",
        "|---|---|---|---|",
        f"| Facts answered (one value, the true one) | {_pct(b['answered'], b['facts'])} | "
        f"{_pct(a['answered'], a['facts'])} | {_pct(s['answered'], s['facts'])} |",
        f"| Facts clean (answered, in a single note) | {_pct(b['clean'], b['facts'])} | "
        f"{_pct(a['clean'], a['facts'])} | {_pct(s['clean'], s['facts'])} |",
        f"| Redundant notes | {b['redundant_notes']} | {a['redundant_notes']} | {s['redundant_notes']} |",
        f"| Notes in the store | {b['notes']} | {a['notes']} | {s['notes']} |",
        f"| Store size (estimated tokens) | {b['tokens']} | {a['tokens']} | {s['tokens']} |",
        f"| Unrelated notes kept verbatim | {_pct(b['unrelated_notes_kept'], b['unrelated_notes'])} | "
        f"{_pct(a['unrelated_notes_kept'], a['unrelated_notes'])} | "
        f"{_pct(s['unrelated_notes_kept'], s['unrelated_notes'])} |",
        "",
        "| Category | Facts | Clean before | Clean after (all) | Clean after (safe) |",
        "|---|---|---|---|---|",
    ]
    for c in CATEGORIES:
        rb, ra, rs = b["by_category"][c], a["by_category"][c], s["by_category"][c]
        out.append(f"| {c} | {rb['facts']} | {rb['clean']} | {ra['clean']} | {rs['clean']} |")
    for w in result["warnings"]:
        out.append(f"\nwarning: {w}")
    return "\n".join(out) + "\n"


__all__ = ["CATEGORIES", "Fact", "Workload", "make_workload", "render", "run_benchmark", "score"]
