"""Deterministic synthetic workloads for performance tests (no randomness across runs)."""

from __future__ import annotations

import json
import random
from pathlib import Path

_NOUNS = ("cache", "router", "parser", "worker", "queue", "bucket", "gateway", "scheduler", "indexer", "crawler",
          "ledger", "mailer", "billing", "search", "auth", "uploader", "renderer", "notifier", "exporter", "importer")
_ADJ = ("primary", "legacy", "staging", "regional", "internal", "public", "nightly", "shadow", "edge", "batch")
_TOOLS = ("pytest", "ruff", "docker", "terraform", "poetry", "pnpm", "yarn", "make", "cargo", "helm")
_PLACES = ("north", "south", "east", "west")


def _word(rng: random.Random, n: int = 6) -> str:
    return "".join(rng.choice("bcdfghjklmnprstvz") + rng.choice("aeiou") for _ in range(n // 2))


def make_large_workspace(root: Path, n_entries: int = 3000, n_sessions: int = 200, msgs_per_session: int = 24,
                         seed: int = 7) -> tuple[Path, Path]:
    """A markdown memory folder with ``n_entries`` bullets (some duplicates / stale values / preference flips)
    plus ``n_sessions`` JSONL transcripts. Returns (memory_dir, sessions_dir)."""
    rng = random.Random(seed)
    mem = root / "mem"
    ses = root / "sessions"
    mem.mkdir(parents=True)
    ses.mkdir(parents=True)
    names = [f"{rng.choice(_ADJ)} {rng.choice(_NOUNS)} {_word(rng)}" for _ in range(n_entries // 3 + 10)]
    files: dict[str, list[str]] = {}
    for i in range(n_entries):
        name = names[i % len(names)]
        kind = i % 10
        if kind < 5:
            line = f"The {name} service runs on port {rng.randint(1000, 9999)}."
        elif kind < 7:
            line = f"The {name} dashboard is {rng.choice(_PLACES)} of the {_word(rng)} panel."
        elif kind == 7:
            line = f"Use {rng.choice(_TOOLS)} for the {name} pipeline."
        elif kind == 8:
            line = f"Don't use {rng.choice(_TOOLS)} for the {name} deploys."
        else:
            line = f"Remember that {name} owners are {_word(rng)} and {_word(rng)} ({_word(rng, 8)})."
        if i % 37 == 0 and i:  # exact-ish duplicate of an earlier note
            line = files[f"notes/{(i - 1) % 30:02d}.md"][-1]
        files.setdefault(f"notes/{i % 30:02d}.md", []).append(line)
    for rel, lines in files.items():
        p = mem / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Notes\n\n" + "".join(f"- {ln}\n" for ln in lines), encoding="utf-8")

    for s in range(n_sessions):
        rows = []
        for m in range(msgs_per_session):
            name = rng.choice(names)
            r = m % 6
            if r == 0:
                text = f"Please check the {name} service, it failed again with a timeout."
            elif r == 1:
                text = (f"Ran `pytest -q` then `ruff check` for {name}. "
                        f"The {name} service runs on port {rng.randint(1000, 9999)}.")
            elif r == 2:
                text = f"Walked {rng.choice(_PLACES)} of the {_word(rng)} panel. No {name} dashboard here."
            elif r == 3:
                text = (f"Found it: the {name} dashboard is {rng.choice(_PLACES)} of the panel. "
                        f"Use {rng.choice(_TOOLS)} next time.")
            elif r == 4:
                text = f"Error: {_word(rng, 8)} missing in {name} config; forgot to set {_word(rng)}."
            else:
                text = f"Let's keep using {rng.choice(_TOOLS)} for {name}; {_word(rng)} {_word(rng)} {_word(rng)}."
            rows.append({"role": "user" if m % 2 == 0 else "assistant", "content": text,
                         "timestamp": f"2026-{1 + s % 12:02d}-{1 + m % 28:02d}T{s % 24:02d}:00:00Z"})
        (ses / f"session-{s:04d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return mem, ses
