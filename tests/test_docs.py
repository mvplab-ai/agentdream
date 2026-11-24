"""The docs must stay true: runnable snippets run, every `agentdream ...` command parses against the real
CLI, Python blocks compile, relative links resolve, and Mermaid diagrams pass a structural lint."""

import ast
import contextlib
import io
import re
import shlex
import shutil
from pathlib import Path

import pytest

from agentdream.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted([ROOT / "README.md", ROOT / "CONTRIBUTING.md", *(ROOT / "docs").glob("*.md"),
               *(ROOT / "examples" / "integrations").glob("*.md")])
FENCE = re.compile(r"(?P<marker><!-- test: run -->\s*\n)?```(?P<lang>[\w-]*)[^\n]*\n(?P<body>.*?)\n```", re.S)


def blocks(lang=None):
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            if lang is None or m.group("lang") == lang:
                line = text[: m.start()].count("\n") + 1
                yield doc, line, m.group("lang"), m.group("body"), bool(m.group("marker"))


def _id(item):
    doc, line, *_ = item
    return f"{doc.relative_to(ROOT)}:{line}"


RUNNABLE = [b for b in blocks("python") if b[4]]
PYTHON = list(blocks("python"))
SHELL = [b for b in blocks() if b[2] in ("bash", "sh", "shell", "console")]
MERMAID = list(blocks("mermaid"))


def test_docs_have_runnable_snippets_and_diagrams():
    assert len(RUNNABLE) >= 8
    assert {b[0].name for b in RUNNABLE} >= {"README.md", "ARCHITECTURE.md", "BACKENDS.md", "CONFIG.md"}
    assert len(MERMAID) >= 4
    kinds = {b[3].strip().split()[0] for b in MERMAID}
    assert {"flowchart", "sequenceDiagram", "stateDiagram-v2"} <= kinds


@pytest.mark.parametrize("item", RUNNABLE, ids=_id)
def test_runnable_snippet(item, tmp_path, monkeypatch):
    doc, line, _lang, body, _ = item
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(body, f"{doc.name}:{line}", "exec"), {"__name__": "__doc_snippet__"})
    assert out.getvalue().strip(), "snippets should print something to show what they do"


@pytest.mark.parametrize("item", PYTHON, ids=_id)
def test_python_blocks_compile(item):
    ast.parse(item[3])


def _cli_invocations(body):
    """Yield argv lists for every `agentdream ...` invocation in a shell block."""
    joined = re.sub(r"\\\n\s*", " ", body)
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        segment: list[str] = []
        for w in [*words, "|"]:
            if w in ("|", "||", "&&", ";"):
                while segment and re.match(r"^[A-Z_][A-Z0-9_]*=", segment[0]):
                    segment.pop(0)  # VAR=value prefixes
                if segment and segment[0] == "agentdream":
                    yield segment[1:]
                elif "--" in segment and segment[segment.index("--") + 1:][:1] == ["agentdream"]:
                    yield segment[segment.index("--") + 2:]  # claude mcp add NAME -- agentdream serve-mcp
                segment = []
            else:
                segment.append(w)


CLI_LINES = [(doc, line, argv) for doc, line, _l, body, _r in SHELL for argv in _cli_invocations(body)]


def test_docs_mention_cli_commands():
    cmds = {argv[0] for _d, _l, argv in CLI_LINES if argv and not argv[0].startswith("-")}
    assert cmds >= {"dream", "show", "accept", "discard", "restore", "schedule", "serve-mcp",
                    "install-claude-hook", "uninstall-claude-hook", "list", "config", "hook"}


@pytest.mark.parametrize(("doc", "line", "argv"), CLI_LINES,
                         ids=[f"{d.relative_to(ROOT)}:{n}:{' '.join(a[:2])}" for d, n, a in CLI_LINES])
def test_documented_cli_lines_parse(doc, line, argv):
    if argv in (["--version"], ["--help"]):
        return
    try:
        build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse error
        pytest.fail(f"{doc.name}:{line}: `agentdream {' '.join(argv)}` does not parse ({exc})")


@pytest.mark.parametrize("cmd", sorted({a[0] for _d, _l, a in CLI_LINES if a and not a[0].startswith("-")}))
def test_cli_help_for_documented_commands(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([cmd, "--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)\s#]+)(#[^)\s]*)?\)")


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: str(d.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    text = doc.read_text(encoding="utf-8")
    for m in LINK.finditer(text):
        target = (doc.parent / m.group(1)).resolve()
        assert target.exists(), f"{doc.name}: broken link {m.group(1)}"
        anchor = m.group(2)
        if anchor and target.suffix == ".md":
            heads = {_slug(h) for h in re.findall(r"^#+\s+(.+)$", target.read_text(encoding="utf-8"), re.M)}
            assert anchor[1:] in heads, f"{doc.name}: missing anchor {m.group(1)}{anchor}"


def _slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[`*_]", "", s)
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def test_no_pypi_claims():
    for doc in DOCS:
        for line in doc.read_text(encoding="utf-8").splitlines():
            assert not re.search(r"pip install agentdream(\s|$)", line), f"{doc.name}: claims PyPI: {line}"


# ------------------------------------------------------------------ mermaid lint

_FLOW_SKIP = re.compile(r"^(subgraph\b|end$|direction\b|classDef\b|class\b|style\b|linkStyle\b|%%)")


def _strip_quoted(line: str) -> str:
    return re.sub(r'"[^"]*"', '""', line)


def lint_mermaid(src: str) -> list[str]:
    errors = []
    lines = [ln.rstrip() for ln in src.strip().splitlines()]
    kind = lines[0].split()[0]
    if kind not in ("flowchart", "graph", "sequenceDiagram", "stateDiagram-v2"):
        return [f"unsupported diagram type {kind!r}"]
    for n, raw in enumerate(lines[1:], start=2):
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue
        if line.count('"') % 2:
            errors.append(f"line {n}: unbalanced quotes")
        if ";" in line:
            errors.append(f"line {n}: ';' ends a statement in Mermaid")
        if kind in ("flowchart", "graph"):
            if _FLOW_SKIP.match(line):
                if line.startswith("subgraph") and "[" in line and '["' not in line:
                    errors.append(f"line {n}: quote subgraph titles")
                continue
            bare = _strip_quoted(line)
            if re.search(r"\w[\[\(\{](?!\")", bare.replace('[(""', '["').replace('[""', '["')
                         .replace('{""', '{"').replace('(""', '("')):
                errors.append(f"line {n}: unquoted node label: {line}")
            if re.search(r"\|[^|\"]+\|", bare):
                errors.append(f"line {n}: unquoted edge label: {line}")
        elif kind == "sequenceDiagram":
            m = re.match(r"^(\w+)\s*(-{1,2}>>|-{1,2}>|-{1,2}x|-{1,2}\))\s*[+-]?(\w+)\s*:\s*(.+)$", line)
            if m and "#" in m.group(4):
                errors.append(f"line {n}: '#' in message text")
        elif kind == "stateDiagram-v2" and re.match(r"^\S+\s*-->\s*\S+\s*:\s*$", line):
            errors.append(f"line {n}: empty transition label")
        elif kind == "stateDiagram-v2" and re.match(r"^\S+\s*-->\s*\S+\s*:.*--", line):
            errors.append(f"line {n}: '--' inside a transition label")
    if kind == "sequenceDiagram":
        declared = set(re.findall(r"^\s*(?:participant|actor)\s+(\w+)", src, re.M))
        for a, b in re.findall(r"^\s*(\w+)\s*-{1,2}>>?[+-]?(\w+)\s*:", src, re.M):
            for p in (a, b):
                if p not in declared:
                    errors.append(f"undeclared participant {p}")
    if kind == "stateDiagram-v2" and src.count("{") != src.count("}"):
        errors.append("unbalanced braces")
    if kind in ("flowchart", "graph"):
        opens = sum(1 for ln in lines if ln.strip().startswith("subgraph"))
        ends = sum(1 for ln in lines if ln.strip() == "end")
        if opens != ends:
            errors.append("unbalanced subgraph/end")
    return errors


@pytest.mark.parametrize("item", MERMAID, ids=_id)
def test_mermaid_diagrams_lint_clean(item):
    assert lint_mermaid(item[3]) == []


def test_mermaid_lint_catches_mistakes():
    assert lint_mermaid("flowchart LR\n  a[load (read-only)] --> b") != []
    assert lint_mermaid('flowchart LR\n  a["ok"] -->|unquoted| b["x"]') != []
    assert lint_mermaid('flowchart LR\n  subgraph s["x"]\n  a["y"]') != []
    assert lint_mermaid("sequenceDiagram\n  participant A\n  A->>B: hi") != []
    assert lint_mermaid("sequenceDiagram\n  participant A\n  A->>A: a; b") != []
    assert lint_mermaid("pie\n  a: 1") != []
    assert lint_mermaid('flowchart LR\n  a["fine"] -->|"label"| b[("db")]') == []
