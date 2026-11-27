"""The GitHub Action example stays a dry run unless apply is the confirmation string."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from agentdream.cli import build_parser

from conftest import hash_tree

ROOT = Path(__file__).resolve().parent.parent
ACTION = ROOT / "examples" / "integrations" / "github-action"
RUN = ACTION / "run.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "dream-example.yml"
CONFIRM = "I-understand-this-writes-the-memory-store"

_CLEARED = (
    "GITHUB_WORKSPACE", "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "RUNNER_TEMP",
    "INPUT_MEMORY", "INPUT_SESSIONS", "INPUT_BACKEND", "INPUT_MODEL", "INPUT_JOBS_DIR",
    "INPUT_OUT", "INPUT_INSTALL", "INPUT_FAIL_ON_CORRECTIONS", "INPUT_APPLY",
    "INPUT_EXTRA_ARGS", "INPUT_ARTIFACT_DIR", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
)


def _run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in _CLEARED:
        env.pop(key, None)
    env["INPUT_INSTALL"] = "skip"
    env.update(overrides)
    return subprocess.run(
        ["bash", str(RUN)], cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
    )


def _outputs(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        out[key] = value
    return out


def test_run_sh_syntax():
    subprocess.run(["bash", "-n", str(RUN)], check=True)
    assert os.access(RUN, os.X_OK)


def test_action_readme_links_resolve():
    text = (ACTION / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((?!https?://)([^)\s#]+)", text)
    assert links
    for link in links:
        assert (ACTION / link).resolve().exists(), link


def test_action_defaults_are_a_dry_run():
    text = (ACTION / "action.yml").read_text(encoding="utf-8")
    assert "default: deterministic" in text
    assert "default: \"false\"" in text
    assert CONFIRM in text
    assert "--apply" in text
    assert "always()" in text
    assert "actions/upload-artifact@v4" in text
    script = RUN.read_text(encoding="utf-8")
    assert "eval " not in script
    assert CONFIRM in script


def test_workflow_is_a_dry_run_with_read_permission():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "contents: read" in text
    assert "uses: ./examples/integrations/github-action" in text
    assert "examples/memory" in text
    assert "examples/sessions" in text
    assert "runner.temp" in text
    assert "I-understand-this-writes-the-memory-store" not in text
    assert "pull_request_target" not in text


def test_documented_show_command_parses():
    """The sample workflow does not shell out, but the action README's dream is the real CLI."""
    build_parser().parse_args([
        "dream", "--memory", "examples/memory", "--sessions", "examples/sessions",
        "--backend", "deterministic", "--jobs-dir", "/tmp/agentdream-jobs", "--json",
    ])


def test_missing_inputs_and_accidental_apply_are_refused(tmp_path):
    missing = _run(tmp_path)
    assert missing.returncode == 2
    assert "memory" in missing.stderr

    memory = ROOT / "examples" / "memory"
    before = hash_tree(memory)
    refused = _run(
        tmp_path,
        INPUT_MEMORY=str(memory),
        INPUT_SESSIONS=str(ROOT / "examples" / "sessions"),
        INPUT_APPLY="true",
    )
    assert refused.returncode == 2
    assert CONFIRM in refused.stderr
    assert hash_tree(memory) == before

    extra = _run(
        tmp_path,
        INPUT_MEMORY=str(memory),
        INPUT_SESSIONS=str(ROOT / "examples" / "sessions"),
        INPUT_EXTRA_ARGS="--apply",
    )
    assert extra.returncode == 2
    assert "--apply" in extra.stderr
    assert hash_tree(memory) == before

    broken = _run(
        tmp_path,
        INPUT_MEMORY=str(memory),
        INPUT_SESSIONS=str(ROOT / "examples" / "sessions"),
        INPUT_EXTRA_ARGS="'--max-sessions",
    )
    assert broken.returncode == 2
    assert "extra-args" in broken.stderr
    assert hash_tree(memory) == before


def test_jobs_dir_inside_the_memory_store_is_refused(tmp_path):
    memory = ROOT / "examples" / "memory"
    before = hash_tree(memory)
    proc = _run(
        tmp_path,
        INPUT_MEMORY=str(memory),
        INPUT_SESSIONS=str(ROOT / "examples" / "sessions"),
        INPUT_JOBS_DIR=str(memory / "jobs"),
    )
    assert proc.returncode == 2
    assert "overlaps" in proc.stderr
    assert hash_tree(memory) == before


def test_anthropic_without_a_key_stops_before_the_dream(tmp_path):
    proc = _run(
        tmp_path,
        INPUT_MEMORY=str(ROOT / "examples" / "memory"),
        INPUT_SESSIONS=str(ROOT / "examples" / "sessions"),
        INPUT_BACKEND="anthropic",
    )
    assert proc.returncode == 2
    assert "ANTHROPIC_API_KEY" in proc.stderr
    assert "sk-" not in proc.stderr + proc.stdout


def test_dry_run_on_bundled_examples_uploads_reports(tmp_path):
    memory = ROOT / "examples" / "memory"
    before = hash_tree(memory)
    art = tmp_path / "artifact"
    jobs = tmp_path / "jobs"
    output = tmp_path / "github-output.txt"
    proc = _run(
        tmp_path,
        INPUT_MEMORY=str(memory),
        INPUT_SESSIONS=str(ROOT / "examples" / "sessions"),
        INPUT_BACKEND="deterministic",
        INPUT_JOBS_DIR=str(jobs),
        INPUT_ARTIFACT_DIR=str(art),
        INPUT_FAIL_ON_CORRECTIONS="false",
        GITHUB_OUTPUT=str(output),
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert hash_tree(memory) == before
    report = json.loads((art / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["corrected"] >= 1
    assert (art / "report.md").is_file()
    assert "deterministic" in (art / "report.md").read_text(encoding="utf-8")
    fields = _outputs(output)
    assert fields["status"] == "completed"
    assert fields["report-md"] == str(art / "report.md")
    assert int(fields["corrected"]) == report["counts"]["corrected"]
    assert fields["job-id"]
    # The staged store is outside the memory tree.
    assert not str(jobs).startswith(str(memory))


def test_fail_on_corrections_keeps_the_report_and_the_store(tmp_path):
    memory = ROOT / "examples" / "memory"
    before = hash_tree(memory)
    art = tmp_path / "artifact"
    proc = _run(
        tmp_path,
        INPUT_MEMORY=str(memory),
        INPUT_SESSIONS=str(ROOT / "examples" / "sessions"),
        INPUT_JOBS_DIR=str(tmp_path / "jobs"),
        INPUT_ARTIFACT_DIR=str(art),
        INPUT_FAIL_ON_CORRECTIONS="true",
    )
    assert proc.returncode == 1
    assert "fail-on-corrections" in proc.stderr
    assert (art / "report.md").is_file()
    assert (art / "report.json").is_file()
    assert hash_tree(memory) == before


def test_confirmation_string_applies_to_a_copy(tmp_path):
    memory = tmp_path / "memory"
    shutil.copytree(ROOT / "examples" / "memory", memory)
    before = hash_tree(memory)
    art = tmp_path / "artifact"
    proc = _run(
        tmp_path,
        INPUT_MEMORY=str(memory),
        INPUT_SESSIONS=str(ROOT / "examples" / "sessions"),
        INPUT_JOBS_DIR=str(tmp_path / "jobs"),
        INPUT_ARTIFACT_DIR=str(art),
        INPUT_APPLY=CONFIRM,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "applied" in proc.stdout
    assert hash_tree(memory) != before
    assert (art / "report.md").is_file()
