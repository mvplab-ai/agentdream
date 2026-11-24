"""Live smoke test against the real `claude` CLI. Skipped unless AGENTDREAM_LIVE=1 and `claude` is on PATH."""

import os
import shutil

import pytest

from agentdream import dream

from conftest import final_contents, write, write_jsonl

pytestmark = pytest.mark.live


@pytest.mark.skipif(os.environ.get("AGENTDREAM_LIVE") != "1" or shutil.which("claude") is None,
                    reason="set AGENTDREAM_LIVE=1 and install/log in to the claude CLI to run")
def test_claude_cli_live(tmp_path):
    mem = write(tmp_path / "m" / "MEMORY.md", "- The PokéCenter in Viridian City is north of the gym.\n").parent
    s = tmp_path / "s"
    write_jsonl(s / "1.jsonl", [
        {"role": "assistant", "content": "Walked north of the gym. No PokéCenter here."},
        {"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."},
    ])
    job = dream(mem, [s], backend="claude-cli")
    assert job.status == "completed", job.error
    assert job.report["backend"].startswith("claude-cli")
    ops = job.report.get("operations", [])
    assert any(o.get("origin") == "claude-cli" for o in ops), job.report.get("warnings")
    store_text = "\n".join(final_contents(job))
    assert "south" in store_text.lower(), store_text
    assert "north of the gym" in (mem / "MEMORY.md").read_text()  # input untouched
