"""Run a dry-run dream over the example memory + sessions and print the report.

    python examples/run_example.py

Nothing in examples/ is modified; output goes to a staging directory under
.agentdream/ (or $AGENTDREAM_HOME).
"""

from pathlib import Path

from agentdream import dream

HERE = Path(__file__).resolve().parent


def main() -> None:
    import os

    os.chdir(HERE.parent)  # keep citations short and relative: examples/sessions/...
    job = dream(memory="examples/memory", sessions=["examples/sessions"], backend="deterministic")
    print(job.report_md)
    print(f"\nstaged store: {job.store_path}\naccept with: agentdream accept {job.id}")


if __name__ == "__main__":
    main()
