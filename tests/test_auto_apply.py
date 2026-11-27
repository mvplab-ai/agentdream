"""Selective auto-apply (types / confidence policy), the /dream skill installer, the benchmark, and the
rule that same-shaped notes about different things are never merged."""

import io
import json

import pytest

from agentdream import accept, dream
from agentdream import claude_hook as ch
from agentdream.bench import CATEGORIES, make_workload, render, run_benchmark, score
from agentdream.cli import main
from agentdream.config import resolve
from agentdream.jobs import DreamJob
from agentdream.models import MemoryEntry
from agentdream.review import ReviewError, policy_split

from conftest import EXAMPLES, hash_tree, write

MEMORY = (
    "- The PokéCenter in Viridian City is north of the gym.\n"
    "- Run the test suite with `pytest -q` before committing.\n"
    "- Run the tests with `pytest -q` before every commit.\n"
)


def cc_line(role, text, ts="2026-11-24T10:00:00Z"):
    return json.dumps({"type": role, "timestamp": ts, "message": {"role": role, "content": text}})


def make_project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    store = tmp_path / "claude-home" / "projects" / "-tmp-proj"
    store.mkdir(parents=True)
    write(proj / "CLAUDE.md", MEMORY)
    session = store / "session-0.jsonl"
    session.write_text("\n".join([
        cc_line("assistant", "Walked north of the gym. No PokéCenter here."),
        cc_line("assistant", "Found the PokéCenter! It is south of the gym.", "2026-11-24T10:05:00Z"),
    ]) + "\n", encoding="utf-8")
    return proj, session


def payload(proj, path, sid):
    return io.StringIO(json.dumps({"session_id": sid, "transcript_path": str(path), "cwd": str(proj),
                                   "hook_event_name": "SessionEnd", "reason": "exit"}))


def example_job():
    return dream(memory=str(EXAMPLES / "memory"), sessions=[str(EXAMPLES / "sessions")])


# ------------------------------------------------------------------ policy

def test_policy_split_by_type_and_confidence():
    ops = example_job().operations
    ok, rest = policy_split(ops, "merge,replace", 0.75)
    by_id = {o.id: o for o in ops}
    assert ok and rest and set(ok) | set(rest) == set(by_id)
    assert all(by_id[i].type in ("merge", "replace") and by_id[i].confidence >= 0.75 for i in ok)
    assert any(by_id[i].type == "correct" for i in rest)
    assert policy_split(ops) == ([o.id for o in ops], [])  # no policy: everything passes
    assert policy_split(ops, min_confidence=1.0)[0] == []


@pytest.mark.parametrize("types, conf", [("merge,bogus", None), ("", None), (None, 1.5), (None, -0.1)])
def test_policy_rejects_bad_values(types, conf):
    with pytest.raises(ReviewError):
        policy_split(example_job().operations, types, conf)


def test_accept_with_policy_applies_only_matching_operations(tmp_path):
    job = example_job()
    res = accept(job, types="merge,replace", min_confidence=0.75, to=str(tmp_path / "out"))
    applied = {o.id: o for o in job.operations if o.id in res.applied_ops}
    assert applied and all(o.type in ("merge", "replace") for o in applied.values())
    assert res.skipped_ops and DreamJob.load(job.id).review == "partially_accepted"
    text = "\n".join(p.read_text(encoding="utf-8") for p in (tmp_path / "out").rglob("*.md"))
    assert "port 9090" in text and "port 8080" not in text  # the stale value was replaced
    assert "north of the gym" in text                         # the correction was left staged


def test_accept_policy_conflicting_with_only_is_an_error(tmp_path):
    job = example_job()
    correction = next(o.id for o in job.operations if o.type == "correct")
    with pytest.raises(ReviewError, match="policy"):
        accept(job, only=[correction], types="merge", to=str(tmp_path / "out"))


def test_accept_policy_matching_nothing_is_an_error(tmp_path):
    with pytest.raises(ReviewError, match="nothing to apply"):
        accept(example_job(), min_confidence=1.0, to=str(tmp_path / "out"))


def test_cli_accept_types(tmp_path, capsys):
    job = example_job()
    assert main(["accept", job.id, "--types", "merge", "--to", str(tmp_path / "out")]) == 0
    assert "applied 1 operation(s)" in capsys.readouterr().out
    assert main(["accept", job.id, "--types", "nonsense", "--force", "--to", str(tmp_path / "out2")]) != 0


# ------------------------------------------------------------------ hook

def test_hook_safe_policy_applies_merges_and_stages_corrections(tmp_path, monkeypatch):
    proj, session = make_project(tmp_path)
    monkeypatch.setenv("AGENTDREAM_HOOK_SYNC", "1")
    opts = ch.HookOptions(every=1, apply=True, apply_types=["merge", "replace"], apply_min_confidence=0.75)
    assert ch.run_hook(opts, payload(proj, session, "s0")) == 0
    text = (proj / "CLAUDE.md").read_text(encoding="utf-8")
    assert text.count("pytest -q") == 1          # the duplicate was merged automatically
    assert "north of the gym" in text            # the correction was not
    job = DreamJob.list(proj / ".agentdream" / "jobs")[0]
    assert job.review == "partially_accepted" and job.backup_path
    assert any(o.type == "correct" for o in job.operations)
    log = (proj / ".agentdream" / "hook.log").read_text(encoding="utf-8")
    assert "left staged by the policy" in log and "types=merge,replace" in log and "confidence>=0.75" in log


def test_hook_policy_matching_nothing_changes_nothing(tmp_path, monkeypatch):
    proj, session = make_project(tmp_path)
    monkeypatch.setenv("AGENTDREAM_HOOK_SYNC", "1")
    before = hash_tree(proj / "CLAUDE.md")
    opts = ch.HookOptions(every=1, apply=True, apply_min_confidence=0.99)
    assert ch.run_hook(opts, payload(proj, session, "s0")) == 0
    assert hash_tree(proj / "CLAUDE.md") == before
    job = DreamJob.list(proj / ".agentdream" / "jobs")[0]
    assert job.status == "completed" and job.review == "pending"
    assert "meet the auto-apply policy" in (proj / ".agentdream" / "hook.log").read_text(encoding="utf-8")


def test_hook_options_round_trip_through_the_command_line():
    opts = ch.HookOptions(every=3, apply=True, apply_types=["merge", "replace"], apply_min_confidence=0.75)
    argv = opts.argv()
    assert argv[argv.index("--apply-types") + 1] == "merge,replace"
    assert argv[argv.index("--apply-min-confidence") + 1] == "0.75"


def test_install_hook_safe_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["install-claude-hook", "--safe", "--every", "3"]) == 0
    out = capsys.readouterr().out
    assert "only types merge, replace and confidence >= 0.75" in out
    cmd = json.loads((tmp_path / ".claude" / "settings.json").read_text())["hooks"]["SessionEnd"][0]["hooks"][0]
    assert "--apply --apply-types merge,replace --apply-min-confidence 0.75" in cmd["command"]


@pytest.mark.parametrize("args", [["--apply-types", "merge"], ["--apply-min-confidence", "0.5"],
                                  ["--apply", "--apply-types", "nope"], ["--apply", "--apply-min-confidence", "2"]])
def test_install_hook_rejects_bad_policy(tmp_path, monkeypatch, capsys, args):
    monkeypatch.chdir(tmp_path)
    assert main(["install-claude-hook", *args]) == 2
    assert "error" in capsys.readouterr().err
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_policy_from_config_file_and_env(tmp_path, monkeypatch):
    write(tmp_path / ".agentdream.toml",
          '[hook]\napply = true\napply_types = ["merge"]\napply_min_confidence = 0.8\n')
    s = resolve({}, cwd=tmp_path)
    assert s.hook["apply_types"] == ["merge"] and s.hook["apply_min_confidence"] == 0.8 and not s.warnings
    monkeypatch.setenv("AGENTDREAM_HOOK_APPLY_TYPES", "merge, replace")
    monkeypatch.setenv("AGENTDREAM_HOOK_APPLY_MIN_CONFIDENCE", "0.6")
    s = resolve({}, cwd=tmp_path)
    assert s.hook["apply_types"] == ["merge", "replace"] and s.hook["apply_min_confidence"] == 0.6
    monkeypatch.setenv("AGENTDREAM_HOOK_APPLY_TYPES", '["replace"]')
    assert resolve({}, cwd=tmp_path).hook["apply_types"] == ["replace"]


# ------------------------------------------------------------------ skill

def test_skill_install_update_uninstall(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".claude" / "skills" / "dream" / "SKILL.md"
    assert main(["install-claude-skill"]) == 0
    text = path.read_text(encoding="utf-8")
    assert text == ch.skill_text() and text.startswith("---\nname: dream\ndescription: ")
    assert "installed" in capsys.readouterr().out
    assert main(["install-claude-skill"]) == 0 and "already up to date" in capsys.readouterr().out
    path.write_text(text + "\nold version\n", encoding="utf-8")          # an older copy of ours is updated
    assert main(["install-claude-skill"]) == 0 and "updated" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8") == ch.skill_text()
    assert main(["uninstall-claude-skill"]) == 0 and not path.exists()
    assert main(["uninstall-claude-skill"]) == 0 and "no /dream skill" in capsys.readouterr().out


def test_skill_install_does_not_overwrite_someone_elses_skill(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".claude" / "skills" / "dream" / "SKILL.md"
    write(path, "---\nname: dream\ndescription: my own dream journal\n---\n")
    assert main(["install-claude-skill"]) != 0
    assert "my own dream journal" in path.read_text(encoding="utf-8")
    assert main(["uninstall-claude-skill"]) == 0 and path.exists()        # and never deletes it
    capsys.readouterr()
    assert main(["install-claude-skill", "--force"]) == 0
    assert path.read_text(encoding="utf-8") == ch.skill_text()


def test_skill_user_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    assert main(["install-claude-skill", "--user"]) == 0
    assert (tmp_path / "home" / ".claude" / "skills" / "dream" / "SKILL.md").is_file()
    assert not (tmp_path / ".claude").exists()


def test_skill_only_names_commands_and_flags_that_exist():
    import re

    from agentdream.cli import build_parser

    text = ch.skill_text()
    helps = {}
    sub = next(a for a in build_parser()._actions if getattr(a, "choices", None))
    for name, parser in sub.choices.items():
        helps[name] = parser.format_help()
    code = "\n".join(re.findall(r"```bash\n(.*?)```", text, re.S))
    commands = re.findall(r"^agentdream ([a-z][a-z\-]+)(.*)$", code, re.M)
    assert len(commands) >= 8
    for cmd, rest in commands:
        assert cmd in helps, cmd
        for flag in re.findall(r"(?<![\w\-])(--[a-z][a-z\-]+)", rest):
            assert flag in helps[cmd], f"agentdream {cmd} has no {flag}"


# ------------------------------------------------------------------ duplicates about different things

def _merged_groups(tmp_path, bullets):
    write(tmp_path / "mem" / "MEMORY.md", "".join(f"- {b}\n" for b in bullets))
    write(tmp_path / "s.jsonl", json.dumps({"role": "user", "content": "hello"}) + "\n")
    job = dream(memory=str(tmp_path / "mem"), sessions=[str(tmp_path / "s.jsonl")])
    return [o.before for o in job.operations if o.type == "merge"]


@pytest.mark.parametrize("bullets", [
    ["The on-call rotation for partner search changes every Monday.",
     "The on-call rotation for sandbox ledger changes every Monday.",
     "The on-call rotation for internal queue changes every Monday."],
    ["Deploys of the shadow queue component go through the `production` branch.",
     "Deploys of the staging exporter component go through the `production` branch."],
    ["Don't use Redis for test environments — it was too slow on CI.",
     "Don't use Webpack for test environments — it was too slow on CI."],
    ["Alice owns the billing dashboard and reviews its alerts weekly.",
     "Bob owns the billing dashboard and reviews its alerts weekly."],
])
def test_same_shape_different_subject_is_not_merged(tmp_path, bullets):
    assert _merged_groups(tmp_path, bullets) == []


@pytest.mark.parametrize("bullets", [
    ["Run the test suite with `pytest -q` before committing.", "Run the tests with `pytest -q` before every commit."],
    ["The staging gateway service listens on port 8443.", "The staging gateway service listens on port 8443"],
    ["Always colour-code the deploy dashboards by region.", "Always color-code the deploy dashboards by region."],
    ["Run `make lint` for the parser module before committing.",
     "Run `make lint` for the parser module before every commit."],
])
def test_real_near_duplicates_are_still_merged(tmp_path, bullets):
    assert len(_merged_groups(tmp_path, bullets)) == 1


def test_correction_keeps_a_direction_word_that_is_part_of_a_name(tmp_path):
    write(tmp_path / "mem" / "MEMORY.md", "- The mailer kiosk is north of the west stairwell.\n")
    rows = [("assistant", "Walked north of the west stairwell. No mailer kiosk here."),
            ("assistant", "Still can't find the mailer kiosk north of the west stairwell."),
            ("assistant", "Found the mailer kiosk! It is just south of the west stairwell."),
            ("assistant", "Restocked the mailer kiosk south of the west stairwell.")]
    write(tmp_path / "s.jsonl", "".join(json.dumps({"role": r, "content": c}) + "\n" for r, c in rows))
    job = dream(memory=str(tmp_path / "mem"), sessions=[str(tmp_path / "s.jsonl")])
    fixes = [o for o in job.operations if o.type == "correct"]
    assert len(fixes) == 1 and "south of the west stairwell" in fixes[0].after[0]
    assert "'south', not 'north'" in fixes[0].reason


# ------------------------------------------------------------------ benchmark

def test_benchmark_workload_is_deterministic_and_labelled(tmp_path):
    a, b = make_workload(tmp_path / "a"), make_workload(tmp_path / "b")
    assert [f.notes for f in a.facts] == [f.notes for f in b.facts]
    assert {f.category for f in a.facts} == set(CATEGORIES)
    files = sorted(p.relative_to(a.memory).as_posix() for p in a.memory.rglob("*.md"))
    assert files == sorted(p.relative_to(b.memory).as_posix() for p in b.memory.rglob("*.md"))
    for f in a.facts:  # every probe recognises its own notes and no other fact's
        assert all(f.probe(n) is not None for n in f.notes), f.id
        others = [n for g in a.facts if g is not f for n in g.notes]
        assert not any(f.probe(n) for n in others), f.id


def test_benchmark_scores_the_untouched_store_as_only_controls_clean(tmp_path):
    wl = make_workload(tmp_path, per_category=4)
    entries = [MemoryEntry(id=str(i), content=n) for i, n in enumerate(n for f in wl.facts for n in f.notes)]
    s = score(wl, entries + [MemoryEntry(id=f"x{i}", content=t) for i, t in enumerate(wl.filler)])
    assert s["clean"] == s["by_category"]["control"]["clean"] == 8
    assert s["unrelated_notes_kept"] == s["unrelated_notes"] == 4
    assert s["redundant_notes"] == 4 * 4  # duplicate, near_duplicate, stale and paraphrase have two notes each


def test_benchmark_deterministic_backend_improves_the_store_without_harm():
    r = run_benchmark()
    before, after, safe = r["before"], r["after_all"], r["after_safe"]
    assert before["clean"] == 20 and before["answered"] == 50 and before["facts"] == 80
    # every control and every unrelated note survives, under both policies
    for s in (after, safe):
        assert s["by_category"]["control"]["clean"] == 20
        assert s["unrelated_notes_kept"] == s["unrelated_notes"] == 10
        assert s["answered"] >= before["answered"] and s["clean"] > before["clean"]
    for cat in ("duplicate", "near_duplicate", "stale"):
        assert after["by_category"][cat]["clean"] == safe["by_category"][cat]["clean"] == 10
    assert after["by_category"]["wrong"]["clean"] == after["by_category"]["reversal"]["clean"] == 10
    assert after["by_category"]["paraphrase"]["clean"] == 0  # rule-based matching does not see paraphrases
    # the safe policy never applies a transcript-based rewrite
    assert safe["by_category"]["wrong"]["clean"] == safe["by_category"]["reversal"]["clean"] == 0
    assert r["operations"]["applied_safe"] < r["operations"]["applied_all"]


def test_benchmark_cli(tmp_path, capsys):
    assert main(["benchmark", "--facts", "2", "--keep", str(tmp_path / "bench")]) == 0
    out = capsys.readouterr().out
    assert "| Facts answered" in out and "| paraphrase | 2 |" in out
    assert (tmp_path / "bench" / "memory").is_dir() and (tmp_path / "bench" / "out" / "report.md").is_file()
    assert main(["benchmark", "--facts", "2", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["before"]["facts"] == 16 and render(data).startswith("agentdream benchmark")
    assert main(["benchmark", "--facts", "99"]) == 2
