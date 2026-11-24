"""Config files, env vars and precedence: CLI > env > project config > user config > defaults."""

import json
import sys

import pytest

from agentdream import _toml
from agentdream.cli import main
from agentdream.config import (
    ConfigError,
    coerce,
    env_config,
    find_project_config,
    parse_config,
    read_config_file,
    resolve,
)
from agentdream.mcp_server import MCPServer

from conftest import write, write_jsonl

FULL = r'''
# agentdream config
backend = "anthropic"
model = 'claude-sonnet-5-5'
memory = "./mem"
sessions = [
  "./logs/*.jsonl",   # trailing comment
  "~/other",
]
instructions = """focus on \
deploys"""

[limits]
max_sessions = 1_0
max_chunk_tokens = 32000
on_limit = "fail"

[redaction]
emails = true
entropy_threshold = 4.5
patterns = ['ACME-\d+', "x\ty"]

[backend_options]
base_url = "http://localhost:11434/v1"
extra = { retries = 2, verbose = false }

[schedule]
every = "6h"
new_sessions = 3

[hook]
every = 2
apply = false
'''


def test_mini_toml_matches_tomllib():
    data = _toml.mini_loads(FULL)
    assert data["sessions"] == ["./logs/*.jsonl", "~/other"]
    assert data["instructions"] == "focus on deploys"
    assert data["limits"] == {"max_sessions": 10, "max_chunk_tokens": 32000, "on_limit": "fail"}
    assert data["redaction"]["patterns"] == ["ACME-\\d+", "x\ty"]
    assert data["backend_options"]["extra"] == {"retries": 2, "verbose": False}
    if sys.version_info >= (3, 11):
        import tomllib

        assert data == tomllib.loads(FULL)


@pytest.mark.parametrize(("text", "msg"), [
    ("a = ", "expected a value"), ("a = 1\na = 2", "duplicate key"), ("[t\nx=1", "expected ']'"),
    ("a = \"open", "unterminated"), ("a = [1 2]", "expected ','"), ("[[arr]]\n", "arrays of tables"),
    ("a = 1979-05-27", "dates"), ("a = 1 b", "unexpected"), ("a = 'x\ny'", "unterminated"),
    ('a = "\\q"', "bad escape"), ("= 1", "expected a key"), ("a 1", "expected '='"),
])
def test_mini_toml_errors_have_line_numbers(text, msg):
    with pytest.raises(_toml.TOMLError, match=msg) as exc:
        _toml.mini_loads(text)
    assert "line" in str(exc.value)


def test_mini_toml_misc_values():
    d = _toml.mini_loads("a = -3\nb = 2.5e3\nc = 0x1f\nd = true\ne = []\nf = {}\ng = '''\nraw\\n'''\n"
                         "h = \"\\u00e9\"\n'quoted key' = 1\nx.y = 2\n[t.u]\nv = [[1], [2, 3]]\n")
    assert d == {"a": -3, "b": 2500.0, "c": 31, "d": True, "e": [], "f": {}, "g": "raw\\n", "h": "é",
                 "quoted key": 1, "x": {"y": 2}, "t": {"u": {"v": [[1], [2, 3]]}}}


def test_loads_falls_back_without_tomllib(monkeypatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", None)
    assert _toml.loads("a = 1\n[b]\nc = 'x'\n") == {"a": 1, "b": {"c": "x"}}
    with pytest.raises(_toml.TOMLError):
        _toml.loads("a = ")


def test_tomllib_errors_are_wrapped():
    if sys.version_info < (3, 11):
        pytest.skip("tomllib needs 3.11")
    with pytest.raises(_toml.TOMLError):
        _toml.loads("a = ")


def test_parse_config_normalises_and_resolves_paths(tmp_path):
    dream, schedule, hook, warns = parse_config(_toml.mini_loads(FULL), "cfg", tmp_path)
    assert dream["memory"] == str(tmp_path / "mem")
    assert dream["sessions"][0] == str(tmp_path / "logs" / "*.jsonl")
    assert not dream["sessions"][1].startswith("~")
    assert dream["max_sessions"] == 10 and dream["on_limit"] == "fail" and dream["max_chunk_tokens"] == 32000
    assert dream["redact_emails"] is True and dream["entropy_threshold"] == 4.5
    assert dream["redact_patterns"] == ["ACME-\\d+", "x\ty"]
    assert dream["backend_options"]["base_url"].startswith("http://localhost")
    assert schedule == {"every": "6h", "new_sessions": 3} and hook == {"every": 2, "apply": False}
    assert warns == []
    d2, _s, _h, warns = parse_config({"memory": "stancelog:rel.db", "nope": 1, "limits": {"bogus": 1},
                                      "redaction": {"zzz": 1}, "schedule": {"x": 1}, "hook": {"y": 1}},
                                     "f.toml", tmp_path)
    assert d2["memory"] == "stancelog:" + str(tmp_path / "rel.db")
    assert len(warns) == 5 and all("unknown key" in w for w in warns)


def test_coercion_errors_name_the_location():
    with pytest.raises(ConfigError, match=r"cfg: max_sessions must be int"):
        coerce("max_sessions", "lots", "cfg")
    with pytest.raises(ConfigError, match="bool"):
        coerce("redact", "maybe", "$AGENTDREAM_REDACT")
    with pytest.raises(ConfigError):
        coerce("max_sessions", True, "x")
    with pytest.raises(ConfigError):
        coerce("entropy_threshold", "hot", "x")
    with pytest.raises(ConfigError):
        coerce("backend", ["a"], "x")
    with pytest.raises(ConfigError):
        coerce("backend_options", "[1]", "x")
    with pytest.raises(ConfigError):
        coerce("sessions", "[1, {]", "x")
    assert coerce("redact", "off", "x") is False and coerce("redact", "YES", "x") is True
    assert coerce("sessions", '["a", "b"]', "x") == ["a", "b"]
    assert coerce("redact_patterns", "A,B", "x") == ["A,B"]
    assert coerce("backend_options", '{"base_url": "u"}', "x") == {"base_url": "u"}


def test_env_config(monkeypatch, tmp_path):
    import os

    env = {"AGENTDREAM_BACKEND": "openai", "AGENTDREAM_MAX_SESSIONS": "7", "AGENTDREAM_REDACT_EMAILS": "1",
           "AGENTDREAM_SESSIONS": os.pathsep.join(["a", "b"]), "AGENTDREAM_BASE_URL": "http://x/v1",
           "AGENTDREAM_HOOK_EVERY": "4", "AGENTDREAM_HOOK_APPLY": "true", "AGENTDREAM_MEMORY": "rel/mem",
           "AGENTDREAM_BACKEND_OPTIONS": '{"timeout": 5}'}
    monkeypatch.chdir(tmp_path)
    dream, hook = env_config(env)
    assert dream["backend"] == "openai" and dream["max_sessions"] == 7 and dream["redact_emails"] is True
    assert dream["sessions"] == ["a", "b"]
    assert dream["backend_options"] == {"timeout": 5, "base_url": "http://x/v1"}
    assert dream["memory"] == str(tmp_path / "rel" / "mem")
    assert hook == {"every": 4, "apply": True}


def test_find_project_config_walks_up(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert find_project_config(deep) is None
    write(tmp_path / "pyproject.toml", "[project]\nname = 'x'\n")
    assert find_project_config(deep) is None  # no [tool.agentdream] table
    write(tmp_path / "pyproject.toml", "[project]\nname = 'x'\n\n[tool.agentdream]\nbackend = 'openai'\n")
    assert find_project_config(deep) == tmp_path / "pyproject.toml"
    assert read_config_file(tmp_path / "pyproject.toml") == {"backend": "openai"}
    write(tmp_path / "a" / ".agentdream.toml", "backend = 'anthropic'\n")
    assert find_project_config(deep) == tmp_path / "a" / ".agentdream.toml"  # nearest wins


def test_bad_config_files(tmp_path):
    write(tmp_path / ".agentdream.toml", "backend = \n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        resolve(cwd=tmp_path)
    with pytest.raises(ConfigError, match="not found"):
        resolve(config_path=tmp_path / "missing.toml")
    with pytest.raises(ConfigError, match="cannot read"):
        read_config_file(tmp_path / "nope.toml")
    (tmp_path / ".agentdream.toml").unlink()
    with pytest.raises(ConfigError, match="unknown option"):
        resolve({"bogus": 1}, cwd=tmp_path)


def test_precedence_cli_env_project_user_defaults(tmp_path, monkeypatch):
    user = tmp_path / ".xdg-config" / "agentdream" / "config.toml"
    write(user, "backend = 'claude-cli'\nmodel = 'user-model'\ninstructions = 'from user'\n"
                "[limits]\nmax_chunks = 3\n")
    proj = tmp_path / "proj"
    write(proj / ".agentdream.toml", "backend = 'openai'\nmodel = 'proj-model'\n[limits]\nmax_sessions = 5\n"
                                     "max_total_tokens = 1000\n")
    env = {"XDG_CONFIG_HOME": str(tmp_path / ".xdg-config"), "AGENTDREAM_MAX_SESSIONS": "7",
           "AGENTDREAM_MODEL": "env-model"}
    s = resolve({"max_total_tokens": 9}, cwd=proj, environ=env)
    eff = s.effective_dream()
    assert eff["backend"] == "openai"            # project > user
    assert eff["instructions"] == "from user"    # user > default
    assert eff["max_chunks"] == 3
    assert eff["model"] == "env-model"           # env > project
    assert eff["max_sessions"] == 7              # env > project
    assert eff["max_total_tokens"] == 9          # cli > project
    assert eff["chunk_overlap"] == 400           # default
    assert s.sources["dream.backend"].endswith(".agentdream.toml") and s.sources["dream.model"] == "env"
    assert s.sources["dream.max_total_tokens"] == "cli"
    assert s.files == [str(user), str(proj / ".agentdream.toml")]
    no_user = resolve(cwd=proj, environ=env, user_config=False)
    assert "instructions" not in no_user.dream
    explicit = resolve(cwd=proj, environ={**env, "AGENTDREAM_CONFIG": str(user)})
    assert explicit.effective_dream()["backend"] == "claude-cli"


def test_cli_config_command_and_precedence(tmp_path, monkeypatch, capsys):
    proj = tmp_path / "proj"
    write(proj / ".agentdream.toml", "backend = 'openai'\n[limits]\nmax_sessions = 5\n[schedule]\nevery = '1h'\n"
                                     "[hook]\nevery = 2\nunknown = 1\n")
    monkeypatch.chdir(proj)
    monkeypatch.setenv("AGENTDREAM_MAX_SESSIONS", "7")
    assert main(["config", "--json"]) == 0
    out = capsys.readouterr()
    d = json.loads(out.out)
    assert d["dream"]["backend"] == "openai" and d["dream"]["max_sessions"] == 7
    assert d["schedule"] == {"every": "1h"} and d["hook"] == {"every": 2}
    assert "unknown key hook.unknown" in out.err
    assert main(["config"]) == 0
    text = capsys.readouterr().out
    assert "backend = 'openai'" in text and "[env]" in text and "schedule.every" in text


def test_cli_dream_takes_memory_and_sessions_from_config(tmp_path, monkeypatch, capsys):
    proj = tmp_path / "proj"
    write(proj / "mem" / "MEMORY.md", "- Use pytest.\n- Use pytest!\n")
    write_jsonl(proj / "logs" / "1.jsonl", [{"role": "user", "content": "hi"}])
    write(proj / ".agentdream.toml", "memory = 'mem'\nsessions = ['logs']\njobs_dir = '.jobs'\n"
                                     "[limits]\nmax_sessions = 3\n")
    monkeypatch.chdir(proj / "logs")  # config is found by walking up; paths are relative to the file
    assert main(["dream", "--json"]) == 0
    job = json.loads(capsys.readouterr().out)
    assert job["status"] == "completed"
    assert job["config"]["memory"] == str(proj / "mem") and job["config"]["max_sessions"] == 3
    assert (proj / ".jobs" / job["id"] / "job.json").exists()
    assert main(["list"]) == 0 and job["id"] in capsys.readouterr().out  # jobs_dir from config too
    assert main(["dream", "--json", "--max-sessions", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["max_sessions"] == 2
    write(proj / ".agentdream.toml", "sessions = ['logs']\n")
    assert main(["dream"]) == 2
    assert "no memory store" in capsys.readouterr().err
    write(proj / ".agentdream.toml", "[limits]\nmax_sessions = 'many'\n")
    assert main(["dream", "-m", str(proj / "mem")]) == 2
    assert "max_sessions must be int" in capsys.readouterr().err


def test_schedule_section_is_used(tmp_path, monkeypatch, capsys):
    proj = tmp_path / "proj"
    write(proj / "mem" / "MEMORY.md", "- x\n")
    write_jsonl(proj / "logs" / "1.jsonl", [{"role": "user", "content": "hi"}])
    write(proj / ".agentdream.toml", "memory = 'mem'\nsessions = ['logs']\n[schedule]\nnew_sessions = 1\n"
                                     "state_file = 'sched.json'\n")
    monkeypatch.chdir(proj)
    assert main(["schedule", "--once"]) == 0
    assert "dreamed:" in capsys.readouterr().out
    assert (proj / "sched.json").exists()


def test_mcp_dream_uses_config_defaults(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    write(proj / "mem" / "MEMORY.md", "- Use pytest.\n- Use pytest!\n")
    write_jsonl(proj / "logs" / "1.jsonl", [{"role": "user", "content": "hi"}])
    write(proj / ".agentdream.toml", "memory = 'mem'\nsessions = ['logs']\n")
    monkeypatch.chdir(proj)
    srv = MCPServer(tmp_path / "jobs")
    resp = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "dream",
                                                                                       "arguments": {}}})
    assert resp["result"]["isError"] is False, resp
    assert resp["result"]["structuredContent"]["status"] == "completed"
    write(proj / ".agentdream.toml", "sessions = ['logs']\n")
    resp = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "dream",
                                                                                       "arguments": {}}})
    assert resp["result"]["isError"] is True and "no memory store" in resp["result"]["content"][0]["text"]
