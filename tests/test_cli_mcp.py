"""CLI end-to-end (main(argv) + a real subprocess) and the MCP stdio server."""

import json
import os
import subprocess
import sys

from agentdream import DreamJob
from agentdream.cli import main
from agentdream.mcp_server import MCPServer

from conftest import hash_tree, write, write_jsonl


def _setup(tmp_path):
    mem = write(tmp_path / "mem" / "MEMORY.md",
                "- The PokéCenter in Viridian City is north of the gym.\n- Use pytest.\n- Use pytest!\n").parent
    s = tmp_path / "s"
    write_jsonl(s / "1.jsonl", [{"role": "assistant", "content": "Walked north of the gym. No PokéCenter here."},
                                {"role": "assistant", "content": "Still no PokéCenter north of the gym."}])
    write_jsonl(s / "2.jsonl", [{"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."}])
    return mem, s


def test_cli_end_to_end(tmp_path, capsys):
    mem, s = _setup(tmp_path)
    before = hash_tree(mem)
    assert main(["dream", "--memory", str(mem), "--sessions", str(s)]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "corrected=1" in out
    assert hash_tree(mem) == before
    job_id = out.split()[1].rstrip(":")

    assert main(["status", job_id]) == 0
    assert "completed" in capsys.readouterr().out
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == job_id
    assert main(["list"]) == 0
    assert job_id in capsys.readouterr().out
    assert main(["show", job_id]) == 0
    assert "## Corrected" in capsys.readouterr().out
    assert main(["show", job_id, "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert main(["show", job_id, "--diff"]) == 0
    assert "+- The PokéCenter in Viridian City is south of the gym." in capsys.readouterr().out

    corr = rep["sections"]["corrected"][0]
    dest = tmp_path / "accepted"
    assert main(["accept", job_id, "--only", corr, "--to", str(dest)]) == 0
    assert "applied 1 operation(s)" in capsys.readouterr().out
    assert "south" in (dest / "MEMORY.md").read_text() and (dest / "MEMORY.md").read_text().count("pytest") == 2
    assert hash_tree(mem) == before

    assert main(["discard", job_id]) == 2  # already (partially) accepted
    assert "error" in capsys.readouterr().err
    assert main(["show", "nope"]) == 2


def test_cli_apply_and_discard(tmp_path, capsys):
    mem, s = _setup(tmp_path)
    assert main(["dream", "-m", str(mem), "-s", str(s), "--apply"]) == 0
    assert "backup" in capsys.readouterr().out
    assert "south of the gym" in (mem / "MEMORY.md").read_text()
    assert main(["dream", "-m", str(mem), "-s", str(s), "--json"]) == 0
    job = json.loads(capsys.readouterr().out)
    assert main(["discard", job["id"]]) == 0
    assert DreamJob.load(job["id"]).review == "discarded"


def test_cli_subprocess(tmp_path):
    mem, s = _setup(tmp_path)
    env = {**os.environ, "AGENTDREAM_HOME": str(tmp_path / "home")}
    r = subprocess.run([sys.executable, "-m", "agentdream", "dream", "--memory", str(mem), "--sessions", str(s),
                        "--instructions", "focus on the PokéCenter"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "completed" in r.stdout
    r2 = subprocess.run([sys.executable, "-m", "agentdream", "list"], capture_output=True, text=True, env=env)
    assert "completed" in r2.stdout


def _rpc(server, method, params=None, id_=1):
    req = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        req["params"] = params
    return server.handle(req)


def test_mcp_handle(tmp_path):
    mem, s = _setup(tmp_path)
    srv = MCPServer()
    init = _rpc(srv, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "test", "version": "0"}})
    assert init["result"]["serverInfo"]["name"] == "agentdream"
    assert init["result"]["capabilities"]["tools"] is not None
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = {t["name"] for t in _rpc(srv, "tools/list")["result"]["tools"]}
    assert tools == {"dream", "dream_status", "accept", "discard", "list_dreams", "show_report"}

    res = _rpc(srv, "tools/call", {"name": "dream", "arguments": {"memory": str(mem), "sessions": [str(s)]}})["result"]
    assert res["isError"] is False
    payload = json.loads(res["content"][0]["text"])
    assert payload["status"] == "completed" and payload["report_summary"]["corrected"] == 1
    jid = payload["id"]
    st = json.loads(_rpc(srv, "tools/call", {"name": "dream_status", "arguments": {"job_id": jid}})["result"]
                    ["content"][0]["text"])
    assert st["status"] == "completed"
    md = _rpc(srv, "tools/call", {"name": "show_report", "arguments": {"job_id": jid}})["result"]["content"][0]["text"]
    assert "# Dream report" in md
    lst = json.loads(_rpc(srv, "tools/call", {"name": "list_dreams", "arguments": {}})["result"]["content"][0]["text"])
    assert [j["id"] for j in lst] == [jid]
    acc = _rpc(srv, "tools/call", {"name": "accept", "arguments": {"job_id": jid, "to": str(tmp_path / "out")}})
    assert json.loads(acc["result"]["content"][0]["text"])["target"] == str(tmp_path / "out")
    bad = _rpc(srv, "tools/call", {"name": "dream_status", "arguments": {"job_id": "nope"}})["result"]
    assert bad["isError"] is True
    assert _rpc(srv, "tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == -32602
    assert _rpc(srv, "bogus/method")["error"]["code"] == -32601
    assert _rpc(srv, "ping")["result"] == {}


def test_mcp_stdio_subprocess(tmp_path):
    mem, s = _setup(tmp_path)
    env = {**os.environ, "AGENTDREAM_HOME": str(tmp_path / "home")}
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "dream", "arguments": {"memory": str(mem), "sessions": [str(s)]}}},
    ]
    stdin = "\n".join(json.dumps(r) for r in reqs) + "\nnot json\n"
    r = subprocess.run([sys.executable, "-m", "agentdream", "serve-mcp"], input=stdin, capture_output=True,
                       text=True, env=env, timeout=60)
    lines = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    assert [l.get("id") for l in lines] == [1, 2, 3, None]
    assert lines[0]["result"]["protocolVersion"] == "2025-06-18"
    assert len(lines[1]["result"]["tools"]) == 6
    assert json.loads(lines[2]["result"]["content"][0]["text"])["status"] == "completed"
    assert lines[3]["error"]["code"] == -32700
