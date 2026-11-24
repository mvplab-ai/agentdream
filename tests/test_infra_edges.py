"""Edge cases of jobs, pipeline limits, review, util, models, MCP stdio loop and backend error paths."""

import io
import json
import os
import subprocess
import urllib.error

import pytest

from agentdream import DreamJob, Pipeline, accept, discard, dream, make_config
from agentdream.backends import BackendError, ClaudeCLIBackend, OpenAIBackend, get_backend, http_post_json
from agentdream.backends.builtin import AnthropicBackend, DeterministicBackend
from agentdream.context import DreamConfig, DreamContext, DreamLimitError
from agentdream.jobs import _pid_alive, default_jobs_dir
from agentdream.mcp_server import MCPServer
from agentdream.models import Evidence, MemoryEntry, Message, Operation, resolve_dependencies
from agentdream.pipeline import phase_limits, phase_load, phase_write
from agentdream.review import ReviewError, select_operations
from agentdream.util import atomic_write_bytes, parse_duration, parse_timestamp

from conftest import EXAMPLES, write, write_jsonl


def _mem_sessions(tmp_path, n_sessions=2, msg="User: hello there\n"):
    mem = write(tmp_path / "m" / "MEMORY.md", "- Use pytest.\n- Use pytest!\n").parent
    for i in range(n_sessions):
        write(tmp_path / "s" / f"{i:02d}.txt", msg)
    return mem, tmp_path / "s"


# ------------------------------------------------------------------ jobs


def test_job_lifecycle_edges(tmp_path, monkeypatch):
    mem, s = _mem_sessions(tmp_path)
    cfg = make_config(mem, [s])
    job = DreamJob(cfg)
    assert repr(job).startswith("DreamJob(") and job.operations == []
    with pytest.raises(FileNotFoundError):
        _ = job.report
    with pytest.raises(FileNotFoundError):
        _ = job.report_md
    job.cancel()  # pending -> canceled without running
    assert job.status == "canceled" and job.done
    assert job.run().status == "canceled"
    with pytest.raises(RuntimeError):
        job.start()
    j2 = DreamJob(cfg).start().wait(30)
    with pytest.raises(RuntimeError, match="already completed"):
        j2.start()
    other = DreamJob.load(j2.id)
    other.review = "accepted"
    other.save()
    assert j2.refresh().review == "accepted"
    # corrupt job files are skipped by list()
    write(j2.jobs_dir / "broken" / "job.json", "{not json")
    assert [j.id for j in DreamJob.list() if j.id in (job.id, j2.id)] == [job.id, j2.id]
    assert DreamJob.list(tmp_path / "nowhere") == []
    monkeypatch.delenv("AGENTDREAM_HOME")
    monkeypatch.chdir(tmp_path)
    assert default_jobs_dir() == tmp_path / ".agentdream" / "jobs"


def test_cancel_requested_through_job_file_and_dead_worker(tmp_path):
    mem, s = _mem_sessions(tmp_path)
    job = DreamJob(make_config(mem, [s]))
    job.save()
    data = json.loads(job.path.read_text())
    data["cancel_requested"] = True
    job.path.write_text(json.dumps(data))
    assert job.run().status == "canceled" and job.error == "canceled by user"
    # a job recorded as running by a process that no longer exists is failed on load
    ghost = DreamJob(make_config(mem, [s]))
    ghost.status, ghost.pid = "running", 2 ** 22 + 12345
    ghost.save()
    loaded = DreamJob.load(ghost.id)
    assert loaded.status == "failed" and "interrupted" in loaded.error


def test_pid_alive():
    assert _pid_alive(os.getpid()) and not _pid_alive(0) and not _pid_alive(-5)


def test_job_load_by_unique_prefix_and_ambiguity(tmp_path):
    mem, s = _mem_sessions(tmp_path)
    a = dream(mem, [s])
    assert DreamJob.load(a.id[:-2]).id == a.id
    b = dream(mem, [s])
    common = a.id[: next(i for i, (x, y) in enumerate(zip(a.id, b.id, strict=True)) if x != y)]
    from agentdream.jobs import JobNotFound

    with pytest.raises(JobNotFound, match="ambiguous"):
        DreamJob.load(common)


def test_input_hash_unavailable_is_tolerated(tmp_path, monkeypatch):
    mem, s = _mem_sessions(tmp_path)

    def boom(_p):
        raise OSError("unreadable")

    monkeypatch.setattr("agentdream.jobs.hash_path", boom)
    job = dream(mem, [s])
    assert job.status == "completed" and job.input_hash is None


# ------------------------------------------------------------------ pipeline limits


def _ctx(tmp_path, **cfg):
    mem, s = _mem_sessions(tmp_path, 3, "User: " + "word " * 200 + "\n")
    ctx = DreamContext(DreamConfig(memory=str(mem), sessions=[str(s)], **cfg))
    phase_load(ctx)
    ctx.entries = list(ctx.original_entries)
    return ctx


def test_limits_drop_sessions_then_trim_messages(tmp_path):
    ctx = _ctx(tmp_path, max_total_tokens=30)
    with pytest.raises(DreamLimitError, match="memory store alone"):
        phase_limits(ctx)
    mem = write(tmp_path / "m2" / "MEMORY.md", "- x\n").parent
    s = tmp_path / "s2"
    write_jsonl(s / "a.jsonl", [{"role": "user", "content": "w " * 40}] * 3)
    write_jsonl(s / "b.jsonl", [{"role": "user", "content": "v " * 40}] * 3)
    ctx = DreamContext(DreamConfig(memory=str(mem), sessions=[str(s)], max_total_tokens=50))  # ~20 tokens/message
    phase_load(ctx)
    ctx.entries = list(ctx.original_entries)
    phase_limits(ctx)
    assert len(ctx.transcripts) == 1 and len(ctx.transcripts[0].messages) < 3
    assert ctx.transcripts[0].metadata["truncated_messages"] >= 1
    assert any("oldest message" in n for n in ctx.notices)


def test_limits_fail_modes(tmp_path):
    with pytest.raises(DreamLimitError, match="max_sessions"):
        phase_limits(_ctx(tmp_path / "a", max_sessions=1, on_limit="fail"))
    with pytest.raises(DreamLimitError, match="max_total_tokens"):
        phase_limits(_ctx(tmp_path / "b", max_total_tokens=50, on_limit="fail"))
    ctx = _ctx(tmp_path / "c", min_sessions=2, max_sessions=1)  # truncation leaves too few sessions
    with pytest.raises(DreamLimitError, match="need at least 2"):
        phase_limits(ctx)


def test_pipeline_errors_and_editing(tmp_path):
    with pytest.raises(ValueError, match="no memory"):
        phase_load(DreamContext(DreamConfig()))
    with pytest.raises(ValueError, match="no output directory"):
        phase_write(DreamContext(DreamConfig()))
    pipe = Pipeline.default().remove("insights")
    assert "insights" not in pipe.names()
    mem, s = _mem_sessions(tmp_path)
    out = tmp_path / "out"
    write(out / "store" / "stale.md", "- left over\n")  # an existing staging store is replaced
    job = dream(mem, [s], out=out, pipeline=pipe)
    assert job.status == "completed" and not (out / "store" / "stale.md").exists()


# ------------------------------------------------------------------ review


def test_review_edges(tmp_path):
    mem, s = _mem_sessions(tmp_path)
    job = dream(mem, [s])
    ops = job.operations
    with pytest.raises(ReviewError, match="unknown or ambiguous"):
        select_operations(ops, only="nope")
    sel, _warns = select_operations(ops, exclude=[ops[0].id])
    assert ops[0].id not in {o.id for o in sel}
    for p in mem.iterdir():
        p.unlink()
    mem.rmdir()
    with pytest.raises(ReviewError, match="no longer exists"):
        accept(job)
    mem2, s2 = _mem_sessions(tmp_path / "x")
    pending = DreamJob(make_config(mem2, [s2]))
    pending.save()
    assert discard(pending.id).review == "discarded" and DreamJob.load(pending.id).status == "canceled"


def test_resolve_dependencies():
    a = Operation(id="a", type="merge", summary="")
    b = Operation(id="b", type="replace", summary="", depends_on=["a"])
    c = Operation(id="c", type="correct", summary="", depends_on=["b", "zzz"])
    assert resolve_dependencies([a, b, c], ["c"]) == ["a", "b", "c"]
    assert resolve_dependencies([a, b, c], ["a", "a"]) == ["a"]


# ------------------------------------------------------------------ models / util


def test_models_small_bits():
    e = MemoryEntry(id="x", content="c", created_at="2026-01-01")
    assert e.timestamp == "2026-01-01"
    m = Message("user", "hi", source_path=None, index=3)
    assert m.citation == "<memory>#3" and m.to_dict()["citation"] == "<memory>#3"
    ev = Evidence.from_message(Message("user", "y" * 500, line_no=2, source_path="/a.jsonl"), "y" * 500)
    assert len(ev.quote) == 240 and ev.quote.endswith("...")
    assert Evidence.from_dict(ev.to_dict()) == ev
    op = Operation(id="o", type="insight", summary="s", evidence=[ev], upsert=[e])
    assert Operation.from_dict(json.loads(json.dumps(op.to_dict()))).to_dict() == op.to_dict()


def test_util_parsing_and_atomic_failure(tmp_path, monkeypatch):
    assert parse_timestamp(1_700_000_000_000) == 1_700_000_000.0  # epoch millis
    assert parse_timestamp("1700000000") == 1_700_000_000.0
    assert parse_timestamp("2026-01-02 garbage") == parse_timestamp("2026-01-02")
    assert parse_timestamp("not a date") is None and parse_timestamp(None) is None
    for bad in ("5 parsecs", "", "m"):
        with pytest.raises(ValueError):
            parse_duration(bad)

    def explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("agentdream.util.os.replace", explode)
    with pytest.raises(OSError):
        atomic_write_bytes(tmp_path / "f.bin", b"data")
    assert not list(tmp_path.glob(".f.bin.*"))  # temp file cleaned up


def test_api_argument_handling(tmp_path):
    mem, s = _mem_sessions(tmp_path)
    assert make_config(mem, None).sessions == []
    assert make_config(mem, s).sessions == [str(s)]
    with pytest.raises(TypeError, match="unknown dream option"):
        make_config(mem, [s], bogus=1)
    with pytest.raises(ValueError, match="apply=True requires wait=True"):
        dream(mem, [s], wait=False, apply=True)


# ------------------------------------------------------------------ MCP stdio loop


def test_mcp_serve_loop(tmp_path):
    mem, s = _mem_sessions(tmp_path)
    srv = MCPServer(tmp_path / "jobs")
    lines = [
        "",
        "{broken",
        json.dumps([1, 2]),
        json.dumps("just a string"),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "method": "initialized"}]),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "dream", "arguments": {"memory": str(mem), "sessions": [str(s)], "wait": False}}}),
    ]
    out = io.StringIO()
    srv.serve(io.StringIO("\n".join(lines) + "\n"), out)
    resps = [json.loads(x) for x in out.getvalue().splitlines()]
    assert resps[0]["error"]["code"] == -32700
    assert len(resps) == 4  # a batch of non-objects and the notifications produce no response
    assert any(isinstance(r, dict) and r.get("error", {}).get("code") == -32600 for r in resps)
    assert [r for r in resps if isinstance(r, list) and r and r[0].get("id") == 1]
    started = next(r for r in resps if isinstance(r, dict) and r.get("id") == 2)["result"]["structuredContent"]
    job = srv._running[started["id"]]
    job.wait(30)
    st = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                     "params": {"name": "dream_status", "arguments": {"job_id": job.id}}})
    assert st["result"]["structuredContent"]["status"] == "completed"
    rep = srv.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                      "params": {"name": "show_report", "arguments": {"job_id": job.id, "format": "json"}}})
    assert rep["result"]["structuredContent"]["counts"]["merged"] == 1
    dis = srv.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                      "params": {"name": "discard", "arguments": {"job_id": job.id}}})
    assert dis["result"]["structuredContent"]["review"] == "discarded"
    unknown = srv.handle({"jsonrpc": "2.0", "id": 6, "method": "nope"})
    assert unknown["error"]["code"] == -32601


# ------------------------------------------------------------------ backends


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_helper_error_paths(monkeypatch):
    def http_error(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "slow down", {}, io.BytesIO(b'{"error": "rate"}'))

    monkeypatch.setattr("urllib.request.urlopen", http_error)
    with pytest.raises(BackendError, match="HTTP 429"):
        http_post_json("http://x/y", {}, {})
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: Resp(b"<html>"))
    with pytest.raises(BackendError, match="non-JSON"):
        http_post_json("http://x/y", {}, {})
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: Resp(b'{"content": []}'))
    with pytest.raises(BackendError, match="no text"):
        AnthropicBackend(api_key="k").complete("s", "p")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: Resp(b'{"choices": []}'))
    with pytest.raises(BackendError, match="unexpected response shape"):
        OpenAIBackend(base_url="http://local/v1").complete("s", "p")


def test_registry_and_misc_backends():
    be = DeterministicBackend()
    assert json.loads(be.complete("s", "p")) == {"operations": []} and not be.is_llm
    assert get_backend(be) is be
    lm = get_backend("lmstudio", model="m")
    assert lm.base_url == "http://localhost:1234/v1"
    from agentdream import available_backends

    assert {"deterministic", "anthropic", "openai", "claude-cli"} <= set(available_backends())
    with pytest.raises(BackendError, match="unknown backend"):
        get_backend("telepathy")


def test_claude_cli_output_variants(monkeypatch):
    monkeypatch.setattr("agentdream.backends.builtin.shutil.which", lambda x: "/usr/bin/claude")
    outputs = iter([
        "plain text answer",
        json.dumps([{"type": "system"}, {"type": "result", "result": "from list"}]),
        json.dumps({"is_error": True, "result": "quota"}),
        json.dumps({"something": "else"}),
    ])

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        return subprocess.CompletedProcess(cmd, 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr("agentdream.backends.builtin.subprocess.run", fake_run)
    be = ClaudeCLIBackend()
    assert be.complete("s", "p") == "plain text answer"
    assert be.complete("s", "p") == "from list"
    with pytest.raises(BackendError, match="quota"):
        be.complete("s", "p")
    assert json.loads(be.complete("s", "p")) == {"something": "else"}

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("claude", 1)

    monkeypatch.setattr("agentdream.backends.builtin.subprocess.run", timeout)
    with pytest.raises(BackendError, match="timed out"):
        be.complete("s", "p")


def test_examples_still_complete_with_every_phase_skipped():
    job = dream(EXAMPLES / "memory", [EXAMPLES / "sessions"],
                instructions="skip merging, don't verify and no insights")
    assert job.status == "completed" and job.operations == []
