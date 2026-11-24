"""Minimal MCP server (JSON-RPC 2.0 over stdio, newline-delimited), stdlib only.

Tools: dream, dream_status, accept, discard, list_dreams, show_report.

Claude Code:  claude mcp add agentdream -- agentdream serve-mcp
Cursor (.cursor/mcp.json): {"mcpServers": {"agentdream": {"command": "agentdream", "args": ["serve-mcp"]}}}
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from . import __version__

PROTOCOL_VERSION = "2025-06-18"

_STR = {"type": "string"}
TOOLS: list[dict[str, Any]] = [
    {"name": "dream",
     "description": "Consolidate an agent memory store against session transcripts (dry run: writes a NEW store "
                    "+ report for review; the input store is never modified).",
     "inputSchema": {"type": "object", "properties": {
         "memory": {"type": "string", "description": "Memory store path (markdown folder, JSON/JSONL, SQLite); "
                                                     "default: `memory` from .agentdream.toml"},
         "sessions": {"type": "array", "items": _STR, "description": "Transcript files, dirs or globs; default: "
                                                                     "`sessions` from .agentdream.toml"},
         "backend": {"type": "string", "description": "deterministic | anthropic | openai | claude-cli",
                     "default": "deterministic"},
         "model": _STR, "instructions": {"type": "string", "description": "Optional focus for the dream"},
         "out": {"type": "string", "description": "Output directory (default: job dir)"},
         "wait": {"type": "boolean", "default": True,
                  "description": "Wait for completion (false: return a job id and poll dream_status)"}},
         "required": []}},
    {"name": "dream_status", "description": "Status of a dream job (pending/running/completed/failed/canceled).",
     "inputSchema": {"type": "object", "properties": {"job_id": _STR}, "required": ["job_id"]}},
    {"name": "accept", "description": "Apply a completed dream to the memory store (backs up the original first). "
                                      "Optionally only some operations.",
     "inputSchema": {"type": "object", "properties": {
         "job_id": _STR, "only": {"type": "array", "items": _STR}, "exclude": {"type": "array", "items": _STR},
         "to": {"type": "string", "description": "Write to this location instead of the original store"}},
         "required": ["job_id"]}},
    {"name": "discard", "description": "Discard a dream's staged output.",
     "inputSchema": {"type": "object", "properties": {"job_id": _STR}, "required": ["job_id"]}},
    {"name": "list_dreams", "description": "List dream jobs.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "show_report", "description": "Show a dream report (markdown or json).",
     "inputSchema": {"type": "object", "properties": {
         "job_id": _STR, "format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown"}},
         "required": ["job_id"]}},
]


class MCPServer:
    def __init__(self, jobs_dir: str | Path | None = None, config_path: str | Path | None = None):
        self.jobs_dir = str(jobs_dir) if jobs_dir else None
        self.config_path = str(config_path) if config_path else None
        self._running: dict[str, Any] = {}
        self.handlers: dict[str, Callable[[dict], Any]] = {
            "dream": self._dream, "dream_status": self._status, "accept": self._accept,
            "discard": self._discard, "list_dreams": self._list, "show_report": self._show,
        }

    # ------------------------------------------------------------ tools
    def _load(self, job_id: str) -> Any:
        from .jobs import DreamJob

        if job_id in self._running:
            return self._running[job_id]
        return DreamJob.load(job_id, self.jobs_dir)

    def _dream(self, a: dict) -> Any:
        from .api import dream
        from .config import resolve

        cli = {k: a.get(k) or None for k in ("memory", "sessions", "backend", "model", "instructions", "out")}
        cli["jobs_dir"] = self.jobs_dir
        settings = resolve(cli, config_path=self.config_path)  # tool args > env > config file > defaults
        opts = dict(settings.dream)
        memory = opts.pop("memory", None)
        if not memory:
            raise ValueError("no memory store: pass `memory` or set it in .agentdream.toml")
        sessions = opts.pop("sessions", None) or []
        backend = opts.pop("backend", None) or "deterministic"
        wait = a.get("wait", True)
        job = dream(memory, sessions, backend, wait=wait, **opts)
        if not wait:
            self._running[job.id] = job
        out = job.to_dict()
        out.pop("config", None)
        if job.status == "completed":
            out["report_summary"] = job.report["counts"]
            out["operations"] = [{"id": o.id, "type": o.type, "kind": o.kind, "summary": o.summary}
                                 for o in job.operations]
        return out

    def _status(self, a: dict) -> Any:
        job = self._load(a["job_id"])
        d = job.to_dict()
        d.pop("config", None)
        return d

    def _accept(self, a: dict) -> Any:
        from .review import accept

        return accept(self._load(a["job_id"]), only=a.get("only"), exclude=a.get("exclude"), to=a.get("to")).to_dict()

    def _discard(self, a: dict) -> Any:
        from .review import discard

        job = discard(self._load(a["job_id"]))
        return {"id": job.id, "review": job.review}

    def _list(self, a: dict) -> Any:
        from .jobs import DreamJob

        return [{"id": j.id, "status": j.status, "review": j.review, "created_at": j.created_at,
                 "counts": j.counts} for j in DreamJob.list(self.jobs_dir)]

    def _show(self, a: dict) -> Any:
        job = self._load(a["job_id"])
        if a.get("format") == "json":
            return job.report
        return job.report_md

    # ------------------------------------------------------------ JSON-RPC
    def handle(self, req: dict) -> dict | None:
        rid = req.get("id")
        method = req.get("method")
        is_notification = "id" not in req

        def ok(result: Any) -> dict | None:
            return None if is_notification else {"jsonrpc": "2.0", "id": rid, "result": result}

        def err(code: int, msg: str) -> dict | None:
            return None if is_notification else {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}

        if method == "initialize":
            params = req.get("params") or {}
            return ok({"protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                       "capabilities": {"tools": {"listChanged": False}},
                       "serverInfo": {"name": "agentdream", "version": __version__},
                       "instructions": "Use `dream` to consolidate memory against transcripts, review the report "
                                       "(show_report), then `accept` or `discard`."})
        if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
            return None
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": TOOLS})
        if method == "tools/call":
            params = req.get("params") or {}
            name = str(params.get("name") or "")
            handler = self.handlers.get(name)
            if handler is None:
                return err(-32602, f"unknown tool: {name}")
            try:
                result = handler(params.get("arguments") or {})
            except Exception as exc:
                return ok({"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})
            text = result if isinstance(result, str) else json.dumps(result, indent=2, default=str)
            payload: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": False}
            if not isinstance(result, str):
                payload["structuredContent"] = result if isinstance(result, dict) else {"items": result}
            return ok(payload)
        return err(-32601, f"method not found: {method}")

    def serve(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError:
                resp: Any = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                if isinstance(req, list):
                    resp = [r for r in (self.handle(x) for x in req if isinstance(x, dict)) if r is not None] or None
                elif isinstance(req, dict):
                    resp = self.handle(req)
                else:
                    resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
            if resp is not None:
                stdout.write(json.dumps(resp, default=str) + "\n")
                stdout.flush()


def main(jobs_dir: str | None = None) -> None:
    MCPServer(jobs_dir).serve()
