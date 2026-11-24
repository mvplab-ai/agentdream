"""Configuration files and environment variables, with layered precedence.

Precedence (highest first)::

    CLI flags / explicit API arguments
    > environment variables (AGENTDREAM_*)
    > project config file (.agentdream.toml, or [tool.agentdream] in pyproject.toml)
    > user config file ($XDG_CONFIG_HOME/agentdream/config.toml, default ~/.config/...)
    > built-in defaults (DreamConfig)

The project file is found by walking up from the working directory; ``--config PATH``
or ``AGENTDREAM_CONFIG`` picks one explicitly. Relative paths inside a config file are
resolved against the file's directory. File format (all keys optional)::

    backend = "anthropic"            # deterministic | anthropic | openai | ollama | claude-cli
    model = "claude-sonnet-5-5"
    memory = "./memory"              # store path (or stancelog:<path>)
    sessions = ["~/.claude/projects/-my-project/*.jsonl"]
    instructions = "focus on deployment notes"
    jobs_dir = ".agentdream/jobs"

    [limits]
    max_sessions = 100
    max_bytes_per_transcript = 2000000
    max_total_tokens = 400000
    on_limit = "truncate"            # or "fail"
    max_chunk_tokens = 60000
    chunk_overlap = 400
    max_chunks = 64

    [redaction]
    enabled = true
    emails = false
    entropy = true
    entropy_threshold = 4.0
    entropy_min_length = 24
    cards = true
    patterns = ["ACME-[0-9]+"]

    [backend_options]                # passed to the backend constructor
    base_url = "http://localhost:11434/v1"

    [schedule]                       # defaults for `agentdream schedule`
    every = "6h"
    idle = "30m"
    new_sessions = 5
    poll = "60s"

    [hook]                           # defaults for the Claude Code hook
    every = 1                        # dream every N sessions
    apply = false                    # true = accept automatically (a backup is still made)
    memory = "CLAUDE.md"             # overrides the top-level memory for the hook

Environment variables: ``AGENTDREAM_<FIELD>`` for any top-level / [limits] / [redaction]
field using its DreamConfig name (``AGENTDREAM_BACKEND``, ``AGENTDREAM_MAX_CHUNK_TOKENS``,
``AGENTDREAM_REDACT_EMAILS``...), ``AGENTDREAM_BASE_URL`` for ``backend_options.base_url``,
``AGENTDREAM_HOOK_EVERY`` / ``AGENTDREAM_HOOK_APPLY`` / ``AGENTDREAM_HOOK_MEMORY``.
Lists (``AGENTDREAM_SESSIONS``) are ``os.pathsep``-separated or a JSON array.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ._toml import TOMLError, loads
from .context import DreamConfig

PROJECT_FILE = ".agentdream.toml"


class ConfigError(ValueError):
    pass


#: [redaction] keys -> DreamConfig fields
_REDACTION = {"enabled": "redact", "emails": "redact_emails", "entropy": "redact_entropy",
              "entropy_threshold": "entropy_threshold", "entropy_min_length": "entropy_min_length",
              "cards": "redact_cards", "patterns": "redact_patterns"}
_LIMITS = ("max_sessions", "min_sessions", "max_bytes_per_transcript", "max_total_tokens", "on_limit",
           "max_chunk_tokens", "chunk_overlap", "max_chunks", "insight_min_sessions", "max_insights")
_PATH_FIELDS = ("memory", "out", "jobs_dir")
_SCHEDULE = {"every": str, "idle": str, "new_sessions": int, "poll": str, "state_file": str}
_HOOK = {"every": int, "apply": bool, "memory": str, "event": str, "sessions": list}

_DREAM_FIELDS = {f.name: f for f in fields(DreamConfig)}
_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off", ""}


@dataclass
class Settings:
    """Resolved settings. ``dream`` holds only DreamConfig fields that some layer set."""

    dream: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    hook: dict[str, Any] = field(default_factory=dict)
    #: "dream.backend" -> "cli" | "env" | "<config file path>"
    sources: dict[str, str] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def effective_dream(self) -> dict[str, Any]:
        """All DreamConfig fields: defaults overlaid with resolved values."""
        base = DreamConfig().to_dict()
        base.update(self.dream)
        return base


# ------------------------------------------------------------------ coercion

def _type_of(name: str) -> str:
    t = str(_DREAM_FIELDS[name].type)
    for key in ("list", "dict", "bool", "int", "float"):
        if t.startswith(key):
            return key
    return "str"


def coerce(name: str, value: Any, where: str, kind: str | None = None) -> Any:
    """Convert ``value`` (from TOML or an env string) to the type of field ``name``."""
    kind = kind or _type_of(name)
    try:
        if kind == "bool":
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            if s in _BOOL_TRUE:
                return True
            if s in _BOOL_FALSE:
                return False
            raise ValueError
        if kind == "int":
            if isinstance(value, bool):
                raise ValueError
            return int(str(value).replace("_", "")) if not isinstance(value, int) else value
        if kind == "float":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if kind == "list":
            if isinstance(value, list):
                return [str(v) for v in value]
            s = str(value).strip()
            if s.startswith("["):
                data = json.loads(s)
                if not isinstance(data, list):
                    raise ValueError
                return [str(v) for v in data]
            return [p for p in s.split(os.pathsep) if p] if name == "sessions" else [s]
        if kind == "dict":
            if isinstance(value, dict):
                return dict(value)
            data = json.loads(str(value))
            if not isinstance(data, dict):
                raise ValueError
            return data
        if value is None or isinstance(value, (dict, list)):
            raise ValueError
        return str(value)
    except (ValueError, TypeError):
        raise ConfigError(f"{where}: {name} must be {kind}, got {value!r}") from None


def _resolve_path(value: str, base: Path) -> str:
    prefix = ""
    if value.startswith("stancelog:"):
        prefix, value = "stancelog:", value[len("stancelog:"):]
    p = Path(os.path.expanduser(value))
    if not p.is_absolute():
        p = base / p
    return prefix + str(p)


# ------------------------------------------------------------------ files

def user_config_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    base = env.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "agentdream" / "config.toml"


def _pyproject_has_section(p: Path) -> bool:
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return False
    return "[tool.agentdream" in text


def find_project_config(start: str | os.PathLike | None = None) -> Path | None:
    """Nearest ``.agentdream.toml`` (or ``pyproject.toml`` with ``[tool.agentdream]``) at or above ``start``."""
    cur = Path(start or os.getcwd()).absolute()
    for d in (cur, *cur.parents):
        f = d / PROJECT_FILE
        if f.is_file():
            return f
        pp = d / "pyproject.toml"
        if pp.is_file() and _pyproject_has_section(pp):
            return pp
    return None


def read_config_file(path: str | os.PathLike) -> dict[str, Any]:
    """Raw agentdream table from a config file (``[tool.agentdream]`` for pyproject.toml)."""
    p = Path(path)
    try:
        data = loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config file {p}: {exc}") from exc
    except TOMLError as exc:
        raise ConfigError(f"invalid TOML in {p}: {exc}") from exc
    if p.name == "pyproject.toml":
        data = (data.get("tool") or {}).get("agentdream") or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: expected a table")
    return data


def parse_config(data: Mapping[str, Any], where: str, base: Path) -> tuple[dict, dict, dict, list[str]]:
    """Normalize a raw config table into (dream fields, schedule, hook, warnings)."""
    dream: dict[str, Any] = {}
    schedule: dict[str, Any] = {}
    hook: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in data.items():
        if key == "limits" and isinstance(value, dict):
            for k, v in value.items():
                if k in _LIMITS:
                    dream[k] = coerce(k, v, f"{where} [limits]")
                else:
                    warnings.append(f"{where}: unknown key limits.{k}")
        elif key == "redaction" and isinstance(value, dict):
            for k, v in value.items():
                if k in _REDACTION:
                    dream[_REDACTION[k]] = coerce(_REDACTION[k], v, f"{where} [redaction]")
                else:
                    warnings.append(f"{where}: unknown key redaction.{k}")
        elif key == "backend_options" and isinstance(value, dict):
            dream["backend_options"] = dict(value)
        elif key == "schedule" and isinstance(value, dict):
            for k, v in value.items():
                if k in _SCHEDULE:
                    schedule[k] = coerce(k, v, f"{where} [schedule]", "int" if _SCHEDULE[k] is int else "str")
                else:
                    warnings.append(f"{where}: unknown key schedule.{k}")
            if "state_file" in schedule:
                schedule["state_file"] = _resolve_path(schedule["state_file"], base)
        elif key == "hook" and isinstance(value, dict):
            for k, v in value.items():
                if k in _HOOK:
                    t = {int: "int", bool: "bool", str: "str", list: "list"}[_HOOK[k]]
                    hook[k] = coerce("sessions" if k == "sessions" else k, v, f"{where} [hook]", t)
                else:
                    warnings.append(f"{where}: unknown key hook.{k}")
            if "memory" in hook:
                hook["memory"] = _resolve_path(hook["memory"], base)
            if "sessions" in hook:
                hook["sessions"] = [_resolve_path(s, base) for s in hook["sessions"]]
        elif key in _DREAM_FIELDS and key not in ("backend_options",):
            dream[key] = coerce(key, value, where)
        else:
            warnings.append(f"{where}: unknown key {key}")
    for k in _PATH_FIELDS:
        if dream.get(k):
            dream[k] = _resolve_path(dream[k], base)
    if dream.get("sessions"):
        dream["sessions"] = [_resolve_path(s, base) for s in dream["sessions"]]
    return dream, schedule, hook, warnings


def env_config(environ: Mapping[str, str] | None = None) -> tuple[dict, dict]:
    """(dream fields, hook settings) from ``AGENTDREAM_*`` environment variables."""
    env = os.environ if environ is None else environ
    dream: dict[str, Any] = {}
    hook: dict[str, Any] = {}
    for name in _DREAM_FIELDS:
        var = "AGENTDREAM_" + name.upper()
        if var in env and name != "backend_options":
            dream[name] = coerce(name, env[var], f"${var}")
    if "AGENTDREAM_BACKEND_OPTIONS" in env:
        dream["backend_options"] = coerce("backend_options", env["AGENTDREAM_BACKEND_OPTIONS"],
                                          "$AGENTDREAM_BACKEND_OPTIONS")
    if env.get("AGENTDREAM_BASE_URL"):
        dream.setdefault("backend_options", {})["base_url"] = env["AGENTDREAM_BASE_URL"]
    cwd = Path(os.getcwd())
    for k in _PATH_FIELDS:
        if dream.get(k):
            dream[k] = _resolve_path(dream[k], cwd)
    for k, t in (("every", "int"), ("apply", "bool"), ("memory", "str")):
        var = f"AGENTDREAM_HOOK_{k.upper()}"
        if var in env:
            hook[k] = coerce(k, env[var], f"${var}", t)
    if hook.get("memory"):
        hook["memory"] = _resolve_path(hook["memory"], cwd)
    return dream, hook


# ------------------------------------------------------------------ resolution

def _layer(settings: Settings, source: str, dream: dict, schedule: dict | None = None,
           hook: dict | None = None) -> None:
    for k, v in dream.items():
        if k == "backend_options":
            settings.dream["backend_options"] = {**settings.dream.get("backend_options", {}), **v}
        else:
            settings.dream[k] = v
        settings.sources[f"dream.{k}"] = source
    for k, v in (schedule or {}).items():
        settings.schedule[k] = v
        settings.sources[f"schedule.{k}"] = source
    for k, v in (hook or {}).items():
        settings.hook[k] = v
        settings.sources[f"hook.{k}"] = source


def resolve(cli: Mapping[str, Any] | None = None, *, cwd: str | os.PathLike | None = None,
            config_path: str | os.PathLike | None = None, environ: Mapping[str, str] | None = None,
            user_config: bool = True, cli_schedule: Mapping[str, Any] | None = None,
            cli_hook: Mapping[str, Any] | None = None) -> Settings:
    """Merge defaults < user config < project config < env < CLI. ``cli`` holds only explicitly
    given DreamConfig fields (``None`` values are ignored)."""
    env = os.environ if environ is None else environ
    s = Settings()
    files: list[Path] = []
    if user_config:
        up = user_config_path(env)
        if up.is_file():
            files.append(up)
    explicit = config_path or env.get("AGENTDREAM_CONFIG")
    if explicit:
        p = Path(os.path.expanduser(str(explicit)))
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        files.append(p)
    else:
        found = find_project_config(cwd)
        if found is not None and found not in files:
            files.append(found)
    for f in files:
        dream, schedule, hook, warns = parse_config(read_config_file(f), str(f), f.parent.absolute())
        _layer(s, str(f), dream, schedule, hook)
        s.files.append(str(f))
        s.warnings.extend(warns)
    e_dream, e_hook = env_config(env)
    _layer(s, "env", e_dream, None, e_hook)
    cli_vals = {k: v for k, v in (cli or {}).items() if v is not None}
    unknown = set(cli_vals) - set(_DREAM_FIELDS)
    if unknown:
        raise ConfigError(f"unknown option(s): {', '.join(sorted(unknown))}")
    _layer(s, "cli", cli_vals, {k: v for k, v in (cli_schedule or {}).items() if v is not None},
           {k: v for k, v in (cli_hook or {}).items() if v is not None})
    return s


__all__ = ["PROJECT_FILE", "ConfigError", "Settings", "coerce", "env_config", "find_project_config",
           "parse_config", "read_config_file", "resolve", "user_config_path"]
