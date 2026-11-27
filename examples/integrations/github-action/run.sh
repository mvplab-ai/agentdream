#!/usr/bin/env bash
# Run one agentdream dream for the composite action in this directory.
# Dry run unless INPUT_APPLY is exactly the confirmation string below.
# Safe to run outside GitHub Actions: set INPUT_MEMORY, INPUT_SESSIONS, and
# optionally GITHUB_OUTPUT / RUNNER_TEMP. See README.md.

set -euo pipefail

CONFIRM="I-understand-this-writes-the-memory-store"

die() {
  echo "agentdream action: $*" >&2
  exit 2
}

fail() {
  echo "agentdream action: $*" >&2
  exit 1
}

is_true() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
  cd "$GITHUB_WORKSPACE"
fi

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  die "python is not on PATH"
fi

MEMORY="${INPUT_MEMORY:-}"
SESSIONS="${INPUT_SESSIONS:-}"
BACKEND="${INPUT_BACKEND:-deterministic}"
[[ -n "$BACKEND" ]] || BACKEND="deterministic"
[[ -n "$MEMORY" ]] || die "input 'memory' is required"
[[ -n "$SESSIONS" ]] || die "input 'sessions' is required"
[[ -e "$MEMORY" ]] || die "memory path does not exist: $MEMORY"

APPLY_NORM="$(printf '%s' "${INPUT_APPLY:-}" | tr -d '[:space:]')"
if [[ -n "$APPLY_NORM" && "$APPLY_NORM" != "$CONFIRM" ]]; then
  die "refusing to apply. true / yes / 1 do not write the store. The only value that does is: $CONFIRM"
fi

EXTRA=()
if [[ -n "${INPUT_EXTRA_ARGS:-}" ]]; then
  extra_file="$(mktemp)"
  if ! "$PY" -c 'import os, shlex, sys
try:
    args = shlex.split(os.environ.get("INPUT_EXTRA_ARGS", ""))
except ValueError as exc:
    print(f"could not parse extra-args: {exc}", file=sys.stderr)
    raise SystemExit(1)
for arg in args:
    print(arg)' >"$extra_file"; then
    rm -f "$extra_file"
    die "could not parse extra-args"
  fi
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    case "$line" in
      --apply|--apply=*)
        rm -f "$extra_file"
        die "--apply is not accepted in extra-args. Leave apply empty for a dry run, or set it to: $CONFIRM"
        ;;
    esac
    EXTRA+=("$line")
  done <"$extra_file"
  rm -f "$extra_file"
fi

if [[ "$BACKEND" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  die "backend anthropic needs ANTHROPIC_API_KEY in the job environment (secrets.ANTHROPIC_API_KEY). Do not pass the key as an action input."
fi

# A path overlaps the memory store when either directory contains the other.
overlaps_memory() {
  "$PY" -c 'import sys
from pathlib import Path
mem = Path(sys.argv[1]).resolve()
other = Path(sys.argv[2]).resolve()
if mem == other or mem in other.parents or other in mem.parents:
    raise SystemExit(1)
' "$MEMORY" "$1"
}

JOBS_DIR="${INPUT_JOBS_DIR:-}"
if [[ -z "$JOBS_DIR" ]]; then
  JOBS_DIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/agentdream-jobs"
fi
OUT="${INPUT_OUT:-$JOBS_DIR/staged}"
if ! overlaps_memory "$JOBS_DIR"; then
  die "jobs-dir overlaps the memory store ($JOBS_DIR). Use a directory outside the store, such as \$RUNNER_TEMP/agentdream-jobs."
fi
if ! overlaps_memory "$OUT"; then
  die "out overlaps the memory store ($OUT). The staged store must sit outside the memory path."
fi
mkdir -p "$JOBS_DIR"

ACTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$ACTION_DIR/../../.." && pwd)"

install_agentdream() {
  local mode="${INPUT_INSTALL:-auto}"
  case "$mode" in
    skip) return 0 ;;
    local|git|auto) ;;
    *) die "unknown install mode '$mode' (expected auto, local, git, or skip)" ;;
  esac
  local target=""
  if [[ "$mode" == "git" ]]; then
    target="git+https://github.com/mvplab-ai/agentdream"
  elif [[ -f "$PACKAGE_ROOT/pyproject.toml" ]] && grep -q '^name = "agentdream"$' "$PACKAGE_ROOT/pyproject.toml"; then
    target="$PACKAGE_ROOT"
  elif [[ "$mode" == "local" ]]; then
    die "install=local but $PACKAGE_ROOT is not the agentdream package"
  elif "$PY" -c 'import agentdream' >/dev/null 2>&1; then
    return 0
  else
    target="git+https://github.com/mvplab-ai/agentdream"
  fi
  echo "agentdream action: installing $target"
  "$PY" -m pip install --disable-pip-version-check --upgrade pip
  "$PY" -m pip install --disable-pip-version-check "$target"
}

install_agentdream

args=(dream --memory "$MEMORY" --sessions "$SESSIONS" --backend "$BACKEND" --jobs-dir "$JOBS_DIR" --out "$OUT" --json)
if [[ -n "${INPUT_MODEL:-}" ]]; then
  args+=(--model "$INPUT_MODEL")
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  args+=("${EXTRA[@]}")
fi

echo "agentdream action: dry run, backend=$BACKEND, jobs-dir=$JOBS_DIR"
json_out="$(mktemp)"
trap 'rm -f "$json_out"' EXIT
if ! "$PY" -m agentdream "${args[@]}" >"$json_out"; then
  fail "dream failed"
fi

parsed="$(JOB_JSON="$json_out" "$PY" -c 'import json, os
from pathlib import Path
job = json.loads(Path(os.environ["JOB_JSON"]).read_text(encoding="utf-8"))
out = Path(job.get("out_dir") or "")
counts = job.get("counts") or {}

def n(key):
    try:
        return int(counts.get(key) or 0)
    except (TypeError, ValueError):
        return 0

print(job.get("id") or "")
print(job.get("status") or "")
print(out / "report.md" if str(out) else "")
print(out / "report.json" if str(out) else "")
print(n("corrected"))
print(n("merged"))
print(n("replaced"))
print(n("contradictions"))
print(n("insights"))
')"

# macOS /bin/bash is 3.2 and has no mapfile. Read the nine lines by hand.
fields=()
while IFS= read -r line || [[ -n "$line" ]]; do
  fields+=("$line")
done <<< "$parsed"
JOB_ID="${fields[0]:-}"
STATUS="${fields[1]:-}"
REPORT_MD="${fields[2]:-}"
REPORT_JSON="${fields[3]:-}"
CORRECTED="${fields[4]:-0}"
MERGED="${fields[5]:-0}"
REPLACED="${fields[6]:-0}"
CONTRADICTIONS="${fields[7]:-0}"
INSIGHTS="${fields[8]:-0}"

if [[ -n "${RUNNER_TEMP:-}" ]]; then
  ARTIFACT_DIR="${INPUT_ARTIFACT_DIR:-$RUNNER_TEMP/agentdream-report}"
else
  ARTIFACT_DIR="${INPUT_ARTIFACT_DIR:-$(mktemp -d)}"
fi
mkdir -p "$ARTIFACT_DIR"
REPORT_MD_OUT=""
REPORT_JSON_OUT=""
if [[ -n "$REPORT_MD" && -f "$REPORT_MD" ]]; then
  cp "$REPORT_MD" "$ARTIFACT_DIR/report.md"
  REPORT_MD_OUT="$ARTIFACT_DIR/report.md"
fi
if [[ -n "$REPORT_JSON" && -f "$REPORT_JSON" ]]; then
  cp "$REPORT_JSON" "$ARTIFACT_DIR/report.json"
  REPORT_JSON_OUT="$ARTIFACT_DIR/report.json"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "job-id=$JOB_ID"
    echo "status=$STATUS"
    echo "corrected=$CORRECTED"
    echo "report-md=$REPORT_MD_OUT"
    echo "report-json=$REPORT_JSON_OUT"
    echo "artifact-dir=$ARTIFACT_DIR"
    echo "jobs-dir=$JOBS_DIR"
  } >> "$GITHUB_OUTPUT"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat >> "$GITHUB_STEP_SUMMARY" <<EOF
### agentdream dry-run

| | |
|---|---|
| Job | \`$JOB_ID\` |
| Status | $STATUS |
| Backend | \`$BACKEND\` |
| Corrected | $CORRECTED |
| Merged | $MERGED |
| Replaced | $REPLACED |
| Contradictions | $CONTRADICTIONS |
| Insights | $INSIGHTS |

The dream command does not modify the memory store.
EOF
fi

echo "agentdream action: job $JOB_ID status=$STATUS corrected=$CORRECTED merged=$MERGED replaced=$REPLACED contradictions=$CONTRADICTIONS insights=$INSIGHTS"
if [[ -n "$REPORT_MD_OUT" ]]; then
  echo "agentdream action: report $REPORT_MD_OUT"
fi

if [[ "$STATUS" != "completed" ]]; then
  fail "dream did not complete (status=${STATUS:-unknown})"
fi
if [[ ! -f "$ARTIFACT_DIR/report.md" || ! -f "$ARTIFACT_DIR/report.json" ]]; then
  fail "dream completed but report.md or report.json is missing"
fi

if is_true "${INPUT_FAIL_ON_CORRECTIONS:-false}" && [[ "$CORRECTED" -gt 0 ]]; then
  fail "$CORRECTED correction(s) to review (fail-on-corrections is set). Nothing was applied."
fi

if [[ "$APPLY_NORM" == "$CONFIRM" ]]; then
  "$PY" -m agentdream accept "$JOB_ID" --jobs-dir "$JOBS_DIR"
  echo "agentdream action: applied $JOB_ID to the memory store. A backup was kept with the job. This action does not commit or push."
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "" >> "$GITHUB_STEP_SUMMARY"
    echo "Applied job \`$JOB_ID\` to the memory store. A backup was kept with the job. This action does not commit or push." >> "$GITHUB_STEP_SUMMARY"
  fi
fi
