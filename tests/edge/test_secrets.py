"""Secrets in odd places never reach the model, report.md, report.json, job.json or logs."""

from __future__ import annotations

import base64
import json

import pytest

from agentdream import Pipeline, dream
from agentdream.claude_hook import HookLog
from agentdream.redact import Redactor

from edgekit import FakeLLM, write, write_jsonl

ANTH = "sk-ant-api03-Q7f9Lk2mN4pR8sT1vW3xY6zA0bC5dE7gH9jK2lM4nP6qR8s"
GH_FILE = "ghp_aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3z5Ab"
AWS = "AKIAQ3EGX7JD2LMN5P0R"
DB_PW = "S3cr3tPassw0rdQx"
URL_TOK = "tok9fK2mP4qR7sT1vW3"
B64_PW = "Zq8mN3pL0vXr"
SPLIT_A, SPLIT_B = "sk-proj-Ab3De5Fg7Hj9Kl1Mn3Pq5Rs7", "Tu9Vw1Xy3Za5Bc7De9Fg1Hj3"
NESTED_PW = "hunter2Xy9Lq"
DOUBLE_PW = "Zx81Lm29Qp47Rt"
FM_SECRET = "xK9mP2qR7sT4vW1yZ3bC5dF8"
HEADING_TOK = "Hd7Kq2Lm9Np4Rs6Tv"
PEM_BODY = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7"

PLANTED = [ANTH, GH_FILE, AWS, DB_PW, URL_TOK, B64_PW, SPLIT_A, SPLIT_B, NESTED_PW, DOUBLE_PW, FM_SECRET,
           HEADING_TOK, PEM_BODY]


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _plant(tmp_path):
    mem = tmp_path / "mem"
    write(mem / "MEMORY.md",
          f"# Notes\n\n## Deploy (token={HEADING_TOK})\n\n"
          f"- The Anthropic key is {ANTH} for the staging bot.\n"
          f"- Database: postgres://admin:{DB_PW}@db.internal:5432/app\n"
          f"- Status API: https://api.example.com/v1/status?token={URL_TOK}&verbose=1\n"
          f"- Legacy config blob: {_b64('password=' + B64_PW)}\n"
          f"- Wrapped key from the email:\n  {SPLIT_A}\n  {SPLIT_B}\n"
          f"- AWS id {AWS} belongs to the deploy role.\n"
          f"- Escaped config: {{\\\"api_key\\\": \\\"{DOUBLE_PW}\\\"}}\n"
          f"- Use pytest for tests.\n- Use pytest for tests!\n")
    write(mem / "creds.md", f"---\nname: creds\napi_token: {FM_SECRET}\n---\nCredentials live in the vault.\n")
    write(mem / "config.json", json.dumps({"db": {"host": "x", "password": NESTED_PW}}))
    ses = tmp_path / "s"
    log = write_jsonl(ses / f"session-{GH_FILE}.jsonl", [
        {"role": "user", "content": f"Here is the key: -----BEGIN PRIVATE KEY-----\n{PEM_BODY}\nabc\n"
                                    "-----END PRIVATE KEY-----"},
        {"role": "assistant", "content": json.dumps({"config": {"auth": {"password": NESTED_PW}}})},
        {"role": "user", "content": f"curl 'https://x.io/api?access_token={URL_TOK}' and key {ANTH}"},
        {"role": "assistant", "content": f"Decoded blob: {_b64('api_key=' + B64_PW + ' secret=' + B64_PW)}"},
        {"role": "user", "content": "Walked north of the gym. No PokéCenter here. Use pytest for tests!"},
    ])
    return mem, ses, log


def _artifacts(job) -> dict[str, str]:
    out = {"job.json": job.path.read_text(encoding="utf-8")}
    for name in ("report.md", "report.json"):
        p = job.out_dir / name
        if p.exists():
            out[name] = p.read_text(encoding="utf-8")
    err = job.dir / "error.log"
    if err.exists():
        out["error.log"] = err.read_text(encoding="utf-8")
    return out


def _assert_clean(where: str, text: str):
    leaks = [s for s in PLANTED if s in text]
    assert not leaks, f"{where} leaks {leaks}"


def test_nothing_secret_reaches_the_llm_backend(tmp_path):
    """Planted secrets (nested JSON, URLs, base64, wrapped lines, filenames, metadata) never hit the model."""
    mem, _ses, log = _plant(tmp_path)
    fake = FakeLLM()
    job = dream(mem, [log], backend=fake)
    assert job.status == "completed", job.error
    assert len(fake.calls) >= 3
    _assert_clean("prompt sent to the model", fake.sent)


def test_nothing_secret_in_reports_job_json(tmp_path):
    """With the deterministic engine, report.md / report.json / job.json contain none of the planted secrets."""
    mem, _ses, log = _plant(tmp_path)
    job = dream(mem, [log], instructions=f"focus on deploy, the token is {URL_TOK}")
    assert job.status == "completed", job.error
    for name, text in _artifacts(job).items():
        _assert_clean(name, text)
    assert job.report["redactions"], "redaction counts should be reported"


def test_secrets_not_in_reports_with_llm_echoing_metadata(tmp_path):
    """A model merging entries whose metadata/keys hold secrets: report.json operations stay clean."""
    mem, _ses, log = _plant(tmp_path)

    def respond(phase, prompt):
        if phase == "consolidate":
            return {"operations": [{"type": "merge", "targets": ["MEMORY.md#8", "MEMORY.md#9"],
                                    "new_content": "Use pytest for tests.", "summary": "dup"}]}
        return {"operations": []}

    job = dream(mem, [log], backend=FakeLLM(respond))
    assert job.status == "completed", job.error
    assert any(o.type == "merge" for o in job.operations)
    for name, text in _artifacts(job).items():
        _assert_clean(name, text)


def test_nested_json_memory_values_and_metadata(tmp_path):
    """JSON map store with secrets in nested values and entry metadata: not sent to the model, not reported."""
    store = write(tmp_path / "m.json", json.dumps({
        "db": {"host": "db.internal", "credentials": {"user": "app", "password": NESTED_PW}},
        "bot": "The staging bot key is " + ANTH,
        "tests": "Use pytest for tests.",
        "tests2": "Use pytest for tests!",
    }))
    log = write_jsonl(tmp_path / "s" / "1.jsonl", [{"role": "user", "content": "Use pytest for tests!"}])
    fake = FakeLLM()
    job = dream(store, [log], backend=fake)
    assert job.status == "completed", job.error
    _assert_clean("prompt", fake.sent)
    for name, text in _artifacts(job).items():
        _assert_clean(name, text)


def test_failed_job_error_and_error_log_are_redacted(tmp_path):
    """A phase that crashes with a secret in its message: job.json error and error.log are redacted."""
    mem, _ses, log = _plant(tmp_path)

    def boom(ctx):
        raise RuntimeError(f"cannot parse note containing {ANTH} and postgres://u:{DB_PW}@h/db")

    job = dream(mem, [log], pipeline=Pipeline.default().replace("verify", boom))
    assert job.status == "failed"
    arts = _artifacts(job)
    assert "error.log" in arts
    for name, text in arts.items():
        _assert_clean(name, text)
    assert "[REDACTED:" in job.error


def test_hook_log_redacts(tmp_path):
    """HookLog lines never contain secrets (hook errors may quote config or paths)."""
    p = tmp_path / "hook.log"
    HookLog(p)(f"dream failed: key {ANTH} url postgres://u:{DB_PW}@h/x")
    _assert_clean("hook.log", p.read_text())


@pytest.mark.parametrize("secret_text", [
    "password=" + B64_PW, "api_key: " + B64_PW, '{"secret": "' + B64_PW + '"}',
    "AWS_SECRET_ACCESS_KEY=" + "a" * 3 + B64_PW, ANTH,
])
def test_base64_encoded_secrets_are_redacted(secret_text):
    """Base64 blobs that decode to key=value secrets (or known key formats) are redacted as base64_secret."""
    blob = _b64(secret_text)
    out = Redactor().redact(f"config blob {blob} end")
    assert blob not in out and "[REDACTED:" in out


def test_base64_of_harmless_text_is_kept():
    """Base64 of ordinary prose (no secret inside) is not redacted."""
    blob = _b64("hello world, this is fine")
    assert Redactor().redact(f"blob {blob}") == f"blob {blob}"


def test_secret_split_across_lines_both_halves_redacted():
    """A key wrapped onto the next line has both halves redacted (prefix match + continuation line)."""
    red = Redactor()
    for text in (f"key:\n{SPLIT_A}\n{SPLIT_B}\n", f"  {ANTH[:30]}\n  {ANTH[30:]}\n", f"{SPLIT_A}\r\n{SPLIT_B}"):
        out = red.redact(text)
        assert SPLIT_A not in out and SPLIT_B not in out and ANTH[30:] not in out, out


def test_continuation_heuristic_leaves_prose_alone():
    """The next line after a redacted key is only swallowed when it is a lone token, not a sentence."""
    out = Redactor().redact(f"{ANTH}\nThen we deployed it.\n")
    assert "Then we deployed it." in out


def test_double_encoded_json_secret_redacted():
    """Escaped JSON inside a string ({\\"api_key\\": \\"...\\"}) is redacted."""
    out = Redactor().redact('{\\"api_key\\": \\"' + DOUBLE_PW + '\\", \\"x\\": 1}')
    assert DOUBLE_PW not in out


@pytest.mark.parametrize("url", [
    f"https://api.example.com/v1?token={URL_TOK}", f"https://x.io/cb?access_token={URL_TOK}&a=1",
    f"https://x.io/?api_key={URL_TOK}", f"mongodb+srv://user:{DB_PW}@cluster0.example.net/db",
    f"redis://:{DB_PW}@cache:6379/0", f"https://user:{DB_PW}@git.example.com/repo.git",
])
def test_url_credentials_and_query_tokens(url):
    """user:pass@ credentials and ?token= / ?access_token= / ?api_key= query values are redacted."""
    out = Redactor().redact(f"see {url} for details")
    assert URL_TOK not in out and DB_PW not in out


def test_redaction_is_idempotent_on_planted_text(tmp_path):
    """Redacting already-redacted text changes nothing and counts nothing new."""
    mem, _ses, log = _plant(tmp_path)
    text = (mem / "MEMORY.md").read_text() + log.read_text()
    red = Redactor()
    once = red.redact(text)
    counts = dict(red.report.counts)
    assert red.redact(once) == once
    assert red.report.counts == counts
