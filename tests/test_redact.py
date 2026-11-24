"""Redaction: every known secret format, false positives, entropy behaviour, wiring into prompts/reports.

Fake secrets are assembled at runtime (prefix + seeded random body) so this file never contains
strings that secret scanners would flag.
"""

import json
import random
import string
import uuid

import pytest

from agentdream import LLMBackend, Redactor, dream, redact
from agentdream.cli import main
from agentdream.redact import PATTERNS, luhn_ok, shannon_entropy

from conftest import write, write_jsonl

RNG = random.Random(20261124)
B62 = string.ascii_letters + string.digits


def rnd(n, alphabet=B62):
    return "".join(RNG.choice(alphabet) for _ in range(n))


def hexs(n):
    return rnd(n, "0123456789abcdef")


def card(prefix="4539", length=16):
    """A Luhn-valid card-like number with the given issuer prefix."""
    body = prefix + "".join(RNG.choice("0123456789") for _ in range(length - len(prefix) - 1))
    for d in "0123456789":
        if luhn_ok(body + d):
            return body + d
    raise AssertionError


# (kind, text containing the secret, the secret itself)
def cases():
    pem_body = "\n".join(rnd(64) for _ in range(4))
    jwt = "eyJ" + rnd(20) + ".eyJ" + rnd(30) + "." + rnd(43)
    out = [
        ("anthropic_key", "ANTHROPIC_API_KEY=" + (s := "sk-ant-api03-" + rnd(60)), s),
        ("openai_key", "key " + (s := "sk-" + "proj-" + rnd(48)) + " in env", s),
        ("openai_key", "old style " + (s := "sk-" + rnd(48)), s),
        ("github_token", "remote uses " + (s := "gh" + "p_" + rnd(36)), s),
        ("github_token", "oauth " + (s := "gh" + "o_" + rnd(36)), s),
        ("github_token", "fine-grained " + (s := "github" + "_pat_" + rnd(22) + "_" + rnd(59)), s),
        ("gitlab_token", "CI_TOKEN is " + (s := "gl" + "pat-" + rnd(20)), s),
        ("slack_token", "bot " + (s := "xo" + "xb-" + "-".join([rnd(11, string.digits), rnd(13, string.digits),
                                                                  rnd(24)])), s),
        ("slack_webhook", "post to " + (s := "https://hooks.slack.com/services/T" + rnd(8) + "/B" + rnd(8) + "/"
                                         + rnd(24)), s),
        ("stripe_key", "billing " + (s := "sk" + "_live_" + rnd(24)), s),
        ("stripe_key", "restricted " + (s := "rk" + "_live_" + rnd(24)), s),
        ("stripe_key", "tests use " + (s := "sk" + "_test_" + rnd(24)), s),
        ("google_api_key", "maps " + (s := "AI" + "za" + rnd(35, B62 + "_-")), s),
        ("google_oauth", "client secret " + (s := "GOC" + "SPX-" + rnd(28)), s),
        ("aws_access_key", "id " + (s := "AK" + "IA" + rnd(16, string.ascii_uppercase + string.digits)), s),
        ("aws_access_key", "temp " + (s := "AS" + "IA" + rnd(16, string.ascii_uppercase + string.digits)), s),
        ("aws_secret_key", "aws_secret_access_key = " + (s := rnd(40, B62 + "/+")), s),
        ("azure_key", "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=" + (s := rnd(86) + "==") + ";", s),
        ("azure_sas", "https://x.blob.core.windows.net/c?sv=2022-11-02&sig=" + (s := rnd(43) + "%3D"), s),
        ("twilio_key", "api key " + (s := "SK" + hexs(32)), s),
        ("twilio_token", "TWILIO_AUTH_TOKEN=" + (s := hexs(32)), s),
        ("sendgrid_key", "mail " + (s := "S" + "G." + rnd(22) + "." + rnd(43)), s),
        ("huggingface_token", "hub " + (s := "h" + "f_" + rnd(34)), s),
        ("npm_token", "publish with " + (s := "np" + "m_" + rnd(36)), s),
        ("pypi_token", "upload " + (s := "py" + "pi-AgE" + rnd(60)), s),
        ("jwt", "Cookie: session=" + jwt, jwt),
        ("bearer_token", "Authorization: Bearer " + (s := rnd(32)), s),
        ("basic_auth", "Authorization: Basic " + (s := rnd(24) + "=="), s),
        ("private_key", "-----BEGIN RSA PRIVATE KEY-----\n" + pem_body + "\n-----END RSA PRIVATE KEY-----", pem_body),
        ("private_key", "-----BEGIN OPENSSH PRIVATE KEY-----\n" + pem_body + "\n-----END OPENSSH PRIVATE KEY-----",
         pem_body),
        ("url_credentials", "DATABASE_URL=postgres://app:" + (s := "pw" + rnd(10)) + "@db.internal:5432/app", s),
        ("url_credentials", "mongodb+srv://admin:" + (s := rnd(14)) + "@cluster0.example.net/db", s),
        ("url_credentials", "redis://:" + (s := rnd(12)) + "@cache:6379/0", s),
        ("connection_string", "Server=db;Database=app;User Id=sa;Password=" + (s := rnd(12)) + ";", s),
        ("secret_assignment", "export DB_PASSWORD=" + (s := rnd(14)), s),
        ("secret_assignment", '{"client_secret": "' + (s := rnd(20)) + '"}', s),
        ("secret_assignment", "api_key: " + (s := rnd(18)), s),
        ("secret_assignment", "slack:\n  signing_secret: '" + (s := hexs(32)) + "'", s),
        ("secret_assignment", 'token = "' + (s := rnd(16)) + '"', s),
        ("secret_flag", "mysql -u root --password " + (s := rnd(12)) + " app", s),
        ("password", "the wifi password is " + (s := "Tr0ub4dor&" + rnd(3, string.digits)), s),
        ("credit_card", "charged card " + (s := card()) + " twice", s),
        ("credit_card", "amex " + (s := " ".join([(c := card("37", 15))[:4], c[4:10], c[10:]])), s),
        ("high_entropy", "the upload signature was " + (s := rnd(40)) + " yesterday", s),
    ]
    return out


CASES = cases()


# ids never include the (fake) secrets themselves
@pytest.mark.parametrize(("kind", "text", "secret"), CASES, ids=[f"{k}-{i}" for i, (k, _t, _s) in enumerate(CASES)])
def test_each_secret_format_is_redacted(kind, text, secret):
    r = Redactor()
    out = r.redact(text)
    assert secret not in out, out
    for part in (secret[:12], secret[-12:]):  # no sizeable fragment survives either
        if len(part) >= 12 and part.strip():
            assert part not in out, out
    assert f"[REDACTED:{kind}]" in out, out
    assert r.report.counts.get(kind, 0) >= 1
    assert secret not in json.dumps(r.report.counts)


def test_every_builtin_pattern_has_a_case():
    covered = {k for k, _t, _s in CASES}
    assert {k for k, *_ in PATTERNS} - covered == {"shopify_token", "digitalocean_token", "databricks_token"}
    r = Redactor()
    for text, kind in [("shp" + "at_" + hexs(32), "shopify_token"), ("do" + "p_v1_" + hexs(64), "digitalocean_token"),
                       ("dapi" + hexs(32), "databricks_token")]:
        assert r.redact(text) == f"[REDACTED:{kind}]"


SAFE = [
    "Fixed the auth bug in commit 9fceb02d0ae598e95dc970b74767f19372d61af8 yesterday.",
    "git checkout 3f2a91c && git rebase -i HEAD~3",
    "sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "md5 d41d8cd98f00b204e9800998ecf8427e matches",
    f"request id {uuid.UUID(int=RNG.getrandbits(128))} returned 500",
    "See src/agentdream/engine/deterministic.py and /usr/local/lib/python3.13/site-packages/foo.py",
    "node_modules/@types/node/index.d.ts and ~/.claude/projects/-workspace-agentdream/abc.jsonl",
    "class AbstractSingletonProxyFactoryBean2 extends testRedactsAwsKey2InConfigFile",
    "Use XMLHttpRequestEventTarget2024 or getUserAccountSettingsV2Handler.",
    "Deploy dream-20261124-164117-0ef35f at 2026-11-24T16:41:17.123456+00:00 (v1.2.3-rc.4).",
    "python3.13-venv-x86_64-linux-gnu and x86_64-unknown-linux-gnu2024Build",
    "The password is stored in 1Password, never in the repo.",
    "max_tokens: 4096, temperature=0.2, token_count=1200, tokenizer: cl100k_base",
    'password: "${DB_PASSWORD}"  # from the environment',
    "secret: <your-secret-here>",
    "API_KEY=os.environ['OPENAI_API_KEY']",
    "Stripe test card 4242 4242 4242 4242 always succeeds.",
    "Order 1234-5678-9012-3456 shipped; phone +1 415 555 2671.",
    "The PokéCenter in Viridian City is south of the gym.",
    "Prefer pnpm over npm; run `pytest -q` before committing.",
    "export PATH=$HOME/.local/bin:$PATH",
    "Rotate the key every 90 days; the token TTL is 3600 seconds.",
]


@pytest.mark.parametrize("text", SAFE)
def test_no_false_positives(text):
    r = Redactor()
    assert r.redact(text) == text
    assert r.report.total == 0


def test_hex_and_uuids_are_secrets_right_after_a_keyword():
    r = Redactor()
    for text in ("api_key: " + hexs(40), "token is " + str(uuid.UUID(int=RNG.getrandbits(128))),
                 "my secret " + hexs(32)):
        assert "[REDACTED:" in r.redact(text), text


def test_entropy_rates_on_random_tokens():
    r = Redactor()
    hits24 = sum(r.is_high_entropy(rnd(24)) for _ in range(400))
    hits40 = sum(r.is_high_entropy(rnd(40)) for _ in range(400))
    assert hits24 >= 0.85 * 400 and hits40 >= 0.97 * 400
    assert not any(r.is_high_entropy(hexs(40)) for _ in range(50))  # git SHAs
    assert not r.is_high_entropy(rnd(20))  # below the minimum length
    assert shannon_entropy("") == 0.0 and shannon_entropy("aaaa") == 0.0
    assert abs(shannon_entropy("abcd") - 2.0) < 1e-9


def test_entropy_threshold_and_min_length_are_configurable():
    tok = rnd(30)
    assert "[REDACTED" in Redactor().redact(f"blob {tok} end")
    assert Redactor(entropy=False).redact(f"blob {tok} end") == f"blob {tok} end"
    assert Redactor(entropy_threshold=6.0).redact(f"blob {tok} end") == f"blob {tok} end"
    assert Redactor(entropy_min_length=40).redact(f"blob {tok} end") == f"blob {tok} end"
    short = rnd(15) + "0aZ"
    assert short in Redactor().redact(f"blob {short} end")
    assert short not in Redactor(entropy_min_length=16, entropy_threshold=3.5).redact(f"blob {short} end")


def test_options_emails_cards_custom_and_disabled():
    text = "mail bob@example.com, card " + card() + ", ticket ACME-12345"
    out = Redactor(emails=True, extra_patterns=[r"ACME-\d+"]).redact(text)
    assert "bob@example.com" not in out and "ACME-12345" not in out and "[REDACTED:credit_card]" in out
    assert "bob@example.com" in Redactor().redact(text)
    assert card() not in Redactor().redact("x " + card())
    no_cards = Redactor(cards=False)
    c = card()
    assert c in no_cards.redact("x " + c)
    assert Redactor(enabled=False).redact("export DB_PASSWORD=" + rnd(14)).startswith("export DB_PASSWORD=")
    assert redact("") == "" and redact("token = " + rnd(16)) == "token = [REDACTED:secret_assignment]"


def test_redaction_is_idempotent():
    text = "\n".join(t for _k, t, _s in CASES)
    once = Redactor().redact(text)
    r2 = Redactor()
    assert r2.redact(once) == once and r2.report.total == 0


# ------------------------------------------------------------------ wiring


class Capture(LLMBackend):
    name = "capture"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prompts = []

    def complete(self, system, prompt, json_schema=None):
        self.prompts.append(system + prompt)
        return '{"operations": []}'


def test_nothing_secret_reaches_the_model_or_the_report(tmp_path):
    gh, pw, ent, key = "gh" + "p_" + rnd(36), rnd(14), rnd(40), "sk-ant-" + rnd(40)
    mem = write(tmp_path / "m" / "MEMORY.md",
                f"# Deploy (token = {rnd(18)})\n\n- Deploy uses {gh} for the registry.\n"
                "- The PokéCenter is north of the gym.\n").parent
    s = tmp_path / "s"
    write_jsonl(s / "1.jsonl", [{"role": "user", "content": f"export DB_PASSWORD={pw} then retry"},
                                {"role": "assistant", "content": f"Uploaded with signature {ent}."},
                                {"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."},
                                {"role": "assistant", "content": "No PokéCenter north of the gym."}])
    be = Capture()
    job = dream(mem, [s], backend=be, instructions=f"focus on deploys; key {key}")
    assert job.status == "completed", job.error
    blob = "\n".join(be.prompts)
    for secret in (gh, pw, ent, key):
        assert secret not in blob
        assert secret not in json.dumps(job.report) and secret not in job.report_md
    red = job.report["redactions"]
    assert red.get("github_token") and red.get("secret_assignment") and red.get("high_entropy")
    assert red.get("anthropic_key")  # caught in the instructions by the final prompt pass
    assert "Redactions" in job.report_md and gh not in job.report_md
    # the staged store still has the real values in untouched text
    from pathlib import Path

    assert gh in (Path(job.store_path) / "MEMORY.md").read_text()


def test_rewriting_an_entry_with_a_secret_is_flagged(tmp_path):
    tok = "gh" + "p_" + rnd(36)
    mem = write(tmp_path / "m" / "MEMORY.md", f"- The deploy server is north of the gym. Log in with {tok} there.\n").parent
    s = tmp_path / "s"
    write_jsonl(s / "1.jsonl", [{"role": "assistant", "content": "No deploy server north of the gym."},
                                {"role": "assistant", "content": "Found the deploy server! It is south of the gym."},
                                {"role": "assistant", "content": "Walked north of the gym again, nothing here."}])
    job = dream(mem, [s])
    assert job.status == "completed", job.error
    (corr,) = [o for o in job.operations if o.type == "correct"]  # north -> south
    assert "[REDACTED:github_token]" in corr.upsert[0].content and tok not in json.dumps(job.report)
    assert any(corr.id in w and "contained a redacted secret" in w for w in job.warnings)


def test_cli_redaction_flags(tmp_path, capsys):
    ent = rnd(40)
    mem = write(tmp_path / "m" / "MEMORY.md", f"- Upload signature {ent} is stale.\n").parent
    s = write_jsonl(tmp_path / "s" / "1.jsonl", [{"role": "user", "content": "hello"}]).parent
    assert main(["dream", "-m", str(mem), "-s", str(s), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["config"]["redact_entropy"] is True
    assert main(["dream", "-m", str(mem), "-s", str(s), "--json", "--no-redact-entropy",
                 "--entropy-threshold", "4.5", "--redact-pattern", "stale"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["config"]["redact_entropy"] is False and d["config"]["entropy_threshold"] == 4.5
    assert d["config"]["redact_patterns"] == ["stale"]
