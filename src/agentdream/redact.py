"""Secret redaction, applied before any text is analysed, sent to an LLM, or quoted in a report.

Two layers:

1. **Known formats** (:data:`PATTERNS`): provider key prefixes (Anthropic, OpenAI, GitHub,
   GitLab, Slack, Stripe, Google, AWS, Azure, Twilio, SendGrid, Hugging Face, npm, PyPI, ...),
   JWTs, PEM private keys, ``password=`` / ``"token": "..."`` / ``secret: ...`` assignments
   in env, JSON and YAML, credentials in URLs (``postgres://user:pass@host``), connection
   strings (``Password=...;``, ``AccountKey=...``), and credit-card numbers (Luhn-checked).
2. **High-entropy tokens**: a long token (default >= 24 chars) that mixes upper case, lower
   case and digits, has Shannon entropy >= ``entropy_threshold`` bits/char (default 4.0) and
   does not read like an identifier (camelCase words) is redacted as ``high_entropy``.

What is deliberately *not* redacted, so normal engineering notes survive: hex digests (git
SHAs, sha256/md5 sums), UUIDs, file paths, version strings, timestamps, and identifiers -
unless they appear right after a secret keyword (``token``, ``secret``, ``password``,
``api key``, ``credential``, ``auth`` ... on the same line, within 40 characters), in which
case they are treated as secrets too. Random 24-character tokens score >= 4.0 bits/char
about 95% of the time (99% from 32 characters), hex never exceeds 4.0 (16 symbols), and
camelCase identifiers are excluded by the word-run check; see ``tests/test_redact.py``.

Redacted text is replaced by ``[REDACTED:<kind>]``; :class:`RedactionReport` counts matches
per kind (the secrets themselves are never stored or reported).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

Validator = Callable[[str], bool]

_PLACEHOLDER = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$|^%[A-Za-z_]+%$|^\{\{.*\}\}$|^<[^>]*>$|^\*+$|^x{4,}$|"
                          r"^\.{3,}$|^(?:true|false|null|none|nil|undefined|required|optional|redacted|"
                          r"\[REDACTED.*)$", re.I)


def _plausible_secret(value: str) -> bool:
    """Assignment values that are clearly not secrets: env references, template vars, numbers,
    booleans, masks, file paths, and lowercase prose words."""
    v = value.strip().strip("\"'`")
    if len(v) < 4 or _PLACEHOLDER.match(v):
        return False
    if re.fullmatch(r"[\d.,_:\-]+", v):  # numbers, versions, times
        return False
    return not v.startswith(("os.environ", "process.env", "env(", "getenv", "/", "./", "~/"))


def _secretish_word(value: str) -> bool:
    """For prose ("the password is X"): require something password-like, not a plain word."""
    return _plausible_secret(value) and bool(re.search(r"\d|[^\w]|[A-Z].*[a-z]|[a-z].*[A-Z]", value))


_KEYWORDS = (r"password|passwd|pwd|passphrase|secret|token|api[_\-]?key|apikey|access[_\-]?key|auth[_\-]?key|"
             r"private[_\-]?key|client[_\-]?secret|credentials?|signing[_\-]?key|encryption[_\-]?key|"
             r"auth[_\-]?token|session[_\-]?key|webhook[_\-]?secret")

# (kind, pattern, group to redact (0 = whole match), optional validator for that group)
PATTERNS: list[tuple[str, re.Pattern[str], int, Validator | None]] = [
    ("private_key", re.compile(
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
        r"(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|\Z)"), 0, None),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{10,}"), 0, None),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}"), 0, None),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), 0, None),
    ("gitlab_token", re.compile(r"\bgl(?:pat|dt|rt|ptt|soat|cbt|ft|imt)-[A-Za-z0-9_\-]{20,}"), 0, None),
    ("slack_webhook", re.compile(r"https://hooks\.slack\.com/(?:services|workflows|triggers)/[A-Za-z0-9/_\-]+"),
     0, None),
    ("slack_token", re.compile(r"\bxox[abposrce]-[A-Za-z0-9\-]{10,}"), 0, None),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}|\bwhsec_[A-Za-z0-9]{20,}"), 0, None),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), 0, None),
    ("google_oauth", re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{20,}|\bya29\.[0-9A-Za-z_\-]{20,}"), 0, None),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|APKA)[0-9A-Z]{16}\b"), 0, None),
    ("aws_secret_key", re.compile(
        r"(?i)\b(?:aws[_\-\s]?)?secret[_\-\s]?(?:access[_\-\s]?)?key(?:[_\-\s]?id)?\b[\"']?\s*[:=]\s*[\"']?"
        r"([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"), 1, None),
    ("azure_key", re.compile(r"(?i)\b(?:AccountKey|SharedAccessKey|SharedAccessSignature)=([A-Za-z0-9+/=%]{20,})"),
     1, None),
    ("azure_sas", re.compile(r"[?&]sig=([A-Za-z0-9%+/=]{20,})"), 1, None),
    ("twilio_key", re.compile(r"\bSK[0-9a-fA-F]{32}\b"), 0, None),
    ("twilio_token", re.compile(
        r"(?i)twilio\S*?\s*(?:auth[_\-\s]?token|secret)[\"']?\s*[:=]?\s*[\"']?([0-9a-f]{32})\b"), 1, None),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,32}\.[A-Za-z0-9_\-]{16,64}"), 0, None),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), 0, None),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"), 0, None),
    ("pypi_token", re.compile(r"\bpypi-AgE[A-Za-z0-9_\-]{50,}"), 0, None),
    ("shopify_token", re.compile(r"\bshp(?:at|ss|ca|pa)_[a-fA-F0-9]{32}\b"), 0, None),
    ("digitalocean_token", re.compile(r"\bdo[por]_v1_[a-f0-9]{64}\b"), 0, None),
    ("databricks_token", re.compile(r"\bdapi[a-f0-9]{32}(?:-\d)?\b"), 0, None),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), 0, None),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_\-\.=/+]{16,})"), 1, None),
    ("basic_auth", re.compile(r"(?i)\bauthorization\s*[:=]\s*[\"']?basic\s+([A-Za-z0-9+/=]{8,})"), 1, None),
    # scheme://user:password@host (postgres, mysql, mongodb+srv, redis://:pw@, amqp, https ...)
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+\-.]*://[^\s:/@]*:([^\s@/]+)@"), 1, _plausible_secret),
    # ADO.NET / ODBC style: Password=...; Pwd=...
    ("connection_string", re.compile(r"(?i)(?:^|[;\s\"'])(?:password|pwd)\s*=\s*([^;\s\"']{3,})"), 1,
     _plausible_secret),
    # KEY=value / "key": "value" / key: value  (env, JSON, YAML, TOML, CLI flags)
    ("secret_assignment", re.compile(
        r"(?i)(?<![A-Za-z0-9])[A-Za-z0-9_.\-]*?(?:" + _KEYWORDS + r")[\"']?\s*(?::|=|:=|=>)\s*[\"']?"
        r"([^\s\"',;}{\]]{4,})"), 1, _plausible_secret),
    ("secret_flag", re.compile(r"(?i)--(?:[a-z0-9]+-)*(?:password|passwd|token|secret|api-key)[= ]\s*[\"']?"
                               r"([^\s\"']{4,})"), 1, _plausible_secret),
    ("password", re.compile(r"(?i)\b(?:password|passwd|passphrase)\s+(?:is|was|of)\s+[\"'`]?([^\s\"'`,;]{4,})"), 1,
     _secretish_word),
]

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_CARD = re.compile(r"(?<![\d.\-])(?:\d[ \-]?){12,18}\d(?![\d.\-])")
_TOKEN = re.compile(r"(?<![A-Za-z0-9_+/=\-])[A-Za-z0-9_+/=\-.]{16,}(?![A-Za-z0-9_+/=\-])")
_HEX = re.compile(r"^(?:0x)?[0-9a-fA-F]+$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_KW_ANY = r"(?:" + _KEYWORDS + r"|\bauth\b|\bbearer\b|\bsignature\b|\bsig\b|\bkey\b)"
#: keyword somewhere in the 40 characters before the token, same line
_NEAR_KEYWORD = re.compile(r"(?i)" + _KW_ANY + r"[^\n]{0,40}$")
#: keyword *directly* before the token, only separators in between ("token: X", "secret is X")
_AFTER_KEYWORD = re.compile(r"(?i)" + _KW_ANY + r"[\"']?\s*(?:[:=]|=>|\bis\b|\bwas\b)?\s*[\"'`]?$")
_WORD_RUN = re.compile(r"[A-Z]?[a-z]{2,}")


def shannon_entropy(s: str) -> float:
    """Bits per character of ``s``'s empirical character distribution."""
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_like(digits: str) -> bool:
    if not 13 <= len(digits) <= 19 or len(set(digits)) < 3:
        return False
    # issuer prefixes: Visa, Mastercard, Amex, Discover, JCB, Diners, UnionPay
    if not re.match(r"4|5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720)|3[47]|6(?:011|5|4[4-9]|2)|35|3(?:0[0-5]|[68])", digits):
        return False
    return luhn_ok(digits)


@dataclass
class RedactionReport:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str, n: int = 1) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + n

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class Redactor:
    """Configurable redactor; ``redact(text)`` returns the redacted text and counts matches.

    Args:
        emails: also redact e-mail addresses.
        extra_patterns: additional regexes (whole match redacted as ``custom``).
        enabled: master switch.
        entropy: enable the high-entropy token check.
        entropy_threshold: bits/char a token must reach to count as random.
        entropy_min_length: minimum token length for the entropy check.
        cards: redact Luhn-valid card numbers with a known issuer prefix.
    """

    def __init__(self, emails: bool = False, extra_patterns: Iterable[str] | None = None, enabled: bool = True, *,
                 entropy: bool = True, entropy_threshold: float = 4.0, entropy_min_length: int = 24,
                 cards: bool = True):
        self.emails = emails
        self.enabled = enabled
        self.extra = [re.compile(p) for p in (extra_patterns or [])]
        self.entropy = entropy
        self.entropy_threshold = float(entropy_threshold)
        self.entropy_min_length = max(8, int(entropy_min_length))
        self.cards = cards
        self.report = RedactionReport()

    def redact(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        out = text
        for kind, pat, group, check in PATTERNS:
            out = self._sub(out, kind, pat, group, check)
        for pat in self.extra:
            out = self._sub(out, "custom", pat, 0, None)
        if self.emails:
            out = self._sub(out, "email", _EMAIL, 0, None)
        if self.cards:
            out = self._sub(out, "credit_card", _CARD, 0, lambda v: _card_like(re.sub(r"\D", "", v)))
        if self.entropy:
            out = self._entropy_pass(out)
        return out

    __call__ = redact

    def _sub(self, text: str, kind: str, pat: re.Pattern[str], group: int, check: Validator | None) -> str:
        def repl(m: re.Match[str]) -> str:
            val = m.group(group) if group else m.group(0)
            if val is None or "[REDACTED" in val or (check is not None and not check(val)):
                return m.group(0)
            self.report.add(kind)
            if group:
                s, e = m.span(group)
                base = m.start(0)
                whole = m.group(0)
                return whole[: s - base] + f"[REDACTED:{kind}]" + whole[e - base:]
            return f"[REDACTED:{kind}]"

        return pat.sub(repl, text)

    # -- entropy
    def is_high_entropy(self, token: str, context: str = "") -> bool:
        """Would ``token`` be redacted by the entropy check? ``context`` is the text before it on
        the same line (exposed for tests and tuning)."""
        tok = token.strip(".=-_")
        if len(tok) < 16:
            return False
        if context and _AFTER_KEYWORD.search(context):
            # "api_key: <anything random-ish>": hex digests and UUIDs count here too
            return bool(_HEX.match(tok) or _UUID.match(tok)) or shannon_entropy(tok) >= 3.0
        near = bool(context) and bool(_NEAR_KEYWORD.search(context))
        if len(tok) < (16 if near else self.entropy_min_length) or _HEX.match(tok) or _UUID.match(tok):
            return False
        if not (re.search(r"[A-Z]", tok) and re.search(r"[a-z]", tok) and re.search(r"\d", tok)):
            return False
        if shannon_entropy(tok) < (min(3.5, self.entropy_threshold) if near else self.entropy_threshold):
            return False
        # identifiers are almost entirely word-like runs ("getUserV2Handler": ~0.9-1.0 of the letters);
        # random base62 strings score 0.3-0.75 (99th percentile ~0.8 at 24 chars)
        letters = sum(ch.isalpha() for ch in tok)
        words = sum(len(w) for w in _WORD_RUN.findall(tok))
        return letters == 0 or words / letters < 0.8

    def _entropy_pass(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            tok = m.group(0)
            if "REDACTED" in tok:
                return tok
            line_start = text.rfind("\n", 0, m.start()) + 1
            context = text[max(line_start, m.start() - 60):m.start()]
            if self.is_high_entropy(tok, context):
                self.report.add("high_entropy")
                return "[REDACTED:high_entropy]"
            return tok

        return _TOKEN.sub(repl, text)


def redact(text: str, emails: bool = False, **options: object) -> str:
    """Convenience one-shot redaction with default settings."""
    return Redactor(emails=emails, **options).redact(text)  # type: ignore[arg-type]


__all__ = ["PATTERNS", "RedactionReport", "Redactor", "luhn_ok", "redact", "shannon_entropy"]
