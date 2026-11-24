"""Secret redaction, applied before any text is analysed or sent to an LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# (kind, pattern, group-to-redact or 0 for whole match)
_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("private_key", re.compile(
        r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|$)"), 0),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{10,}"), 0),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"), 0),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), 0),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"), 0),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), 0),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b"), 0),
    ("aws_secret_key", re.compile(
        r"(?i)\baws_?secret_?(?:access_?)?key\b\s*[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{30,})"), 1),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), 0),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_\-\.=/+]{16,})"), 1),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+\-.]*://[^\s:/@]+:([^\s@/]+)@"), 1),
    ("secret_assignment", re.compile(
        r"(?i)\b(?:password|passwd|pwd|passphrase|secret|client_secret|api[_\-]?key|access[_\-]?token|auth[_\-]?token|token)"
        r"\b\s*[\"']?\s*[:=]\s*[\"']?([^\s\"',;]{4,})"), 1),
]
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


@dataclass
class RedactionReport:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str, n: int = 1) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + n

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class Redactor:
    def __init__(self, emails: bool = False, extra_patterns: list[str] | None = None, enabled: bool = True):
        self.emails = emails
        self.enabled = enabled
        self.extra = [re.compile(p) for p in (extra_patterns or [])]
        self.report = RedactionReport()

    def redact(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        out = text
        for kind, pat, group in _PATTERNS:
            out = self._sub(out, kind, pat, group)
        for pat in self.extra:
            out = self._sub(out, "custom", pat, 0)
        if self.emails:
            out = self._sub(out, "email", _EMAIL, 0)
        return out

    def _sub(self, text: str, kind: str, pat: re.Pattern[str], group: int) -> str:
        def repl(m: re.Match[str]) -> str:
            val = m.group(group) if group else m.group(0)
            if val.startswith("[REDACTED"):
                return m.group(0)
            self.report.add(kind)
            if group:
                s, e = m.span(group)
                base = m.start(0)
                whole = m.group(0)
                return whole[: s - base] + f"[REDACTED:{kind}]" + whole[e - base:]
            return f"[REDACTED:{kind}]"

        return pat.sub(repl, text)


def redact(text: str, emails: bool = False) -> str:
    return Redactor(emails=emails).redact(text)
