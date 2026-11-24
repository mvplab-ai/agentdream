"""Lightweight, dependency-free text heuristics used by the deterministic engine.

Nothing here is tied to specific test strings: we normalise text, extract
"claims" (subject / value / polarity) with a handful of generic English
patterns, and compare them with token-set similarity plus typed value slots
(directions, numbers, on/off states).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

STOPWORDS = frozenset("""
a an the this that these those it its it's is are was were be been being am do does did done doing
of in on at to for from by with without about into onto over under up down out off as than then so
and or but nor if else when while where which who whom whose what why how there here
i me my mine we us our ours you your yours he him his she her hers they them their theirs
will would shall should can could may might must just also too very really quite actually now still
some any each every all both few more most other such own same only
s t d ll re ve m
let let's lets ok okay yes yeah hmm oh well maybe perhaps probably please
go going gone get got make made one thing things way like
""".split())

NEGATIONS = frozenset({"not", "no", "never", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "isn't",
                       "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't", "werent", "won't", "wont",
                       "can't", "cant", "cannot", "shouldn't", "shouldnt", "mustn't", "nothing", "nowhere",
                       "none", "neither", "nor", "without"})

# Typed value slots. Two claims about the same subject conflict when they put
# different words in the same slot category.
SLOT_VOCAB: dict[str, frozenset[str]] = {
    "direction": frozenset({"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
                            "left", "right", "above", "below", "upstairs", "downstairs", "inside", "outside",
                            "behind", "front", "top", "bottom"}),
    "state": frozenset({"enabled", "disabled", "true", "false", "on", "off", "open", "closed", "required",
                        "optional", "allowed", "forbidden", "public", "private"}),
}
_NUM = re.compile(r"(?<![\w.])v?\d+(?:\.\d+)*(?!\w|\.\d)")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_'+#.\-]*[a-z0-9+#]|[a-z0-9]")

PREPOSITIONS = frozenset({"in", "at", "of", "on", "for", "from", "near", "by", "with", "inside", "within",
                          "under", "next", "beside", "behind", "to", "during"})

FAILURE_MARKERS = re.compile(
    r"\b(error|errors|failed|fail|fails|failing|failure|mistake|wrong|oops|forgot|forgotten|traceback|exception|"
    r"doesn'?t work|didn'?t work|not working|not found|no such|can'?t find|couldn'?t find|cannot find|dead end|"
    r"nothing here|isn'?t here|not here|wasted|again|stuck|loop(?:ing)?|crash(?:ed|es)?|broke|broken|"
    r"undefined|missing|denied|refused|timed out|timeout|no [\wÀ-ɏ'\-]+ (?:here|there|found|anywhere))\b",
    re.I)

CONFIRM_MARKERS = re.compile(
    r"\b(found|finally|actually|turns out|turned out|confirmed|confirm|verified|indeed|correct|arrived|reached|"
    r"entered|located|is at|it'?s at|there it is|got it|works|worked|success|successfully|real(?:ly)?)\b", re.I)

INTENT_MARKERS = re.compile(
    r"\b(heading|head|going|walk(?:ing)? (?:to|toward)|try(?:ing)?|will|plan|planning|let me|i'?ll|"
    r"looking for|to find|search(?:ing)? for|should be|supposed to be|according to|memory says|my notes say|"
    r"i think|i believe|probably|expect(?:ed|ing)?)\b", re.I)


def fold(text: str) -> str:
    """Lowercase + strip accents (PokéCenter -> pokecenter) + normalise quotes."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return t.lower()


def strip_markup(text: str) -> str:
    t = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?", "", text)
    t = re.sub(r"[*_`]{1,3}", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t.strip()


def strip_dates(text: str) -> str:
    t = _DATE.sub(" ", text)
    t = re.sub(r"[\(\[]\s*[\)\]]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def stem(tok: str) -> str:
    t = tok
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
        if len(t) > 2 and t[-1] == t[-2] and t[-1] not in "ls":
            t = t[:-1]
    elif len(t) > 4 and t.endswith("ied"):
        t = t[:-3] + "y"
    elif len(t) > 4 and t.endswith("ed") and not t.endswith("eed"):
        t = t[:-2]
        if len(t) > 2 and t[-1] == t[-2] and t[-1] not in "ls":
            t = t[:-1]
    elif len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 3 and t.endswith("es") and t[-3] in "sxz":
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    return t


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(fold(strip_dates(text)))


def content_tokens(text: str, keep_negations: bool = False) -> list[str]:
    out = []
    for t in tokens(text):
        t = t.strip(".'")
        if not t:
            continue
        if t in NEGATIONS:
            if keep_negations:
                out.append(t)
            continue
        if t in STOPWORDS:
            continue
        out.append(stem(t))
    return out


def token_set(text: str) -> frozenset[str]:
    return frozenset(content_tokens(text))


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Fraction of ``a`` contained in ``b``."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def normalize(text: str) -> str:
    t = fold(strip_dates(strip_markup(text)))
    t = re.sub(r"[^a-z0-9#+]+", " ", t)
    return " ".join(t.split())


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def numbers(text: str) -> frozenset[str]:
    return frozenset(n.lstrip("v") for n in _NUM.findall(fold(strip_dates(text))))


def is_negated(text: str) -> bool:
    toks = set(tokens(text))
    if toks & NEGATIONS:
        return True
    return bool(re.search(r"\b(avoid|stop using|no longer|rejected|reject|deprecated|banned)\b", fold(text)))


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\s+[—–]\s+|;\s+", line):
            s = s.strip()
            if s:
                parts.append(s)
    return parts


def _slot_scan(text: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    toks = [t.strip(".'") for t in tokens(text)]
    pos: dict[str, set[str]] = {}
    neg: dict[str, set[str]] = {}
    for i, t in enumerate(toks):
        for cat, vocab in SLOT_VOCAB.items():
            if t in vocab:
                window = toks[max(0, i - 2): i]
                target = neg if any(w in NEGATIONS for w in window) else pos
                target.setdefault(cat, set()).add(t)
    nums = numbers(text)
    if nums:
        pos["number"] = set(nums)
    return pos, neg


def slots(text: str) -> dict[str, set[str]]:
    """Return slot-category -> words asserted (directions, states, numbers); negated words excluded."""
    return _slot_scan(text)[0]


def negated_slots(text: str) -> dict[str, set[str]]:
    """Slot words that appear negated ("not north")."""
    return _slot_scan(text)[1]


# ---------------------------------------------------------------- claims

_COPULA = re.compile(
    r"\s(?:is|are|was|were|lives|sits|stands|runs on|listens on|located|lies|=|->|:)\s+|\s*[:=]\s+")

_NEG_PREF = re.compile(
    r"(?:^|\b)(?:do not|don't|dont|never|avoid|stop using|stop|no longer use|shouldn't|should not|must not|mustn't|"
    r"can't|cannot|won't|will not|we rejected|rejected|reject|ditch|drop)\s+"
    r"(?:use |using |run |running |call |calling |choose |pick |try |trying |install |installing |rely on |"
    r"depend on |go with |switch to |use of )?(?P<obj>.+)", re.I)
_POS_PREF = re.compile(
    r"(?:^|\b)(?:always use|please use|let's use|lets use|let us use|we should use|you should use|should use|"
    r"use|prefer|prefers|preferred|switch to|switched to|switching to|go with|going with|stick with|sticking with|"
    r"choose|chose|pick|adopt|adopted|recommend|we use|i use|keep using|start using|move to|moved to)\s+"
    r"(?P<obj>.+)", re.I)
_EVAL = re.compile(
    r"^(?P<obj>.+?)\s+(?:is|are|was|were|seems|seem|looks|look|might be|may be|could be|would be|turned out|"
    r"turns out to be|ended up)\s+(?:actually\s+|much\s+|way\s+|really\s+|definitely\s+|clearly\s+|a\s+|the\s+|"
    r"probably\s+)*(?P<val>better|best|preferred|preferable|the way to go|the right choice|good|great|fine|faster|"
    r"recommended|nicer|superior|more reliable|winner|worse|worst|bad|broken|slower|terrible|flaky|buggy|"
    r"unreliable|deprecated|rejected|banned|not recommended|a bad idea|a mistake|not good|not great|"
    r"not better|not the way to go)\b", re.I)
_POS_EVAL = frozenset({"better", "best", "preferred", "preferable", "the way to go", "the right choice", "good",
                       "great", "fine", "faster", "recommended", "nicer", "superior", "more reliable", "winner"})
_OBJ_STOP = re.compile(
    r"\s+(?:for|because|since|as|instead of|rather than|over|than|when|while|in|on|to|unless|until|if|so|"
    r"but|and then|which|that|after|before|anymore|any more|again|here|there)\b|[,;:.!?()\[\]\"]|\s+[—–-]\s+")
_LEADING_FILLER = re.compile(r"^(?:(?:ok(?:ay)?|so|well|hmm|actually|maybe|perhaps|honestly|also|and|but|please|"
                             r"i think|i guess|on reflection|after all|in the end|ultimately|note|rule)[,:]?\s+)+", re.I)


@dataclass
class Claim:
    text: str
    kind: str  # "preference" | "fact"
    subject: frozenset[str]
    subject_core: frozenset[str]
    value: str = ""
    value_tokens: frozenset[str] = field(default_factory=frozenset)
    polarity: int = 1  # +1 positive, -1 negative
    slots: dict[str, set[str]] = field(default_factory=dict)
    object_text: str = ""

    def same_subject(self, other: "Claim") -> bool:
        if not self.subject or not other.subject:
            return False
        if self.kind == "preference" or other.kind == "preference":
            return self.subject == other.subject or (
                jaccard(self.subject, other.subject) >= 0.5 and bool(self.subject_core & other.subject_core))
        if self.subject_core and other.subject_core and not (self.subject_core & other.subject_core):
            return False
        return jaccard(self.subject, other.subject) >= 0.6 or (
            bool(self.subject_core) and self.subject_core <= other.subject and
            containment(self.subject, other.subject) >= 0.5)


def _clean_object(raw: str) -> str:
    obj = _OBJ_STOP.split(" " + raw.strip(), maxsplit=1)[0].strip()
    obj = re.sub(r"^(?:the|a|an|using|use)\s+", "", obj, flags=re.I)
    return obj.strip(" '\"`*")


def _core(subject_text: str) -> frozenset[str]:
    """Head of a noun phrase: tokens before the first preposition."""
    head: list[str] = []
    for t in tokens(subject_text):
        t = t.strip(".'")
        if t in PREPOSITIONS:
            break
        if t in STOPWORDS or t in NEGATIONS:
            continue
        head.append(stem(t))
    return frozenset(head)


def parse_claim(sentence: str) -> Claim | None:
    """Extract a preference or fact claim from one sentence, or None."""
    raw = strip_dates(strip_markup(sentence)).strip()
    if not raw:
        return None
    text = _LEADING_FILLER.sub("", fold(raw)).strip()
    text = re.sub(r"^(?:we|i|you|the team|team)\s+(?=(?:should|must|do|don't|dont|never|always|prefer|use|rejected|"
                  r"avoid|stopped|switched|chose|decided|will|won't)\b)", "", text)
    text = re.sub(r"^(?:decided to|decided against)\s+", lambda m: "use " if "to" in m.group(0) else "avoid ", text)

    m = _EVAL.match(text)
    if m:
        obj = _clean_object(m.group("obj"))
        obj = re.sub(r"^(?:using|use)\s+", "", obj)
        subj = token_set(obj)
        if subj:
            val = m.group("val").lower()
            pol = 1 if val in _POS_EVAL else -1
            if re.search(r"\b(not|n't|never)\b", text[: m.start("val")][-12:]):
                pol = -pol
            return Claim(sentence, "preference", subj, _core(obj) or subj, val, frozenset({val}), pol,
                         object_text=obj)

    m = _NEG_PREF.search(text)
    if m and m.start() <= 3:
        obj = _clean_object(m.group("obj"))
        subj = token_set(obj)
        if subj:
            return Claim(sentence, "preference", subj, _core(obj) or subj, "avoid", frozenset({"avoid"}), -1,
                         object_text=obj)

    m = _POS_PREF.search(text)
    if m and m.start() <= 3:
        obj = _clean_object(m.group("obj"))
        subj = token_set(obj)
        if subj:
            return Claim(sentence, "preference", subj, _core(obj) or subj, "use", frozenset({"use"}), 1,
                         object_text=obj)

    # Facts: "<subject> is <value>"
    m = _COPULA.search(" " + text + " ")
    if m:
        left = (" " + text + " ")[: m.start()].strip()
        right = (" " + text + " ")[m.end():].strip().rstrip(".!")
        left = re.sub(r"^(?:that|note that|remember that|remember:|fact:)\s+", "", left)
        subj = token_set(left)
        if subj and right:
            neg = bool(re.match(r"^(?:not|never|no longer|n't)\b", right)) or left.endswith(("n't", " not"))
            right_clean = re.sub(r"^(?:not|never|no longer|actually|really|now|currently|just)\s+", "", right)
            return Claim(sentence, "fact", subj, _core(left) or subj, right_clean, token_set(right_clean),
                         -1 if neg else 1, slots(right_clean))
    return None


def claims_in(text: str) -> list[Claim]:
    out = []
    for s in split_sentences(strip_markup(text)):
        c = parse_claim(s)
        if c:
            out.append(c)
    return out


def values_conflict(a: Claim, b: Claim) -> bool:
    """Same subject, different value (typed slots first, then token sets)."""
    shared = set(a.slots) & set(b.slots)
    for cat in shared:
        if a.slots[cat] != b.slots[cat] and not (a.slots[cat] & b.slots[cat] and cat == "number"):
            return True
    if shared:
        return False
    if a.kind == "fact" and b.kind == "fact" and a.value_tokens and b.value_tokens:
        return jaccard(a.value_tokens, b.value_tokens) < 0.34
    return False


def replace_word(text: str, old: str, new: str) -> str:
    """Replace whole-word ``old`` (accent/case-insensitive) with ``new``, preserving capitalisation."""
    pat = re.compile(r"(?<![\w])" + re.escape(old) + r"(?![\w])", re.I)

    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        if w.isupper() and len(w) > 1:
            return new.upper()
        if w[:1].isupper():
            return new[:1].upper() + new[1:]
        return new

    return pat.sub(repl, text)


def flip_preference(text: str, to_positive: bool) -> str:
    """Rewrite a preference sentence to the opposite polarity, best effort."""
    t = text.strip()
    if to_positive:
        rules = [
            (r"\b(?:do not|don't|dont|never|shouldn't|should not|must not)\s+(use|run|call|choose|pick|try|install)\b",
             r"\1"),
            (r"\bavoid\s+using\b", "use"),
            (r"\bavoid\b", "use"),
            (r"\bstop using\b", "use"),
            (r"\bno longer use\b", "use"),
            (r"\bis (?:rejected|deprecated|banned|bad|worse|broken|not recommended)\b", "is preferred"),
            (r"\bare (?:rejected|deprecated|banned|bad|worse|broken|not recommended)\b", "are preferred"),
        ]
    else:
        rules = [
            (r"\b(?:always use|please use|use)\b", "don't use"),
            (r"\bprefer\b", "avoid"),
            (r"\bis (?:better|preferred|recommended|good|great|the way to go)\b", "is rejected"),
        ]
    for pat, rep in rules:
        new, n = re.subn(pat, rep, t, count=1, flags=re.I)
        if n:
            new = new.strip()
            return new[:1].upper() + new[1:]
    return ""


def shorten(text: str, n: int = 160) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 3] + "..."
