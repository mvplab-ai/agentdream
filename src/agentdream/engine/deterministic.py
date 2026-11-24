"""Deterministic (no-LLM) dreaming heuristics.

* consolidate: normalised-text / token-set near-duplicate detection; subject +
  polarity extraction to find contradictions; timestamps / store order decide
  which note is stale.
* verify: every memory claim is checked against transcript sentences that
  mention its subject. Typed value slots (directions, numbers, states) and
  generic "X is Y" values are compared; failure / confirmation / intent cues
  weight the evidence. Corrections cite ``file:line`` + quote.
* insights: frequency counting across sessions for recurring mistakes,
  repeated preferences, preference reversals and converged command workflows.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from ..context import DreamContext, SentenceRec
from ..models import Evidence, MemoryEntry, Operation, stable_hash
from ..text import (CONFIRM_MARKERS, FAILURE_MARKERS, INTENT_MARKERS, SLOT_VOCAB, Claim, claims_in, containment,
                    flip_preference, fold, is_negated, jaccard, negated_slots, normalize, numbers, parse_claim,
                    ratio, replace_word, shorten, slots, strip_dates, strip_markup, token_set, values_conflict)
from ..util import parse_timestamp

_PRONOUN_START = re.compile(r"^(?:it|it's|its|there|that|this|they|they're|which)\b", re.I)
_SLOT_PRIORITY = ("direction", "state", "number")


# ============================================================== helpers

def _when(e: MemoryEntry) -> str:
    return e.updated_at or e.created_at or ""


def _clean(text: str) -> str:
    return " ".join(strip_markup(text).split())


def _is_duplicate(a: MemoryEntry, b: MemoryEntry) -> bool:
    na, nb = normalize(a.content), normalize(b.content)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if is_negated(a.content) != is_negated(b.content):
        return False
    if numbers(a.content) != numbers(b.content):
        return False
    sa, sb = slots(a.content), slots(b.content)
    for cat in set(sa) & set(sb):
        if sa[cat] != sb[cat]:
            return False
    if set(sa) != set(sb) and (sa.keys() ^ sb.keys()) & {"direction", "state"}:
        return False
    ca, cb = _main_claim(a), _main_claim(b)
    if ca and cb and ca.kind == cb.kind == "fact" and ca.same_subject(cb) and values_conflict(ca, cb):
        return False  # same subject, different value: a staleness question, not a duplicate
    ta, tb = token_set(a.content), token_set(b.content)
    j = jaccard(ta, tb)
    if j >= 0.75:
        return True
    if j >= 0.5 and ratio(a.content, b.content) >= 0.85:
        return True
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(small) >= 3 and containment(small, big) >= 0.9 and j >= 0.6:
        return True
    return False


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _components(n: int, pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    uf = _UF(n)
    for a, b in pairs:
        uf.union(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def _main_claim(e: MemoryEntry) -> Claim | None:
    cs = claims_in(e.content)
    return cs[0] if cs else None


_CHATTY = re.compile(r"^(?:(?:ok(?:ay)?|so|well|hmm|actually|maybe|perhaps|honestly|on reflection|after all|please|"
                     r"i think|i guess|yeah|right)[,:]?\s+)+", re.I)


def _imperative(text: str) -> str:
    t = _CHATTY.sub("", _clean(text)).rstrip(".!")
    t = re.sub(r"^(?:let'?s|let us|we should|you should|i want to|we will|we'll)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+(?:from now on|going forward|after all|instead)$", "", t, flags=re.I)
    return t[:1].upper() + t[1:] if t else t


def resolved_preference_content(older_text: str, newer_text: str, newer: Claim) -> str:
    """Current-value text for a preference reversal.

    Prefer flipping the older note's own claim clause (keeps its wording/scope, drops stale
    justifications); otherwise turn the newer statement into an imperative note."""
    clause = re.split(r"\s*[,;:—–]\s*|\s+-\s+|\s+(?:because|since|as|but)\s+", strip_dates(_clean(older_text)))[0]
    flipped = flip_preference(clause, to_positive=newer.polarity > 0)
    if flipped:
        return flipped.rstrip(".")
    if newer.value in ("use", "avoid"):
        imp = _imperative(newer_text)
        if imp:
            return imp
    obj = newer.object_text or " ".join(sorted(newer.subject))
    return f"Prefer {obj}" if newer.polarity > 0 else f"Avoid {obj}"


def history_suffix(older_text: str, older_when: str | None, newer_ref: str | None) -> str:
    when = f" ({older_when[:10]})" if older_when else ""
    later = f"; reversed later ({newer_ref})" if newer_ref else "; reversed by a later note"
    quote = shorten(strip_dates(_clean(older_text)), 120).rstrip(".")
    return f". (History: earlier said \"{quote}\"{when}{later}.)"


# ============================================================== consolidate

def consolidate(ctx: DreamContext) -> None:
    _merge_duplicates(ctx)
    ctx.check_cancel()
    _memory_conflicts(ctx)


def _merge_duplicates(ctx: DreamContext) -> None:
    entries = list(ctx.entries)
    n = len(entries)
    toks = [token_set(e.content) for e in entries]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not toks[i] or not toks[j]:
                continue
            if len(toks[i] & toks[j]) == 0:
                continue
            if _is_duplicate(entries[i], entries[j]):
                pairs.append((i, j))
    for group in _components(n, pairs):
        members = [entries[i] for i in group]
        survivor = max(members, key=lambda e: (DreamContext.recency_key(e), len(e.content)))
        # keep the most informative wording if the newest one is just a shorter restatement
        richest = max(members, key=lambda e: len(token_set(e.content)))
        content = survivor.content
        if len(token_set(richest.content)) > len(token_set(survivor.content)) + 1:
            content = richest.content
        others = [e for e in members if e.id != survivor.id]
        merged = survivor.copy()
        merged.content = content
        merged.metadata = {**merged.metadata, "merged_from": sorted(e.id for e in members)}
        stamps = [parse_timestamp(_when(e)) for e in members]
        created = [e.created_at for e in members if e.created_at]
        if created:
            merged.created_at = min(created)
        op = Operation(
            id=Operation.make_id("merge", [e.id for e in members], [content]),
            type="merge",
            summary=f"Merged {len(members)} duplicate notes into {survivor.id}",
            reason="Notes say the same thing (normalised text / token overlap); kept the most recent wording."
            if any(s is not None for s in stamps) else
            "Notes say the same thing (normalised text / token overlap).",
            targets=[e.id for e in members], remove=[e.id for e in others], upsert=[merged],
            before=[e.content for e in members], after=[content], confidence=0.85, phase="consolidate",
        )
        ctx.add_op(op)


def _memory_conflicts(ctx: DreamContext) -> None:
    entries = list(ctx.entries)
    claims = [_main_claim(e) for e in entries]
    pref_pairs, fact_pairs = [], []
    for i in range(len(entries)):
        ci = claims[i]
        if ci is None:
            continue
        for j in range(i + 1, len(entries)):
            cj = claims[j]
            if cj is None or ci.kind != cj.kind or not ci.same_subject(cj):
                continue
            if ci.kind == "preference":
                if ci.polarity != cj.polarity:
                    pref_pairs.append((i, j))
            else:
                if ci.polarity == cj.polarity and values_conflict(ci, cj):
                    fact_pairs.append((i, j))
                elif ci.polarity != cj.polarity and not values_conflict(ci, cj):
                    fact_pairs.append((i, j))
    used: set[int] = set()
    for group in _components(len(entries), pref_pairs):
        group = [g for g in group if g not in used]
        if len(group) < 2:
            continue
        used.update(group)
        ordered = sorted(group, key=lambda k: DreamContext.recency_key(entries[k]))
        newest = entries[ordered[-1]]
        newest_claim = claims[ordered[-1]]
        older_idx = [k for k in ordered[:-1] if claims[k].polarity != newest_claim.polarity] or ordered[:-1]
        oldest = entries[older_idx[0]]
        current = resolved_preference_content(claims[older_idx[0]].text, newest.content, newest_claim)
        if _when(newest):
            current += history_suffix(oldest.content, _when(oldest), _when(newest)[:10])
        else:
            current += history_suffix(oldest.content, _when(oldest), None)
        resolved = newest.copy()
        resolved.content = current
        others = [entries[k] for k in ordered[:-1]]
        resolved.metadata = {**resolved.metadata, "history": [
            {"content": e.content, "when": _when(e) or None, "id": e.id} for e in others]}
        has_ts = any(_when(entries[k]) for k in ordered)
        op = Operation(
            id=Operation.make_id("contradiction", [e.id for e in others] + [newest.id], [current]),
            type="contradiction", kind="temporal",
            summary=f"Preference reversed over time: current value is \"{shorten(_clean(newest.content), 80)}\"",
            reason=("Opposite-polarity notes about the same subject; the latest one "
                    + ("(by timestamp)" if has_ts else "(by position in the store)") +
                    " is kept as the current value and the earlier stance is recorded as history."),
            targets=[e.id for e in others] + [newest.id], remove=[e.id for e in others], upsert=[resolved],
            before=[e.content for e in others] + [newest.content], after=[current],
            history=[f"{_when(e) or 'earlier'}: {e.content}" for e in others] +
                    [f"{_when(newest) or 'later'}: {newest.content}"],
            confidence=0.75 if has_ts else 0.55, phase="consolidate",
        )
        ctx.add_op(op)

    for group in _components(len(entries), fact_pairs):
        group = [g for g in group if g not in used]
        if len(group) < 2:
            continue
        used.update(group)
        ordered = sorted(group, key=lambda k: DreamContext.recency_key(entries[k]))
        newest = entries[ordered[-1]]
        stale = [entries[k] for k in ordered[:-1]]
        has_ts = any(_when(entries[k]) for k in ordered)
        kept = newest.copy()
        kept.metadata = {**kept.metadata, "supersedes": [
            {"id": e.id, "content": e.content, "when": _when(e) or None} for e in stale]}
        opposite = any(claims[k].polarity != claims[ordered[-1]].polarity for k in ordered[:-1])
        op = Operation(
            id=Operation.make_id("replace", [e.id for e in stale] + [newest.id], [newest.content]),
            type="replace",
            kind="conflict" if opposite else None,
            summary=f"Stale value replaced by the latest: \"{shorten(_clean(newest.content), 90)}\"",
            reason=("Notes give different values for the same subject; the most recent "
                    + ("(by timestamp)" if has_ts else "(by position in the store)") + " wins."),
            targets=[e.id for e in stale] + [newest.id], remove=[e.id for e in stale], upsert=[kept],
            before=[e.content for e in stale], after=[newest.content],
            history=[f"{_when(e) or 'earlier'}: {e.content}" for e in stale],
            confidence=0.75 if has_ts else 0.55, phase="consolidate",
        )
        ctx.add_op(op)


# ============================================================== verify

def _mentions(claim: Claim, rec: SentenceRec) -> bool:
    if claim.subject_core and claim.subject_core <= rec.toks:
        return True
    return len(claim.subject) >= 2 and containment(claim.subject, rec.toks) >= 0.67


def _sentence_windows(ctx: DreamContext, claim: Claim, rivals: list[Claim]):
    """Yield (rec, prev_rec_same_message_or_None, via_pronoun) for sentences about ``claim``."""
    recs = ctx.sentences()
    qualifiers = claim.subject - claim.subject_core
    rival_quals = [r.subject - r.subject_core for r in rivals]
    all_rival = frozenset().union(*rival_quals) if rival_quals else frozenset()
    self_d = qualifiers - all_rival
    rival_d = [q - qualifiers for q in rival_quals]
    active: dict[int, str] = {}  # transcript index -> "self" / "rival" (which qualifier was seen last)
    prev: SentenceRec | None = None
    prev_mentioned = False
    for rec in recs:
        if rivals:
            self_hit = bool(self_d & rec.toks)
            rival_hit = any(rd & rec.toks for rd in rival_d)
            if self_hit and not rival_hit:
                active[rec.t_index] = "self"
            elif rival_hit and not self_hit:
                active[rec.t_index] = "rival"
        same_msg_prev = prev if (prev is not None and prev.t_index == rec.t_index and
                                 prev.m_index == rec.m_index) else None
        mentioned = _mentions(claim, rec)
        pronoun = (not mentioned and same_msg_prev is not None and prev_mentioned
                   and bool(_PRONOUN_START.match(rec.text.strip())))
        ok_scope = not rivals or active.get(rec.t_index) == "self" or bool(self_d and self_d <= rec.toks)
        if (mentioned or pronoun) and ok_scope:
            yield rec, same_msg_prev, pronoun
        prev = rec
        prev_mentioned = mentioned or pronoun


def _rivals(ctx: DreamContext, entry: MemoryEntry, claim: Claim) -> list[Claim]:
    out = []
    for e in ctx.entries:
        if e.id == entry.id:
            continue
        for c in claims_in(e.content):
            if c.kind == claim.kind and c.subject_core and c.subject_core == claim.subject_core \
                    and c.subject != claim.subject:
                out.append(c)
    return out


def verify(ctx: DreamContext) -> None:
    if not ctx.transcripts:
        return
    entries = list(ctx.entries)
    for k, entry in enumerate(entries):
        ctx.check_cancel()
        ctx.progress("verify", k / max(1, len(entries)))
        current = ctx.entry(entry.id)
        if current is None:
            continue
        for claim in claims_in(current.content):
            if claim.kind == "fact" and claim.polarity > 0:
                cat = next((c for c in _SLOT_PRIORITY if c in claim.slots), None)
                done = _verify_slot(ctx, current, claim, cat) if cat else _verify_generic(ctx, current, claim)
            elif claim.kind == "preference":
                done = _verify_preference(ctx, current, claim)
            else:
                done = False
            if done:
                break


_NEG_RE = re.compile(r"\b(?:no|not|never|nothing|nowhere|isn'?t|wasn'?t|aren'?t|can'?t|couldn'?t|cannot|"
                     r"didn'?t|doesn'?t|don'?t|without)\b")
_SLOT_RE = re.compile(r"\b(?:" + "|".join(sorted({w for ws in SLOT_VOCAB.values() for w in ws}, key=len,
                                                   reverse=True)) + r")\b|\d")


def _negated_before_slot(folded: str) -> bool:
    """True when a negation precedes the first value word (e.g. "No PokéCenter north of here")."""
    neg = _NEG_RE.search(folded)
    if not neg:
        return False
    slot = _SLOT_RE.search(folded)
    return slot is None or neg.start() < slot.start()


def _verify_slot(ctx: DreamContext, entry: MemoryEntry, claim: Claim, cat: str) -> bool:
    old_words = set(claim.slots[cat])
    vocab_ctx = claim.value_tokens - {w for ws in claim.slots.values() for w in ws}
    vocab_ctx = frozenset(t for t in vocab_ctx if not t.replace(".", "").isdigit())
    support: Counter[str] = Counter()
    refute: Counter[str] = Counter()
    ev_support: dict[str, list[Evidence]] = defaultdict(list)
    ev_refute: list[Evidence] = []
    rivals = _rivals(ctx, entry, claim)
    for rec, prev, _pronoun in _sentence_windows(ctx, claim, rivals):
        s_pos = slots(rec.text).get(cat, set())
        s_neg = negated_slots(rec.text).get(cat, set())
        src_text = rec.text
        if not s_pos and not s_neg and prev is not None:
            s_pos = slots(prev.text).get(cat, set())
            s_neg = negated_slots(prev.text).get(cat, set())
            src_text = prev.text + " " + rec.text
        if not s_pos and not s_neg:
            continue
        if vocab_ctx and containment(vocab_ctx, frozenset(token_set(src_text))) < 0.5:
            continue
        subject_negated = _negated_before_slot(rec.folded)
        failure = bool(FAILURE_MARKERS.search(rec.text))
        confirm = (bool(CONFIRM_MARKERS.search(rec.text)) and not subject_negated) or (
            prev is not None and bool(CONFIRM_MARKERS.search(prev.text)) and not _negated_before_slot(prev.folded)
            and not FAILURE_MARKERS.search(prev.text))
        intent = bool(INTENT_MARKERS.search(rec.text)) and not confirm
        if intent and not subject_negated:
            continue  # beliefs / plans ("my notes say X", "heading north") are not evidence
        if (failure or subject_negated) and not confirm:
            for w in s_pos | s_neg:
                refute[w] += 1
            if (s_pos | s_neg) & old_words:
                ev_refute.append(Evidence.from_message(rec.msg, src_text, "refutes"))
            continue
        if intent:
            continue
        weight = 2 if confirm else 1
        for w in s_pos:
            support[w] += weight
            ev_support[w].append(Evidence.from_message(rec.msg, src_text, "supports"))
        for w in s_neg:
            refute[w] += 1
            if w in old_words:
                ev_refute.append(Evidence.from_message(rec.msg, src_text, "refutes"))

    score = {w: support[w] - 2 * refute[w] for w in set(support) | set(refute)}
    old_score = sum(score.get(w, 0) for w in old_words)
    alts = [w for w in support if w not in old_words and score[w] > 0]
    if alts:
        best = max(alts, key=lambda w: (score[w], support[w]))
        if score[best] >= 2 and score[best] > old_score:
            new_content = entry.content
            for w in old_words:
                new_content = replace_word(new_content, w, best)
            if new_content == entry.content:
                new_content = f"{_clean(entry.content).rstrip('.')} — corrected: {best}"
            evidence = (ev_refute[:4] + ev_support[best][:4])
            evidence.sort(key=lambda e: (e.source_path or "", e.line_no or 0))
            return _emit_correction(ctx, entry, new_content, evidence,
                                    f"Transcripts show '{best}', not '{'/'.join(sorted(old_words))}' "
                                    f"({support[best]} supporting, {sum(refute[w] for w in old_words)} refuting "
                                    "observation(s)).", 0.8 if refute else 0.65)
    if sum(refute[w] for w in old_words) >= 2 and old_score < 0:
        _emit_flag(ctx, entry, ev_refute[:6],
                   f"Transcripts repeatedly contradict '{'/'.join(sorted(old_words))}' but give no confirmed "
                   "alternative; review this note.")
        return True
    return False


def _orig_span(original: str, folded_sub: str) -> str:
    f = fold(original)
    i = f.find(folded_sub)
    if i >= 0 and len(f) == len(original):
        return original[i:i + len(folded_sub)]
    return folded_sub


def _verify_generic(ctx: DreamContext, entry: MemoryEntry, claim: Claim) -> bool:
    alt_score: Counter[str] = Counter()
    alt_raw: dict[str, str] = {}
    alt_ev: dict[str, list[Evidence]] = defaultdict(list)
    confirm_old = 0
    for rec, _prev, _pronoun in _sentence_windows(ctx, claim, _rivals(ctx, entry, claim)):
        c = parse_claim(rec.text)
        if c is None or c.kind != "fact" or not c.same_subject(claim) or c.polarity != claim.polarity:
            continue
        confirm = bool(CONFIRM_MARKERS.search(rec.text))
        if INTENT_MARKERS.search(rec.text) and not confirm:
            continue
        if values_conflict(c, claim):
            key = normalize(c.value)
            if not key:
                continue
            alt_score[key] += 2 if (confirm or re.search(r"\b(now|changed|moved|renamed|instead)\b", rec.folded)) \
                else 1
            alt_raw[key] = _orig_span(rec.text, c.value)
            alt_ev[key].append(Evidence.from_message(rec.msg, rec.text, "supports"))
        elif jaccard(c.value_tokens, claim.value_tokens) >= 0.5:
            confirm_old += 1
    if not alt_score:
        return False
    best = max(alt_score, key=lambda k: (alt_score[k], len(alt_ev[k])))
    if alt_score[best] < 2 or alt_score[best] <= confirm_old:
        return False
    old_raw = _orig_span(entry.content, claim.value)
    new_value = alt_raw[best].rstrip(".")
    if old_raw and old_raw in entry.content:
        new_content = entry.content.replace(old_raw, new_value, 1)
    else:
        new_content = _clean(alt_ev[best][-1].quote)
    return _emit_correction(ctx, entry, new_content, alt_ev[best][:5],
                            f"Transcripts state a different value ('{shorten(new_value, 60)}') for this subject.",
                            0.7)


def _pref_claims(ctx: DreamContext) -> list[tuple[SentenceRec, Claim]]:
    cache = getattr(ctx, "_pref_cache", None)
    if cache is not None:
        return cache
    out = []
    for rec in ctx.sentences():
        role = (rec.msg.role or "").lower()
        if role in ("tool", "system"):
            continue
        c = parse_claim(rec.text)
        if c is None or c.kind != "preference":
            continue
        if role not in ("user", "log", "unknown"):
            # assistant narration ("Use X to ...") is weak; require evaluative or explicit wording
            if c.value in ("use", "avoid") and not re.search(
                    r"\b(prefer|should|recommend|better|always|never|don'?t|avoid|let'?s)\b", rec.folded):
                continue
        out.append((rec, c))
    ctx._pref_cache = out  # type: ignore[attr-defined]
    return out


def _verify_preference(ctx: DreamContext, entry: MemoryEntry, claim: Claim) -> bool:
    stmts = [(rec, c) for rec, c in _pref_claims(ctx) if c.same_subject(claim)]
    if not stmts:
        return False
    stmts.sort(key=lambda rc: rc[0].order)
    last_rec, last = stmts[-1]
    if last.polarity == claim.polarity:
        return False
    mem_ts = parse_timestamp(_when(entry))
    msg_ts = parse_timestamp(last_rec.msg.timestamp)
    evidence = [Evidence.from_message(r.msg, r.text, "supports" if c.polarity == last.polarity else "mentions")
                for r, c in stmts[-5:]]
    if mem_ts is not None and msg_ts is not None and mem_ts > msg_ts:
        _emit_flag(ctx, entry, evidence, "Transcripts expressed the opposite preference, but this note is newer "
                                         "than that evidence; kept as is.", kind="temporal")
        return True
    current = resolved_preference_content(claim.text, last_rec.text, last)
    current += history_suffix(entry.content, _when(entry) or None, last_rec.msg.citation)
    new = entry.copy()
    new.content = current
    new.metadata = {**new.metadata, "history": [{"content": entry.content, "when": _when(entry) or None}]}
    op = Operation(
        id=Operation.make_id("contradiction", [entry.id], [current]),
        type="contradiction", kind="temporal",
        summary=f"Preference reversed in later sessions: now \"{shorten(_clean(current.split(' (History:')[0]), 80)}\"",
        reason="The note's stance was contradicted by a later transcript statement about the same subject; the "
               "latest stance becomes the current value and the earlier one is kept as history.",
        targets=[entry.id], remove=[], upsert=[new], before=[entry.content], after=[current],
        evidence=evidence, history=[f"{_when(entry) or 'memory'}: {entry.content}",
                                    f"{last_rec.msg.citation}: {last_rec.text}"],
        confidence=0.7, phase="verify",
    )
    ctx.add_op(op)
    return True


def _emit_correction(ctx: DreamContext, entry: MemoryEntry, new_content: str, evidence: list[Evidence],
                     reason: str, confidence: float) -> bool:
    # an inline date on the old note would now be misleading: move it to the evidence date
    ev_dates = sorted(e.timestamp[:10] for e in evidence if e.timestamp and re.match(r"\d{4}-\d{2}-\d{2}", e.timestamp)
                      and e.stance == "supports")
    if ev_dates and re.search(r"\b\d{4}-\d{2}-\d{2}\b", new_content):
        new_content = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", ev_dates[-1], new_content, count=1)
    new = entry.copy()
    new.content = new_content
    if ev_dates:
        new.updated_at = ev_dates[-1]
    new.metadata = {**new.metadata, "corrected_from": entry.content,
                    "evidence": [e.citation for e in evidence]}
    op = Operation(
        id=Operation.make_id("correct", [entry.id], [new_content]),
        type="correct", summary=f"Corrected from transcript evidence: \"{shorten(_clean(new_content), 90)}\"",
        reason=reason, targets=[entry.id], upsert=[new], before=[entry.content], after=[new_content],
        evidence=evidence, confidence=confidence, phase="verify",
    )
    ctx.add_op(op)
    return True


def _emit_flag(ctx: DreamContext, entry: MemoryEntry, evidence: list[Evidence], reason: str,
               kind: str = "evidence") -> None:
    op = Operation(
        id=Operation.make_id("contradiction", [entry.id], ["flag", reason]),
        type="contradiction", kind=kind, summary=f"Note contradicted by transcripts: \"{shorten(_clean(entry.content), 80)}\"",
        reason=reason, targets=[entry.id], before=[entry.content], evidence=evidence, confidence=0.5,
        phase="verify",
    )
    ctx.add_op(op)


# ============================================================== insights

_GENERIC = frozenset({"try", "tri", "still", "just", "now", "again", "back", "went", "go", "walk", "look", "see",
                      "think", "know", "want", "need", "use", "run", "get", "time", "step", "first", "next",
                      "turn", "check", "seem", "appear"})
_FAIL_STEMS = frozenset({"error", "fail", "failure", "mistake", "wrong", "oop", "forgot", "forgott", "traceback",
                         "exception", "work", "found", "find", "dead", "end", "noth", "nothing", "stuck", "loop",
                         "crash", "broke", "broken", "missing", "denied", "refus", "timed", "timeout", "wast",
                         "undefin", "such"})
_CMD_START = re.compile(r"^[a-z][a-z0-9_.\-]*$")
_KNOWN_CLI = frozenset({"git", "npm", "npx", "pnpm", "yarn", "pip", "pip3", "python", "python3", "pytest", "uv",
                        "make", "cargo", "go", "docker", "kubectl", "terraform", "poetry", "ruff", "black", "mypy",
                        "node", "deno", "bun", "just", "gradle", "mvn", "bundle", "rake", "rails", "dotnet",
                        "helm", "aws", "gcloud", "az", "tox", "nox", "pre-commit", "eslint", "tsc", "jest",
                        "vitest", "claude", "agentdream", "ls", "cd", "curl", "gh"})


def _already_known(ctx: DreamContext, text: str) -> bool:
    t = token_set(text)
    if not t:
        return True
    for e in ctx.entries:
        et = token_set(e.content)
        if jaccard(t, et) >= 0.5 or (len(t) >= 3 and containment(t, et) >= 0.8):
            return True
    return False


def _evidence_cited(ctx: DreamContext) -> set[str]:
    return {ev.citation for op in ctx.ops for ev in op.evidence}


def _new_entry(content: str, key: str, evidence: list[Evidence], extra: dict | None = None) -> MemoryEntry:
    return MemoryEntry(
        id=f"insight-{stable_hash(content, length=8)}", content=content, path=None, key=key,
        metadata={"origin": "agentdream", "evidence": [e.citation for e in evidence], **(extra or {})},
        source="agentdream",
    )


def insights(ctx: DreamContext) -> None:
    if not ctx.transcripts:
        return
    budget = [ctx.config.max_insights]
    _preference_insights(ctx, budget)
    ctx.check_cancel()
    _mistake_insights(ctx, budget)
    ctx.check_cancel()
    _workflow_insights(ctx, budget)


def _preference_insights(ctx: DreamContext, budget: list[int]) -> None:
    mem_claims = [c for e in ctx.entries for c in claims_in(e.content) if c.kind == "preference"]
    stmts = [(r, c) for r, c in _pref_claims(ctx) if not any(c.same_subject(m) for m in mem_claims)]
    clusters: list[list[tuple[SentenceRec, Claim]]] = []
    for rc in stmts:
        for cl in clusters:
            if cl[0][1].same_subject(rc[1]):
                cl.append(rc)
                break
        else:
            clusters.append([rc])
    for cl in clusters:
        if budget[0] <= 0:
            return
        cl.sort(key=lambda rc: rc[0].order)
        last_rec, last = cl[-1]
        flipped = [rc for rc in cl if rc[1].polarity != last.polarity]
        sessions = {r.session for r, _ in cl}
        if flipped:
            first_rec, _first = flipped[0]
            current = resolved_preference_content(first_rec.text, last_rec.text, last)
            current += history_suffix(first_rec.text, first_rec.msg.timestamp, last_rec.msg.citation)
            ev = [Evidence.from_message(first_rec.msg, first_rec.text, "mentions"),
                  Evidence.from_message(last_rec.msg, last_rec.text, "supports")]
            entry = _new_entry(current, "Preferences", ev, {"history": [first_rec.text, last_rec.text]})
            ctx.add_op(Operation(
                id=Operation.make_id("contradiction", [], [current]), type="contradiction", kind="temporal",
                summary=f"Preference reversed across sessions; current: \"{shorten(_clean(last_rec.text), 80)}\"",
                reason="An earlier session rejected this and a later one preferred it (or vice versa); the latest "
                       "stance is recorded as current with the earlier one as history.",
                upsert=[entry], after=[current], evidence=ev,
                history=[f"{first_rec.msg.citation}: {first_rec.text}", f"{last_rec.msg.citation}: {last_rec.text}"],
                confidence=0.6, phase="insights",
            ))
            budget[0] -= 1
        elif len(sessions) >= ctx.config.insight_min_sessions:
            rep_rec = max(cl, key=lambda rc: (len(rc[0].toks), rc[0].order))[0]
            rep = re.split(r",\s*(?:remember|ok|okay|right|please|thanks)\b", _imperative(rep_rec.text), flags=re.I)[0]
            rep = rep.rstrip(".?! ")
            if _already_known(ctx, rep):
                continue
            content = f"Preference (stated in {len(sessions)} sessions): {rep}."
            ev = [Evidence.from_message(r.msg, r.text, "supports") for r, _ in cl[:5]]
            ctx.add_op(Operation(
                id=Operation.make_id("insight", [], [content]), type="insight",
                summary=f"Repeated preference: {shorten(rep, 80)}", reason="Same stance expressed in several sessions.",
                upsert=[_new_entry(content, "Preferences", ev)], after=[content], evidence=ev,
                confidence=0.6, phase="insights",
            ))
            budget[0] -= 1


def _signature(text: str) -> frozenset[str]:
    return frozenset(t for t in token_set(text) if t not in _GENERIC and t not in _FAIL_STEMS and len(t) > 1)


def _mistake_insights(ctx: DreamContext, budget: list[int]) -> None:
    cands: list[tuple[SentenceRec, frozenset[str]]] = []
    for rec in ctx.sentences():
        if (rec.msg.role or "") == "system":
            continue
        if not FAILURE_MARKERS.search(rec.text):
            continue
        if INTENT_MARKERS.search(rec.text) and not re.search(r"\b(again|error|failed|wrong|mistake)\b", rec.folded):
            continue
        sig = _signature(rec.text)
        if len(sig) < 2:
            continue
        cands.append((rec, sig))
    clusters: list[dict] = []
    for rec, sig in cands:
        best, best_j = None, 0.0
        for cl in clusters:
            j = jaccard(sig, cl["core"])
            shared = len(sig & cl["core"])
            if (j >= 0.4 or shared >= 3) and j >= best_j:
                best, best_j = cl, j
        if best is None:
            clusters.append({"core": sig, "members": [(rec, sig)], "counts": Counter(sig)})
        else:
            best["members"].append((rec, sig))
            best["counts"].update(sig)
    scored = []
    for cl in clusters:
        sessions = {r.session for r, _ in cl["members"]}
        n = len(cl["members"])
        if len(sessions) >= ctx.config.insight_min_sessions or n >= 3:
            scored.append((len(sessions), n, cl))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    for n_sessions, n, cl in scored:
        if budget[0] <= 0:
            return
        counts = cl["counts"]
        rep_rec, _ = max(cl["members"], key=lambda m: (sum(counts[t] for t in m[1]) / (len(m[1]) ** 0.5),
                                                       -len(m[0].text)))
        rep = _clean(rep_rec.text).rstrip(".")
        sess_txt = f"across {n_sessions} sessions" if n_sessions > 1 else "within one session"
        content = f"Recurring mistake ({n} occurrences {sess_txt}): \"{shorten(rep, 140)}\" — check before repeating."
        if _already_known(ctx, rep):
            continue
        ev = [Evidence.from_message(r.msg, r.text, "supports") for r, _ in cl["members"][:6]]
        ctx.add_op(Operation(
            id=Operation.make_id("insight", [], [content]), type="insight",
            summary=f"Recurring mistake {sess_txt}: {shorten(rep, 70)}",
            reason=f"Failure-marked statements with overlapping content ({', '.join(sorted(counts, key=lambda t: -counts[t])[:4])}) recur.",
            upsert=[_new_entry(content, "Recurring mistakes", ev)], after=[content], evidence=ev,
            confidence=0.6 if n_sessions > 1 else 0.45, phase="insights",
        ))
        budget[0] -= 1


def _normalize_cmd(cmd: str) -> str | None:
    cmd = cmd.strip().lstrip("$ ").strip()
    parts = cmd.split()
    if not parts or not _CMD_START.match(parts[0]):
        return None
    if parts[0] not in _KNOWN_CLI and len(parts) < 2:
        return None
    if parts[0] not in _KNOWN_CLI and not re.match(r"^[a-z][a-z0-9_\-]{1,20}$", parts[0]):
        return None
    out = []
    for p in parts[:5]:
        if p[0] in "'\"" or "/" in p or "=" in p or p in ("&&", "|", ";", ">", "2>&1"):
            break
        out.append(p)
    if not out:
        return None
    if out[0] not in _KNOWN_CLI and len(out) < 2:
        return None
    return " ".join(out)


def _commands_in(msg) -> list[str]:
    text = msg.content
    cmds = []
    m = re.match(r"^\[tool\]\s+(?:Bash|bash|shell|terminal|run_command|exec)\s*:\s*(.+)$", text, re.S)
    if m:
        for part in re.split(r"\s*(?:&&|;|\|\|)\s*", m.group(1).splitlines()[0]):
            c = _normalize_cmd(part)
            if c:
                cmds.append(c)
        return cmds
    for span in re.findall(r"`([^`\n]{2,120})`", text):
        c = _normalize_cmd(span)
        if c and c.split()[0] in _KNOWN_CLI:
            cmds.append(c)
    for line in text.splitlines():
        if line.strip().startswith("$ "):
            c = _normalize_cmd(line.strip()[2:])
            if c:
                cmds.append(c)
    return cmds


def _workflow_insights(ctx: DreamContext, budget: list[int]) -> None:
    per_session: list[list[tuple[str, object]]] = []
    for t in ctx.transcripts:
        seq: list[tuple[str, object]] = []
        seen: set[str] = set()
        for m in t.messages:
            for c in _commands_in(m):
                if c not in seen:
                    seen.add(c)
                    seq.append((c, m))
        per_session.append(seq)
    pair_sessions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for si, seq in enumerate(per_session):
        names = [c for c, _ in seq]
        seen_pairs = set()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pr = (names[i], names[j])
                if pr not in seen_pairs:
                    seen_pairs.add(pr)
                    pair_sessions[pr].append(si)
    need = max(2, ctx.config.insight_min_sessions)
    good = {pr: ss for pr, ss in pair_sessions.items() if len(ss) >= need}
    if not good:
        return
    # chain building: greedily extend the most frequent pair
    used: set[tuple[str, str]] = set()
    emitted = 0
    for pr, ss in sorted(good.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if budget[0] <= 0 or emitted >= 3:
            return
        if pr in used:
            continue
        chain = list(pr)
        sess = set(ss)
        extended = True
        while extended:
            extended = False
            for (a, b), s2 in good.items():
                if a == chain[-1] and b not in chain and sess <= set(s2) | set():
                    if set(s2) >= sess:
                        chain.append(b)
                        extended = True
                        break
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                used.add((chain[i], chain[j]))
        flow = " → ".join(f"`{c}`" for c in chain)
        content = f"Converged workflow (used in {len(sess)} sessions): {flow}."
        if _already_known(ctx, " ".join(chain)):
            continue
        ev = []
        for si in sorted(sess)[:4]:
            for c, m in per_session[si]:
                if c == chain[0]:
                    ev.append(Evidence.from_message(m, m.content, "supports"))
                    break
        ctx.add_op(Operation(
            id=Operation.make_id("insight", [], [content]), type="insight",
            summary=f"Converged workflow: {flow}", reason="The same command sequence recurs across sessions.",
            upsert=[_new_entry(content, "Workflows", ev)], after=[content], evidence=ev, confidence=0.55,
            phase="insights",
        ))
        budget[0] -= 1
        emitted += 1
