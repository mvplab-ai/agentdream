"""Candidate generation ("blocking") so similarity checks scale to large stores.

Comparing every pair of N entries is O(N^2); on a 5k-entry store that is 12.5M
comparisons of non-trivial heuristics. Instead we only compare pairs that *can*
pass the similarity test, using exact filters (no recall is lost):

* :func:`similar_pairs` - **prefix filtering** for Jaccard thresholds. Order
  tokens globally by rarity; if ``J(x, y) >= t`` then ``x`` and ``y`` share at
  least one token among the first ``|x| - ceil(t*|x|) + 1`` tokens of each
  (Chaudhuri et al., 2006; Bayardo et al., 2007). Rare tokens make the posting
  lists short, so the candidate set is close to the true result set.
* :class:`TokenIndex` - a plain inverted index (token -> ascending item ids)
  with helpers for "contains all of these tokens" and "contains at least k of
  these tokens" queries.

Everything here is deterministic: results are sorted and independent of hash
seeds.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence


def _prefix_len(n: int, threshold: float) -> int:
    # tiny epsilon so 0.5 * 4 == 2.0 doesn't become 3 through float noise
    return n - math.ceil(threshold * n - 1e-9) + 1


def similar_pairs(sets: Sequence[frozenset[str] | set[str]], threshold: float) -> set[tuple[int, int]]:
    """All index pairs ``(i, j)``, ``i < j``, that *may* have ``jaccard(sets[i], sets[j]) >= threshold``.

    Superset of the true pairs (callers verify each candidate), never misses one. Empty sets never pair.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    df: Counter[str] = Counter()
    for s in sets:
        df.update(s)
    postings: dict[str, list[int]] = defaultdict(list)
    out: set[tuple[int, int]] = set()
    for i, s in enumerate(sets):
        if not s:
            continue
        ordered = sorted(s, key=lambda t: (df[t], t))
        n = len(ordered)
        for tok in ordered[:_prefix_len(n, threshold)]:
            plist = postings[tok]
            for j in plist:
                m = len(sets[j])
                # size filter: jaccard <= min/max, so very different sizes can't reach the threshold
                if min(m, n) >= threshold * max(m, n) - 1e-9:
                    out.add((j, i))
            plist.append(i)
    return out


class TokenIndex:
    """Inverted index over token sets: token -> ascending list of item positions."""

    def __init__(self, sets: Iterable[Iterable[str]] = ()):
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.size = 0
        for s in sets:
            self.add(s)

    def add(self, tokens: Iterable[str]) -> int:
        pos = self.size
        for t in set(tokens):
            self.postings[t].append(pos)
        self.size += 1
        return pos

    def df(self, token: str) -> int:
        return len(self.postings.get(token, ()))

    def with_any(self, tokens: Iterable[str]) -> set[int]:
        out: set[int] = set()
        for t in tokens:
            out.update(self.postings.get(t, ()))
        return out

    def with_all(self, tokens: Iterable[str]) -> set[int]:
        """Positions whose token set contains every token in ``tokens`` (empty query -> empty result)."""
        toks = sorted(set(tokens), key=self.df)
        if not toks:
            return set()
        result = set(self.postings.get(toks[0], ()))
        for t in toks[1:]:
            if not result:
                break
            result.intersection_update(self.postings.get(t, ()))
        return result

    def overlap_counts(self, tokens: Iterable[str]) -> Counter[int]:
        """position -> number of query tokens it contains (only positions with >= 1)."""
        c: Counter[int] = Counter()
        for t in set(tokens):
            c.update(self.postings.get(t, ()))
        return c


def cluster_order(n: int, pairs: Iterable[tuple[int, int]]) -> list[int]:
    """A permutation of ``range(n)`` that keeps connected items (by ``pairs``) adjacent.

    Components appear in order of their smallest member; members keep their relative order.
    Used to put likely duplicates into the same LLM chunk."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [i for root in sorted(groups) for i in groups[root]]


__all__ = ["TokenIndex", "cluster_order", "similar_pairs"]
