"""Local infini-gram: next-token statistics after the longest context match in a corpus.

Index = suffix array of the *reversed* token corpus (tokens < 65536, encoded as
big-endian uint16 byte pairs for divsufsort; only even byte offsets kept, so
order == token-lexicographic order). Querying the reversed context
[x_t, x_{t-1}, ...] extends the match one token at a time by narrowing an SA
range, and all query positions are processed in parallel with a vectorized
binary search.

The token that *follows* the matched context in the original corpus is the
token *preceding* the suffix in the reversed corpus: rev[sa - 1].
"""

from __future__ import annotations

import numpy as np


def build_index(tokens: np.ndarray):
    rev = np.ascontiguousarray(tokens[::-1].astype(np.uint16))
    import pydivsufsort

    by = rev.astype(">u2").view(np.uint8).copy()
    sa = pydivsufsort.divsufsort(by)
    sa = sa[(sa & 1) == 0] // 2
    return rev, sa.astype(np.int64)


class InfiniGram:
    def __init__(self, rev: np.ndarray, sa: np.ndarray, vocab: int):
        self.rev, self.sa, self.V = rev.astype(np.int64), sa, vocab
        self.n = len(rev)
        self.uni = np.bincount(rev, minlength=vocab).astype(np.float64)
        self.uni /= self.uni.sum()

    def _col(self, idx, k):
        """token at offset k of suffix sa[idx]; -1 past the end."""
        p = self.sa[idx] + k
        out = np.full(p.shape, -1, dtype=np.int64)
        ok = p < self.n
        out[ok] = self.rev[p[ok]]
        return out

    def _bound(self, lo, hi, k, q, upper):
        """vectorized lower/upper bound of q in column k over [lo, hi)."""
        l, r = lo.copy(), hi.copy()
        while True:
            act = l < r
            if not act.any():
                return l
            mid = (l + r) // 2
            v = np.where(act, self._col(np.minimum(mid, self.n - 1), k), 0)
            go_right = (v <= q) if upper else (v < q)
            l = np.where(act & go_right, mid + 1, l)
            r = np.where(act & ~go_right, mid, r)

    def longest_match(self, ctx: np.ndarray, max_len: int = 64):
        """ctx (N, L): each row is a reversed context, ctx[:, 0] = current token,
        -1 padded. Returns (match_len, lo, hi) of the longest suffix with count>=1
        whose continuation exists (sa > 0)."""
        N = ctx.shape[0]
        lo = np.zeros(N, dtype=np.int64)
        hi = np.full(N, len(self.sa), dtype=np.int64)
        best = (np.zeros(N, dtype=np.int64), lo.copy(), hi.copy())
        for k in range(min(max_len, ctx.shape[1])):
            q = ctx[:, k]
            alive = (q >= 0) & (hi > lo)
            if not alive.any():
                break
            nlo = self._bound(lo, hi, k, q, upper=False)
            nhi = self._bound(lo, hi, k, q, upper=True)
            ok = alive & (nhi > nlo)
            lo = np.where(ok, nlo, lo)
            hi = np.where(ok, nhi, lo)  # dead rows -> empty range
            best = (np.where(ok, k + 1, best[0]), np.where(ok, lo, best[1]), np.where(ok, hi, best[2]))
        return best

    def next_counts(self, lo, hi, cap: int = 2000):
        """Sparse next-token counts for each range (sampling up to cap occurrences)."""
        out = []
        for a, b in zip(lo.tolist(), hi.tolist()):
            if b <= a:
                out.append((np.zeros(0, dtype=np.int64), np.zeros(0)))
                continue
            idx = np.arange(a, b) if b - a <= cap else np.linspace(a, b - 1, cap).astype(np.int64)
            s = self.sa[idx]
            s = s[s > 0]
            nxt = self.rev[s - 1]
            u, c = np.unique(nxt, return_counts=True)
            scale = (b - a) / max(len(idx), 1)
            out.append((u, c * scale))
        return out
