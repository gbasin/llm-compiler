"""The callable-expert API that discovery agents write against (Pile / BPE track).

An expert is a plain function of a query context. For each queried position p it returns M items: a token
id (what the expert proposes is relevant to predicting the token after p, -1 = no item) plus F small integer
features per item (each in [0, size_f)). The core reads the items with attention between loops.

    from llmc.xapi import expert, table

    @table
    def bigram(corpus, vocab):            # built once from the index corpus, cached; arrays only
        ...
        return {"next": arr}

    @expert(items=2, feats=(16,))
    def my_expert(ctx):
        t0 = ctx.back(0)                  # current token at each position (N,)
        ...
        return tokens (N, 2) int64, feats (N, 2, 1) int64

Query context `ctx` (all code + data, nothing learned):
  ctx.n               number of queried positions N
  ctx.back(j)         token j steps back from each position (j=0: the current token), -1 before the text start
  ctx.window(L)       (N, L) matrix of back(0..L-1)
  ctx.vocab           list of 50304 token strings (Pythia BPE; leading space = word start)
  ctx.index           suffix array over Pile tokens 0..30M (see Index)
  ctx.menu            the built-in call-1 answers: dict(cont (N,7,4), cnt (N,7)) for k in (1,2,3,4,5,6,8)
  ctx.tables[name]    dict of arrays returned by your @table function `name`
  ctx.rng             numpy Generator
Experts must be causal: output for position p may depend only on tokens at positions <= p.
"""
from __future__ import annotations

import numpy as np

EXPERTS: dict = {}
TABLES: dict = {}


def expert(items: int, feats: tuple = ()):
    def deco(f):
        EXPERTS[f.__name__] = dict(fn=f, items=int(items), feats=tuple(int(x) for x in feats))
        return f
    return deco


def table(f):
    TABLES[f.__name__] = f
    return f


class Index:
    """Suffix array over the REVERSED index corpus (Pile tokens 0..30M). A match range [lo, hi) is a set of
    corpus occurrences of a token string, built from its LAST token backwards:
        lo, hi = ix.root(N)
        lo, hi = ix.narrow(lo, hi, 0, ctx.back(0))   # occurrences of the current token
        lo, hi = ix.narrow(lo, hi, 1, ctx.back(1))   # ... preceded by the previous token
    narrow(lo, hi, col, q): keep occurrences whose token `col` steps back from the string's end equals q
    (q < 0 or empty range -> empty). col must equal the number of tokens matched so far.
    You may put any token at col 0 (e.g. a guessed next token y) and then the context at cols 1.. to ask
    "how often does y follow this context".
    count(lo, hi) -> occurrences.  follow(lo, hi, j) -> the token j+1 steps after the matched string's end
    in one sampled occurrence (-1 if none).  follow_top(lo, hi, j, k, samples) -> (tokens (N,k), counts (N,k))
    most common such tokens among `samples` sampled occurrences."""

    def __init__(self, ig, rng):
        self.ig, self.rng, self.size = ig, rng, len(ig.sa)

    def root(self, n):
        return np.zeros(n, dtype=np.int64), np.full(n, len(self.ig.sa), dtype=np.int64)

    def narrow(self, lo, hi, col, q):
        q = np.asarray(q, dtype=np.int64)
        alive = (q >= 0) & (hi > lo)
        if not alive.any():
            return lo, lo.copy()
        qq = np.where(alive, q, 0)
        nlo = self.ig._bound(lo, hi, col, qq, upper=False)
        nhi = self.ig._bound(lo, hi, col, qq, upper=True)
        return np.where(alive, nlo, lo), np.where(alive, nhi, lo)

    def count(self, lo, hi):
        return hi - lo

    def follow(self, lo, hi, j=0):
        n = hi - lo
        pick = lo + (self.rng.random(len(lo)) * np.maximum(n, 1)).astype(np.int64)
        s = self.ig.sa[np.minimum(pick, len(self.ig.sa) - 1)]
        ok = (n > 0) & (s - 1 - j >= 0)
        return np.where(ok, self.ig.rev[np.maximum(s - 1 - j, 0)], -1)

    def follow_top(self, lo, hi, j=0, k=4, samples=32):
        N = len(lo)
        S = np.stack([self.follow(lo, hi, j) for _ in range(samples)], 1)  # (N, samples)
        toks = np.full((N, k), -1, dtype=np.int64)
        cnts = np.zeros((N, k), dtype=np.int64)
        Ss = np.sort(S, 1)
        for i in range(N):
            u, c = np.unique(Ss[i][Ss[i] >= 0], return_counts=True)
            o = np.argsort(-c)[:k]
            toks[i, :len(o)], cnts[i, :len(o)] = u[o], c[o]
        return toks, cnts


class Ctx:
    def __init__(self, stream, pos, vocab, index, menu, tables, rng):
        self.stream, self.pos, self.vocab, self.index, self.menu, self.tables, self.rng = \
            stream, pos, vocab, index, menu, tables, rng
        self.n = len(pos)
        self._cache = {}

    def back(self, j):
        j = int(j)
        if j < 0:
            raise ValueError("ctx.back(j) needs j >= 0 (experts are causal)")
        if j not in self._cache:
            p = self.pos - j
            self._cache[j] = np.where(p >= 0, np.asarray(self.stream[np.maximum(p, 0)]).astype(np.int64), -1)
        return self._cache[j]

    def window(self, L):
        return np.stack([self.back(j) for j in range(L)], 1)
