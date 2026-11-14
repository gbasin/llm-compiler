"""Generic integer op library for the register file.

Each op: fn(model, *slots, **kw) -> LongTensor (B, T), causal (position t only
uses positions <= t). Ops are *generic primitives*; task-specific glue lives in
the network (readers/routers) or in teacher-written lookup tables (`map`).

Domain functions give the output slot's value range (for reader tables).
"""

from __future__ import annotations

import torch

OPS = {}


def op(domain):
    def deco(fn):
        OPS[fn.__name__] = (fn, domain)
        return fn
    return deco


V_ = lambda m, s: m.V
BOOL = lambda m, s: 2


def get(name):
    return OPS[name]


@op(V_)
def xval(m, tok):
    """Item value: tok if it is an item token (< V) else 0."""
    return torch.where(tok < m.V, tok, torch.zeros_like(tok))


@op(lambda m, s: m.domains["tok"])
def task(m, tok):
    """Task token broadcast to every position (position 0 holds it)."""
    return tok[:, :1].expand_as(tok)


@op(lambda m, s: m.domains[s.inputs[0]])
def shift(m, a, k: int = 1):
    out = torch.zeros_like(a)
    out[:, k:] = a[:, :-k]
    return out


@op(V_)
def add_mod(m, a, b):
    return (a + b) % m.V


@op(V_)
def mul_mod(m, a, b):
    return (a * b) % m.V


@op(V_)
def cumsum_mod(m, a):
    return torch.cumsum(a, 1) % m.V


@op(lambda m, s: m.domains[s.inputs[0]])
def cummax(m, a):
    return torch.cummax(a, 1).values


@op(BOOL)
def eq(m, a, b):
    return (a == b).long()


@op(lambda m, s: m.domains[s.inputs[0]])
def mask(m, a, keep):
    return torch.where(keep.bool(), a, torch.zeros_like(a))


@op(lambda m, s: max(s.kwargs["table"]) + 1)
def map(m, a, table: list[int]):
    """Teacher-written lookup table: out = table[a]."""
    t = torch.tensor(table, device=a.device)
    return t[a.clamp(0, len(table) - 1)]


@op(V_)
def count_prefix_eq(m, a):
    """Number of positions j <= t with a_j == a_t, mod V."""
    oh = torch.nn.functional.one_hot(a, int(a.max()) + 1).cumsum(1)
    return oh.gather(2, a[..., None])[..., 0] % m.V


@op(lambda m, s: m.domains[s.inputs[1]])
def prev_occ_next(m, key, val):
    """val at (j+1) for the latest j < t with key_j == key_t, else 0 (induction)."""
    B, T = key.shape
    out = torch.zeros_like(val)
    idx = torch.arange(T, device=key.device)
    same = (key[:, :, None] == key[:, None, :]) & (idx[None, None, :] < idx[None, :, None])  # (B,t,j)
    j = torch.where(same, idx[None, None, :], -1).max(-1).values  # (B,T)
    nxt = val.gather(1, (j + 1).clamp(max=T - 1))
    return torch.where(j >= 0, nxt, out)


@op(BOOL)
def isitem(m, tok):
    """1 at item positions (tok < V), else 0."""
    return (tok < m.V).long()
