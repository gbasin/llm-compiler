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


def _suffix_match(tok, cap: int = 15):
    """For each t: the j < t whose preceding suffix best matches the suffix
    ending at t (longest, latest on ties). Returns (next token after j, length)."""
    B, T = tok.shape
    M = torch.zeros(B, T, dtype=torch.long, device=tok.device)  # match len for previous t
    best_next = torch.zeros_like(tok)
    best_len = torch.zeros_like(tok)
    jidx = torch.arange(T, device=tok.device)
    for t in range(T):
        eq = tok[:, t:t + 1] == tok  # (B, T) over j
        prevM = torch.zeros_like(M)
        prevM[:, 1:] = M[:, :-1]  # M_{t-1}[j-1]
        Mt = torch.where(eq, (prevM + 1).clamp(max=cap), torch.zeros_like(M))
        Mt = torch.where(jidx[None] < t, Mt, torch.zeros_like(Mt))
        score = Mt * T + jidx[None]  # longest, then latest
        j = score.argmax(1)
        L = Mt.gather(1, j[:, None])[:, 0]
        nxt = tok.gather(1, (j + 1).clamp(max=T - 1)[:, None])[:, 0]
        best_next[:, t] = torch.where(L > 0, nxt, torch.zeros_like(nxt))
        best_len[:, t] = L
        M = Mt
    return best_next, best_len


def _suffix_cached(m, tok):
    c = getattr(m, "_suffix_cache", None)
    if c is None or c[0] is not tok:
        c = (tok, _suffix_match(tok))
        m._suffix_cache = c
    return c[1]


@op(lambda m, s: m.domains[s.inputs[0]])
def suffix_next(m, tok):
    """Continuation of the longest earlier match of the current suffix (generic copy/induction)."""
    return _suffix_cached(m, tok)[0]


@op(lambda m, s: 16)
def suffix_len(m, tok):
    """Length (capped at 15) of that longest earlier suffix match."""
    return _suffix_cached(m, tok)[1]


@op(lambda m, s: 17)
def since(m, tok, value: int, cap: int = 15):
    """Positions since the last occurrence of token `value` (0 = here), cap+1 if none/far."""
    B, T = tok.shape
    idx = torch.arange(T, device=tok.device).expand(B, T)
    last = torch.where(tok == value, idx, torch.full_like(idx, -10_000))
    last = torch.cummax(last, 1).values
    return (idx - last).clamp(max=cap + 1)


# ---------------------------------------------------------------- text / arithmetic ops
# These read characters through m.itos (set by the trainer for char-level data).
NUMWORDS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
     "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"])}
NUMWORDS["dozen"] = 12
BIG = lambda m, s: 4096


@op(BIG)
def parse_num(m, tok):
    """Causal: value of the digit run, or number word, that ends at position t (uses chars
    <= t only), else -1. 'seven' inside 'seventeen' marks 7 at 'n' then 17 at the end."""
    B, T = tok.shape
    out = torch.full_like(tok, -1)
    txt = ["".join(m.itos[i] if i < len(m.itos) else "#" for i in row) for row in tok.tolist()]
    for b, s in enumerate(txt):
        run_start = 0
        for t in range(T):
            c = s[t]
            if t == 0 or (c.isdigit() != s[t - 1].isdigit()) or (c.isalpha() != s[t - 1].isalpha()):
                run_start = t
            if c.isdigit() and t - run_start < 3:
                out[b, t] = int(s[run_start:t + 1])
            elif c.isalpha():
                w = s[run_start:t + 1].lower()
                if w in NUMWORDS:
                    out[b, t] = NUMWORDS[w]
    return out


@op(BIG)
def hold(m, val, role, r: int):
    """Latest val (>=0) at a position where role == r, carried forward; -1 before any."""
    B, T = val.shape
    idx = torch.arange(T, device=val.device).expand(B, T)
    hit = (role == r) & (val >= 0)
    last = torch.cummax(torch.where(hit, idx, torch.full_like(idx, -1)), 1).values
    return torch.where(last >= 0, val.gather(1, last.clamp(min=0)), torch.full_like(val, -1))


@op(BIG)
def arith(m, a, b, opc):
    """opc: 0 add, 1 sub, 2 mul. -1 if any input missing or result out of range."""
    r = torch.where(opc == 0, a + b, torch.where(opc == 1, a - b, a * b))
    bad = (a < 0) | (b < 0) | (opc < 0) | (opc > 2) | (r < 0) | (r >= 4096)
    return torch.where(bad, torch.full_like(r, -1), r)


@op(lambda m, s: len(m.itos) + 1)
def render(m, val, pos):
    """Char id of the next char to emit when writing str(val) followed by newline, given
    pos = number of answer chars already emitted. -1 if val invalid."""
    B, T = val.shape
    out = torch.full_like(val, -1)
    stoi = {c: i for i, c in enumerate(m.itos)}
    for b in range(B):
        for t in range(T):
            v, p = int(val[b, t]), int(pos[b, t])
            if v < 0 or p < 0:
                continue
            s = str(v) + "\n"
            if p < len(s):
                out[b, t] = stoi[s[p]]
    return out


@op(lambda m, s: 64)
def since_char(m, tok, ch: str, cap: int = 62):
    """Positions since the last occurrence of character `ch` (0 = here); cap+1 if none."""
    c = m.itos.index(ch)
    B, T = tok.shape
    idx = torch.arange(T, device=tok.device).expand(B, T)
    last = torch.cummax(torch.where(tok == c, idx, torch.full_like(idx, -10_000)), 1).values
    return (idx - last).clamp(max=cap + 1)


@op(lambda m, s: m.domains[s.inputs[0]])
def offset(m, a, k: int):
    """a + k (keeps -1 as missing when a < 0)."""
    return torch.where(a >= 0, a + k, a)
