"""Hybrid model: a growable tiny transformer core + a shared register file.

Register file: per-position integer *slots* (B, T) that code and network share.
  - slot "tok" holds the raw input tokens.
  - Op:      code, slots -> new slot (pure, causal, integer). Runs as soon as its
             inputs are available (at the latest input's site).
  - Decoder: network -> slot. LN + linear head at a site; value = argmax. Trained
             only by its contract (CE to teacher labels), never through code.
  - Reader:  slot -> network. Embedding table (a lookup table, zero-init) added at
             a site through a per-position router  sigmoid(r . LN(h)).  Zero table
             => function-preserving at insertion, and the table gets gradient
             immediately (no zero-gate saddle).

Core blocks keep attention heads (fixed head_dim) and MLP neurons as separately
growable parameters (see growth.py). Sites: 0..n_layers, as in model.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import oplib


def _p(*shape, std=None, zero=False):
    t = torch.zeros(*shape)
    if not zero:
        fan_in = shape[-1] if len(shape) > 1 else 1
        nn.init.normal_(t, std=std or 1 / math.sqrt(fan_in))
    return nn.Parameter(t)


class GBlock(nn.Module):
    """Pre-LN block with growable heads (H x head_dim) and MLP width (n_mlp)."""

    def __init__(self, d: int, n_heads: int, head_dim: int, n_mlp: int, identity: bool = False):
        super().__init__()
        self.d, self.hd = d, head_dim
        A = n_heads * head_dim
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wq, self.Wk, self.Wv = _p(A, d), _p(A, d), _p(A, d)
        self.Wo = _p(d, A, zero=identity) if A else _p(d, 0, zero=True)
        self.W1, self.b1 = _p(n_mlp, d), _p(n_mlp, zero=True)
        self.W2 = _p(d, n_mlp, zero=identity) if n_mlp else _p(d, 0, zero=True)
        self.b2 = _p(d, zero=True)

    @property
    def n_heads(self):
        return self.Wq.shape[0] // self.hd

    @property
    def n_mlp(self):
        return self.W1.shape[0]

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        if H:
            h = self.ln1(x)
            sh = lambda W: (h @ W.T).view(B, T, H, self.hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(sh(self.Wq), sh(self.Wk), sh(self.Wv), is_causal=True)
            x = x + a.transpose(1, 2).reshape(B, T, H * self.hd) @ self.Wo.T
        if self.n_mlp:
            act = F.gelu(self.ln2(x) @ self.W1.T + self.b1)
            self.last_act = act.detach()
            x = x + act @ self.W2.T
        return x + self.b2


@dataclass
class OpSpec:
    name: str  # oplib function name
    inputs: list[str]
    out: str
    kwargs: dict = field(default_factory=dict)


class Reader(nn.Module):
    def __init__(self, slot: str, domain: int, site: int, d: int):
        super().__init__()
        self.slot, self.domain, self.site = slot, domain, site
        self.table = nn.Parameter(torch.zeros(domain, d))
        self.ln = nn.LayerNorm(d)
        self.router = nn.Linear(d, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.enabled = True

    def gate(self, h):
        return torch.sigmoid(self.router(self.ln(h)))

    def forward(self, h, val):
        return self.gate(h) * self.table[val.clamp(0, self.domain - 1)]


class Decoder(nn.Module):
    def __init__(self, slot: str, domain: int, site: int, d: int, labels: str):
        super().__init__()
        self.slot, self.domain, self.site, self.labels = slot, domain, site, labels
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, domain)


class Hybrid(nn.Module):
    def __init__(self, vocab: int, d: int, blocks: list[tuple[int, int]], head_dim: int, V: int):
        """blocks: list of (n_heads, n_mlp)."""
        super().__init__()
        self.V, self.d, self.hd = V, d, head_dim
        self.tok_emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(GBlock(d, h, head_dim, m) for h, m in blocks)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.ops: list[OpSpec] = []
        self.domains: dict[str, int] = {"tok": vocab}
        self.readers = nn.ModuleList()
        self.decoders = nn.ModuleList()

    # ------------------------------------------------------------ structure
    @property
    def n_layers(self):
        return len(self.blocks)

    def arch(self):
        return [(b.n_heads, b.n_mlp) for b in self.blocks]

    def add_op(self, name, inputs, out, **kw):
        fn, dom = oplib.get(name)
        for i in inputs:
            assert i in self.domains, f"unknown slot {i}"
        spec = OpSpec(name, list(inputs), out, kw)
        self.ops.append(spec)
        self.domains[out] = dom(self, spec)
        return spec

    def add_reader(self, slot, site):
        r = Reader(slot, self.domains[slot], site, self.d).to(self.tok_emb.weight.device)
        self.readers.append(r)
        return r

    def add_decoder(self, slot, domain, site, labels):
        dec = Decoder(slot, domain, site, self.d, labels).to(self.tok_emb.weight.device)
        self.decoders.append(dec)
        self.domains[slot] = domain
        return dec

    def param_counts(self):
        core = sum(p.numel() for n, p in self.named_parameters()
                   if not n.startswith(("readers", "decoders")))
        iface = sum(p.numel() for n, p in self.named_parameters() if n.startswith(("readers", "decoders")))
        code = dict(ops=len(self.ops),
                    table_entries=sum(len(o.kwargs.get("table", [])) for o in self.ops))
        return dict(core=core, interface=iface, **code)

    # ------------------------------------------------------------ forward
    def _run_ops(self, regs, ready_site, site):
        changed = True
        while changed:
            changed = False
            for op in self.ops:
                if op.out in regs or not all(i in regs for i in op.inputs):
                    continue
                fn, _ = oplib.get(op.name)
                regs[op.out] = fn(self, *[regs[i] for i in op.inputs], **op.kwargs)
                ready_site[op.out] = site
                changed = True

    def forward(self, tokens, capture=False):
        x = self.tok_emb(tokens)
        regs = {"tok": tokens}
        ready = {"tok": 0}
        dec_logits = {}
        resid = []
        for i in range(self.n_layers + 1):
            for dec in self.decoders:
                if dec.site == i:
                    lg = dec.head(dec.ln(x))
                    dec_logits[dec.slot] = lg
                    regs[dec.slot] = lg.argmax(-1)
                    ready[dec.slot] = i
            self._run_ops(regs, ready, i)
            h = x
            for r in self.readers:
                if r.site == i and r.enabled:
                    assert r.slot in regs, f"slot {r.slot} not available at site {i}"
                    x = x + r(h, regs[r.slot])
            if capture:
                if x.requires_grad:
                    x.retain_grad()
                resid.append(x)
            if i < self.n_layers:
                x = self.blocks[i](x)
        logits = self.head(self.ln_f(x))
        self.last = dict(regs=regs, dec_logits=dec_logits, resid=resid)
        return logits
