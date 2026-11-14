"""Tiny pre-LN GPT with addressable residual *sites* for code-module injection.

Site i (0 <= i <= n_layers) is the residual stream entering block i; site
n_layers is the stream entering the final LayerNorm. An injected module reads
the residual at `read` and adds its output to the residual at `write`
(write >= read). Reads see the stream *before* any writes at the same site.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    max_len: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    mlp_mult: int = 4
    pos: str = "none"  # "none" (NoPE) | "learned"


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_heads = c.n_heads
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(c.d_model, c.mlp_mult * c.d_model),
            nn.GELU(),
            nn.Linear(c.mlp_mult * c.d_model, c.d_model),
        )

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.max_len, c.d_model) if c.pos == "learned" else None
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.injections = nn.ModuleList()

    @property
    def n_sites(self):
        return self.cfg.n_layers + 1

    def inject(self, module):
        assert 0 <= module.read <= module.write <= self.cfg.n_layers
        self.injections.append(module)
        return module

    def neural_params(self):
        """Parameter count excluding injected modules (the 'core')."""
        inj = {id(p) for p in self.injections.parameters()}
        return sum(p.numel() for p in self.parameters() if id(p) not in inj)

    def forward(self, idx, capture: bool = False):
        """Returns logits, and if capture, list of residual tensors per site
        (with retain_grad so phantom gradients can be read after backward)."""
        T = idx.shape[1]
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        pending: dict[int, list[torch.Tensor]] = {}
        resid = []
        L = self.cfg.n_layers
        for i in range(L + 1):
            for m in self.injections:
                if m.read == i and m.enabled:
                    pending.setdefault(m.write, []).append(m(x))
            for c in pending.pop(i, []):
                x = x + c
            if capture:
                if x.requires_grad:
                    x.retain_grad()
                resid.append(x)
            if i < L:
                x = self.blocks[i](x)
        logits = self.head(self.ln_f(x))
        return (logits, resid) if capture else logits
