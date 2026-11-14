"""Function-preserving growth of the hybrid core, with Adam state carried over.

  add_mlp(model, opt, layer, k):   k new MLP neurons (fresh in-weights, zero out-weights)
  add_heads(model, opt, layer, k): k new heads (fresh q/k/v, zero output projection)
  add_layer(model, opt, at, heads, mlp): new identity block (zero output projections)

Growth replaces Parameters with larger ones; `_swap` copies the old data and the
optimizer state (exp_avg, exp_avg_sq; step) into the matching slice and zeros
the new part.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .hybrid import GBlock


def _swap(opt, owner: nn.Module, name: str, new: torch.Tensor, region: tuple):
    old = getattr(owner, name)
    newp = nn.Parameter(new)
    with torch.no_grad():
        newp[region] = old.data
    if opt is not None:
        st = opt.state.pop(old, None)
        for g in opt.param_groups:
            g["params"] = [newp if p is old else p for p in g["params"]]
        if st:
            nst = {}
            for k, v in st.items():
                if torch.is_tensor(v) and v.shape == old.shape:
                    z = torch.zeros_like(new)
                    z[region] = v
                    nst[k] = z
                else:
                    nst[k] = v
            opt.state[newp] = nst
    setattr(owner, name, newp)


def _fresh(rows, cols):
    t = torch.empty(rows, cols)
    nn.init.normal_(t, std=1 / math.sqrt(cols))
    return t


@torch.no_grad()
def add_mlp(model, opt, layer: int, k: int):
    b = model.blocks[layer]
    n, d = b.W1.shape
    dev = b.W1.device
    _swap(opt, b, "W1", torch.cat([b.W1.data, _fresh(k, d).to(dev)]), (slice(0, n),))
    _swap(opt, b, "b1", torch.zeros(n + k, device=dev), (slice(0, n),))
    _swap(opt, b, "W2", torch.zeros(d, n + k, device=dev), (slice(None), slice(0, n)))


@torch.no_grad()
def add_heads(model, opt, layer: int, k: int):
    b = model.blocks[layer]
    A, d = b.Wq.shape
    dev = b.Wq.device
    add = k * b.hd
    for nm in ("Wq", "Wk", "Wv"):
        _swap(opt, b, nm, torch.cat([getattr(b, nm).data, _fresh(add, d).to(dev)]), (slice(0, A),))
    _swap(opt, b, "Wo", torch.zeros(d, A + add, device=dev), (slice(None), slice(0, A)))


@torch.no_grad()
def add_layer(model, opt, at: int, heads: int, mlp: int):
    blk = GBlock(model.d, heads, model.hd, mlp, identity=True).to(model.tok_emb.weight.device)
    model.blocks.insert(at, blk)
    # shift sites of readers/decoders at or after the insertion point
    for m in list(model.readers) + list(model.decoders):
        if m.site > at:
            m.site += 1
    if opt is not None:
        opt.add_param_group(dict(params=list(blk.parameters()), **{k: v for k, v in opt.defaults.items()}))
    return blk
