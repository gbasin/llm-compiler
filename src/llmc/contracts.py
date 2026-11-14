"""Interface contracts: keep non-differentiable code modules connected.

A contract says: "at module.read, the quantity this code consumes is linearly
decodable by module.W_in". It does not claim anything about what the code does,
so no surrogate gradient through the code is needed.

Pieces:
  labels    - per-position ints (-1 = don't care) on a fixed anchor set of inputs.
              mode="truth":  from a teacher-supplied label fn (e.g. task.code_inputs)
              mode="anchor": whatever W_in decoded on the anchor set at insertion
                             time (label-free; works for network-invented features)
  loss      - CE(W_in(LN(h_read)), labels) on the anchor set; gradient flows into
              W_in *and* the core, telling the network to keep the feature readable.
  monitor   - decode agreement on the anchor set.
  refit     - if agreement < threshold, ridge-refit W_in on the anchor set.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .interface import fit_probe_W
from .tasks import Batch


class Contract:
    def __init__(self, module, anchor: Batch, labels: torch.Tensor, weight: float = 0.1,
                 refit_below: float = 0.98, scale: float = 10.0):
        self.module, self.anchor, self.labels = module, anchor, labels
        self.weight, self.refit_below, self.scale = weight, refit_below, scale
        self.refits = 0

    @classmethod
    def create(cls, module, model, anchor: Batch, mode: str, label_fn: Callable | None = None, **kw):
        if mode == "truth":
            labels = label_fn(anchor)
        elif mode == "anchor":
            with torch.no_grad():
                _, resid = model(anchor.tokens, capture=True)
                labels = module.W_in(module.ln(resid[module.read])).argmax(-1)
                labels = torch.where(anchor.tokens == getattr(model, "_pad_id", -1), -1, labels)
        else:
            raise KeyError(mode)
        return cls(module, anchor, labels, **kw)

    def _logits(self, model, idx=None):
        tok = self.anchor.tokens if idx is None else self.anchor.tokens[idx]
        _, resid = model(tok, capture=True)
        return self.module.W_in(self.module.ln(resid[self.module.read]))

    def loss(self, model, n: int = 32):
        """Contract loss on a random subset of n anchor sequences (cheap per step)."""
        idx = torch.randperm(self.anchor.tokens.shape[0])[:n]
        lg = self._logits(model, idx)
        lab = self.labels[idx]
        sel = lab >= 0
        return self.weight * F.cross_entropy(lg[sel], lab[sel])

    @torch.no_grad()
    def agreement(self, model) -> float:
        lg = self._logits(model)
        sel = self.labels >= 0
        return (lg[sel].argmax(-1) == self.labels[sel]).float().mean().item()

    @torch.no_grad()
    def refit(self, model):
        _, resid = model(self.anchor.tokens, capture=True)
        X = self.module.ln(resid[self.module.read])
        sel = self.labels >= 0
        w, b = fit_probe_W(X[sel].float(), self.labels[sel], self.module.program.in_dim)
        self.module.W_in.weight.copy_(self.scale * w)
        self.module.W_in.bias.copy_(self.scale * b)
        self.refits += 1

    def maintain(self, model) -> dict:
        a = self.agreement(model)
        rec = dict(agree=a, refit=False)
        if a < self.refit_below:
            self.refit(model)
            rec.update(refit=True, agree_after=self.agreement(model))
        return rec
