"""Teacher-designed module interfaces.

A zero gate with random W_in/W_out sits on a saddle: the gate's gradient
<dL/dh, W_out P(W_in h)> is noise and W_in/W_out get no gradient at all. Early
in training loss noise kicks the gate off zero; later it often never opens.
The teacher can break the symmetry by designing the interface:

  init_in="probe":    fit W_in by ridge regression from LN(h_read) to the
                      program's intended input labels (task.code_inputs).
  init_out="unembed": point W_out's columns at the unembedding directions of
                      the corresponding answer tokens, so a positive gate
                      immediately raises the logit of the program's output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def fit_probe_W(X: torch.Tensor, y: torch.Tensor, n_cls: int, lam: float = 1e-2):
    """Ridge onto one-hot labels. Returns (weight (n_cls, d), bias (n_cls,))."""
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    Y = F.one_hot(y, n_cls).float()
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1], device=X.device)
    W = torch.linalg.solve(A, Xb.T @ Y)
    return W[:-1].T.contiguous(), W[-1].contiguous()


@torch.no_grad()
def probe_init_in(module, model, task, batch, scale: float = 10.0):
    _, resid = model(batch.tokens, capture=True)
    X = module.ln(resid[module.read])
    y = task.code_inputs(batch)
    sel = y >= 0
    w, b = fit_probe_W(X[sel].float(), y[sel], module.program.in_dim)
    module.W_in.weight.copy_(scale * w)
    module.W_in.bias.copy_(scale * b)
    acc = ((X[sel] @ w.T + b).argmax(-1) == y[sel]).float().mean().item()
    return acc


@torch.no_grad()
def unembed_init_out(module, model, task, scale: float = 1.0):
    # answer token for value v is V + v
    rows = model.head.weight[task.V: 2 * task.V]  # (V, d)
    W = rows / rows.norm(dim=-1, keepdim=True)
    module.W_out.weight.copy_(scale * W.T)
