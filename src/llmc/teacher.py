"""Diagnostics for the teacher (Claude Code) to decide what code to write and where.

    uv run python -m llmc.teacher runs/<name>/final.pt [--split ood]

Reports, on one eval split:
  * accuracy by item position -> where does the model break down?
  * phantom gradients: |dL/dh| at each residual site, on wrong vs right
    predictions -> which sites the loss wants changed.
  * linear probes at each site for task quantities (current input, answer)
    -> a good READ site is where module inputs are decodable; if the answer
    is never decodable, the computation is missing (a candidate for code).
  * injected modules: gate value and ablation effect.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .tasks import Batch
from .train import evaluate, load_ckpt, make_eval_sets, parse_splits


def phantom_grads(model, b: Batch):
    model.zero_grad(set_to_none=True)
    logits, resid = model(b.tokens, capture=True)
    loss = F.cross_entropy(logits.flatten(0, 1), b.targets.flatten(), ignore_index=-100, reduction="sum")
    loss.backward()
    wrong = (logits.argmax(-1) != b.targets) & b.mask
    right = (~wrong) & b.mask
    rows = []
    for i, h in enumerate(resid):
        gn = h.grad.norm(dim=-1)
        rows.append(dict(site=i, wrong=gn[wrong].mean().item() if wrong.any() else 0.0,
                         right=gn[right].mean().item() if right.any() else 0.0))
    model.zero_grad(set_to_none=True)
    return rows, logits.detach(), [h.detach() for h in resid]


def linear_probe(X: torch.Tensor, y: torch.Tensor, n_cls: int, lam: float = 1e-2, train_frac: float = 0.7):
    """Ridge regression onto one-hot labels; returns held-out accuracy."""
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    X, y = X[perm], y[perm]
    k = int(train_frac * n)
    Xb = torch.cat([X, torch.ones(n, 1)], 1)
    Y = F.one_hot(y, n_cls).float()
    A = Xb[:k].T @ Xb[:k] + lam * torch.eye(Xb.shape[1])
    W = torch.linalg.solve(A, Xb[:k].T @ Y[:k])
    return ((Xb[k:] @ W).argmax(-1) == y[k:]).float().mean().item()


def report(path: str, split: str = "ood", n: int = 512):
    cfg, task, model = load_ckpt(path)
    sets = make_eval_sets(task, cfg)
    full = sets[split]
    b = Batch(*(t[:n] for t in (full.tokens, full.targets, full.mask, full.lengths)))
    print(f"== {path}  task={cfg.task} V={cfg.V} layers={cfg.n_layers} d={cfg.d_model} split={split} "
          f"lengths={parse_splits(cfg.eval_splits)[split]}")

    ev = evaluate(model, task, sets, "cpu")
    print("eval seq/tok acc:", {k: f"{v['seq_acc']:.3f}/{v['tok_acc']:.3f}" for k, v in ev.items()})

    rows, logits, resid = phantom_grads(model, b)
    correct = (logits.argmax(-1) == b.targets) & b.mask
    # accuracy by item index (bucketed)
    step = task.max_len(2) - task.max_len(1)
    item_idx = (torch.arange(b.tokens.shape[1]) - 1) // step
    buckets = {}
    for lo in range(0, int(b.lengths.max()), 8):
        sel = b.mask & (item_idx >= lo)[None] & (item_idx < lo + 8)[None]
        if sel.any():
            buckets[f"{lo + 1}-{lo + 8}"] = round(correct[sel].float().mean().item(), 3)
    print("token acc by item position:", buckets)

    print("phantom grad |dL/dh| per site (wrong / right):")
    for r in rows:
        print(f"  site {r['site']}: {r['wrong']:.4f} / {r['right']:.4f}")

    labels = task.probe_targets(b)
    print("linear probe held-out acc per site:")
    for name, y in labels.items():
        sel = y >= 0
        accs = [linear_probe(h[sel].float(), y[sel], task.V) for h in resid]
        base = (y[sel] == y[sel].mode().values).float().mean().item()
        print(f"  {name:12s} " + " ".join(f"s{i}={a:.2f}" for i, a in enumerate(accs)) + f"  (majority={base:.2f})")

    for m in model.injections:
        m.enabled = False
        ab = evaluate(model, task, {split: full}, "cpu")[split]
        m.enabled = True
        print(f"module {m.describe()}  ablated {split} seq/tok: {ab['seq_acc']:.3f}/{ab['tok_acc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", default="ood")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    report(a.ckpt, a.split, a.n)


if __name__ == "__main__":
    main()
