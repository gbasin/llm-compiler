"""Rung 0 of the looped reasoning core: can ONE shared block, applied T times, do k-hop pointer
chasing as well as a stack of distinct blocks with T times the weights, and solve more hops when
given more loops at test time?

Task (generated): a random permutation P on N symbols shown as shuffled pairs "x P(x)", then
HOP^k PAD^(12-k) QUERY s; target = P^k(s) at the last position.

Models:
  stacked  : L distinct pre-LN blocks, one pass.
  looped   : 1 block (or B blocks) applied T times. Each step: x <- block(A [x ; e0] (+ step emb)),
             A = input-injection adapter (Huginn-style recurrent depth). T sampled per batch in training.

    uv run python -m llmc.loop --arm looped --name r0_loop
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid import GBlock


# ------------------------------------------------------------------ task
class PointerTask:
    """Hop count k is UNARY: k HOP marks then PAD up to kmax slots, so unseen k is representable."""

    def __init__(self, n=12, kmax=12):
        self.n, self.kmax = n, kmax
        self.QUERY, self.HOP, self.PAD = n, n + 1, n + 2
        self.vocab = n + 3
        self.length = 2 * n + 2 + kmax

    def sample(self, B, g, kmin=1, kmax=None):
        n = self.n
        kmax = kmax or self.kmax
        P = torch.argsort(torch.rand(B, n, generator=g), dim=1)  # P[b, x] = next of x
        order = torch.argsort(torch.rand(B, n, generator=g), dim=1)
        xs = order
        ys = P.gather(1, xs)
        pairs = torch.stack([xs, ys], -1).view(B, 2 * n)
        s = torch.randint(0, n, (B,), generator=g)
        k = torch.randint(kmin, kmax + 1, (B,), generator=g)
        cur = s.clone()
        for step in range(1, kmax + 1):
            nxt = P.gather(1, cur[:, None])[:, 0]
            cur = torch.where(k >= step, nxt, cur)
        slots = torch.where(torch.arange(self.kmax)[None] < k[:, None], self.HOP, self.PAD)
        tok = torch.cat([pairs, slots, torch.full((B, 1), self.QUERY), s[:, None]], 1)  # read out at s
        return tok, cur, k


# ------------------------------------------------------------------ model
class Looped(nn.Module):
    def __init__(self, vocab, length, d=64, heads=4, mlp=128, n_blocks=1, step_emb=False, max_steps=64, inject=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(length, d) * 0.02)
        self.blocks = nn.ModuleList(GBlock(d, heads, d // heads, mlp) for _ in range(n_blocks))
        self.inject = nn.Linear(2 * d, d, bias=False) if inject else None
        if inject:
            with torch.no_grad():  # start as x + e0
                self.inject.weight.copy_(torch.cat([torch.eye(d), torch.eye(d)], 1) * 0.5)
        self.step = nn.Parameter(torch.zeros(max_steps, d)) if step_emb else None
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def core_params(self):
        """Core = blocks + injection adapter + step emb (lexicon = emb/pos/head excluded)."""
        skip = ("emb.", "pos", "head.", "ln.")
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith(skip))

    def forward(self, tok, T=1, all_steps=False):
        e0 = self.emb(tok) + self.pos[:tok.shape[1]]
        x = e0
        outs = []
        for t in range(T):
            if self.inject is not None and t > 0:
                x = self.inject(torch.cat([x, e0], -1))
            if self.step is not None:
                x = x + self.step[min(t, len(self.step) - 1)]
            for b in self.blocks:
                x = b(x)
            if all_steps:
                outs.append(self.head(self.ln(x[:, -1])))
        return outs if all_steps else self.head(self.ln(x[:, -1]))


# ------------------------------------------------------------------ train / eval
@torch.no_grad()
def eval_grid(model, task, Ts, ks, n=512, seed=123):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    grid = {}
    for k in ks:
        tok, y, _ = task.sample(n, g, kmin=k, kmax=k)
        outs = model(tok, T=max(Ts), all_steps=True)
        for T in Ts:
            grid[(k, T)] = (outs[T - 1].argmax(-1) == y).float().mean().item()
    model.train()
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", choices=["looped", "stacked"], required=True)
    ap.add_argument("--blocks", type=int, default=1, help="distinct blocks (stacked: depth)")
    ap.add_argument("--tmin", type=int, default=8)
    ap.add_argument("--tmax", type=int, default=12)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp", type=int, default=128)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k_train", type=int, default=6)
    ap.add_argument("--curriculum", action="store_true", help="start at 1 hop, add a hop when train acc > 0.9")
    ap.add_argument("--step_emb", action="store_true")
    ap.add_argument("--no_inject", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out_dir", default="runs/loop")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    task = PointerTask(a.n, kmax=12)
    model = Looped(task.vocab, task.length, a.d, a.heads, a.mlp, a.blocks, a.step_emb, inject=not a.no_inject)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500))
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    f = (out / "log.jsonl").open("w")
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    ks = list(range(1, 13))
    Ts = [1, 2, 4, 6, 8, 12, 16, 24] if a.arm == "looped" else [1]
    kcur, accs = (1 if a.curriculum else a.k_train), []
    for step in range(a.steps + 1):
        if step % 2000 == 0 or step == a.steps:
            grid = eval_grid(model, task, Ts, ks)
            T_main = a.tmax if a.arm == "looped" else 1
            rec = dict(step=step, kcur=kcur, arm=a.arm, blocks=a.blocks, core=model.core_params(), time=round(time.time() - t0),
                       acc_by_k={k: round(grid[(k, T_main)], 3) for k in ks},
                       grid={f"k{k}_T{T}": round(v, 3) for (k, T), v in grid.items()})
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps({x: rec[x] for x in ("step", "core", "time", "acc_by_k")}), flush=True)
        if step == a.steps:
            break
        tok, y, k = task.sample(a.bs, g, kmin=1, kmax=kcur)
        T = int(torch.randint(a.tmin, a.tmax + 1, (1,), generator=g)) if a.arm == "looped" else 1
        lg = model(tok, T=T)
        loss = F.cross_entropy(lg, y)
        accs = (accs + [(lg.argmax(-1) == y).float().mean().item()])[-50:]
        if a.curriculum and kcur < a.k_train and len(accs) == 50 and sum(accs) / 50 > 0.9:
            kcur, accs = kcur + 1, []
            print(f"step {step}: curriculum -> k <= {kcur}", flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()


if __name__ == "__main__":
    main()
