"""Callable retrieval on real text (steps 1+2 of the callable plan), built on lm7.

Expert (code + data only, run live in data workers): the Pile suffix-array index (tokens 0-30M).
For every position it answers a MENU of candidate queries: "what followed the last k tokens?" for
k in KS. Each answer = the 4 tokens that followed one corpus occurrence (a retrieved continuation),
plus the occurrence count. That gives M = len(KS) * 4 retrieved items per position.
Core: lm7's (looped) blocks. Between loops, the core READS the items through per-position
attention (query = its state; keys/values = item token embedding + candidate/offset/count features),
i.e. it chooses which lookup to trust and what to take from it (the soft candidate method). The
read result is added to the state for the next loop. A proposal port turns the attention mass on each
candidate into a logit bonus for that candidate's first continuation token.
Ablation `--uniform`: the read weights are fixed uniform (no choice), everything else equal.

    uv run python -m llmc.lm8 --name read_loop2x3 --blocks 2 --loops 3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid import GBlock
from .lm7 import D, IG_CTX, TRAIN_HI, TRAIN_LO, VOCAB, copy_props, ig_model, val_tokens, ProposalPort

KS = (1, 2, 3, 4, 5, 6, 8)
NCONT = 4


def multi_query(stream: np.ndarray, pos: np.ndarray):
    """For each position p and each k in KS: the continuation (NCONT tokens) after one corpus
    occurrence of stream[p-k+1..p], and log2 occurrence count (0 = no match). -1 = no token."""
    ig = ig_model()
    N = len(pos)
    ctx = np.stack([np.where(pos - j >= 0, np.asarray(stream[np.maximum(pos - j, 0)]).astype(np.int64), -1)
                    for j in range(max(KS))], 1)
    lo = np.zeros(N, dtype=np.int64)
    hi = np.full(N, len(ig.sa), dtype=np.int64)
    cont = np.full((N, len(KS), NCONT), -1, dtype=np.int64)
    cnt = np.zeros((N, len(KS)), dtype=np.int64)
    rng = np.random.default_rng(int(pos[0]) % (2 ** 31))
    ki = 0
    for k in range(max(KS)):
        q = ctx[:, k]
        alive = (q >= 0) & (hi > lo)
        if alive.any():
            nlo = ig._bound(lo, hi, k, q, upper=False)
            nhi = ig._bound(lo, hi, k, q, upper=True)
            lo = np.where(alive, nlo, lo)
            hi = np.where(alive, nhi, lo)
        else:
            hi = lo.copy()
        if k + 1 == KS[ki]:
            n = hi - lo
            cnt[:, ki] = np.where(n > 0, np.floor(np.log2(np.maximum(n, 1))) + 1, 0)
            pick = lo + (rng.random(N) * np.maximum(n, 1)).astype(np.int64)
            s = ig.sa[np.minimum(pick, len(ig.sa) - 1)]
            for j in range(NCONT):
                ok = (n > 0) & (s - 1 - j >= 0)
                cont[:, ki, j] = np.where(ok, ig.rev[np.maximum(s - 1 - j, 0)], -1)
            ki += 1
            if ki == len(KS):
                break
    return torch.from_numpy(cont), torch.from_numpy(np.minimum(cnt, 31))


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, bs, seq, seed):
        self.bs, self.seq, self.seed = bs, seq, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        tok = np.load(D / "train_tokens.npy", mmap_mode="r")
        while True:
            starts = rng.integers(TRAIN_LO + IG_CTX, TRAIN_HI - self.seq - 1, self.bs)
            pos = (starts[:, None] + np.arange(self.seq + 1)[None]).reshape(-1)
            x = torch.from_numpy(np.asarray(tok[pos]).astype(np.int64)).view(self.bs, self.seq + 1)
            p = (starts[:, None] + np.arange(self.seq)[None]).reshape(-1)
            cont, cnt = multi_query(tok, p)
            yield dict(x=x, cont=cont.view(self.bs, self.seq, len(KS), NCONT), cnt=cnt.view(self.bs, self.seq, len(KS)))


def val_batches(seq, n_windows):
    v = val_tokens()
    starts = np.linspace(IG_CTX, len(v) - seq - 2, n_windows).astype(np.int64)
    out = []
    for i in range(0, n_windows, 16):
        st = starts[i:i + 16]
        pos = (st[:, None] + np.arange(seq + 1)[None]).reshape(-1)
        p = (st[:, None] + np.arange(seq)[None]).reshape(-1)
        cont, cnt = multi_query(v, p)
        out.append(dict(x=torch.from_numpy(v[pos]).view(len(st), seq + 1),
                        cont=cont.view(len(st), seq, len(KS), NCONT), cnt=cnt.view(len(st), seq, len(KS))))
    return out


class Reader(nn.Module):
    """Per-position attention over that position's retrieved items (+ a null item)."""

    def __init__(self, d, uniform=False):
        super().__init__()
        self.uniform = uniform
        self.cand = nn.Embedding(len(KS), d)
        self.off = nn.Embedding(NCONT, d)
        self.cnt = nn.Embedding(32, d)
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, h, emb, cont, cnt):
        B, T, C, J = cont.shape
        d = h.shape[-1]
        ok = cont >= 0
        it = emb(cont.clamp(min=0)) + self.cand.weight[None, None, :, None] + self.off.weight[None, None, None] \
            + self.cnt(cnt)[..., None, :]
        it = it.view(B, T, C * J, d)
        ok = ok.view(B, T, C * J)
        k, v = self.k(it), self.v(it)
        if self.uniform:
            w = ok.float() / ok.float().sum(-1, keepdim=True).clamp(min=1)
        else:
            s = torch.einsum("btd,btmd->btm", self.q(self.ln(h)), k) / math.sqrt(d)
            s = s.masked_fill(~ok, -1e4)
            s = torch.cat([s, torch.zeros_like(s[..., :1])], -1)  # null item
            w = s.softmax(-1)[..., :-1]
        r = torch.einsum("btm,btmd->btd", w, v)
        return self.out(r), w.view(B, T, C, J)


class CallLM(nn.Module):
    def __init__(self, d=128, heads=4, mlp=512, blocks=2, loops=3, seq=256, uniform=False, copy=True, propose=True):
        super().__init__()
        self.loops, self.propose = loops, propose
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.blocks = nn.ModuleList(GBlock(d, heads, d // heads, mlp) for _ in range(blocks))
        self.reader = Reader(d, uniform)
        self.ln = nn.LayerNorm(d)
        self.copy_port = ProposalPort(d, 3) if copy else None
        self.prop_port = ProposalPort(d, len(KS)) if propose else None

    def core_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith("blocks"))

    def forward(self, b, return_w=False):
        x = b["x"][:, :-1]
        T = x.shape[1]
        h = self.emb(x) + self.pos[:T]
        w = None
        for t in range(self.loops):
            for blk in self.blocks:
                h = blk(h)
            r, w = self.reader(h, self.emb, b["cont"], b["cnt"])  # call: read the retrieved items
            if t < self.loops - 1:
                h = h + r
        hf = self.ln(h)
        logits = hf @ self.emb.weight.T
        if self.prop_port is not None:  # candidate c proposes its first continuation token, scaled by read mass
            mass = w.sum(-1)  # (B,T,C)
            logits = self.prop_port(logits, hf, b["cont"][..., 0], torch.log(mass.clamp(min=1e-4)))
        if self.copy_port is not None:
            logits = self.copy_port(logits, hf, copy_props(x))
        return (logits, w) if return_w else logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--no_propose", action="store_true")
    ap.add_argument("--no_copy", action="store_true")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/lm8")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    model = CallLM(blocks=a.blocks, loops=a.loops, seq=a.seq, uniform=a.uniform, copy=not a.no_copy,
                   propose=not a.no_propose).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    dl = torch.utils.data.DataLoader(Stream(a.bs, a.seq, a.seed), batch_size=None, num_workers=a.workers,
                                     persistent_workers=True, prefetch_factor=4)
    vb = val_batches(a.seq, 256)
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    f = (out / "log.jsonl").open("w")
    to = lambda b: {k: v.to(dev) for k, v in b.items()}

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = n = 0
        wsum = torch.zeros(len(KS), device=dev)
        for b in vb:
            b = to(b)
            lg, w = model(b, return_w=True)
            tot += F.cross_entropy(lg.flatten(0, 1).float(), b["x"][:, 1:].flatten(), reduction="sum").item()
            n += b["x"][:, 1:].numel()
            wsum += w.sum((-1)).mean((0, 1))
        model.train()
        return tot / n, (wsum / len(vb)).tolist()

    t0 = time.time()
    it = iter(dl)
    for step in range(a.steps + 1):
        if step % 1000 == 0 or step == a.steps:
            v, wk = evaluate()
            rec = dict(step=step, val=round(v, 4), core=model.core_params(), loops=a.loops, blocks=a.blocks,
                       uniform=a.uniform, read_mass_by_k={k: round(x, 3) for k, x in zip(KS, wk)}, time=round(time.time() - t0))
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps(rec), flush=True)
        if step == a.steps:
            break
        b = to(next(it))
        loss = F.cross_entropy(model(b).flatten(0, 1), b["x"][:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
