"""Chained callable retrieval (step 2 of the callable plan), built on lm8.

Two calls per forward, the second built from the answers to the first:
  call 1 (menu)   : lm8's multi_query. For each k in KS, "what followed the last k tokens?" -> one
                    retrieved 4-token continuation + count. Its first token y_c is candidate c's guess.
  call 2 (verify) : for every candidate c, the expert asks the index about the EXTENDED context
                    (last k tokens + y_c) for every k: how often does y_c follow, relative to the
                    context alone (a count ratio per k), and what followed y_c (3 tokens) at the
                    deepest match. So pass 2 looks two steps ahead along each candidate.
The core reads call-1 items after loop 1 (as in lm8) and call-2 items after loop 2. In the `chain`
mode, pass-2 attention is biased by log(pass-1 read mass on c) * learned scale, so what the core chose
to read on pass 1 steers which verification it reads (soft candidate method; the expert answers all
candidates, no labels). `--no_chain` drops that bias (verification still available, not steered).

    uv run python -m llmc.lm9 --name chain_2x3 --blocks 2 --loops 3
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
from .lm8 import KS, NCONT, multi_query

NC2 = 3  # tokens after y_c returned by the verify call
NRB = 18  # ratio buckets (0 = context unseen, 1 = extension unseen, 2.. = log2 ratio buckets)


def verify_query(stream, pos, y):
    """y (N, C) candidate next tokens (-1 none). For each candidate and each k in KS: ratio bucket of
    count(ctx_k + y) / count(ctx_k); plus NC2 tokens that followed y at the deepest matching k (k=0 = y alone)."""
    ig = ig_model()
    N, C = y.shape
    K = max(KS)
    ctx = np.stack([np.where(pos - j >= 0, np.asarray(stream[np.maximum(pos - j, 0)]).astype(np.int64), -1)
                    for j in range(K)], 1)  # (N, K): ctx[:, j] = token j back
    # base counts n(ctx_k) for every k (shared over candidates)
    lo = np.zeros(N, dtype=np.int64); hi = np.full(N, len(ig.sa), dtype=np.int64)
    base = np.zeros((N, K + 1), dtype=np.int64); base[:, 0] = len(ig.sa)
    for k in range(K):
        q = ctx[:, k]; alive = (q >= 0) & (hi > lo)
        if alive.any():
            nlo, nhi = ig._bound(lo, hi, k, q, False), ig._bound(lo, hi, k, q, True)
            lo, hi = np.where(alive, nlo, lo), np.where(alive, nhi, lo)
        else:
            hi = lo.copy()
        base[:, k + 1] = hi - lo
    # extended queries, all candidates flattened: reversed query = [y, ctx0, ctx1, ...]
    yy = y.reshape(-1)
    cx = np.repeat(ctx, C, 0)
    M = len(yy)
    lo = np.zeros(M, dtype=np.int64); hi = np.full(M, len(ig.sa), dtype=np.int64)
    ok = yy >= 0
    nlo, nhi = ig._bound(lo, hi, 0, np.where(ok, yy, 0), False), ig._bound(lo, hi, 0, np.where(ok, yy, 0), True)
    lo, hi = np.where(ok, nlo, lo), np.where(ok, nhi, lo)
    ext = np.zeros((M, K + 1), dtype=np.int64); ext[:, 0] = hi - lo
    best_lo, best_n = lo.copy(), hi - lo
    for k in range(K):
        q = cx[:, k]; alive = (q >= 0) & (hi > lo)
        if alive.any():
            nlo, nhi = ig._bound(lo, hi, k + 1, q, False), ig._bound(lo, hi, k + 1, q, True)
            lo, hi = np.where(alive, nlo, lo), np.where(alive, nhi, lo)
        else:
            hi = lo.copy()
        n = hi - lo
        ext[:, k + 1] = n
        better = n > 0
        best_lo, best_n = np.where(better, lo, best_lo), np.where(better, n, best_n)
    bn = np.repeat(base, C, 0)
    kidx = np.array(KS)
    e, b = ext[:, kidx], bn[:, kidx]
    r = np.log2((e + 0.5) / np.maximum(b, 1))  # <= 0
    rb = np.clip(np.floor(-r), 0, NRB - 3).astype(np.int64) + 2
    rb = np.where(e == 0, 1, rb)
    rb = np.where((b == 0) | ~ok[:, None], 0, rb)
    rng = np.random.default_rng(int(pos[0]) % (2 ** 31) + 7)
    pick = best_lo + (rng.random(M) * np.maximum(best_n, 1)).astype(np.int64)
    s = ig.sa[np.minimum(pick, len(ig.sa) - 1)]
    cont2 = np.full((M, NC2), -1, dtype=np.int64)
    for j in range(NC2):
        okj = ok & (best_n > 0) & (s - 1 - j >= 0)
        cont2[:, j] = np.where(okj, ig.rev[np.maximum(s - 1 - j, 0)], -1)
    return (torch.from_numpy(rb.reshape(N, C, len(KS))), torch.from_numpy(cont2.reshape(N, C, NC2)))


def make_batch(stream, starts, seq):
    pos = (starts[:, None] + np.arange(seq + 1)[None]).reshape(-1)
    p = (starts[:, None] + np.arange(seq)[None]).reshape(-1)
    B = len(starts)
    cont, cnt = multi_query(stream, p)
    rb, cont2 = verify_query(stream, p, cont[..., 0].numpy())
    C = len(KS)
    return dict(x=torch.from_numpy(np.asarray(stream[pos]).astype(np.int64)).view(B, seq + 1),
                cont=cont.view(B, seq, C, NCONT), cnt=cnt.view(B, seq, C),
                rb=rb.view(B, seq, C, len(KS)), cont2=cont2.view(B, seq, C, NC2))


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, bs, seq, seed):
        self.bs, self.seq, self.seed = bs, seq, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        tok = np.load(D / "train_tokens.npy", mmap_mode="r")
        while True:
            yield make_batch(tok, rng.integers(TRAIN_LO + IG_CTX, TRAIN_HI - self.seq - 1, self.bs), self.seq)


def val_batches(seq, n_windows):
    v = val_tokens()
    starts = np.linspace(IG_CTX, len(v) - seq - 2, n_windows).astype(np.int64)
    return [make_batch(v, starts[i:i + 16], seq) for i in range(0, n_windows, 16)]


class Reader(nn.Module):
    """Per-position attention over items (+ null). items: token ids (B,T,M), feature embedding sum
    (B,T,M,d), mask; optional additive score bias (B,T,M)."""

    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, h, it, ok, bias=None):
        k, v = self.k(it), self.v(it)
        s = torch.einsum("btd,btmd->btm", self.q(self.ln(h)), k) / math.sqrt(h.shape[-1])
        if bias is not None:
            s = s + bias
        s = s.masked_fill(~ok, -1e4)
        w = torch.cat([s, torch.zeros_like(s[..., :1])], -1).softmax(-1)[..., :-1]
        return self.out(torch.einsum("btm,btmd->btd", w, v)), w


class ChainLM(nn.Module):
    def __init__(self, d=128, heads=4, mlp=512, blocks=2, loops=3, seq=256, chain=True, verify=True):
        super().__init__()
        self.loops, self.chain, self.verify = loops, chain, verify
        C = len(KS)
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.blocks = nn.ModuleList(GBlock(d, heads, d // heads, mlp) for _ in range(blocks))
        # call-1 item features (port, outside the core)
        self.cand = nn.Embedding(C, d); self.off = nn.Embedding(NCONT, d); self.cnt = nn.Embedding(32, d)
        # call-2 item features: per-k ratio buckets, offset after y
        self.ratio = nn.Embedding(len(KS) * NRB, d); self.off2 = nn.Embedding(NC2 + 1, d)
        self.r1, self.r2 = Reader(d), Reader(d)
        self.chain_scale = nn.Parameter(torch.ones(())) if chain else None
        self.ln = nn.LayerNorm(d)
        self.copy_port = ProposalPort(d, 3)

    def core_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith("blocks"))

    def items1(self, b):
        B, T, C, J = b["cont"].shape
        it = self.emb(b["cont"].clamp(min=0)) + self.cand.weight[None, None, :, None] + self.off.weight[None, None, None] \
            + self.cnt(b["cnt"])[..., None, :]
        return it.view(B, T, C * J, -1), (b["cont"] >= 0).view(B, T, C * J)

    def items2(self, b):
        """per candidate c: NC2+1 items = [y_c, then the tokens that followed y_c], each carrying c's ratio profile."""
        y = b["cont"][..., :1]
        toks = torch.cat([y, b["cont2"]], -1)  # (B,T,C,NC2+1)
        B, T, C, J = toks.shape
        kofs = torch.arange(len(KS), device=y.device) * NRB
        prof = self.ratio(b["rb"] + kofs).sum(-2)  # (B,T,C,d)
        it = self.emb(toks.clamp(min=0)) + prof[..., None, :] + self.off2.weight[None, None, None] \
            + self.cand.weight[None, None, :, None]
        ok = (toks >= 0) & (y >= 0)
        return it.view(B, T, C * J, -1), ok.view(B, T, C * J), J

    def forward(self, b, return_w=False):
        x = b["x"][:, :-1]
        T = x.shape[1]
        h = self.emb(x) + self.pos[:T]
        it1, ok1 = self.items1(b)
        it2 = None
        w1 = w2 = None
        for t in range(self.loops):
            for blk in self.blocks:
                h = blk(h)
            if t == self.loops - 1:
                break
            if t == 0 or not self.verify:
                r, w1 = self.r1(h, it1, ok1)
            else:
                if it2 is None:
                    it2, ok2, J = self.items2(b)
                bias = None
                if self.chain:  # pass-1 choice steers pass-2 reading
                    m1 = w1.view(*w1.shape[:2], len(KS), NCONT).sum(-1)  # (B,T,C)
                    bias = (self.chain_scale * torch.log(m1.clamp(min=1e-4)))[..., None].expand(-1, -1, -1, J).flatten(2)
                r, w2 = self.r2(h, it2, ok2, bias)
            h = h + r
        hf = self.ln(h)
        logits = self.copy_port(hf @ self.emb.weight.T, hf, copy_props(x))
        return (logits, w1, w2) if return_w else logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--no_chain", action="store_true")
    ap.add_argument("--no_verify", action="store_true", help="re-read call 1 on every pass (lm8-equivalent)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/lm9")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    model = ChainLM(blocks=a.blocks, loops=a.loops, seq=a.seq, chain=not a.no_chain, verify=not a.no_verify).to(dev)
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
        m1 = torch.zeros(len(KS), device=dev); m2 = torch.zeros((), device=dev)
        for b in vb:
            b = to(b)
            lg, w1, w2 = model(b, return_w=True)
            tot += F.cross_entropy(lg.flatten(0, 1).float(), b["x"][:, 1:].flatten(), reduction="sum").item()
            n += b["x"][:, 1:].numel()
            m1 += w1.view(*w1.shape[:2], len(KS), NCONT).sum(-1).mean((0, 1))
            if w2 is not None:
                m2 += w2.sum(-1).mean()
        model.train()
        return tot / n, (m1 / len(vb)).tolist(), (m2 / len(vb)).item()

    t0 = time.time()
    it = iter(dl)
    for step in range(a.steps + 1):
        if step % 1000 == 0 or step == a.steps:
            v, m1, m2 = evaluate()
            rec = dict(step=step, val=round(v, 4), core=model.core_params(), loops=a.loops, blocks=a.blocks,
                       chain=not a.no_chain, verify=not a.no_verify, read1_by_k={k: round(x, 3) for k, x in zip(KS, m1)},
                       read2_mass=round(m2, 3),
                       chain_scale=round(model.chain_scale.item(), 3) if model.chain_scale is not None else None,
                       time=round(time.time() - t0))
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
