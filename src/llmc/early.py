"""Replication (scaled down) of doubleAI's sliding-window + exact-match lookup timing result,
plus our question: does code present from the start let a smaller core do the job?

Model: char-level transformer whose attention only sees the last W positions (sliding window).
Lookup (code, parameter-free): for each position t, find the latest j < t whose last k chars
equal the last k chars ending at t; the char after it (tokens[j+1]) is "retrieved". Every head in
every layer gets it as an extra key/value: k_r, v_r = Wk, Wv applied to LN(emb(retrieved) + p),
with p a fixed (non-learned) random vector marking "retrieved". The head adds
    g_h * sigmoid(q . k_r / sqrt(hd)) * v_r
with g_h a learned per-head gate initialised to 0 (function-preserving at insertion).
(doubleAI put the retrieved token into each head's softmax; a sigmoid side channel keeps
insertion exactly function-preserving.)

Arms: none | start (lookup from step 0) | mid (inserted at --insert_at) | full (no window).
Metrics on fixed val sequences: overall loss, and loss on "beyond-window copy" tokens (the
lookup's match is farther than W back AND its retrieved char is the true next char), i.e. tokens
the windowed network cannot get from attention.

    uv run python -m llmc.early --arm start --name e_start_s0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lmdata import CharSource


def lookup(tok, k: int):
    """tok (B,T) -> retrieved (B,T) (char after latest earlier k-gram match, -1 if none), dist (B,T)."""
    B, T = tok.shape
    V = int(tok.max()) + 1
    key = tok.clone()
    for i in range(1, k):
        sh = torch.full_like(tok, -1)
        sh[:, i:] = tok[:, :-i]
        key = key * (V + 1) + (sh + 1)
    idx = torch.arange(T)
    valid_key = idx >= k - 1
    same = (key[:, :, None] == key[:, None, :]) & (idx[None, None, :] < idx[None, :, None]) & valid_key[None, None, :]
    j = torch.where(same, idx[None, None, :], -1).max(-1).values
    ret = tok.gather(1, (j + 1).clamp(max=T - 1))
    ret = torch.where(j >= 0, ret, torch.full_like(ret, -1))
    return ret, torch.where(j >= 0, idx[None] - j, torch.zeros_like(j))


class Block(nn.Module):
    def __init__(self, d, H, hd, n_mlp):
        super().__init__()
        self.H, self.hd = H, hd
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * H * hd, bias=False)
        self.o = nn.Linear(H * hd, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, n_mlp), nn.GELU(), nn.Linear(n_mlp, d))
        self.g = nn.Parameter(torch.zeros(H))  # lookup gate per head

    def forward(self, x, mask, r_emb=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if r_emb is not None:
            _, kr, vr = self.qkv(self.ln1(r_emb)).view(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)
            w = torch.sigmoid((q * kr).sum(-1, keepdim=True) / math.sqrt(self.hd))
            a = a + self.g.view(1, -1, 1, 1) * w * vr
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        return x + self.mlp(self.ln2(x))


class SWA(nn.Module):
    def __init__(self, vocab, d, L, H, hd, n_mlp, window, T, k):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, H, hd, n_mlp) for _ in range(L))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        i = torch.arange(T)
        self.register_buffer("mask", (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < window))
        self.register_buffer("p_ret", torch.randn(d) * 0.5)
        self.k, self.use_lookup = k, False

    def core_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.endswith(".g"))

    def forward(self, tok):
        T = tok.shape[1]
        x = self.emb(tok) + self.pos.weight[:T]
        r_emb = None
        if self.use_lookup:
            ret, _ = lookup(tok, self.k)
            r_emb = torch.where((ret >= 0)[..., None], self.emb(ret.clamp(min=0)) + self.p_ret, torch.zeros_like(x))
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m, r_emb)
        return self.head(self.ln(x))


@torch.no_grad()
def evaluate(model, X, window, k):
    model.eval()
    lg = torch.cat([model(X[i:i + 64, :-1]) for i in range(0, len(X), 64)])
    model.train()
    y = X[:, 1:]
    nll = F.cross_entropy(lg.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
    ret, dist = lookup(X[:, :-1], k)
    far = (ret == y) & (dist >= window)
    near = (ret == y) & (dist < window) & (ret >= 0)
    rest = ~(far | near)
    return dict(loss=round(nll.mean().item(), 4), far_copy=round(nll[far].mean().item(), 4),
                near_copy=round(nll[near].mean().item(), 4), other=round(nll[rest].mean().item(), 4),
                frac_far=round(far.float().mean().item(), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", choices=["none", "start", "mid", "full"], required=True)
    ap.add_argument("--out_dir", default="runs/early")
    ap.add_argument("--arch", default="2x4x64", help="LxHxMLP")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--hd", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--insert_at", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    src = CharSource("data/tinyshakespeare.txt")
    X = src.sample(256, a.T + 1, torch.Generator().manual_seed(999), split="val")
    L, H, M = map(int, a.arch.split("x"))
    window = a.T if a.arm == "full" else a.window
    model = SWA(src.vocab, a.d, L, H, a.hd, M, window, a.T, a.k)
    model.use_lookup = a.arm == "start"
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps))))
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    f = (out / "log.jsonl").open("w")
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for step in range(a.steps + 1):
        if a.arm == "mid" and step == a.insert_at:
            model.use_lookup = True
        if step % a.eval_every == 0:
            rec = dict(step=step, arm=a.arm, arch=a.arch, core=model.core_params(), lookup=model.use_lookup,
                       gate=round(sum(b.g.abs().mean().item() for b in model.blocks) / L, 4),
                       time=round(time.time() - t0, 1), **evaluate(model, X, a.window, a.k))
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps(rec), flush=True)
        if step == a.steps:
            break
        x = src.sample(a.bs, a.T + 1, g)
        loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()


if __name__ == "__main__":
    main()
