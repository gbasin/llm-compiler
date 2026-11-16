"""Rung 7: real text (Pile, Pythia/GPT-NeoX BPE tokens) with a looped reasoning core and experts.

Lexicon (outside the core): tied token embedding / unembedding, learned positions.
Core: a unit of `blocks` pre-LN blocks, applied `loops` times with shared weights (rung 0 recipe).
Experts (code), wired as next-token PROPOSAL ports (gated logit bonuses, gate = f(final hidden state)):
  ig   : local infini-gram over the first 30M Pile training tokens (suffix array). For each position,
         the longest earlier-corpus match of the context; top-4 next tokens with probabilities and the
         match length. The LM trains on tokens 30M..100M (never indexed) and validates on Pile val docs.
  copy : in-context copy: the token that followed the latest earlier occurrence of the last 1/2/3 tokens.
Arms differ only in core shape, loops and experts. Metric: val loss (nats/token) on held-out Pile docs.

    uv run python -m llmc.lm7 --name pile_loop2x3_ig --blocks 2 --loops 3 --experts ig,copy
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

from .audit.infinigram import InfiniGram
from .hybrid import GBlock

D = Path("data/pile")
VOCAB = 50304
TRAIN_LO, TRAIN_HI = 30_000_000, 100_000_000
IG_CTX, IG_K, IG_SAMPLES = 24, 4, 64


# ------------------------------------------------------------------ data + infini-gram expert
def val_tokens():
    p = D / "valid_tokens.npy"
    if not p.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        docs = json.loads((D / "valid_docs.json").read_text())
        ids = []
        for d in docs:
            ids += tok(d)["input_ids"] + [0]
        np.save(p, np.asarray(ids, dtype=np.uint16))
    return np.load(p).astype(np.int64)


_IG = None


def ig_model():
    global _IG
    if _IG is None:
        _IG = InfiniGram(np.load(D / "rev_30M.npy"), np.load(D / "sa_30M.npy"), VOCAB)
    return _IG


def ig_query(stream: np.ndarray, pos: np.ndarray):
    """Expert: for each absolute position p in `stream`, infini-gram next-token proposals given
    stream[..p]. Returns top-k tokens (N,K) (-1 = none), probs (N,K), match length (N,)."""
    ig = ig_model()
    ctx = np.stack([np.where(pos - k >= 0, np.asarray(stream[np.maximum(pos - k, 0)]).astype(np.int64), -1)
                    for k in range(IG_CTX)], 1)
    ml, lo, hi = ig.longest_match(ctx.astype(np.int64), IG_CTX)
    n = hi - lo
    S = IG_SAMPLES
    off = (np.arange(S)[None, :] * np.maximum(n, 1)[:, None]) // S
    idx = np.minimum(lo[:, None] + off, len(ig.sa) - 1)
    s = ig.sa[idx]
    nxt = np.where((s > 0) & (n[:, None] > 0) & (ml[:, None] > 0), ig.rev[np.maximum(s - 1, 0)], -1)
    nx = torch.from_numpy(nxt)
    srt, _ = nx.sort(1)
    # run-length counts per row -> top-k
    new = torch.ones_like(srt, dtype=torch.bool)
    new[:, 1:] = srt[:, 1:] != srt[:, :-1]
    grp = new.cumsum(1) - 1
    cnt = torch.zeros_like(srt).scatter_add_(1, grp, torch.ones_like(srt))
    val = torch.full_like(srt, -1).scatter_(1, grp, srt)
    cnt = torch.where(val >= 0, cnt, torch.zeros_like(cnt))
    top = cnt.topk(IG_K, 1)
    toks = val.gather(1, top.indices)
    toks = torch.where(top.values > 0, toks, torch.full_like(toks, -1))
    probs = top.values.float() / cnt.sum(1, keepdim=True).clamp(min=1)
    return toks, probs, torch.from_numpy(np.minimum(ml, 15))


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, bs, seq, seed, use_ig):
        self.bs, self.seq, self.seed, self.use_ig = bs, seq, seed, use_ig

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        tok = np.load(D / "train_tokens.npy", mmap_mode="r")
        while True:
            starts = rng.integers(TRAIN_LO + IG_CTX, TRAIN_HI - self.seq - 1, self.bs)
            pos = (starts[:, None] + np.arange(self.seq + 1)[None]).reshape(-1)
            x = torch.from_numpy(np.asarray(tok[pos]).astype(np.int64)).view(self.bs, self.seq + 1)
            out = dict(x=x)
            if self.use_ig:
                p = (starts[:, None] + np.arange(self.seq)[None]).reshape(-1)
                t_, pr, ml = ig_query(tok, p)
                out.update(ig_tok=t_.view(self.bs, self.seq, IG_K), ig_p=pr.view(self.bs, self.seq, IG_K),
                           ig_len=ml.view(self.bs, self.seq))
            yield out


def val_batches(seq, n_windows, use_ig):
    v = val_tokens()
    starts = np.linspace(IG_CTX, len(v) - seq - 2, n_windows).astype(np.int64)
    out = []
    for i in range(0, n_windows, 16):
        st = starts[i:i + 16]
        pos = (st[:, None] + np.arange(seq + 1)[None]).reshape(-1)
        b = dict(x=torch.from_numpy(v[pos]).view(len(st), seq + 1))
        if use_ig:
            p = (st[:, None] + np.arange(seq)[None]).reshape(-1)
            t_, pr, ml = ig_query(v, p)
            b.update(ig_tok=t_.view(len(st), seq, IG_K), ig_p=pr.view(len(st), seq, IG_K), ig_len=ml.view(len(st), seq))
        out.append(b)
    return out


# ------------------------------------------------------------------ model
def copy_props(x, ks=(1, 2, 3)):
    """Expert: for each k, the token after the latest earlier occurrence of the last k tokens (-1 none)."""
    B, T = x.shape
    idx = torch.arange(T, device=x.device)
    outs = []
    h = torch.zeros_like(x)
    for k in range(1, max(ks) + 1):
        sh = torch.full_like(x, -1)
        sh[:, k - 1:] = x[:, :T - k + 1]
        h = h * 50321 + (sh + 1)
        if k in ks:
            same = (h[:, :, None] == h[:, None, :]) & (idx[None, None, :] < idx[None, :, None]) & (idx[None, None, :] >= k - 1)
            j = torch.where(same, idx[None, None, :], torch.full_like(idx[None, None, :], -1)).max(-1).values
            nxt = x.gather(1, (j + 1).clamp(max=T - 1))
            outs.append(torch.where(j >= 0, nxt, torch.full_like(nxt, -1)))
    return torch.stack(outs, -1)  # (B,T,len(ks))


class ProposalPort(nn.Module):
    """logits[v] += softplus(gate(h)) * (a + b * feat) for each proposed token v (gate init 0 -> 0.69)."""

    def __init__(self, d, n_props, n_feat_buckets=16):
        super().__init__()
        self.gate = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_props))
        nn.init.zeros_(self.gate[1].weight)
        nn.init.constant_(self.gate[1].bias, -2.0)
        self.feat = nn.Embedding(n_feat_buckets, n_props)
        nn.init.zeros_(self.feat.weight)
        self.scale = nn.Parameter(torch.full((n_props,), 1.0))

    def forward(self, logits, h, props, logp=None, bucket=None):
        g = F.softplus(self.gate(h))  # (B,T,P)
        if bucket is not None:
            g = g * torch.exp(self.feat(bucket))
        bonus = g * self.scale * (1.0 if logp is None else (logp + 4.0).clamp(min=0))
        ok = props >= 0
        return logits.scatter_add(-1, props.clamp(min=0), torch.where(ok, bonus, torch.zeros_like(bonus)))


class LoopLM(nn.Module):
    def __init__(self, d=128, heads=4, mlp=512, blocks=2, loops=1, seq=256, experts=()):
        super().__init__()
        self.loops, self.experts = loops, set(experts)
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.blocks = nn.ModuleList(GBlock(d, heads, d // heads, mlp) for _ in range(blocks))
        self.ln = nn.LayerNorm(d)
        self.ig_port = ProposalPort(d, IG_K) if "ig" in self.experts else None
        self.copy_port = ProposalPort(d, 3) if "copy" in self.experts else None

    def core_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith("blocks"))

    def forward(self, b):
        x_in = b["x"][:, :-1]
        T = x_in.shape[1]
        h = self.emb(x_in) + self.pos[:T]
        for _ in range(self.loops):
            for blk in self.blocks:
                h = blk(h)
        h = self.ln(h)
        logits = h @ self.emb.weight.T
        if self.ig_port is not None:
            logits = self.ig_port(logits, h, b["ig_tok"], torch.log(b["ig_p"].clamp(min=1e-4)), b["ig_len"])
        if self.copy_port is not None:
            logits = self.copy_port(logits, h, copy_props(x_in))
        return logits


# ------------------------------------------------------------------ train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--mlp", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--experts", default="")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/lm7")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    experts = [e for e in a.experts.split(",") if e]
    use_ig = "ig" in experts
    dev = torch.device(a.device)
    model = LoopLM(a.d, a.heads, a.mlp, a.blocks, a.loops, a.seq, experts).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    dl = torch.utils.data.DataLoader(Stream(a.bs, a.seq, a.seed, use_ig), batch_size=None,
                                     num_workers=a.workers, persistent_workers=a.workers > 0)
    vb = val_batches(a.seq, 256, use_ig)
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    f = (out / "log.jsonl").open("w")

    def to(b):
        return {k: v.to(dev) for k, v in b.items()}

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = n = 0
        for b in vb:
            b = to(b)
            lg = model(b)
            tot += F.cross_entropy(lg.flatten(0, 1).float(), b["x"][:, 1:].flatten(), reduction="sum").item()
            n += b["x"][:, 1:].numel()
        model.train()
        return tot / n

    t0 = time.time()
    it = iter(dl)
    for step in range(a.steps + 1):
        if step % 1000 == 0 or step == a.steps:
            rec = dict(step=step, val=round(evaluate(), 4), core=model.core_params(), loops=a.loops,
                       blocks=a.blocks, experts=experts, time=round(time.time() - t0))
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps(rec), flush=True)
        if step == a.steps:
            break
        b = to(next(it))
        lg = model(b)
        loss = F.cross_entropy(lg.flatten(0, 1), b["x"][:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
