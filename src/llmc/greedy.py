"""Greedy, early code insertion: score every candidate code feature at ANY training step, without
waiting for a plateau, and insert the winners as zero-init readers.

Candidate = (code op on the raw tokens) x (read site). Inserting one means adding a reader: a
zero-init embedding table over the op's output values, added to the residual stream at that site.
Because the table starts at zero the model is unchanged, and the table's gradient right now tells
us how much it would help:

  gradscore  = sum_v || G_v - n_v * gbar ||^2 / N   (G_v = summed dL/dh_site over positions where
               the op outputs v; subtracting the mean gradient removes what a bias could do).
               One backward pass scores every candidate at every site.
  fitscore   = held-out loss drop after training ONLY that reader for a few steps on the frozen
               model (an honest, more expensive check; doubleAI's "score on held-out docs").

Arms (fixed core, Tiny Shakespeare, same data order):
  none   no code
  all    every candidate at site 0 from step 0
  greedy every --every steps from step 0: gradscore all, fitscore the top --k_fit, insert the best
         if its fitscore > --min_gain (one insertion per check)
  late   doubleAI-like timing: the same selection, but only from --late_at on

    uv run python -m llmc.greedy run --arm greedy --name g_s0
    uv run python -m llmc.greedy validate          # gradscore vs fitscore vs real trial gain
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import oplib
from .hybrid import Hybrid
from .ladder import parse_arch
from .lmdata import CharSource
from .oplib import op


@op(lambda m, s: m.V + 1)
def kgram_next(m, tok, k: int):
    """Char after the latest earlier occurrence of the last k chars (V if none). Causal."""
    B, T = tok.shape
    key = tok.clone()
    for i in range(1, k):
        sh = torch.full_like(tok, -1)
        sh[:, i:] = tok[:, :-i]
        key = key * (m.V + 1) + (sh + 1)
    idx = torch.arange(T, device=tok.device)
    same = (key[:, :, None] == key[:, None, :]) & (idx[None, None, :] < idx[None, :, None]) & (idx >= k - 1)[None, None, :]
    j = torch.where(same, idx[None, None, :], -1).max(-1).values
    nxt = tok.gather(1, (j + 1).clamp(max=T - 1))
    return torch.where(j >= 0, nxt, torch.full_like(nxt, m.V))


@op(lambda m, s: 9)
def match_len(m, tok, kmax: int = 8):
    """Longest k <= kmax with an earlier match of the last k chars (0..kmax)."""
    out = torch.zeros_like(tok)
    for k in range(1, kmax + 1):
        out = torch.where(kgram_next(m, tok, k) < m.V, torch.full_like(tok, k), out)
    return out


@op(lambda m, s: 6)
def char_class(m, tok):
    """0 lower vowel, 1 lower consonant, 2 upper, 3 space, 4 newline, 5 punctuation/other."""
    def cls(c):
        if c in "aeiou": return 0
        if c.islower(): return 1
        if c.isupper(): return 2
        if c == " ": return 3
        if c == "\n": return 4
        return 5
    tab = torch.tensor([cls(c) for c in m.itos], device=tok.device)
    return tab[tok]


def candidates(src):
    nl, sp = src.stoi["\n"], src.stoi[" "]
    c = [(f"kgram{k}", "kgram_next", dict(k=k)) for k in (1, 2, 3, 4, 6, 8)]
    c += [("match_len", "match_len", {}), ("since_nl", "since", dict(value=nl)),
          ("since_sp", "since", dict(value=sp)), ("class", "char_class", {})]
    c += [(f"shift{k}", "shift", dict(k=k)) for k in (1, 2, 3)]
    return c


def add_candidate(model, cand, site):
    name, fn, kw = cand
    if name not in model.domains:
        model.add_op(fn, ["tok"], name, **kw)
    return model.add_reader(name, site)


def lm_loss(model, x):
    return F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())


@torch.no_grad()
def eval_loss(model, X):
    model.eval()
    out = sum(lm_loss(model, X[i:i + 256]).item() * len(X[i:i + 256]) for i in range(0, len(X), 256)) / len(X)
    model.train()
    return out


def gradscores(model, cands, src, n_batches=4, bs=64, T=64, seed=0):
    """One copy of the model with a zero-init reader for every (candidate, site); score by
    centred gradient energy of each reader table."""
    probe = copy.deepcopy(model)
    taken = {(r.slot, r.site) for r in model.readers}
    pairs = [(c, s) for c in cands for s in range(probe.n_layers + 1) if (c[0], s) not in taken]
    for p in probe.parameters():
        p.requires_grad_(False)
    rd = [add_candidate(probe, c, s) for c, s in pairs]
    g = torch.Generator().manual_seed(seed)
    acc = [torch.zeros_like(r.table) for r in rd]
    for _ in range(n_batches):
        x = src.sample(bs, T + 1, g)
        for r in rd:
            r.table.grad = None
        lm_loss(probe, x).backward()
        regs = probe.last["regs"]
        for i, r in enumerate(rd):
            G = r.table.grad
            n = torch.bincount(regs[r.slot].flatten().clamp(0, r.domain - 1), minlength=r.domain).float()
            gbar = G.sum(0) / n.sum()
            acc[i] += G - n[:, None] * gbar[None]
    return {(c[0], s): float((a ** 2).sum()) * 1e4 for (c, s), a in zip(pairs, acc)}


def fitscore(model, cand, site, src, X, steps=150, bs=64, T=64, seed=0):
    """Held-out loss drop from training only this candidate's reader on the frozen model."""
    m2 = copy.deepcopy(model)
    for p in m2.parameters():
        p.requires_grad_(False)
    r = add_candidate(m2, cand, site)
    opt = torch.optim.Adam(r.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(seed + 1)
    base = eval_loss(m2, X)
    for _ in range(steps):
        loss = lm_loss(m2, src.sample(bs, T + 1, g))
        opt.zero_grad(); loss.backward(); opt.step()
    return base - eval_loss(m2, X)


def make(src, arch, d, seed):
    torch.manual_seed(seed)
    m = Hybrid(src.vocab, d, parse_arch(arch), 8, V=src.vocab)
    m.itos = src.itos
    return m


def run(a):
    torch.set_num_threads(a.threads)
    src = CharSource("data/tinyshakespeare.txt")
    X = src.sample(512, a.T + 1, torch.Generator().manual_seed(999), split="val")
    Xs = src.sample(256, a.T + 1, torch.Generator().manual_seed(4242), split="train")  # selection set
    model = make(src, a.arch, a.d, a.seed)
    cands = candidates(src)
    if a.arm == "all":
        for c in cands:
            add_candidate(model, c, 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    f = (out / "log.jsonl").open("w")
    g = torch.Generator().manual_seed(a.seed)
    t0, sel_time, inserted = time.time(), 0.0, []
    for step in range(a.steps + 1):
        check = a.arm == "greedy" or (a.arm == "late" and step >= a.late_at)
        if check and step % a.every == 0 and step < a.steps and len(inserted) < a.max_insert:
            ts = time.time()
            gs = gradscores(model, cands, src, seed=step)
            top = sorted(gs, key=gs.get, reverse=True)[:a.k_fit]
            byname = {c[0]: c for c in cands}
            fits = {k: fitscore(model, byname[k[0]], k[1], src, Xs, seed=step) for k in top}
            best = max(fits, key=fits.get)
            rec = dict(step=step, event="check", top=[[k[0], k[1], round(gs[k], 3), round(fits[k], 4)] for k in top])
            if fits[best] > a.min_gain:
                r = add_candidate(model, byname[best[0]], best[1])
                opt.add_param_group(dict(params=list(r.parameters())))
                inserted.append(list(best))
                rec["insert"] = list(best)
            sel_time += time.time() - ts
            f.write(json.dumps(rec) + "\n"); f.flush()
        if step % a.eval_every == 0:
            pc = model.param_counts()
            rec = dict(step=step, event="eval", loss=round(eval_loss(model, X), 4), core=pc["core"],
                       interface=pc["interface"], inserted=inserted, sel_time=round(sel_time, 1),
                       time=round(time.time() - t0, 1))
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps(rec), flush=True)
        if step == a.steps:
            break
        x = src.sample(a.bs, a.T + 1, g)
        loss = lm_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def validate(a):
    """At a few checkpoints of a no-code run: gradscore vs fitscore vs real trial gain
    (insert + train the whole model --trial steps, minus a no-insert control)."""
    torch.set_num_threads(a.threads)
    src = CharSource("data/tinyshakespeare.txt")
    X = src.sample(512, a.T + 1, torch.Generator().manual_seed(999), split="val")
    Xs = src.sample(256, a.T + 1, torch.Generator().manual_seed(4242), split="train")
    model = make(src, a.arch, a.d, a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    cands = candidates(src)
    g = torch.Generator().manual_seed(a.seed)
    res, step = [], 0
    for ck in [int(s) for s in a.checkpoints.split(",")]:
        while step < ck:
            loss = lm_loss(model, src.sample(a.bs, a.T + 1, g))
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        gs = gradscores(model, cands, src, seed=ck)

        def trial(cs):
            m2, o2 = copy.deepcopy((model, opt))
            if cs is not None:
                r = add_candidate(m2, cs[0], cs[1])
                o2.add_param_group(dict(params=list(r.parameters())))
            g2 = torch.Generator().manual_seed(77 + ck)
            for _ in range(a.trial):
                l2 = lm_loss(m2, src.sample(a.bs, a.T + 1, g2))
                o2.zero_grad(); l2.backward(); o2.step()
            return eval_loss(m2, X)
        ctrl = trial(None)
        for c in cands:
            for s in (0, model.n_layers):
                k = (c[0], s)
                res.append(dict(step=ck, cand=c[0], site=s, grad=round(gs[k], 4),
                                fit=round(fitscore(model, c, s, src, Xs, seed=ck), 4),
                                trial=round(ctrl - trial((c, s)), 4)))
                print(json.dumps(res[-1]), flush=True)
    out = Path(a.out_dir) / "validate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "validate"])
    ap.add_argument("--name", default="g")
    ap.add_argument("--arm", choices=["none", "all", "greedy", "late"], default="greedy")
    ap.add_argument("--out_dir", default="runs/greedy")
    ap.add_argument("--arch", default="2x2x64")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--late_at", type=int, default=3000)
    ap.add_argument("--k_fit", type=int, default=3)
    ap.add_argument("--min_gain", type=float, default=0.005)
    ap.add_argument("--max_insert", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--checkpoints", default="50,500,3000")
    ap.add_argument("--trial", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    {"run": run, "validate": validate}[a.cmd](a)


if __name__ == "__main__":
    main()
