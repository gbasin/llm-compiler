"""Next-token LM training with grow-on-demand (and optional code in the register file).

Growth policy (trial-based): train in rounds. If the eval loss plateaus while
still above target, run a short trial for each candidate growth AND a
no-growth control from identical state and data. Grow only if some candidate
beats the control by > `min_gain`; pick the best gain per added parameter, and
adopt that trial's model. If nothing beats the control, the core isn't
capacity-bound: stop.

    uv run python -m llmc.lm --name hmm_grow --source hmm --arch 1x0x0
    uv run python -m llmc.lm --name tiny --source char:data/tinyshakespeare.txt --arch 1x1x8
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import growth
from .hybrid import Hybrid
from .ladder import apply_plan, parse_arch
from .lmdata import CharSource, HMMSource


def lm_loss(model, x):
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())


@torch.no_grad()
def eval_loss(model, X, bs=256):
    model.eval()
    tot = 0.0
    for i in range(0, len(X), bs):
        tot += lm_loss(model, X[i:i + bs]).item() * len(X[i:i + bs])
    model.train()
    return tot / len(X)


def candidates(model, k_mlp):
    c = [("none", {})]
    for i, (h, m) in enumerate(model.arch()):
        c.append((f"+head@{i}", dict(grow="heads", layer=i, k=1)))
        c.append((f"+{k_mlp}mlp@{i}", dict(grow="mlp", layer=i, k=k_mlp)))
    c.append(("+layer", dict(grow="layer", at=model.n_layers, heads=1, mlp=k_mlp)))
    return c


def apply_growth(model, opt, a):
    if a.get("grow") == "heads":
        growth.add_heads(model, opt, a["layer"], a["k"])
    elif a.get("grow") == "mlp":
        growth.add_mlp(model, opt, a["layer"], a["k"])
    elif a.get("grow") == "layer":
        growth.add_layer(model, opt, a["at"], a["heads"], a["mlp"])


def nparams(model):
    return model.param_counts()["core"]


def train_steps(model, opt, src, n, args, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        x = src.sample(args.batch_size, args.seq_len + 1, g)
        loss = lm_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="runs/m4")
    ap.add_argument("--source", default="hmm")  # hmm | hmm_copy | char:<path>
    ap.add_argument("--arch", default="1x0x0")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--head_dim", type=int, default=8)
    ap.add_argument("--plan", default="")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--round_steps", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--trial_steps", type=int, default=400)
    ap.add_argument("--k_mlp", type=int, default=8)
    ap.add_argument("--plateau", type=float, default=0.01)
    ap.add_argument("--min_gain", type=float, default=0.01)
    ap.add_argument("--target_gap", type=float, default=0.02)
    ap.add_argument("--no_grow", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    if args.source.startswith("hmm"):
        src = HMMSource(vocab=16, states=16, branch=2, emit_alpha=0.1, seed=0,
                        p_copy=0.5 if args.source == "hmm_copy" else 0.0)
        X = src.sample(1024, args.seq_len + 1, torch.Generator().manual_seed(999))
        floor = src.bayes_loss(X) if src.p_copy == 0 else float("nan")
    else:
        src = CharSource(args.source.split(":", 1)[1])
        X = src.sample(1024, args.seq_len + 1, torch.Generator().manual_seed(999), split="val")
        floor = float("nan")
    model = Hybrid(src.vocab, args.d, parse_arch(args.arch), args.head_dim, V=src.vocab)
    log_plan = []
    apply_plan(model, json.loads(args.plan) if args.plan else [], log_plan)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    mf = (out / "log.jsonl").open("w")
    t0 = time.time()
    hist = []
    step = 0
    L = eval_loss(model, X)

    def log(event, **kw):
        rec = dict(step=step, loss=round(L, 4), floor=round(floor, 4), core=nparams(model),
                   arch=model.arch(), event=event, time=round(time.time() - t0, 1), **kw)
        mf.write(json.dumps(rec) + "\n"); mf.flush()
        print(json.dumps(rec), flush=True)

    log("start", plan=log_plan)
    for r in range(args.rounds):
        train_steps(model, opt, src, args.round_steps, args, seed=args.seed * 1000 + r)
        step += args.round_steps
        prev, L = L, eval_loss(model, X)
        hist.append(L)
        log("round")
        if L - floor < args.target_gap:
            log("reached_target")
            break
        if args.no_grow or prev - L > args.plateau:
            continue
        # plateau: trial each candidate from identical state/data
        results = []
        for name, a in candidates(model, args.k_mlp):
            m2, o2 = copy.deepcopy((model, opt))
            before = nparams(m2)
            apply_growth(m2, o2, a)
            train_steps(m2, o2, src, args.trial_steps, args, seed=10_000 + step)
            results.append((name, a, eval_loss(m2, X), nparams(m2) - before, m2, o2))
        base = next(res for res in results if res[0] == "none")
        trials = {n: dict(loss=round(l, 4), added=dp) for n, _, l, dp, _, _ in results}
        best = None
        for res in results:
            n, a, l, dp, m2, o2 = res
            gain = base[2] - l
            if n != "none" and gain > args.min_gain and (best is None or gain / dp > best[0]):
                best = (gain / dp, res)
        step += args.trial_steps
        if best is None:
            model, opt = base[4], base[5]
            L = base[2]
            log("plateau_no_growth_helps", trials=trials)
            break
        n, a, l, dp, m2, o2 = best[1]
        model, opt, L = m2, o2, l
        log("grow", choice=n, trials=trials)

    torch.save(dict(arch=model.arch(), state=model.state_dict()), out / "model.pt")
    log("end")


if __name__ == "__main__":
    main()
