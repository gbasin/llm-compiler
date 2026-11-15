"""Train a hybrid (or pure) model on messy-language word problems (direction 4).

Contract label functions (teacher-known, from generator metadata):
  role_labels: at the last char of each number mention -> 1 (operand a), 2 (b), 3 (distractor);
               0 at every other position.
  op_labels:   op index (0 add, 1 sub, 2 mul) at every position from the 'A' of "A:" on; else -1.

    uv run python -m llmc.wm_train --name pure --arch 2x4x128
    uv run python -m llmc.wm_train --name hyb --arch 2x4x64 --plan ref
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .hybrid import Hybrid
from .ladder import apply_plan, parse_arch
from .wordmath import OPS, WordMath

REF_PLAN = [
    {"op": "parse_num", "inputs": ["tok"], "out": "num"},
    {"decoder": "role", "domain": 4, "site": -1, "labels": "role_labels"},
    {"op": "hold", "inputs": ["num", "role"], "out": "a", "kwargs": {"r": 1}},
    {"op": "hold", "inputs": ["num", "role"], "out": "b", "kwargs": {"r": 2}},
    {"decoder": "opc", "domain": 3, "site": -1, "labels": "op_labels"},
    {"op": "arith", "inputs": ["a", "b", "opc"], "out": "res"},
    {"op": "since_char", "inputs": ["tok"], "out": "sc", "kwargs": {"ch": ":"}},
    {"op": "offset", "inputs": ["sc"], "out": "pos", "kwargs": {"k": -1}},
    {"op": "render", "inputs": ["res", "pos"], "out": "ch"},
    {"logit_slot": "ch"},
]


def role_labels(probs, T):
    lab = torch.zeros(len(probs), T, dtype=torch.long)
    for i, p in enumerate(probs):
        lab[i, len(p.text):] = -1
        for s, e, v, r in p.spans:
            lab[i, e - 1] = {"a": 1, "b": 2, "distractor": 3}[r]
    return lab


def op_labels(probs, T):
    lab = torch.full((len(probs), T), -1, dtype=torch.long)
    for i, p in enumerate(probs):
        a_pos = p.text.rindex("A:")
        lab[i, a_pos:len(p.text)] = OPS.index(p.op)
    return lab


LABELS = dict(role_labels=role_labels, op_labels=op_labels)


def build(args, wm):
    model = Hybrid(wm.vocab, args.d, parse_arch(args.arch), args.head_dim, V=wm.vocab)
    model.itos = wm.itos
    plan = REF_PLAN if args.plan == "ref" else (json.loads(args.plan) if args.plan else [])
    for a in plan:
        a = dict(a)
        if "decoder" in a and a["site"] < 0:
            a["site"] = model.n_layers
        if "logit_slot" in a:
            model.add_logit_slot(a["logit_slot"])
        else:
            apply_plan(model, [a], [])
    return model, plan


def evaluate(model, wm, n=300, seed=123):
    out = {}
    for name, kw in dict(id=dict(), distract=dict(p_distract=1.0), big=dict(max_n=50)).items():
        rng = random.Random(seed)
        _, _, probs = wm.batch(n, rng, **kw)
        out[name] = round(wm.exact_match(model, probs), 4)
    return out


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    wm = WordMath()
    model, plan = build(args, wm)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed)
    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            ev = evaluate(model, wm)
            model.train()
            rec = dict(step=step, eval=ev, time=round(time.time() - t0, 1))
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if step == args.steps:
            break
        tokens, targets, probs = wm.batch(args.batch_size, rng)
        logits = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        for dec in model.decoders:
            lab = LABELS[dec.labels](probs, tokens.shape[1])
            lg = model.last["dec_logits"][dec.slot]
            sel = lab >= 0
            loss = loss + args.contract_weight * F.cross_entropy(lg[sel], lab[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    res = dict(name=args.name, plan=args.plan[:40], params=model.param_counts(), arch=model.arch(),
               final=log[-1]["eval"], log=log)
    (out / "result.json").write_text(json.dumps(res, indent=2))
    torch.save(model.state_dict(), out / "model.pt")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="runs/wm")
    ap.add_argument("--arch", default="2x4x128")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--head_dim", type=int, default=16)
    ap.add_argument("--plan", default="")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--contract_weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
