"""Curriculum ladder: continual learning with a minimal core + growing code.

Each *stage* adds one task. The teacher (Claude Code) may first apply a *plan*:
add code ops / readers / decoders to the register file, or grow the core. Then
the model trains on the new task with replay of every earlier task, and we check
that earlier tasks did not regress.

    # stage 0 from scratch
    uv run python -m llmc.ladder stage --out runs/m3/s0 --task summod --arch "1x1x8" --d 16
    # next stage from a checkpoint, with a plan
    uv run python -m llmc.ladder stage --ckpt runs/m3/s0/model.pt --out runs/m3/s1 --task max \\
        --plan '[{"op":"cummax","inputs":["x"],"out":"mx"},{"reader":"mx","site":1}]'
    uv run python -m llmc.ladder diagnose runs/m3/s1/model.pt

arch "LxHxM" = L layers, H heads each, M MLP neurons each (or "1x1x8,1x0x4").
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import growth
from .hybrid import Hybrid
from .multitask import MultiScan
from .tasks import Batch
from .train import loss_and_acc

LADDER = ["summod", "max", "lag1", "summod_pair", "summod_even", "depth", "count_eq", "lag2", "induct"]
V = 5
TRAIN_LEN = (2, 16)
OOD_LEN = (17, 32)


def make_task():
    return MultiScan(V, tuple(LADDER))


def parse_arch(s: str):
    blocks = []
    for part in s.split(","):
        L, H, M = map(int, part.split("x"))
        blocks += [(H, M)] * L
    return blocks


# ------------------------------------------------------------------ checkpoint
def save(path, model: Hybrid, meta: dict):
    torch.save(dict(
        d=model.d, hd=model.hd, arch=model.arch(), vocab=model.tok_emb.num_embeddings,
        ops=[o.__dict__ for o in model.ops],
        readers=[dict(slot=r.slot, site=r.site) for r in model.readers],
        decoders=[dict(slot=d.slot, domain=d.domain, site=d.site, labels=d.labels) for d in model.decoders],
        state=model.state_dict(), meta=meta), path)


def load(path):
    ck = torch.load(path, weights_only=False)
    m = Hybrid(ck["vocab"], ck["d"], ck["arch"], ck["hd"], V)
    for o in ck["ops"]:
        m.add_op(o["name"], o["inputs"], o["out"], **o["kwargs"])
    for dcd in ck["decoders"]:
        m.add_decoder(**dcd)
    for r in ck["readers"]:
        m.add_reader(r["slot"], r["site"])
    m.load_state_dict(ck["state"])
    return m, ck["meta"]


# ------------------------------------------------------------------ plan
def apply_plan(model: Hybrid, plan: list[dict], log: list):
    for a in plan:
        if "op" in a:
            model.add_op(a["op"], a["inputs"], a["out"], **a.get("kwargs", {}))
        elif "reader" in a:
            model.add_reader(a["reader"], a["site"])
        elif "decoder" in a:
            model.add_decoder(a["decoder"], a["domain"], a["site"], a["labels"])
        elif a.get("grow") == "mlp":
            growth.add_mlp(model, None, a["layer"], a["k"])
        elif a.get("grow") == "heads":
            growth.add_heads(model, None, a["layer"], a["k"])
        elif a.get("grow") == "layer":
            growth.add_layer(model, None, a["at"], a.get("heads", 1), a.get("mlp", 0))
        else:
            raise ValueError(a)
        log.append(a)


# ------------------------------------------------------------------ eval
_EVAL = {}


def eval_sets(task, n=256):
    if not _EVAL:
        g = torch.Generator().manual_seed(777)
        for t in task.tasks:
            _EVAL[(t, "id")] = task.sample(torch.randint(*TRAIN_LEN, (n,), generator=g) + 0, g, task=t)
            _EVAL[(t, "ood")] = task.sample(torch.randint(OOD_LEN[0], OOD_LEN[1] + 1, (n,), generator=g), g, task=t)
    return _EVAL


@torch.no_grad()
def evaluate(model, task, tasks):
    model.eval()
    out = {}
    for t in tasks:
        r = {}
        for sp in ("id", "ood"):
            b = eval_sets(task)[(t, sp)]
            _, tok, seq = loss_and_acc(model(b.tokens), b)
            r[sp] = round(seq, 4)
        out[t] = r
    model.train()
    return out


# ------------------------------------------------------------------ training
def contract_loss(model, task, batch):
    loss = 0.0
    for dec in model.decoders:
        lab = getattr(task, dec.labels)(batch)
        lg = model.last["dec_logits"][dec.slot]
        sel = lab >= 0
        if sel.any():
            loss = loss + F.cross_entropy(lg[sel], lab[sel])
    return loss


def mixed_batch(task, g, new, old, n, replay):
    lengths = torch.randint(TRAIN_LEN[0], TRAIN_LEN[1] + 1, (n,), generator=g)
    if not old:
        return task.sample(lengths, g, task=new)
    k = int(replay * n)
    ba = task.sample(lengths[:k], g, task=old)
    bb = task.sample(lengths[k:], g, task=new)
    T = max(ba.tokens.shape[1], bb.tokens.shape[1])
    pad = lambda t, v: F.pad(t, (0, T - t.shape[1]), value=v)
    return Batch(torch.cat([pad(ba.tokens, task.PAD), pad(bb.tokens, task.PAD)]),
                 torch.cat([pad(ba.targets, -100), pad(bb.targets, -100)]),
                 torch.cat([pad(ba.mask, False), pad(bb.mask, False)]),
                 torch.cat([ba.lengths, bb.lengths]))


def stage(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    task = make_task()
    if args.ckpt:
        model, meta = load(args.ckpt)
    else:
        model = Hybrid(task.vocab_size, args.d, parse_arch(args.arch), args.head_dim, V)
        meta = dict(learned=[], history=[])
    learned = list(meta["learned"])
    plan_log = []
    apply_plan(model, json.loads(args.plan) if args.plan else [], plan_log)
    all_tasks = learned + ([args.task] if args.task not in learned else [])
    before = evaluate(model, task, all_tasks)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve = []
    t0 = time.time()
    solved_at = None
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, task, all_tasks)
            curve.append(dict(step=step, eval=ev))
            new = ev[args.task]
            worst_old = min([ev[t]["id"] - before[t]["id"] for t in learned], default=0.0)
            worst_old_ood = min([ev[t]["ood"] - before[t]["ood"] for t in learned], default=0.0)
            print(f"[{args.task}] {step:5d} new={new['id']:.3f}/{new['ood']:.3f} "
                  f"worst_old_delta={worst_old:+.3f} ({time.time() - t0:.0f}s)", flush=True)
            if (new["id"] >= 0.99 and new["ood"] >= args.ood_target and step > 0
                    and worst_old >= -args.max_regress and worst_old_ood >= -args.max_regress):
                solved_at = step
                break
        if step == args.steps:
            break
        lr = args.lr * min(1, (step + 1) / 50)
        for gr in opt.param_groups:
            gr["lr"] = lr
        b = mixed_batch(task, g, args.task, learned, args.batch_size, args.replay)
        loss, _, _ = loss_and_acc(model(b.tokens), b)
        if model.decoders:
            loss = loss + args.contract_weight * contract_loss(model, task, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    after = evaluate(model, task, all_tasks)
    rec = dict(task=args.task, plan=plan_log, steps=step, solved_at=solved_at,
               before=before, after=after, params=model.param_counts(), arch=model.arch(),
               regress={t: round(after[t]["id"] - before[t]["id"], 4) for t in learned},
               regress_ood={t: round(after[t]["ood"] - before[t]["ood"], 4) for t in learned},
               time=round(time.time() - t0, 1))
    meta = dict(learned=all_tasks if (solved_at or args.force_learned) else learned,
                history=meta["history"] + [rec])
    save(out / "model.pt", model, meta)
    (out / "stage.json").write_text(json.dumps(rec, indent=2))
    (out / "curve.jsonl").write_text("\n".join(json.dumps(c) for c in curve))
    print(json.dumps(dict(task=args.task, solved_at=solved_at, after=after, params=rec["params"],
                          arch=rec["arch"], regress=rec["regress"], regress_ood=rec["regress_ood"])))


# ------------------------------------------------------------------ diagnose
@torch.no_grad()
def diagnose(args):
    task = make_task()
    model, meta = load(args.ckpt)
    tasks = meta["learned"] + ([args.task] if args.task and args.task not in meta["learned"] else [])
    print("arch", model.arch(), "params", model.param_counts())
    print("ops", [(o.name, o.inputs, o.out) for o in model.ops])
    print("readers", [(r.slot, r.site) for r in model.readers], "decoders", [(d.slot, d.site) for d in model.decoders])
    print("eval", evaluate(model, task, tasks))
    acts = {}
    for t in tasks:
        b = eval_sets(task)[(t, "id")]
        model(b.tokens, capture=True)
        resid = model.last["resid"]
        gates = {f"{r.slot}@{r.site}": round(r.gate(resid[r.site] if r.site == 0 else resid[r.site])[..., 0][b.mask].mean().item(), 3)
                 for r in model.readers}
        for i, blk in enumerate(model.blocks):
            if blk.n_mlp:
                acts.setdefault(i, []).append(blk.last_act.abs()[b.mask].mean(0))
        print(f"  {t:12s} reader gates (approx, post-write stream) {gates}")
    for i, us in acts.items():
        u = torch.stack(us).max(0).values
        print(f"layer {i} mlp: {len(u)} neurons, used(max over tasks >0.1*mean) {(u > 0.1 * u.mean()).sum().item()}, "
              f"utilization dist {[round(v, 2) for v in u.sort(descending=True).values.tolist()]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--ckpt")
    s.add_argument("--out", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--arch", default="1x1x8")
    s.add_argument("--d", type=int, default=16)
    s.add_argument("--head_dim", type=int, default=8)
    s.add_argument("--plan", default="")
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--eval_every", type=int, default=100)
    s.add_argument("--batch_size", type=int, default=128)
    s.add_argument("--lr", type=float, default=3e-3)
    s.add_argument("--replay", type=float, default=0.5)
    s.add_argument("--contract_weight", type=float, default=0.3)
    s.add_argument("--ood_target", type=float, default=0.95)
    s.add_argument("--max_regress", type=float, default=0.01)
    s.add_argument("--force_learned", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=4)
    d = sub.add_parser("diagnose")
    d.add_argument("ckpt")
    d.add_argument("--task")
    a = ap.parse_args()
    (stage if a.cmd == "stage" else diagnose)(a)


if __name__ == "__main__":
    main()
