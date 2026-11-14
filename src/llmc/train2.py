"""Milestone 2: does moving skill A into code free neural capacity for skill B?

Phase A (steps_A): train on tasks_A. At inject_step, insert a code module with a
routed gate and teacher-designed interface. From inject_step on, apply the
substitution `pressure` arm.
Phase B (steps_B): train on tasks_B with `replay` fraction of A, pressure on.

    uv run python -m llmc.train2 --name m2_test --program py_summod --pressure sparsity

Key outputs per eval: per-task seq acc (id / ood), effective core size (neuron
and head groups above a norm threshold), contract agreement and refits, gates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Contract
from .interface import probe_init_in, unembed_init_out
from .model import GPT, GPTConfig
from .modules import CodeModule
from .multitask import MultiScan
from .programs import build_program
from .tasks import Batch
from .train import loss_and_acc


@dataclass
class Config2:
    name: str = "m2"
    seed: int = 0
    threads: int = 2
    out_dir: str = "runs/m2"
    # tasks
    V: int = 5
    tasks_A: str = "summod,max"
    tasks_B: str = "induct,count_eq,lag2"
    train_len_min: int = 2
    train_len_max: int = 16
    ood_len: str = "17-32"
    eval_size: int = 256
    # model
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    mlp_mult: int = 4
    # schedule
    steps_A: int = 3000
    steps_B: int = 3000
    replay: float = 0.3
    batch_size: int = 128
    lr: float = 1e-3
    module_lr: float = 3e-3
    weight_decay: float = 0.01
    warmup: int = 100
    eval_every: int = 500
    # module
    program: str = "none"  # none | soft_summod | py_summod | control ...
    inject_step: int = 1000
    read: int = 0
    write: int = -1
    gate_mode: str = "routed"
    gate_init: float = 0.0
    out_scale: float = 20.0
    # contract
    contract: str = "truth"  # none | truth | anchor
    contract_weight: float = 0.1
    # pressure (from inject_step on; applied even with program=none for fair comparison)
    # none | sparsity (inject..end) | window (inject..end of A) | window+recycle
    # (window, then reinit all pruned units fresh at the start of B) | redo (activation-based)
    pressure: str = "none"
    l1: float = 3e-4  # group-lasso coefficient
    redo_every: int = 250
    redo_tau: float = 0.05
    prune_eps: float = 0.05  # group-norm threshold for counting a unit as alive


# ---------------------------------------------------------------- core groups
def unit_groups(model: GPT):
    """Yield (kind, layer, index, [tensor slices...]) for each prunable unit:
    MLP hidden neurons and attention heads."""
    out = []
    for li, blk in enumerate(model.blocks):
        W1, b1, W2 = blk.mlp[0].weight, blk.mlp[0].bias, blk.mlp[2].weight
        out.append(("mlp", li, torch.cat([W1, b1[:, None], W2.T], 1)))  # (H, 2d+1) rows = neurons
        D, H = blk.qkv.weight.shape[1], blk.n_heads
        hd = D // H
        q, k, v = blk.qkv.weight.split(D, 0)
        per_head = torch.cat([t.view(H, hd, D).flatten(1) for t in (q, k, v)]
                             + [blk.proj.weight.view(D, H, hd).permute(1, 0, 2).flatten(1)], 1)
        out.append(("head", li, per_head))  # (H, 4*hd*D)
    return out


def group_norms(model):
    return {(k, li): rows.norm(dim=1) for k, li, rows in unit_groups(model)}


def group_lasso(model):
    return sum(rows.norm(dim=1).sum() for _, _, rows in unit_groups(model))


@torch.no_grad()
def core_size(model, eps):
    """Effective core: params in alive units + always-on params (embeddings, LNs,
    biases of attention/proj/MLP out, unembedding)."""
    alive, total, per = 0, 0, {}
    for k, li, rows in unit_groups(model):
        n = rows.norm(dim=1)
        a = int((n > eps).sum())
        per[f"{k}{li}"] = a
        alive += a * rows.shape[1]
        total += rows.numel()
    fixed = model.neural_params() - total
    return dict(alive_units=per, core_params=alive + fixed, full_params=model.neural_params())


# ---------------------------------------------------------------- ReDo
class ActStats:
    def __init__(self, model):
        self.sums, self.hooks = {}, []
        for li, blk in enumerate(model.blocks):
            self.hooks.append(blk.mlp[1].register_forward_hook(self._hook(li)))

    def _hook(self, li):
        def f(mod, inp, out):
            if torch.is_grad_enabled():
                s = out.detach().abs().flatten(0, 1).mean(0)
                self.sums[li] = self.sums.get(li, 0) + s
        return f

    def reset(self):
        self.sums = {}


@torch.no_grad()
def redo(model, stats: ActStats, opt, tau: float):
    n_reset = 0
    for li, s in stats.sums.items():
        score = s / (s.mean() + 1e-8)
        dead = score <= tau
        if not dead.any():
            continue
        blk = model.blocks[li]
        W1, b1, W2 = blk.mlp[0].weight, blk.mlp[0].bias, blk.mlp[2].weight
        fresh = torch.empty_like(W1)
        nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
        W1[dead] = fresh[dead]
        b1[dead] = 0
        W2[:, dead] = 0
        for p, idx in ((W1, (dead,)), (b1, (dead,)), (W2, (slice(None), dead))):
            st = opt.state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key][idx] = 0
        n_reset += int(dead.sum())
    stats.reset()
    return n_reset


def _zero_adam(opt, p, idx):
    st = opt.state.get(p, {})
    for key in ("exp_avg", "exp_avg_sq"):
        if key in st:
            st[key][idx] = 0


@torch.no_grad()
def recycle(model, opt, eps):
    """Reinitialize every unit whose group norm is below eps: fresh input
    weights, zero output weights (so the function is unchanged at reset)."""
    counts = {}
    for li, blk in enumerate(model.blocks):
        W1, b1, W2 = blk.mlp[0].weight, blk.mlp[0].bias, blk.mlp[2].weight
        dead = torch.cat([W1, b1[:, None], W2.T], 1).norm(dim=1) <= eps
        fresh = torch.empty_like(W1)
        nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
        W1[dead], b1[dead], W2[:, dead] = fresh[dead], 0, 0
        for p, idx in ((W1, (dead,)), (b1, (dead,)), (W2, (slice(None), dead))):
            _zero_adam(opt, p, idx)
        counts[f"mlp{li}"] = int(dead.sum())
        D, H = blk.qkv.weight.shape[1], blk.n_heads
        hd = D // H
        norms = unit_groups(model)[2 * li + 1][2].norm(dim=1)
        fresh = torch.empty_like(blk.qkv.weight)
        nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
        nh = 0
        for h in range(H):
            if norms[h] <= eps:
                for part in range(3):
                    sl = slice(part * D + h * hd, part * D + (h + 1) * hd)
                    blk.qkv.weight[sl] = fresh[sl]
                    blk.qkv.bias[sl] = 0
                    _zero_adam(opt, blk.qkv.weight, (sl,))
                blk.proj.weight[:, h * hd:(h + 1) * hd] = 0
                _zero_adam(opt, blk.proj.weight, (slice(None), slice(h * hd, (h + 1) * hd)))
                nh += 1
        counts[f"head{li}"] = nh
    return counts


# ---------------------------------------------------------------- eval
def make_evals(task: MultiScan, cfg: Config2):
    g = torch.Generator().manual_seed(12345)
    a, b = map(int, cfg.ood_len.split("-"))
    sets = {}
    for t in task.tasks:
        sets[(t, "id")] = task.sample(torch.randint(cfg.train_len_min, cfg.train_len_max + 1, (cfg.eval_size,), generator=g), g, task=t)
        sets[(t, "ood")] = task.sample(torch.randint(a, b + 1, (cfg.eval_size,), generator=g), g, task=t)
    return sets


@torch.no_grad()
def evaluate(model, sets):
    model.eval()
    res = {}
    for (t, sp), b in sets.items():
        logits, resid = model(b.tokens, capture=True)
        _, tok, seq = loss_and_acc(logits, b)
        r = res.setdefault(t, {})
        r[sp] = round(seq, 4)
        r[sp + "_tok"] = round(tok, 4)
        for m in model.injections:
            if m.gate_mode == "routed":
                # resid[write] is post-write; router sees pre-write in forward, close enough for logging
                gv = m.gate_value(resid[m.write])[..., 0]
                r[f"gate_{m.name}_{sp}"] = round(gv[b.mask].mean().item(), 4)
    model.train()
    return res


def lr_mult(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


# ---------------------------------------------------------------- main loop
def train(cfg: Config2):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    A, B = cfg.tasks_A.split(","), cfg.tasks_B.split(",")
    task = MultiScan(cfg.V, tuple(A + B))
    mcfg = GPTConfig(vocab_size=task.vocab_size, max_len=128, d_model=cfg.d_model, n_layers=cfg.n_layers,
                     n_heads=cfg.n_heads, mlp_mult=cfg.mlp_mult)
    model = GPT(mcfg)
    model._pad_id = task.PAD
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    mf = (out / "metrics.jsonl").open("w")
    sets = make_evals(task, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed)
    total = cfg.steps_A + cfg.steps_B
    contract, stats, events = None, None, []
    t0 = time.time()

    def sample(names, n):
        lengths = torch.randint(cfg.train_len_min, cfg.train_len_max + 1, (n,), generator=g)
        return task.sample(lengths, g, task=names)

    for step in range(total + 1):
        pressure_on = step >= cfg.inject_step and cfg.pressure != "none"
        if cfg.pressure.startswith("window") and step >= cfg.steps_A:
            pressure_on = False
        if step == cfg.steps_A and cfg.pressure == "window+recycle":
            n = recycle(model, opt, cfg.prune_eps)
            events.append(dict(step=step, event="recycle", **n))
            print(f"[{cfg.name}] step {step}: recycled {n}", flush=True)
        if step == cfg.inject_step:
            if cfg.program != "none":
                prog = build_program(cfg.program, cfg.V)
                write = cfg.n_layers if cfg.write < 0 else cfg.write
                m = model.inject(CodeModule(cfg.d_model, prog, cfg.read, write, cfg.gate_init,
                                            name=cfg.program, gate_mode=cfg.gate_mode))
                pb = sample(A, 512)
                fit = probe_init_in(m, model, task, pb)
                unembed_init_out(m, model, task, cfg.out_scale)
                opt.add_param_group(dict(params=[p for n, p in m.named_parameters() if n != "gate"],
                                         lr=cfg.module_lr, weight_decay=cfg.weight_decay))
                opt.add_param_group(dict(params=[m.gate], lr=cfg.module_lr, weight_decay=0.0))
                if cfg.contract != "none":
                    contract = Contract.create(m, model, sample(A + B, 128), cfg.contract, task.code_inputs,
                                               weight=cfg.contract_weight)
                events.append(dict(step=step, event="inject", probe_fit=fit, **m.describe()))
                print(f"[{cfg.name}] step {step}: injected {m.describe()} probe_fit={fit:.3f}", flush=True)
            if "redo" in cfg.pressure:
                stats = ActStats(model)
        for grp in opt.param_groups:
            grp.setdefault("base_lr", grp["lr"])
            grp["lr"] = grp["base_lr"] * lr_mult(step, total, cfg.warmup)

        if step % cfg.eval_every == 0 or step == total:
            ev = evaluate(model, sets)
            rec = dict(step=step, phase="A" if step < cfg.steps_A else "B", time=round(time.time() - t0, 1),
                       eval=ev, core=core_size(model, cfg.prune_eps),
                       gates={m.name: float(m.gate.detach()) for m in model.injections})
            if contract is not None:
                rec["contract"] = contract.maintain(model)
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            short = " ".join(f"{t}={ev[t]['id']:.2f}/{ev[t]['ood']:.2f}" for t in task.tasks)
            print(f"[{cfg.name}] {step:5d} {rec['phase']} {short} core={rec['core']['core_params']}"
                  + (f" contract={rec['contract']['agree']:.3f}" if contract else ""), flush=True)
        if step == total:
            break

        if step < cfg.steps_A:
            b = sample(A, cfg.batch_size)
        else:
            nA = int(cfg.replay * cfg.batch_size)
            ba, bb = sample(A, nA), sample(B, cfg.batch_size - nA)
            T = max(ba.tokens.shape[1], bb.tokens.shape[1])
            pad = lambda t, v: F.pad(t, (0, T - t.shape[1]), value=v)
            b = Batch(torch.cat([pad(ba.tokens, task.PAD), pad(bb.tokens, task.PAD)]),
                      torch.cat([pad(ba.targets, -100), pad(bb.targets, -100)]),
                      torch.cat([pad(ba.mask, False), pad(bb.mask, False)]),
                      torch.cat([ba.lengths, bb.lengths]))
        loss, _, _ = loss_and_acc(model(b.tokens), b)
        if contract is not None:
            loss = loss + contract.loss(model)
        if pressure_on and ("sparsity" in cfg.pressure or cfg.pressure.startswith("window")):
            loss = loss + cfg.l1 * group_lasso(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if pressure_on and stats is not None and step % cfg.redo_every == 0 and step > cfg.inject_step:
            n = redo(model, stats, opt, cfg.redo_tau)
            if n:
                events.append(dict(step=step, event="redo", reset=n))

    final = evaluate(model, sets)
    summary = dict(name=cfg.name, final=final, core=core_size(model, cfg.prune_eps), events=events,
                   refits=contract.refits if contract else 0)
    if model.injections:
        for m in model.injections:
            m.enabled = False
        summary["ablated"] = evaluate(model, sets)
        for m in model.injections:
            m.enabled = True
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(dict(cfg=dataclasses.asdict(cfg), state=model.state_dict()), out / "final.pt")
    print(f"[{cfg.name}] done in {time.time() - t0:.0f}s", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config2):
        typ = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(f.default, bool) else type(f.default)
        ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    train(Config2(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
