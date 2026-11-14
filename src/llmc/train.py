"""Train a tiny GPT on a toy task, optionally injecting a code module mid-run.

    uv run python -m llmc.train --name baseline
    uv run python -m llmc.train --name inject --inject_step 2000 --program soft_cummax

Writes runs/<name>/{config.json, metrics.jsonl, summary.json, final.pt}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import GPT, GPTConfig
from .interface import probe_init_in, unembed_init_out
from .modules import CodeModule
from .programs import build_program
from .tasks import Batch, make_task


@dataclass
class Config:
    name: str = "run"
    seed: int = 0
    device: str = "cpu"  # tiny models run faster on CPU than MPS
    threads: int = 2  # per process; run several experiments in parallel
    # task
    task: str = "running_sum_mod"
    V: int = 5
    interleave: bool = False
    train_len_min: int = 2
    train_len_max: int = 16
    eval_splits: str = "id:2-16,ood:17-32,ood2:33-64"
    eval_size: int = 1024
    # model
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    pos: str = "none"
    # optim
    steps: int = 4000
    batch_size: int = 128
    lr: float = 1e-3
    module_lr: float = 3e-3
    weight_decay: float = 0.01
    warmup: int = 100
    eval_every: int = 200
    # injection
    inject_step: int = -1  # -1: never; 0: from the start
    program: str = "soft_summod"
    read: int = 0
    write: int = -1  # -1: last site (n_layers)
    gate_init: float = 0.0
    init_in: str = "random"  # random | probe   (see interface.py)
    init_out: str = "random"  # random | unembed
    out_scale: float = 1.0
    out_dir: str = "runs"


def parse_splits(s: str) -> dict[str, tuple[int, int]]:
    out = {}
    for part in s.split(","):
        k, rng = part.split(":")
        a, b = rng.split("-")
        out[k] = (int(a), int(b))
    return out


def pick_device(d: str) -> torch.device:
    if d == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(d)


def loss_and_acc(logits, b: Batch):
    loss = F.cross_entropy(logits.flatten(0, 1), b.targets.flatten(), ignore_index=-100)
    pred = logits.argmax(-1)
    correct = (pred == b.targets) & b.mask
    tok_acc = correct.sum().item() / b.mask.sum().item()
    seq_acc = ((correct | ~b.mask).all(1)).float().mean().item()
    return loss, tok_acc, seq_acc


def make_eval_sets(task, cfg: Config):
    g = torch.Generator().manual_seed(10_000 + 0)  # fixed across runs
    sets = {}
    for k, (a, b) in parse_splits(cfg.eval_splits).items():
        lengths = torch.randint(a, b + 1, (cfg.eval_size,), generator=g)
        sets[k] = task.sample(lengths, g)
    return sets


@torch.no_grad()
def evaluate(model, task, sets, device, bs=256):
    model.eval()
    res = {}
    for k, full in sets.items():
        tl = ta = sa = 0.0
        n = full.tokens.shape[0]
        for i in range(0, n, bs):
            b = Batch(*(t[i:i + bs] for t in (full.tokens, full.targets, full.mask, full.lengths)))
            # trim padding to this chunk's max length
            T = task.max_len(int(b.lengths.max()))
            b = Batch(b.tokens[:, :T], b.targets[:, :T], b.mask[:, :T], b.lengths).to(device)
            loss, tok, seq = loss_and_acc(model(b.tokens), b)
            w = b.tokens.shape[0]
            tl += loss.item() * w; ta += tok * w; sa += seq * w
        res[k] = dict(loss=tl / n, tok_acc=ta / n, seq_acc=sa / n)
    model.train()
    return res


def make_module(cfg: Config, model: GPT, device):
    prog = build_program(cfg.program, cfg.V)
    write = model.cfg.n_layers if cfg.write < 0 else cfg.write
    m = CodeModule(cfg.d_model, prog, cfg.read, write, cfg.gate_init, name=cfg.program).to(device)
    return model.inject(m)


def lr_at(step, cfg: Config):
    if step < cfg.warmup:
        return (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def build_model(cfg: Config, task, device) -> GPT:
    mcfg = GPTConfig(vocab_size=task.vocab_size, max_len=task.max_len(64), d_model=cfg.d_model,
                     n_layers=cfg.n_layers, n_heads=cfg.n_heads, pos=cfg.pos)
    return GPT(mcfg).to(device)


def save_ckpt(path, cfg: Config, model: GPT):
    torch.save(dict(cfg=dataclasses.asdict(cfg), state=model.state_dict(),
                    injections=[m.describe() for m in model.injections]), path)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    task = make_task(cfg.task, V=cfg.V, interleave=cfg.interleave)
    model = build_model(cfg, task, device)
    for spec in ck["injections"]:
        c2 = dataclasses.replace(cfg, program=spec["name"], read=spec["read"], write=spec["write"])
        make_module(c2, model, device)
    model.load_state_dict(ck["state"])
    return cfg, task, model


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    device = pick_device(cfg.device)
    task = make_task(cfg.task, V=cfg.V, interleave=cfg.interleave)
    model = build_model(cfg, task, device)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    mf = (out / "metrics.jsonl").open("w")
    eval_sets = make_eval_sets(task, cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed)
    t0 = time.time()
    for step in range(cfg.steps + 1):
        if step == cfg.inject_step:
            if step > 0:
                save_ckpt(out / "pre_inject.pt", cfg, model)
            m = make_module(cfg, model, device)
            if cfg.init_in == "probe":
                pb = task.sample(torch.randint(cfg.train_len_min, cfg.train_len_max + 1, (512,), generator=g), g).to(device)
                print(f"[{cfg.name}] probe-init W_in, fit acc {probe_init_in(m, model, task, pb):.3f}")
            if cfg.init_out == "unembed":
                unembed_init_out(m, model, task, cfg.out_scale)
            gate = [m.gate]
            rest = [p for n, p in m.named_parameters() if n != "gate"]
            opt.add_param_group(dict(params=rest, lr=cfg.module_lr, weight_decay=cfg.weight_decay))
            opt.add_param_group(dict(params=gate, lr=cfg.module_lr, weight_decay=0.0))
            for grp in opt.param_groups[-2:]:
                grp["base_lr"] = grp["lr"]
            print(f"[{cfg.name}] step {step}: injected {m.describe()}")
        for grp in opt.param_groups:
            grp.setdefault("base_lr", grp["lr"])
            grp["lr"] = grp["base_lr"] * lr_at(step, cfg)

        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = evaluate(model, task, eval_sets, device)
            rec = dict(step=step, time=round(time.time() - t0, 1), eval=ev,
                       gates={m.name: float(m.gate.detach()) for m in model.injections})
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            msg = " ".join(f"{k}={v['seq_acc']:.3f}/{v['tok_acc']:.3f}" for k, v in ev.items())
            print(f"[{cfg.name}] step {step:5d} {msg} gates={rec['gates']} ({rec['time']}s)", flush=True)
        if step == cfg.steps:
            break

        lengths = torch.randint(cfg.train_len_min, cfg.train_len_max + 1, (cfg.batch_size,), generator=g)
        b = task.sample(lengths, g).to(device)
        loss, _, _ = loss_and_acc(model(b.tokens), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    summary = dict(name=cfg.name, final=evaluate(model, task, eval_sets, device),
                   neural_params=model.neural_params(),
                   module_params=sum(p.numel() for p in model.injections.parameters()),
                   injections=[m.describe() for m in model.injections])
    if model.injections:
        for m in model.injections:
            m.enabled = False
        summary["ablated"] = evaluate(model, task, eval_sets, device)
        for m in model.injections:
            m.enabled = True
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    save_ckpt(out / "final.pt", cfg, model)
    print(f"[{cfg.name}] done in {time.time() - t0:.0f}s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        typ = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(f.default, bool) else type(f.default)
        ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    train(Config(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
