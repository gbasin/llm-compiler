"""Direction 3 on text: can a headless Claude Code agent rediscover n-gram tables and copying
(and find more) for Tiny Shakespeare, starting from a tiny core with no code?

The agent edits ops.py (ops, readers' inputs, and TABLE builders) and plan.json. Score =
validation loss (nats/char) of a fixed core (2 layers, 2 heads, 64 MLP, d=32) after a fixed
training budget. Lower is better; a change is kept if it lowers val loss by > 0.005.
The agent does NOT see llmc/ngram.py or the suffix ops (hidden).

    uv run python -m llmc.discover_lm run --ws runs/discover/shake1 --iters 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from . import oplib
from .hybrid import Hybrid
from .ladder import apply_plan, parse_arch
from .lmdata import CharSource

HIDDEN = ["suffix_next", "suffix_len", "parse_num", "hold", "arith", "render", "since_char", "offset"]
ARCH, D, HD = "2x2x64", 32, 8
STEPS, SEQ, BS = 4000, 64, 64

OPS_TEMPLATE = '''"""Agent-written ops and table builders for char-level Tiny Shakespeare. Edit freely.

Ops: @op(domain_fn) fn(m, *slots, **kw) -> LongTensor (B, T); CAUSAL. m.itos: id -> char.
Tables: TABLES[name] = builder(train: LongTensor (N,) of char ids, vocab: int) -> obj with
  obj.logprobs(tok: LongTensor (B,T), targets: LongTensor (B,T) | None) -> FloatTensor (B,T,vocab)
  = log p(next char | context up to t). `targets` is given during TRAINING ONLY (the true next
  chars); if your table is built from the training text it has memorized those targets, so use
  them to apply leave-one-out (remove the current occurrence) or the network will learn to
  over-trust it. Must be causal. At most one table is used (plan action {"table": name}); it is
  added to the output logits with a learned per-position weight.
"""
import torch

from llmc.oplib import op

TABLES = {}
'''

API = """
## Task: next-character prediction on Tiny Shakespeare (~1MB, 65 chars). Score = validation
loss in nats/char (last 10% of the text) after training a FIXED tiny core (2 layers x 2 heads x
64 MLP, d=32, ~17k params) for 4000 steps of batch 64 x 64 chars. Lower is better.
Baseline (no code) is about 2.2. You can't change the core; improve it with code.

## Plan actions (plan.json, a list):
  {"op": NAME, "inputs": [slots], "out": slot, "kwargs": {...}}  code op -> new int slot
  {"reader": slot, "site": i}   slot -> network via zero-init embedding + router at site 0..2
                                (small domains only, < ~200 values)
  {"table": NAME}               TABLES[NAME](train_ids, vocab) -> log-prob table added to logits
Slot "tok" = char ids. Generic ops available (src/llmc/oplib.py): xval, task, shift(k), eq, mask,
map(table), count_prefix_eq, prev_occ_next, isitem, since(value), cummax, add_mod, mul_mod,
cumsum_mod. Ignore other ops in that file. Write anything new in ops.py.
Training data for tables: the first 90% of the text (char ids); validation is the rest.
"""

PROMPT = """You are the teacher in a research loop that moves work from a tiny neural network into
code (ops, lookup tables). Lower the validation loss below by editing ONLY {ws}/ops.py and
{ws}/plan.json. Do not edit anything else in the repository.
{api}
## Current state
{state}
## Workflow
1. Read {ws}/report.md and {ws}/history.json.
2. Edit {ws}/ops.py and {ws}/plan.json (one focused idea per iteration).
3. Validate: uv run python -m llmc.discover_lm check --ws {ws}  (builds, runs a batch, prints
   the table-only val loss if a table is used). Fix errors until it passes. Keep ops fast
   (vectorized torch); a trial must finish in a few minutes.
4. Stop and summarize in 2-3 sentences what you changed and why.
"""


def load_mod(ws: Path):
    for n in HIDDEN:
        oplib.OPS.pop(n, None)
    spec = importlib.util.spec_from_file_location(f"agent_lm_{time.time_ns()}", ws / "ops.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(ws, src, mod):
    model = Hybrid(src.vocab, D, parse_arch(ARCH), HD, V=src.vocab)
    model.itos = src.itos
    plan = json.loads((ws / "plan.json").read_text())
    for a in plan:
        if "table" in a:
            model.add_logit_table(mod.TABLES[a["table"]](src.train, src.vocab))
        else:
            apply_plan(model, [a], [])
    return model, plan


def val_loss(model, X):
    model.eval()
    with torch.no_grad():
        tot = sum(F.cross_entropy(model(X[i:i + 128, :-1]).flatten(0, 1), X[i:i + 128, 1:].flatten()).item()
                  * len(X[i:i + 128]) for i in range(0, len(X), 128))
    model.train()
    return tot / len(X)


def trial(ws, seed=0):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    src = CharSource("data/tinyshakespeare.txt")
    X = src.sample(512, SEQ + 1, torch.Generator().manual_seed(999), split="val")
    mod = load_mod(ws)
    model, plan = build(ws, src, mod)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(STEPS):
        x = src.sample(BS, SEQ + 1, g)
        loss = F.cross_entropy(model(x[:, :-1], targets=x[:, 1:]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return dict(val_loss=round(val_loss(model, X), 4), params=model.param_counts(),
                train_time=round(time.time() - t0, 1))


def check(ws):
    src = CharSource("data/tinyshakespeare.txt")
    mod = load_mod(ws)
    model, plan = build(ws, src, mod)
    x = src.sample(8, SEQ + 1, torch.Generator().manual_seed(1))
    loss = F.cross_entropy(model(x[:, :-1], targets=x[:, 1:]).flatten(0, 1), x[:, 1:].flatten())
    loss.backward()
    print("OK: builds, forward/backward. params:", model.param_counts())
    if model.table is not None:
        X = src.sample(256, SEQ + 1, torch.Generator().manual_seed(999), split="val")
        lp = model.table.logprobs(X[:, :-1], None)
        print("table-only val loss:", round(F.nll_loss(lp.flatten(0, 1), X[:, 1:].flatten()).item(), 4))
        Xt = src.sample(256, SEQ + 1, torch.Generator().manual_seed(5), split="train")
        lpt = model.table.logprobs(Xt[:, :-1], Xt[:, 1:])
        print("table-only train loss with targets (should be close to val if leave-one-out works):",
              round(F.nll_loss(lpt.flatten(0, 1), Xt[:, 1:].flatten()).item(), 4))
    for k, v in model.last["regs"].items():
        if k != "tok":
            print(f"slot {k}: first row {v[0, :40].tolist()}")


def run(ws: Path, iters: int):
    ws.mkdir(parents=True, exist_ok=True)
    if not (ws / "ops.py").exists():
        (ws / "ops.py").write_text(OPS_TEMPLATE)
        (ws / "plan.json").write_text("[]")
    hp = ws / "history.json"
    hist = json.loads(hp.read_text()) if hp.exists() else []
    if not hist:
        res = trial(ws)
        hist.append(dict(iter=0, kept=True, summary="baseline: no code", **res))
        hp.write_text(json.dumps(hist, indent=2))
        report(ws, res, hist)
        print("baseline", res, flush=True)
    best = min(h["val_loss"] for h in hist if h["kept"])
    for it in range(len(hist), len(hist) + iters):
        for f in ("ops.py", "plan.json"):
            shutil.copy(ws / f, ws / (f + ".prev"))
        state = f"Iteration {it}. Best val loss so far: {best:.4f}."
        t0 = time.time()
        proc = subprocess.run(["claude", "-p", PROMPT.format(ws=ws, api=API, state=state), "--output-format", "json",
                               "--max-turns", "40", "--permission-mode", "acceptEdits", "--allowedTools",
                               "Read,Write,Edit,Glob,Grep,Bash(uv run python -m llmc.discover_lm check:*)"],
                              capture_output=True, text=True, timeout=2400)
        agent_time = round(time.time() - t0, 1)
        try:
            o = json.loads(proc.stdout)
            summary, cost, turns = o.get("result", "")[-800:], o.get("total_cost_usd"), o.get("num_turns")
        except Exception:
            summary, cost, turns = (proc.stdout + proc.stderr)[-800:], None, None
        try:
            res = trial(ws)
            ok = res["val_loss"] < best - 0.005
        except Exception:
            res, ok = dict(val_loss=99.0, error=traceback.format_exc()[-1500:]), False
        if ok:
            best = res["val_loss"]
        else:
            for f in ("ops.py", "plan.json"):
                shutil.copy(ws / (f + ".prev"), ws / f)
        hist.append(dict(iter=it, kept=ok, summary=summary, agent_time=agent_time, agent_cost_usd=cost,
                         agent_turns=turns, **res))
        hp.write_text(json.dumps(hist, indent=2))
        report(ws, res, hist)
        print(f"iter {it}: kept={ok} val={res['val_loss']} agent {agent_time}s cost={cost}", flush=True)


def report(ws, res, hist):
    lines = ["# Latest trial", json.dumps({k: v for k, v in res.items() if k != "error"})]
    if "error" in res:
        lines += ["## ERROR (reverted)", "```", res["error"], "```"]
    lines += ["", "## History"] + [f"- iter {h['iter']}: kept={h['kept']} val={h['val_loss']} :: "
                                   f"{str(h.get('summary', ''))[:300]}" for h in hist]
    (ws / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check", "trial"])
    ap.add_argument("--ws", required=True)
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()
    ws = Path(a.ws)
    {"run": lambda: run(ws, a.iters), "check": lambda: check(ws), "trial": lambda: print(trial(ws))}[a.cmd]()


if __name__ == "__main__":
    main()
