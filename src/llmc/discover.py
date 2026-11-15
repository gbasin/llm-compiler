"""Direction 3: automated discovery. A headless Claude Code agent (`claude -p`) iterates:
read diagnostics -> write code ops / contract label functions / plan -> harness trains a
trial -> keep if better, else revert.

The agent starts WITHOUT the word-problem ops I wrote by hand (parse_num, hold, arith,
render, since_char, offset); it only gets the generic M3 library plus the hybrid API.

    uv run python -m llmc.discover run --ws runs/discover/wm1 --iters 6
    uv run python -m llmc.discover check --ws runs/discover/wm1     # agent's self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from . import oplib

TASK_V2 = False
TASK_V3 = False
HIDDEN = ["parse_num", "hold", "arith", "render", "since_char", "offset"]
ARCH, D, HD = "2x4x64", 64, 16
import os
TRIAL_STEPS, EVAL_N = int(os.environ.get("TRIAL_STEPS", 1500)), 200

OPS_TEMPLATE = '''"""Agent-written ops and contract label functions. Edit freely.

Ops: decorate with @op(domain_fn); signature fn(m, *slots, **kwargs) -> LongTensor (B, T).
  - slots are LongTensors (B, T); must be CAUSAL (output at t may only use inputs at <= t).
  - m.itos: list mapping char id -> char; m.domains: slot -> domain size.
  - domain_fn(m, spec) -> int, the number of distinct values (used for reader tables).
Label functions: LABELS[name] = fn(probs, T) -> LongTensor (B, T) of target classes, -1 = ignore.
  probs: list of Problem(text, prompt_len, a, b, op, ans, spans) where op in {"add","sub","mul"}
  and spans = [(start, end, value, role)] with role in {"a","b","distractor"} (char offsets).
"""
import torch

from llmc.oplib import op

LABELS = {}
'''

API_DOC = """
## Task
Char-level word problems; operands written as digits or words ("seven", "a dozen"), varied
phrasing, optional distractor sentences that can contain numbers. Example:
  "Tom had 9 books and found three more. He is nine years old. Q: how many books now? A: 12\\n"
The model must generate the answer digits + newline after "A: ". Metric: greedy exact match on
three splits (v2 if the workspace has a V2 marker file: distractor numbers can also appear
INSIDE the operand sentence, e.g. "Tom has 5 bags in room 8 with 10 stones each.", or
between the two operands; v3 if the workspace has a V3 marker: the question is always the neutral
"Q: how many <obj>?", so the operation must be inferred from the story; the score also reports a
"novel" split with SECRET held-out phrasings/verbs/distractors (stored outside the repo; you cannot
see them) -- it is NOT part of the score, but it measures whether your solution generalizes, so
prefer general solutions over memorizing template wording): id (operands 0..20), distract (always a distractor sentence), big (operands 0..50,
written as digits above 20). Score = mean of the three.

## Model: tiny transformer core, default arch 2 layers x 4 heads x 64 MLP, d=64 (~57k core params).
You MAY shrink it with a plan action {"arch": "LxHxM"} (L layers, H heads of dim 16, M MLP units),
e.g. "1x2x32". Width d=64 is fixed.
## Objective (lexicographic): a change is kept if it raises the score by > 0.005, OR keeps the
score within 0.005 of the best while reducing core params. So once the score is maxed, shrink the
neural core by moving more work into code.
It is a `Hybrid` with a register file of per-position integer slots ("tok" = char ids).
Trial = 1500 training steps (LM loss on answer chars + contract losses), then eval.

## Plan (plan.json): a list of actions applied in order:
  {"op": NAME, "inputs": [slot,...], "out": slot, "kwargs": {...}}  code op -> new slot
  {"reader": slot, "site": i}             slot -> network: zero-init embedding table added at
                                          residual site i (0..2) through a learned router.
                                          Only for slots with small domains (< ~200).
  {"decoder": slot, "domain": K, "site": -1, "labels": LABELNAME}
                                          network -> slot: linear head on the residual at site
                                          (-1 = last); value = argmax; trained ONLY by a contract
                                          loss against LABELS[LABELNAME](probs, T) (-1 = ignore).
  {"logit_slot": slot}                    code -> output: slot value is a char id; adds a learned,
                                          gated bias to that char's logit (how code can emit
                                          answer characters directly).
Ops run as soon as their inputs exist (decoders at the last site become available there).
Everything must be causal.

## Available generic ops (llmc/oplib.py): xval, task, shift(k), add_mod, mul_mod, cumsum_mod,
cummax, eq, mask, map(table), count_prefix_eq, prev_occ_next, isitem, suffix_next, suffix_len,
since(value). Read src/llmc/oplib.py for semantics (ignore ops not in this list).
You may write ANY new ops and label functions in ops.py in your workspace.
"""

PROMPT = """You are the teacher in a research loop that moves computation from a tiny neural
network into code. Improve the score on the task below by editing ONLY two files in your
workspace {ws}: ops.py (new code ops + contract label functions) and plan.json (the plan).
Do not edit anything else in the repository.

{api}

## Current state
{state}

## Workflow
1. Read {ws}/report.md (latest diagnostics, error examples) and {ws}/history.json.
2. Edit {ws}/ops.py and {ws}/plan.json.
3. Validate with: uv run python -m llmc.discover check --ws {ws}
   (builds the model, runs one batch, reports slot values and, if possible, whether the code
   path alone would produce correct answers given correct labels). Fix errors until it passes.
4. Stop. The harness then trains a trial and keeps your change only if the score improves.
Keep the change focused (one idea per iteration). End with a 2-3 sentence summary of what you
changed and why.
"""


def load_agent_module(ws: Path):
    for name in HIDDEN:
        oplib.OPS.pop(name, None)
    spec = importlib.util.spec_from_file_location(f"agent_ops_{time.time_ns()}", ws / "ops.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(ws: Path, wm, mod):
    from .hybrid import Hybrid
    from .ladder import apply_plan, parse_arch

    plan = json.loads((ws / "plan.json").read_text())
    arch = next((a["arch"] for a in plan if "arch" in a), ARCH)
    model = Hybrid(wm.vocab, D, parse_arch(arch), HD, V=wm.vocab)
    model.itos = wm.itos
    for a in plan:
        a = dict(a)
        if "arch" in a:
            continue
        if "decoder" in a and a.get("site", -1) < 0:
            a["site"] = model.n_layers
        if "logit_slot" in a:
            model.add_logit_slot(a["logit_slot"])
        else:
            apply_plan(model, [a], [])
    return model, plan


def contract_loss(model, mod, probs, T):
    loss = 0.0
    for dec in model.decoders:
        lab = mod.LABELS[dec.labels](probs, T)
        lg = model.last["dec_logits"][dec.slot]
        sel = lab >= 0
        if sel.any():
            loss = loss + F.cross_entropy(lg[sel], lab[sel])
    return loss


def evaluate(model, wm, n=EVAL_N):
    out = {}
    for name, kw in dict(id=dict(), distract=dict(p_distract=1.0), big=dict(max_n=50)).items():
        _, _, probs = wm.batch(n, random.Random(123), **kw)
        out[name] = round(wm.exact_match(model, probs), 4)
    out["score"] = round(sum(out.values()) / 3, 4)
    # held-out paraphrases (never trained on, not part of the score)
    _, _, probs = wm.batch(n, random.Random(321), novel=True)
    out["novel"] = round(wm.exact_match(model, probs), 4)
    return out


def failures(model, wm, k=12):
    rows = []
    for name, kw in dict(id=dict(), distract=dict(p_distract=1.0), big=dict(max_n=50)).items():
        _, _, probs = wm.batch(60, random.Random(7), **kw)
        for p in probs:
            ids = wm.encode(p.text[:p.prompt_len])
            for _ in range(5):
                nxt = int(model(torch.tensor([ids]))[0, -1].argmax())
                if nxt in (wm.stoi["\n"], wm.PAD):
                    break
                ids.append(nxt)
            got = "".join(wm.itos[i] for i in ids[p.prompt_len:])
            if got != str(p.ans):
                rows.append(f"[{name}] {p.text[:p.prompt_len]!r} -> got {got!r}, want {p.ans} ({p.op})")
    random.Random(0).shuffle(rows)
    return rows[:k]


def trial(ws: Path, steps=TRIAL_STEPS, seed=0):
    from .wordmath import WordMath

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    wm = WordMath(v2=TASK_V2, v3=TASK_V3)
    mod = load_agent_module(ws)
    model, plan = build(ws, wm, mod)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = random.Random(seed)
    t0 = time.time()
    for step in range(steps):
        tokens, targets, probs = wm.batch(64, rng)
        logits = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        if model.decoders:
            loss = loss + 0.5 * contract_loss(model, mod, probs, tokens.shape[1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    ev = evaluate(model, wm)
    fails = failures(model, wm)
    return dict(eval=ev, params=model.param_counts(), train_time=round(time.time() - t0, 1),
                plan_len=len(plan)), fails


def check(ws: Path):
    """Agent-facing validation: build, forward one batch, show slots for one example."""
    from .wordmath import WordMath

    wm = WordMath(v2=TASK_V2, v3=TASK_V3)
    mod = load_agent_module(ws)
    model, plan = build(ws, wm, mod)
    tokens, targets, probs = wm.batch(8, random.Random(1))
    logits = model(tokens)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
    if model.decoders:
        loss = loss + contract_loss(model, mod, probs, tokens.shape[1])
    loss.backward()
    print("OK: plan builds, forward/backward pass. params:", model.param_counts())
    p = probs[0]
    print("example:", repr(p.text))
    regs = model.last["regs"]
    for name, v in regs.items():
        if name == "tok":
            continue
        vals = v[0, :len(p.text)].tolist()
        print(f"  slot {name:10s}", " ".join(f"{p.text[i]}:{x}" for i, x in enumerate(vals) if x not in (0, -1))[:400])
    # oracle: replace decoder outputs with their labels and recompute ops
    if model.decoders:
        for dec in model.decoders:
            regs_lab = mod.LABELS[dec.labels](probs, tokens.shape[1])
            print(f"  decoder {dec.slot}: label agreement (untrained) "
                  f"{((regs[dec.slot] == regs_lab) | (regs_lab < 0)).float().mean():.3f}")
    print("Note: decoders are untrained here, so code outputs depending on them are not yet meaningful.")


def run(ws: Path, iters: int, model_name: str | None):
    ws.mkdir(parents=True, exist_ok=True)
    if not (ws / "ops.py").exists():
        (ws / "ops.py").write_text(OPS_TEMPLATE)
        (ws / "plan.json").write_text("[]")
    hist_path = ws / "history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
    if not hist:
        res, fails = trial(ws)
        hist.append(dict(iter=0, kept=True, summary="baseline: pure network, empty plan", **res))
        write_report(ws, res, fails, hist)
        hist_path.write_text(json.dumps(hist, indent=2))
        print("baseline", res["eval"], flush=True)
    best = max(h["eval"]["score"] for h in hist if h["kept"])
    best_core = [h for h in hist if h["kept"]][-1]["params"]["core"]
    for it in range(len(hist), len(hist) + iters):
        shutil.copy(ws / "ops.py", ws / "ops.prev.py")
        shutil.copy(ws / "plan.json", ws / "plan.prev.json")
        state = f"Iteration {it}. Best score so far: {best:.4f} with core params {best_core}. Current plan has {len(json.loads((ws / 'plan.json').read_text()))} actions."
        prompt = PROMPT.format(ws=ws, api=API_DOC, state=state)
        t0 = time.time()
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "40",
               "--allowedTools", f"Read,Write,Edit,Glob,Grep,Bash(uv run python -m llmc.discover check:*)",
               "--permission-mode", "acceptEdits"]
        if model_name:
            cmd += ["--model", model_name]
        import os
        env = {k: v for k, v in os.environ.items() if k != "WM_HELDOUT"}  # never expose the secret set
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        agent_time = round(time.time() - t0, 1)
        try:
            out = json.loads(proc.stdout)
            summary, cost = out.get("result", "")[-800:], out.get("total_cost_usd")
            turns = out.get("num_turns")
        except Exception:
            summary, cost, turns = proc.stdout[-800:] + proc.stderr[-400:], None, None
        try:
            res, fails = trial(ws)
            sc, core = res["eval"]["score"], res["params"]["core"]
            ok = sc > best + 0.005 or (sc >= best - 0.005 and core < best_core)
        except Exception:
            res, fails, ok = dict(eval=dict(score=-1), error=traceback.format_exc()[-1500:]), [], False
        if ok:
            best, best_core = max(best, res["eval"]["score"]), res["params"]["core"]
        else:
            shutil.copy(ws / "ops.prev.py", ws / "ops.py")
            shutil.copy(ws / "plan.prev.json", ws / "plan.json")
        hist.append(dict(iter=it, kept=ok, summary=summary, agent_time=agent_time, agent_cost_usd=cost,
                         agent_turns=turns, **res))
        hist_path.write_text(json.dumps(hist, indent=2))
        write_report(ws, res, fails, hist)
        print(f"iter {it}: kept={ok} eval={res['eval']} agent {agent_time}s cost={cost}", flush=True)


def write_report(ws, res, fails, hist):
    lines = ["# Latest trial", json.dumps(res.get("eval")), "", f"params: {res.get('params')}", ""]
    if "error" in res:
        lines += ["## ERROR (change was reverted)", "```", res["error"], "```"]
    lines += ["## Sample failures (prompt -> model output)"] + [f"- {f}" for f in fails]
    lines += ["", "## History"] + [f"- iter {h['iter']}: kept={h['kept']} score={h['eval'].get('score')} :: "
                                   f"{str(h.get('summary', ''))[:300]}" for h in hist]
    (ws / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check", "trial"])
    ap.add_argument("--ws", required=True)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--model")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--v3", action="store_true")
    a = ap.parse_args()
    ws = Path(a.ws)
    global TASK_V2
    global TASK_V3
    TASK_V3 = a.v3 or (ws / "V3").exists()
    TASK_V2 = a.v2 or (ws / "V2").exists() or TASK_V3
    if a.v3:
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "V3").touch()
    if a.v2:
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "V2").touch()
    if a.cmd == "check":
        check(ws)
    elif a.cmd == "trial":
        print(trial(ws))
    else:
        run(ws, a.iters, a.model)


if __name__ == "__main__":
    main()
