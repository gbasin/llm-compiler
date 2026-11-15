"""Discovery with information: the same Shakespeare task as discover_lm (fixed 2x2x64 core, 4000
steps, val loss), but the agent gets two tools that the blind loop lacked.

  diagnose  failure clusters of the CURRENT best model (doubleAI-style): the highest-loss tokens on a
            fixed selection set (train split), clustered by [target-centred phantom gradient at every
            site, final hidden state]. Per cluster: share of excess loss, demand per site, simple
            statistics (is the target char earlier in the context? does a 4-gram lookup get it? chars
            since newline), and example contexts with target and the model's top guesses.
  score     seconds-fast check of a candidate (extra plan actions using ops/tables from ops.py) on the
            frozen current model: trains only the new interface params for --steps (default 150), and
            reports the loss change overall and PER CLUSTER, so the agent sees whether it fixed the
            failure it aimed at.

The keep/discard decision is unchanged: a full 4000-step trial on the validation split.

    uv run python -m llmc.discover_fast run --ws runs/discover/fast1 --iters 6
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from . import discover_lm as D
from .greedy import kgram_next  # noqa: F401  (registers the op)
from .hybrid import Hybrid
from .ladder import apply_plan, parse_arch
from .lmdata import CharSource

SEL_SEED, SEL_N, K = 4242, 256, 10


def build_from(plan, src, mod):
    model = Hybrid(src.vocab, D.D, parse_arch(D.ARCH), D.HD, V=src.vocab)
    model.itos = src.itos
    apply_actions(model, plan, src, mod)
    return model


def apply_actions(model, plan, src, mod):
    new = []
    for a in plan:
        if "table" in a:
            model.add_logit_table(mod.TABLES[a["table"]](src.train, src.vocab))
            new += list(model.bias_gate.parameters())
        else:
            before = {id(p) for p in model.parameters()}
            apply_plan(model, [a], [])
            new += [p for p in model.parameters() if id(p) not in before]
    return new


def trial_and_save(ws, seed=0):
    """D.trial plus a checkpoint of the trained model for the tools."""
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    src = CharSource("data/tinyshakespeare.txt")
    X = src.sample(512, D.SEQ + 1, torch.Generator().manual_seed(999), split="val")
    mod = D.load_mod(ws)
    plan = json.loads((ws / "plan.json").read_text())
    model = build_from(plan, src, mod)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(D.STEPS):
        x = src.sample(D.BS, D.SEQ + 1, g)
        loss = F.cross_entropy(model(x[:, :-1], targets=x[:, 1:]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    res = dict(val_loss=round(D.val_loss(model, X), 4), params=model.param_counts(),
               train_time=round(time.time() - t0, 1))
    return res, dict(plan=plan, state=model.state_dict(), ops_src=(ws / "ops.py").read_text())


FIT_FRAC = 0.9  # score: tables/corpus stats fit on train[:90%], measured on the unseen last 10%


def fit_split(src):
    """Copy of src whose .train is the first FIT_FRAC of train (what tables may see) plus .dev, the rest."""
    n = int(len(src.train) * FIT_FRAC)
    f = copy.copy(src)
    f.train, f.dev = src.train[:n], src.train[n:]
    return f


def load_ckpt(ws, src=None):
    ck = torch.load(ws / "ckpt.pt", weights_only=False)
    src = src or CharSource("data/tinyshakespeare.txt")
    # build with the ops source the checkpoint was trained with
    tmp = ws / "_ckpt_ops"
    tmp.mkdir(exist_ok=True)
    (tmp / "ops.py").write_text(ck["ops_src"])
    model = build_from(ck["plan"], src, D.load_mod(tmp))
    model.load_state_dict(ck["state"])
    return model, src, ck


def sel_set(src):
    return src.sample(SEL_N, D.SEQ + 1, torch.Generator().manual_seed(SEL_SEED), split="train")


def per_token_nll(model, X, eval_mode=False):
    was = model.training
    model.train(not eval_mode)
    with torch.no_grad():
        lg = model(X[:, :-1])
    model.train(was)
    return F.cross_entropy(lg.flatten(0, 1), X[:, 1:].flatten(), reduction="none").view(X.shape[0], -1)


def kmeans(Z, k, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = Z[torch.randperm(len(Z), generator=g)[:k]].clone()
    for _ in range(iters):
        lab = (Z @ C.T).argmax(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = F.normalize(Z[lab == j].mean(0), dim=0)
    return lab


def diagnose(ws):
    model, src, _ = load_ckpt(ws)
    X = sel_set(src)
    x, y = X[:, :-1], X[:, 1:]
    model.eval()
    lg = model(x, capture=True)
    nll = F.cross_entropy(lg.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
    nll.sum().backward()
    resid = model.last["resid"]
    thr = nll.detach().quantile(0.8)
    hard = nll.detach() > thr
    feats, demand = [], []
    for h in resid:
        g = h.grad
        gm = torch.zeros(src.vocab, g.shape[-1]).index_add_(0, y.flatten(), g.flatten(0, 1))
        gm /= torch.bincount(y.flatten(), minlength=src.vocab).clamp(min=1)[:, None]
        gc = g - gm[y]
        demand.append(gc.norm(dim=-1))
        feats.append(F.normalize(gc[hard], dim=-1))
    hl = F.normalize(resid[-1].detach()[hard], dim=-1)
    Z = F.normalize(torch.cat([torch.cat(feats, -1) / len(feats) ** 0.5, hl], -1), dim=-1)
    lab = kmeans(Z, K)
    labels = torch.full_like(y, -1)
    labels[hard] = lab
    torch.save(labels, ws / "clusters.pt")
    # statistics
    probs = lg.detach().softmax(-1)
    top3 = probs.topk(3, -1)
    kg = kgram_next(model, x, 4)
    T = x.shape[1]
    idx = torch.arange(T)
    seen = ((x[:, None, :] == y[:, :, None]) & (idx[None, None, :] <= idx[None, :, None])).any(-1)
    nl = src.stoi["\n"]
    last_nl = torch.cummax(torch.where(x == nl, idx.expand_as(x), torch.full_like(x, -1)), 1).values
    since_nl = idx - last_nl
    excess = (nll.detach() - thr).clamp(min=0)
    show = lambda s: s.replace("\n", "⏎")
    out = ["# Failure clusters of the current best model",
           f"Selection set: {SEL_N} x {D.SEQ} chars from the TRAIN split (seed {SEL_SEED}). Mean loss "
           f"{nll.mean():.4f}. 'Hard' tokens = top 20% by loss (> {thr:.2f} nats); they hold "
           f"{nll.detach()[hard].sum() / nll.sum():.0%} of total loss. Clusters group hard tokens whose "
           "phantom gradient (dL/dh, centred per target char) and final hidden state point the same way.",
           "Columns: share = this cluster's share of hard-token loss; demand = mean centred |dL/dh| per site "
           "0..2 (where the network wants change); seen = target char occurs earlier in the 64-char context; "
           "kg4 = the 4-gram lookup (char after latest earlier match of last 4 chars) predicts the target; "
           "nl = median chars since newline.", ""]
    tot = nll.detach()[hard].sum()
    order = sorted(range(K), key=lambda j: -float(nll.detach()[labels == j].sum()))
    for j in order:
        mk = labels == j
        if not mk.any():
            continue
        share = float(nll.detach()[mk].sum() / tot)
        dem = [round(float(d[mk].mean()), 2) for d in demand]
        out.append(f"## cluster {j}: n={int(mk.sum())}, share={share:.0%}, mean loss {nll.detach()[mk].mean():.2f}, "
                   f"demand={dem}, seen={seen[mk].float().mean():.0%}, kg4={(kg[mk] == y[mk]).float().mean():.0%}, "
                   f"nl={int(since_nl[mk].float().median())}")
        tc = torch.bincount(y[mk], minlength=src.vocab).topk(5)
        out.append("targets: " + ", ".join(f"{show(src.itos[i])!r}x{int(c)}" for c, i in zip(tc.values, tc.indices) if c > 0))
        pos = mk.nonzero()
        pos = pos[torch.randperm(len(pos), generator=torch.Generator().manual_seed(j))[:8]]
        for b, t in pos.tolist():
            ctx = "".join(src.itos[i] for i in x[b, max(0, t - 47):t + 1].tolist())
            guesses = " ".join(f"{show(src.itos[i])!r}:{p:.2f}" for p, i in zip(top3.values[b, t].tolist(), top3.indices[b, t].tolist()))
            out.append(f"    ...{show(ctx)!r} -> target {show(src.itos[y[b, t]])!r} (loss {nll[b, t]:.2f}; model {guesses})")
        out.append("")
    (ws / "clusters.md").write_text("\n".join(out))
    print("\n".join(out[:6]))
    print(f"wrote {ws / 'clusters.md'}")


def score(ws, cand_path, steps=150):
    """Tables (base and candidate) are rebuilt from train[:FIT_FRAC]; new params train on that part;
    the headline number is measured in eval mode on train[FIT_FRAC:], text no table has counted.
    (Measuring on text the table was built from rewards memorization: it ranked tables backwards.)"""
    os.environ["CTX_FRAC"] = str(FIT_FRAC)  # ops that read the corpus themselves must honour this
    full = CharSource("data/tinyshakespeare.txt")
    src = fit_split(full)
    model, _, _ = load_ckpt(ws, src)
    labels = torch.load(ws / "clusters.pt")
    X = sel_set(full)
    gd = torch.Generator().manual_seed(SEL_SEED + 1)
    Xd = torch.stack([src.dev[i:i + D.SEQ + 1] for i in torch.randint(0, len(src.dev) - D.SEQ - 1, (512,), generator=gd)])
    base = per_token_nll(model, X)
    dbase = per_token_nll(model, Xd, eval_mode=True).mean()
    for p in model.parameters():
        p.requires_grad_(False)
    cand = json.loads(Path(cand_path).read_text())
    t0 = time.time()
    new = apply_actions(model, cand, src, D.load_mod(ws))
    for p in new:
        p.requires_grad_(True)
    opt = torch.optim.Adam(new, lr=1e-2)
    g = torch.Generator().manual_seed(7)
    for _ in range(steps):
        x = src.sample(D.BS, D.SEQ + 1, g)
        loss = F.cross_entropy(model(x[:, :-1], targets=x[:, 1:]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    after = per_token_nll(model, X)
    dafter = per_token_nll(model, Xd, eval_mode=True).mean()
    print(f"candidate {cand} ({time.time() - t0:.0f}s, {steps} steps, only new params trained)")
    print(f"loss change on unseen dev text (last {1 - FIT_FRAC:.0%} of train, no table has counted it): "
          f"{dafter - dbase:+.4f}  (negative = better; this is the number to trust, for tables too)")
    rows = [f"per-cluster changes below are on the in-sample selection set (useful for WHICH failures "
            f"moved, not for magnitude): overall {(after - base).mean():+.4f}"]
    for j in range(K):
        mk = labels == j
        if mk.any():
            rows.append(f"cluster {j}: {float((after - base)[mk].mean()):+.3f} nats/token (n={int(mk.sum())})")
    easy = labels == -1
    rows.append(f"non-hard tokens: {float((after - base)[easy].mean()):+.3f}")
    print("\n".join(rows))


PROMPT = """You are the teacher in a research loop that moves work from a tiny neural network into
code (ops, lookup tables). Lower the validation loss below by editing ONLY {ws}/ops.py and
{ws}/plan.json. Do not edit or read anything else under runs/.
{api}
## Tools (use them; they are cheap)
- {ws}/clusters.md: where the CURRENT best model (trained with the current plan) fails: clusters of
  its highest-loss tokens with examples, the model's guesses, and simple statistics. Read it first and
  design code aimed at a specific cluster.
- Score an idea in seconds before committing to it: write extra plan actions to a JSON file (e.g.
  {ws}/cand.json, a list of actions in the same format as plan.json, using ops/TABLES you added to
  ops.py) and run
    uv run python -m llmc.discover_fast score --ws {ws} --cand {ws}/cand.json
  It adds those actions on top of the current trained model, trains only the new params briefly, and
  prints the loss change on dev text no table has seen (tables are rebuilt from the first 90% of
  train for scoring; ops that read the corpus directly must use only
  train[:int(len(train) * float(os.environ.get("CTX_FRAC", "1.0")))]) and per cluster. Use it to test several ideas and variants, then put
  the winners into plan.json. (A table replaces the current one if the plan already has one.)
## Current state
{state}
## Workflow
1. Read {ws}/clusters.md, {ws}/report.md and {ws}/history.json.
2. Hypothesize what information or procedure would fix the biggest clusters; write it as code in ops.py;
   score it; iterate.
3. Put the best into plan.json (the full plan to train from scratch next) and validate:
   uv run python -m llmc.discover_lm check --ws {ws}
4. Stop and summarize in 2-3 sentences what you changed, why, and what scoring showed.
"""


def run(ws: Path, iters: int):
    ws.mkdir(parents=True, exist_ok=True)
    if not (ws / "ops.py").exists():
        (ws / "ops.py").write_text(D.OPS_TEMPLATE + "\nfrom llmc.greedy import kgram_next  # noqa: F401 (op kgram_next(tok, k))\n")
        (ws / "plan.json").write_text("[]")
    hp = ws / "history.json"
    hist = json.loads(hp.read_text()) if hp.exists() else []
    if not hist:
        res, ck = trial_and_save(ws)
        torch.save(ck, ws / "ckpt.pt")
        diagnose(ws)
        hist.append(dict(iter=0, kept=True, summary="baseline: no code", **res))
        hp.write_text(json.dumps(hist, indent=2))
        D.report(ws, res, hist)
        print("baseline", res, flush=True)
    best = min(h["val_loss"] for h in hist if h["kept"])
    api = D.API + "\nAlso available: kgram_next(tok, k=K) -> char after the latest earlier occurrence of the last K chars (V if none).\n"
    for it in range(len(hist), len(hist) + iters):
        for f in ("ops.py", "plan.json"):
            shutil.copy(ws / f, ws / (f + ".prev"))
        state = f"Iteration {it}. Best val loss so far: {best:.4f}."
        t0 = time.time()
        proc = subprocess.run(["claude", "-p", PROMPT.format(ws=ws, api=api, state=state), "--output-format", "json",
                               "--max-turns", "60", "--permission-mode", "acceptEdits", "--allowedTools",
                               "Read,Write,Edit,Glob,Grep,Bash(uv run python -m llmc.discover_lm check:*),"
                               "Bash(uv run python -m llmc.discover_fast score:*)"],
                              capture_output=True, text=True, timeout=3600)
        agent_time = round(time.time() - t0, 1)
        try:
            o = json.loads(proc.stdout)
            summary, cost, turns = o.get("result", "")[-800:], o.get("total_cost_usd"), o.get("num_turns")
        except Exception:
            summary, cost, turns = (proc.stdout + proc.stderr)[-800:], None, None
        try:
            res, ck = trial_and_save(ws)
            ok = res["val_loss"] < best - 0.005
        except Exception:
            res, ck, ok = dict(val_loss=99.0, error=traceback.format_exc()[-1500:]), None, False
        if ok:
            best = res["val_loss"]
            torch.save(ck, ws / "ckpt.pt")
            diagnose(ws)
        else:
            for f in ("ops.py", "plan.json"):
                shutil.copy(ws / (f + ".prev"), ws / f)
        hist.append(dict(iter=it, kept=ok, summary=summary, agent_time=agent_time, agent_cost_usd=cost,
                         agent_turns=turns, **res))
        hp.write_text(json.dumps(hist, indent=2))
        D.report(ws, res, hist)
        print(f"iter {it}: kept={ok} val={res['val_loss']} agent {agent_time}s cost={cost}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "diagnose", "score"])
    ap.add_argument("--ws", required=True)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--cand")
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    ws = Path(a.ws)
    if a.cmd == "run":
        run(ws, a.iters)
    elif a.cmd == "diagnose":
        diagnose(ws)
    else:
        score(ws, a.cand, a.steps)


if __name__ == "__main__":
    main()
