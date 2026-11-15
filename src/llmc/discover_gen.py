"""General-algorithm discovery: can an agent write code that makes a tiny core better at text in
general (not one corpus), so the core can shrink?

Differences from discover_lm / discover_fast:
  * No pre-written program library. The agent starts from an empty ops.py; oplib's generic ops,
    greedy's kgram_next, the n-gram tables, and the repo source are hidden (OPS is cleared before the
    agent's module loads; repo src/docs/runs are denied to the agent).
  * No corpus access from code. Ops see only the token window they are given. A static check bans
    file/IO/imports beyond torch/math/stdlib helpers, embedded constant collections > 64 items and
    long string literals, and table plan actions. (Learned memory still exists in the core and in
    readers; reader interface is capped.)
  * Three corpora at once (Shakespeare, TinyStories prose, Python stdlib source). Score = mean val
    loss over the three. A fourth corpus (Pile validation docs) is a hidden transfer test, never shown
    to the agent and never used for decisions.
  * Shrinking: after every iteration the harness tries the next smaller core on the current plan and
    adopts it if mean val loss gets worse by <= SHRINK_TOL. The point is a smaller, more general core.
  * Agent tools: failure clusters (phantom gradients, neutral statistics only) and a fast `score`.

    uv run python -m llmc.discover_gen run --ws runs/gen/gen1 --iters 8
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from . import oplib
from .hybrid import Hybrid
from .ladder import apply_plan, parse_arch

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CORPORA = ["shakespeare", "stories", "python"]
CHARS = ["\n"] + [chr(i) for i in range(32, 127)] + ["¤"]  # ¤ = any other char
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
CORES = [("2x2x64", 32), ("2x2x32", 32), ("1x2x32", 32), ("1x2x32", 16), ("1x1x16", 16), ("1x0x0", 16)]
HD, SEQ, BS, STEPS = 8, 128, 32, 8000
KEEP_MARGIN, CORPUS_MARGIN, SHRINK_TOL = 0.005, 0.015, 0.01
MAX_READER_DOMAIN, MAX_INTERFACE = 128, 6000
SEL_SEED, SEL_N, K = 4242, 96, 10  # per corpus selection rows


# ------------------------------------------------------------------ data
def _enc(text):
    o = STOI["¤"]
    return torch.tensor([STOI.get(c, o) for c in text.replace("\t", "    ")], dtype=torch.long)


def _corpus_text(name, n=1_200_000):
    cache = DATA / "gen" / f"{name}.txt"
    if cache.exists():
        return cache.read_text()
    cache.parent.mkdir(exist_ok=True)
    if name == "shakespeare":
        t = (DATA / "tinyshakespeare.txt").read_text()
    elif name == "stories":
        with open(DATA / "tinystories" / "TinyStories-valid.txt") as f:
            t = f.read(n)
    elif name == "python":
        root = Path(sysconfig.get_paths()["stdlib"])
        parts, tot = [], 0
        for p in sorted(root.glob("*.py")) + sorted(root.glob("[a-z]*/*.py")):
            if "test" in str(p.relative_to(root)):
                continue
            s = p.read_text(errors="ignore")
            parts.append(s); tot += len(s)
            if tot > n:
                break
        t = "\n".join(parts)
    elif name == "pile":
        t = "\n\n".join(json.loads((DATA / "pile" / "valid_docs.json").read_text()))
    t = t[:n]
    cache.write_text(t)
    return t


class MultiSource:
    """Per corpus: train 80% / dev 10% (agent tools) / val 10% (keep decisions). Pile: transfer only."""

    def __init__(self):
        self.itos, self.stoi, self.vocab = CHARS, STOI, VOCAB
        self.parts = {}
        for c in CORPORA:
            d = _enc(_corpus_text(c))
            a, b = int(len(d) * 0.8), int(len(d) * 0.9)
            self.parts[c] = dict(train=d[:a], dev=d[a:b], val=d[b:])
        self.transfer = _enc(_corpus_text("pile"))

    def sample(self, B, L, g, split="train", corpus=None):
        cs = torch.randint(0, len(CORPORA), (B,), generator=g) if corpus is None else torch.full((B,), CORPORA.index(corpus))
        rows = []
        for c in cs.tolist():
            d = self.parts[CORPORA[c]][split]
            i = int(torch.randint(0, len(d) - L, (1,), generator=g))
            rows.append(d[i:i + L])
        return torch.stack(rows), cs

    def fixed(self, split, n, seed, corpus):
        return self.sample(n, SEQ + 1, torch.Generator().manual_seed(seed), split, corpus)[0]


# ------------------------------------------------------------------ agent code: sandbox
ALLOWED_IMPORTS = {"torch", "math", "torch.nn.functional", "functools", "itertools", "collections", "llmc.oplib"}
BANNED_NAMES = {"open", "eval", "exec", "__import__", "compile", "globals", "locals", "vars", "breakpoint",
                "input", "getattr", "setattr", "importlib"}
BANNED_ATTRS = {"read_text", "read_bytes", "load", "loadtxt", "fromfile", "system", "popen", "OPS", "__dict__",
                "__globals__", "__builtins__", "__subclasses__"}


def static_check(src_text):
    errs = []
    if len(src_text) > 40_000:
        errs.append(f"ops.py is {len(src_text)} bytes (max 40000)")
    tree = ast.parse(src_text)
    doc_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant):
            doc_ids.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    errs.append(f"line {node.lineno}: import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS or (node.module == "llmc.oplib" and [a.name for a in node.names] != ["op"]):
                errs.append(f"line {node.lineno}: from {node.module} import ... not allowed (only `from llmc.oplib import op`)")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f"line {node.lineno}: {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            errs.append(f"line {node.lineno}: .{node.attr} not allowed")
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) > 64:
            errs.append(f"line {node.lineno}: literal collection with {len(node.elts)} items (max 64)")
        elif isinstance(node, ast.Dict) and len(node.keys) > 64:
            errs.append(f"line {node.lineno}: literal dict with {len(node.keys)} items (max 64)")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) > 120 \
                and id(node) not in doc_ids:
            errs.append(f"line {node.lineno}: string literal of {len(node.value)} chars (max 120)")
    return errs


def load_mod(path: Path):
    errs = static_check(path.read_text())
    if errs:
        raise ValueError("ops.py failed the sandbox check:\n  " + "\n  ".join(errs))
    oplib.OPS.clear()  # no pre-written ops: only what the agent defines
    spec = importlib.util.spec_from_file_location(f"agent_gen_{time.time_ns()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_plan(plan):
    for a in plan:
        if set(a) - {"op", "inputs", "out", "kwargs", "reader", "site"} or not ("op" in a or "reader" in a):
            raise ValueError(f"plan action not allowed: {a} (only op and reader actions)")


def build(plan, core_idx):
    validate_plan(plan)
    arch, d = CORES[core_idx]
    blocks = parse_arch(arch)
    model = Hybrid(VOCAB, d, blocks, HD, V=VOCAB)
    model.itos = CHARS
    nsites = len(blocks)
    for a in plan:
        a = dict(a)
        if "reader" in a:
            a["site"] = min(int(a["site"]), nsites)  # clamp when the core shrinks
        apply_plan(model, [a], [])
    for r in model.readers:
        if r.domain > MAX_READER_DOMAIN:
            raise ValueError(f"reader on slot {r.slot}: domain {r.domain} > {MAX_READER_DOMAIN}")
    pc = model.param_counts()
    if pc["interface"] > MAX_INTERFACE:
        raise ValueError(f"interface params {pc['interface']} > {MAX_INTERFACE}")
    return model


def val_loss(model, X):
    model.eval()
    with torch.no_grad():
        tot = sum(F.cross_entropy(model(X[i:i + 128, :-1]).flatten(0, 1), X[i:i + 128, 1:].flatten()).item()
                  * len(X[i:i + 128]) for i in range(0, len(X), 128))
    model.train()
    return tot / len(X)


def trial(ws, core_idx, seed=0, steps=STEPS):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    src = MultiSource()
    plan = json.loads((ws / "plan.json").read_text())
    load_mod(ws / "ops.py")
    model = build(plan, core_idx)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        x, _ = src.sample(BS, SEQ + 1, g)
        loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    per = {c: round(val_loss(model, src.fixed("val", 256, 999, c)), 4) for c in CORPORA}
    T = src.transfer
    gt = torch.Generator().manual_seed(999)
    Xt = torch.stack([T[i:i + SEQ + 1] for i in torch.randint(0, len(T) - SEQ - 1, (256,), generator=gt).tolist()])
    res = dict(val_loss=round(sum(per.values()) / len(per), 4), per_corpus=per,
               transfer_pile=round(val_loss(model, Xt), 4), core=f"{CORES[core_idx][0]} d={CORES[core_idx][1]}",
               core_idx=core_idx, params=model.param_counts(), train_time=round(time.time() - t0, 1))
    return res, dict(plan=plan, core_idx=core_idx, state=model.state_dict(), ops_src=(ws / "ops.py").read_text())


def load_ckpt(ws):
    ck = torch.load(ws / "ckpt.pt", weights_only=False)
    tmp = ws / "_ckpt_ops"
    tmp.mkdir(exist_ok=True)
    (tmp / "ops.py").write_text(ck["ops_src"])
    load_mod(tmp / "ops.py")
    model = build(ck["plan"], ck["core_idx"])
    model.load_state_dict(ck["state"])
    return model, ck


# ------------------------------------------------------------------ agent tools
def sel_set(src):
    X = torch.cat([src.fixed("train", SEL_N, SEL_SEED + i, c) for i, c in enumerate(CORPORA)])
    corp = torch.arange(len(CORPORA)).repeat_interleave(SEL_N)
    return X, corp


def per_token_nll(model, X):
    model.eval()
    with torch.no_grad():
        lg = model(X[:, :-1])
    model.train()
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
    """Failure clusters with neutral statistics only (no hints about which algorithm would help)."""
    model, ck = load_ckpt(ws)
    src = MultiSource()
    X, corp = sel_set(src)
    x, y = X[:, :-1], X[:, 1:]
    model.eval()
    lg = model(x, capture=True)
    nll = F.cross_entropy(lg.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
    nll.sum().backward()
    resid = model.last["resid"]
    nd = nll.detach()
    thr = nd.quantile(0.8)
    hard = nd > thr
    feats, demand = [], []
    for h in resid:
        gr = h.grad
        gm = torch.zeros(VOCAB, gr.shape[-1]).index_add_(0, y.flatten(), gr.flatten(0, 1))
        gm /= torch.bincount(y.flatten(), minlength=VOCAB).clamp(min=1)[:, None]
        gc = gr - gm[y]
        demand.append(gc.norm(dim=-1))
        feats.append(F.normalize(gc[hard], dim=-1))
    hl = F.normalize(resid[-1].detach()[hard], dim=-1)
    Z = F.normalize(torch.cat([torch.cat(feats, -1) / len(feats) ** 0.5, hl], -1), dim=-1)
    labels = torch.full_like(y, -1)
    labels[hard] = kmeans(Z, K)
    torch.save(labels, ws / "clusters.pt")
    probs = lg.detach().softmax(-1)
    top3 = probs.topk(3, -1)
    show = lambda s: s.replace("\n", "⏎")
    cm = corp[:, None].expand_as(y)
    per = {c: round(float(nd[corp == i].mean()), 3) for i, c in enumerate(CORPORA)}
    out = ["# Failure clusters of the current best model",
           f"Core {CORES[ck['core_idx']][0]} d={CORES[ck['core_idx']][1]}. Selection set: {SEL_N} x {SEQ} chars "
           f"of TRAIN text per corpus. Mean loss per corpus: {per}. 'Hard' tokens = top 20% by loss "
           f"(> {thr:.2f} nats), holding {nd[hard].sum() / nd.sum():.0%} of total loss. Clusters group hard "
           "tokens whose phantom gradient (dL/dh, centred per target char) and final hidden state point the "
           "same way.",
           "Columns: share = share of hard-token loss; demand = mean centred |dL/dh| per site (where the "
           "network wants change); corpora = where the cluster's tokens come from; pos = median position in "
           "the window.", ""]
    tot = nd[hard].sum()
    order = sorted(range(K), key=lambda j: -float(nd[labels == j].sum()))
    pos_t = torch.arange(y.shape[1]).expand_as(y)
    for j in order:
        mk = labels == j
        if not mk.any():
            continue
        share = float(nd[mk].sum() / tot)
        dem = [round(float(dd[mk].mean()), 2) for dd in demand]
        cc = torch.bincount(cm[mk], minlength=len(CORPORA)).float()
        cs = ", ".join(f"{c} {v / cc.sum():.0%}" for c, v in zip(CORPORA, cc.tolist()))
        out.append(f"## cluster {j}: n={int(mk.sum())}, share={share:.0%}, mean loss {nd[mk].mean():.2f}, "
                   f"demand={dem}, corpora: {cs}, pos={int(pos_t[mk].float().median())}")
        tc = torch.bincount(y[mk], minlength=VOCAB).topk(5)
        out.append("targets: " + ", ".join(f"{show(CHARS[i])!r}x{int(c)}" for c, i in zip(tc.values, tc.indices) if c > 0))
        pos = mk.nonzero()
        pos = pos[torch.randperm(len(pos), generator=torch.Generator().manual_seed(j))[:10]]
        for b, t in pos.tolist():
            ctx = "".join(CHARS[i] for i in x[b, max(0, t - 59):t + 1].tolist())
            gs = " ".join(f"{show(CHARS[i])!r}:{p:.2f}" for p, i in zip(top3.values[b, t].tolist(), top3.indices[b, t].tolist()))
            out.append(f"    [{CORPORA[int(corp[b])]}] ...{show(ctx)!r} -> target {show(CHARS[y[b, t]])!r} "
                       f"(loss {nll[b, t]:.2f}; model {gs})")
        out.append("")
    (ws / "clusters.md").write_text("\n".join(out))
    print("\n".join(out[:4]))


def score(ws, cand_path, steps=200):
    """Add candidate actions to the trained current model, train only their params, report the change
    on DEV text (not used for keep decisions), per corpus and per failure cluster."""
    src = MultiSource()
    model, ck = load_ckpt(ws)
    labels = torch.load(ws / "clusters.pt")
    X, corp = sel_set(src)
    D = {c: src.fixed("dev", 192, 77, c) for c in CORPORA}
    base = per_token_nll(model, X)
    dbase = {c: per_token_nll(model, D[c]).mean() for c in CORPORA}
    for p in model.parameters():
        p.requires_grad_(False)
    cand = json.loads(Path(cand_path).read_text())
    load_mod(ws / "ops.py")  # current agent ops (may contain new ones)
    validate_plan(cand)
    t0 = time.time()
    before = {id(p) for p in model.parameters()}
    nsites = len(model.blocks)
    for a in cand:
        a = dict(a)
        if "reader" in a:
            a["site"] = min(int(a["site"]), nsites)
        apply_plan(model, [a], [])
    new = [p for p in model.parameters() if id(p) not in before]
    if not new:
        print("candidate adds no trainable parameters (needs a reader)"); return
    for p in new:
        p.requires_grad_(True)
    opt = torch.optim.Adam(new, lr=1e-2)
    g = torch.Generator().manual_seed(7)
    for _ in range(steps):
        x, _ = src.sample(BS, SEQ + 1, g)
        loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    after = per_token_nll(model, X)
    dd = {c: float(per_token_nll(model, D[c]).mean() - dbase[c]) for c in CORPORA}
    print(f"candidate {cand} ({time.time() - t0:.0f}s, {steps} steps, only new params trained; "
          f"interface now {model.param_counts()['interface']} / {MAX_INTERFACE})")
    print(f"loss change on dev text, mean over corpora: {sum(dd.values()) / len(dd):+.4f}  "
          f"({', '.join(f'{c} {v:+.4f}' for c, v in dd.items())}; negative = better; a full trial typically gains 1-3x this)")
    rows = ["per-cluster change on the selection set:"]
    for j in range(K):
        mk = labels == j
        if mk.any():
            rows.append(f"  cluster {j}: {float((after - base)[mk].mean()):+.3f} nats/token (n={int(mk.sum())})")
    rows.append(f"  non-hard tokens: {float((after - base)[labels == -1].mean()):+.3f}")
    print("\n".join(rows))


def check(ws):
    src = MultiSource()
    plan = json.loads((ws / "plan.json").read_text())
    load_mod(ws / "ops.py")
    core_idx = json.loads((ws / "state.json").read_text())["core_idx"] if (ws / "state.json").exists() else 0
    model = build(plan, core_idx)
    x, _ = src.sample(BS, SEQ + 1, torch.Generator().manual_seed(1))
    t0 = time.time()
    loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
    loss.backward()
    dt = time.time() - t0
    # causality: changing the future must not change ops' outputs at earlier positions
    x2 = x.clone(); x2[:, 70:] = torch.randint(0, VOCAB, x2[:, 70:].shape)
    with torch.no_grad():
        model(x[:, :-1]); r1 = {k: v.clone() for k, v in model.last["regs"].items()}
        model(x2[:, :-1]); r2 = model.last["regs"]
    bad = [k for k in r1 if not torch.equal(r1[k][:, :70], r2[k][:, :70])]
    print("OK: sandbox, builds, forward/backward." if not bad else f"CAUSALITY VIOLATION in slots {bad}")
    print(f"params: {model.param_counts()} (interface cap {MAX_INTERFACE}); forward+backward {dt:.2f}s "
          f"for a {BS}x{SEQ} batch (keep under ~0.3s or trials get slow)")
    for k, v in r1.items():
        if k != "tok":
            print(f"slot {k}: {v[0, :48].tolist()}")
    if bad:
        sys.exit(1)


# ------------------------------------------------------------------ loop
OPS_TEMPLATE = '''"""Your ops. Write everything yourself; nothing is pre-defined.

@op(domain_fn) def name(m, *slots, **kw) -> LongTensor (B, T) of ints in [0, domain).
  slots: LongTensor (B, T) inputs (slot "tok" = char ids; m.itos[i] is char i, vocab size m.V = 97).
  domain_fn(m, spec) -> int: number of distinct output values; spec.kwargs has the action's kwargs.
  Must be CAUSAL: output at t may only use inputs at positions <= t.
"""
import torch

from llmc.oplib import op
'''

API = f"""
## Task
Next-character prediction on THREE corpora at once: Shakespeare plays, simple children's stories
(TinyStories), and Python standard-library source. 97 chars (newline, printable ASCII, '¤' = other).
Score = val loss (nats/char) on each corpus after training a tiny transformer core from scratch
for {STEPS} steps of batch {BS} x {SEQ} chars (lower is better). A change is KEPT if the mean over the
three improves by > {KEEP_MARGIN}, OR one corpus improves by > {CORPUS_MARGIN} while no corpus gets
worse by > {KEEP_MARGIN}.

## The point
We want the smallest neural network, with capability moved into code. Think of each op as a small
SYMBOLIC EXPERT: exact, narrow, and cheap. It does NOT need to help everywhere: an expert that
tracks some structure that only exists in one kind of text is welcome. The network does the
soft part: deciding per position whether and how to use each expert (every reader has a learned
per-position router gate) and everything fuzzy. Several specialists side by side are fine.
"General" here means algorithmic (computed from the window by rules, so it would work on unseen text
of that kind), not memorized statistics, and not necessarily useful on every corpus.
After every iteration the harness tries the next smaller core ({' -> '.join(f'{a}/d{d}' for a, d in CORES)}; "LxHxM" = layers x heads x MLP
width) and adopts it if mean loss worsens by <= {SHRINK_TOL}. Code that lets a smaller core match a bigger one
is the best outcome. The core has NO positional encoding.

## Rules (enforced)
- Write every op yourself in ops.py. There is no library: no pre-written ops are available, and
  repository source is off-limits. Imports: torch, math, torch.nn.functional, functools, itertools,
  collections, and `from llmc.oplib import op` only.
- Ops see only their input slots (the current {SEQ}-char window). No file access, no corpus statistics,
  no embedded word lists or tables (literal collections > 64 items or strings > 120 chars are rejected).
  Rules and procedures, not memorized knowledge.
- Ops must be causal and reasonably fast (vectorized torch).
- Plan actions (plan.json, a list), only these two:
    {{"op": NAME, "inputs": [slots], "out": slot, "kwargs": {{...}}}}  run your op -> new int slot
    {{"reader": slot, "site": i}}  feed a slot to the network (zero-init embedding of its value, gated by a
                                  learned router) before layer i (0..number of layers; clamped if the core shrinks)
  Reader domains <= {MAX_READER_DOMAIN}; total reader params (domain x d + router) <= {MAX_INTERFACE}.
"""

PROMPT = """You are the teacher in a research loop that moves capability from a tiny neural network into
general algorithms written as code. Lower the score by editing ONLY ops.py and plan.json in the current
directory ({ws}).
{api}
## Tools (cheap; use them)
- clusters.md: where the CURRENT best model fails: clusters of its highest-loss tokens (by phantom
  gradient), with examples from each corpus and the model's top guesses. Read it first.
- ./tool check : sandbox check, build, forward/backward, causality test, timing, sample slot values.
- ./tool score cand.json : cand.json = extra plan actions (same format) using ops in ops.py. Adds them to
  the current trained model, trains only the new reader params briefly (~10-30 s), prints the dev-loss
  change per corpus and per failure cluster.
## Current state
{state}
## Workflow
1. Read clusters.md, report.md, history.json.
2. Think about what exact procedure (computable from the window alone) would resolve a big cluster,
   on whichever corpus it lives in. Write it in ops.py, check it, score it, iterate. Try several ideas.
3. Put the best into plan.json (the full plan; the next trial trains from scratch) and run ./tool check.
4. Stop and summarize in 3-4 sentences: what algorithms you wrote, why, and what scoring showed.
"""


def improved(res, st):
    """Mean gain > KEEP_MARGIN, or a specialist win: one corpus gains > CORPUS_MARGIN, none loses > KEEP_MARGIN."""
    if res["val_loss"] < st["best"] - KEEP_MARGIN:
        return True
    bp = st.get("best_per")
    if not bp:
        return False
    d = {c: res["per_corpus"][c] - bp[c] for c in CORPORA}
    return min(d.values()) < -CORPUS_MARGIN and max(d.values()) <= KEEP_MARGIN


def report(ws, res, hist):
    lines = ["# Latest trial", json.dumps({k: v for k, v in res.items() if k not in ("error", "transfer_pile")})]
    if "error" in res:
        lines += ["## ERROR (reverted)", "```", res["error"], "```"]
    lines += ["", "## History"] + [
        f"- iter {h['iter']}: kept={h['kept']} val={h['val_loss']} core={h.get('core')} per={h.get('per_corpus')} :: "
        f"{str(h.get('summary', ''))[:400]}" for h in hist]
    (ws / "report.md").write_text("\n".join(lines))


def _agent_hist(hist):
    return [{k: v for k, v in h.items() if k != "transfer_pile"} for h in hist]


def run(ws: Path, iters: int):
    ws = ws.resolve()
    ws.mkdir(parents=True, exist_ok=True)
    if not (ws / "ops.py").exists():
        (ws / "ops.py").write_text(OPS_TEMPLATE)
        (ws / "plan.json").write_text("[]")
    tool = ws / "tool"
    tool.write_text(f'''#!/bin/bash
# ./tool check | ./tool score cand.json
cmd=$1; cand=""
if [ -n "$2" ]; then cand=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); fi
cd {REPO} && exec uv run python -m llmc.discover_gen "$cmd" --ws {ws} ${{cand:+--cand "$cand"}}
''')
    tool.chmod(0o755)
    hp, sp = ws / "history_full.json", ws / "state.json"
    hist = json.loads(hp.read_text()) if hp.exists() else []
    st = json.loads(sp.read_text()) if sp.exists() else dict(core_idx=0)
    log = lambda *a: print(*a, flush=True)

    def save():
        hp.write_text(json.dumps(hist, indent=2))
        (ws / "history.json").write_text(json.dumps(_agent_hist(hist), indent=2))
        sp.write_text(json.dumps(st))

    def try_shrink(it):
        if st["core_idx"] + 1 >= len(CORES):
            return
        try:
            res, ck = trial(ws, st["core_idx"] + 1)
        except Exception:
            log("shrink failed:", traceback.format_exc()[-500:]); return
        ok = res["val_loss"] <= st["best"] + SHRINK_TOL
        hist.append(dict(iter=it, kind="shrink", kept=ok, summary=f"harness: try core {res['core']}", **res))
        log(f"iter {it} shrink -> {res['core']}: val={res['val_loss']} (best {st['best']}) adopted={ok} "
            f"transfer={res['transfer_pile']}")
        if ok:
            st["core_idx"] += 1
            st["best"], st["best_per"] = res["val_loss"], res["per_corpus"]
            torch.save(ck, ws / "ckpt.pt")
            diagnose(ws)
        save()

    if not hist:
        res, ck = trial(ws, 0)
        torch.save(ck, ws / "ckpt.pt")
        st["best"], st["best_per"] = res["val_loss"], res["per_corpus"]
        hist.append(dict(iter=0, kind="agent", kept=True, summary="baseline: no code", **res))
        save(); report(ws, res, hist); diagnose(ws)
        log("baseline", json.dumps(res))
        try_shrink(0)
    repo = str(REPO)
    deny = [f"{t}(/{repo}/{p}/**)" for t in ("Read", "Edit", "Write", "Glob", "Grep")
            for p in ("src", "docs", "scripts", ".venv", "runs/discover", "runs/greedy", "runs/m4", "runs/m4h",
                      "runs/decompile", "runs/early", "runs/m6site", "runs/wm", "runs/audit_ts")]
    deny += [f"Read(/{repo}/README.md)", f"Read(/{repo}/CLAUDE.md)"]
    for it in range(1 + max(h["iter"] for h in hist), 1 + max(h["iter"] for h in hist) + iters):
        for f in ("ops.py", "plan.json"):
            shutil.copy(ws / f, ws / (f + ".prev"))
        core = CORES[st["core_idx"]]
        state = (f"Iteration {it}. Best mean val loss so far: {st['best']:.4f} (per corpus {st.get('best_per')}) on core {core[0]} d={core[1]} "
                 f"(the next trial uses this core).")
        t0 = time.time()
        proc = subprocess.run(["claude", "-p", PROMPT.format(ws=ws, api=API, state=state), "--output-format", "json",
                               "--max-turns", "80", "--permission-mode", "acceptEdits",
                               "--allowedTools", "Read,Write,Edit,Glob,Grep,Bash(./tool:*),Bash(./tool *)",
                               "--disallowedTools", ",".join(deny)],
                              capture_output=True, text=True, timeout=5400, cwd=ws)
        agent_time = round(time.time() - t0, 1)
        try:
            o = json.loads(proc.stdout)
            summary, cost, turns = o.get("result", "")[-1200:], o.get("total_cost_usd"), o.get("num_turns")
        except Exception:
            summary, cost, turns = (proc.stdout + proc.stderr)[-1200:], None, None
        try:
            res, ck = trial(ws, st["core_idx"])
            ok = improved(res, st)
        except Exception:
            res, ck, ok = dict(val_loss=99.0, error=traceback.format_exc()[-1500:]), None, False
        if ok:
            st["best"], st["best_per"] = res["val_loss"], res["per_corpus"]
            torch.save(ck, ws / "ckpt.pt")
            diagnose(ws)
        else:
            for f in ("ops.py", "plan.json"):
                shutil.copy(ws / (f + ".prev"), ws / f)
        hist.append(dict(iter=it, kind="agent", kept=ok, summary=summary, agent_time=agent_time, agent_cost_usd=cost,
                         agent_turns=turns, **res))
        save(); report(ws, res, hist)
        log(f"iter {it}: kept={ok} val={res['val_loss']} per={res.get('per_corpus')} transfer={res.get('transfer_pile')} "
            f"core={res.get('core')} agent {agent_time}s turns={turns} cost={cost}")
        try_shrink(it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check", "score", "diagnose", "trial"])
    ap.add_argument("--ws", required=True)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--cand")
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    ws = Path(a.ws)
    if a.cmd == "run":
        run(ws, a.iters)
    elif a.cmd == "check":
        check(ws)
    elif a.cmd == "score":
        score(ws, a.cand)
    elif a.cmd == "diagnose":
        diagnose(ws)
    else:
        res, ck = trial(ws, a.core, steps=a.steps)
        torch.save(ck, ws / "ckpt.pt")
        print(json.dumps(res))


if __name__ == "__main__":
    main()
