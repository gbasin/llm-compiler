"""Discovery loop for CALLABLE experts on Pile (BPE), built on lm9's chained-retrieval core.

An agent writes experts (llmc.xapi) in <ws>/experts.py. Each expert returns items (token + features) per
position; the looped core reads them by attention on pass 1, next to lm9's built-in menu (suffix-array
continuations), then reads lm9's verify call on pass 2. Nothing is pre-written: experts.py starts empty.

Signals for the agent (all computed on TRAINING text, never on val):
  diagnose.md : where the current best model loses (token categories, coverage of the true next token by
                each existing expert, read mass per expert, decoded high-loss examples nobody covers)
  ./tool hits : for experts in experts.py: coverage, NEW coverage (true token offered by this expert and no
                existing one), and the oracle gap = the best model's loss on those newly covered tokens
                (an upper bound on what the expert can save), plus speed
  ./tool score: short training (1-block core, 3k steps) with experts.py vs the kept experts; dev-loss delta
  ./tool check: sandbox + shapes + causality + timing
Harness (driven by the caller, one iteration = agent turn then `finish`): full trial (2-block x3 loops,
10k steps); keep if dev loss improves by MARGIN; val (held-out Pile docs) is reported but never shown to the agent.

    uv run python -m llmc.discover_call init --ws runs/xcall/x1
    uv run python -m llmc.discover_call finish --ws runs/xcall/x1
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lm9, xapi
from .lm7 import D, IG_CTX, TRAIN_LO, VOCAB, copy_props, ig_model, val_tokens
from .lm8 import KS, NCONT

REPO = Path(__file__).resolve().parents[2]
DEV_LO = 99_000_000  # training draws from [TRAIN_LO, DEV_LO); dev windows from [DEV_LO, 100M)
MARGIN = 0.01
import os
_SM = int(os.environ.get("XC_STEPS", "0"))  # smoke-test override
FULL = dict(blocks=2, loops=3, steps=_SM or 10000)
SCORE = dict(blocks=1, loops=3, steps=_SM or 3000)
SEQ, BS = 256, 16
MAX_BATCH_S = 0.6  # expert time per 4096 positions

ALLOWED_IMPORTS = {"numpy", "math", "re", "collections", "itertools", "functools", "heapq", "bisect", "string",
                   "llmc.xapi"}
BANNED_NAMES = {"open", "eval", "exec", "__import__", "compile", "globals", "locals", "vars", "breakpoint", "input",
                "getattr", "setattr", "delattr"}
BANNED_ATTRS = {"read_text", "read_bytes", "load", "loadtxt", "fromfile", "save", "savez", "system", "popen",
                "__dict__", "__globals__", "__subclasses__", "__builtins__", "memmap", "stream", "pos"}

TEMPLATE = '''"""Your callable experts. Write everything yourself; nothing is pre-defined. API: see the prompt."""
import numpy as np
from llmc.xapi import expert, table
'''


# ------------------------------------------------------------------ sandbox + loading
def static_check(src):
    errs = []
    if len(src) > 1_000_000:
        errs.append("experts.py larger than 1MB")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            errs += [f"line {node.lineno}: import {a.name} not allowed" for a in node.names if a.name not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and node.module not in ALLOWED_IMPORTS:
            errs.append(f"line {node.lineno}: from {node.module} not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f"line {node.lineno}: {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            errs.append(f"line {node.lineno}: .{node.attr} not allowed (use ctx.back / ctx.window)")
    return errs


def load_experts(path: Path):
    src = path.read_text()
    errs = static_check(src)
    if errs:
        raise ValueError("experts.py failed the sandbox check:\n  " + "\n  ".join(errs))
    xapi.EXPERTS.clear(); xapi.TABLES.clear()
    spec = importlib.util.spec_from_file_location(f"xexp_{time.time_ns()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(xapi.EXPERTS), dict(xapi.TABLES)


_VOCAB = None


def vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = json.loads((D / "vocab_str.json").read_text())
    return _VOCAB


def build_tables(ws: Path, tables: dict, log=print):
    out = {}
    cache = ws / "cache"
    cache.mkdir(exist_ok=True)
    corpus = None
    for name, fn in tables.items():
        h = hashlib.sha1(inspect.getsource(fn).encode()).hexdigest()[:12]
        f = cache / f"{name}-{h}.npz"
        if not f.exists():
            if corpus is None:
                corpus = np.load(D / "train_tokens.npy", mmap_mode="r")[:30_000_000]
            t0 = time.time()
            arrs = fn(corpus, vocab())
            if not isinstance(arrs, dict):
                raise ValueError(f"table {name} must return a dict of numpy arrays")
            np.savez(f, **{k: np.asarray(v) for k, v in arrs.items()})
            log(f"built table {name} in {time.time() - t0:.1f}s")
        z = np.load(f)
        out[name] = {k: z[k] for k in z.files}
    return out


class XSpec:
    """Layout of all agent items: expert order, item counts, global feature-id offsets."""

    def __init__(self, experts: dict):
        self.names = list(experts)
        self.items = [experts[n]["items"] for n in self.names]
        self.feats = [experts[n]["feats"] for n in self.names]
        self.M = sum(self.items)
        self.F = max([len(f) for f in self.feats] + [1])
        self.nfeat = 1 + sum(sum(f) for f in self.feats)

    def desc(self):
        return dict(names=self.names, items=self.items, feats=self.feats)


def run_experts(experts, tables, spec, stream, p, menu, seed):
    N = len(p)
    tok = np.full((N, max(spec.M, 1)), -1, dtype=np.int64)
    fid = np.zeros((N, max(spec.M, 1), spec.F), dtype=np.int64)
    eid = np.zeros((N, max(spec.M, 1)), dtype=np.int64)
    sid = np.zeros((N, max(spec.M, 1)), dtype=np.int64)
    m0, f0 = 0, 1
    for e, name in enumerate(spec.names):
        x = experts[name]
        rng = np.random.default_rng(seed)
        ctx = xapi.Ctx(stream, p, vocab(), xapi.Index(ig_model(), rng), menu, tables, rng)
        r = x["fn"](ctx)
        t, fe = (r if isinstance(r, tuple) else (r, None))
        t = np.asarray(t, dtype=np.int64)
        M = x["items"]
        if t.shape != (N, M):
            raise ValueError(f"expert {name}: tokens shape {t.shape}, expected {(N, M)}")
        if ((t < -1) | (t >= VOCAB)).any():
            raise ValueError(f"expert {name}: token ids outside [-1, {VOCAB})")
        tok[:, m0:m0 + M] = t
        eid[:, m0:m0 + M] = e
        sid[:, m0:m0 + M] = np.arange(m0, m0 + M)
        for j, size in enumerate(x["feats"]):
            if fe is None:
                raise ValueError(f"expert {name} declares feats but returned none")
            fe = np.asarray(fe, dtype=np.int64)
            if fe.shape != (N, M, len(x["feats"])):
                raise ValueError(f"expert {name}: feats shape {fe.shape}, expected {(N, M, len(x['feats']))}")
            v = fe[..., j]
            if ((v < 0) | (v >= size)).any():
                raise ValueError(f"expert {name}: feature {j} outside [0, {size})")
            fid[:, m0:m0 + M, j] = v + f0
            f0 += size
        m0 += M
    return tok, fid, eid, sid


class XState:
    """Per-process cache of the agent module (data workers load it themselves)."""
    path = None
    experts = tables = spec = None

    @classmethod
    def get(cls, path: Path | None):
        if path is None:
            return None
        if cls.path != path:
            ex, tb = load_experts(path)
            cls.experts, cls.tables, cls.spec, cls.path = ex, build_tables(path.parent, tb, log=lambda *a: None), XSpec(ex), path
        return cls


def make_batch(stream, starts, seq, xpath):
    b = lm9.make_batch(stream, starts, seq)
    X = XState.get(xpath)
    if X is not None and X.spec.M > 0:
        B = len(starts)
        p = (starts[:, None] + np.arange(seq)[None]).reshape(-1)
        menu = dict(cont=b["cont"].view(B * seq, len(KS), NCONT).numpy(), cnt=b["cnt"].view(B * seq, len(KS)).numpy())
        tok, fid, eid, sid = run_experts(X.experts, X.tables, X.spec, stream, p, menu, int(starts[0]) % (2 ** 31))
        b.update(xt=torch.from_numpy(tok).view(B, seq, -1), xf=torch.from_numpy(fid).view(B, seq, tok.shape[1], -1),
                 xe=torch.from_numpy(eid).view(B, seq, -1), xs=torch.from_numpy(sid).view(B, seq, -1))
    return b


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, seed, xpath):
        self.seed, self.xpath = seed, xpath

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        tok = np.load(D / "train_tokens.npy", mmap_mode="r")
        while True:
            yield make_batch(tok, rng.integers(TRAIN_LO + IG_CTX, DEV_LO - SEQ - 1, BS), SEQ, self.xpath)


def windows(kind, n):
    if kind == "dev":
        st = np.linspace(DEV_LO + IG_CTX, 100_000_000 - SEQ - 2, n).astype(np.int64)
        return np.load(D / "train_tokens.npy", mmap_mode="r"), st
    if kind == "sel":  # training text the agent may inspect
        st = np.random.default_rng(12345).integers(TRAIN_LO + IG_CTX, DEV_LO - SEQ - 1, n)
        return np.load(D / "train_tokens.npy", mmap_mode="r"), st
    v = val_tokens()
    return v, np.linspace(IG_CTX, len(v) - SEQ - 2, n).astype(np.int64)


def batches(kind, n, xpath):
    s, st = windows(kind, n)
    return [make_batch(s, st[i:i + 16], SEQ, xpath) for i in range(0, n, 16)]


# ------------------------------------------------------------------ model
class XLM(lm9.ChainLM):
    def __init__(self, spec: XSpec | None, **kw):
        super().__init__(**kw)
        d = self.emb.weight.shape[1]
        self.nx = len(spec.names) if spec else 0
        if spec and spec.M > 0:
            self.xexp = nn.Embedding(self.nx, d)
            self.xslot = nn.Embedding(spec.M, d)
            self.xfeat = nn.Embedding(spec.nfeat, d, padding_idx=0)

    def items1(self, b):
        it, ok = super().items1(b)
        if "xt" in b and self.nx:
            xt = b["xt"]
            xi = self.emb(xt.clamp(min=0)) + self.xexp(b["xe"]) + self.xslot(b["xs"]) + self.xfeat(b["xf"]).sum(-2)
            it, ok = torch.cat([it, xi], 2), torch.cat([ok, xt >= 0], 2)
        return it, ok


def make_model(spec, cfg):
    return XLM(spec, blocks=cfg["blocks"], loops=cfg["loops"], seq=SEQ)


@torch.no_grad()
def evaluate(model, bs, dev, per_token=False):
    model.eval()
    tot = n = 0
    nll, mass = [], []
    for b in bs:
        b = {k: v.to(dev) for k, v in b.items()}
        lg, w1, _ = model(b, return_w=True)
        l = F.cross_entropy(lg.flatten(0, 1).float(), b["x"][:, 1:].flatten(), reduction="none")
        tot += l.sum().item(); n += l.numel()
        if per_token:
            nll.append(l.view(lg.shape[:2]).cpu()); mass.append(w1.cpu())
    model.train()
    return (tot / n, nll, mass) if per_token else tot / n


def trial(ws: Path, cfg, xpath: Path | None, seed=0, log=print, with_val=True):
    dev = torch.device("mps")
    torch.manual_seed(seed)
    X = XState.get(xpath)
    spec = X.spec if X else None
    model = make_model(spec, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01, betas=(0.9, 0.95))
    S = cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, S) / S))))
    dl = torch.utils.data.DataLoader(Stream(seed, xpath), batch_size=None, num_workers=6, persistent_workers=False,
                                     prefetch_factor=4)
    devb = batches("dev", 128, xpath)
    t0 = time.time()
    it = iter(dl)
    for step in range(S):
        b = {k: v.to(dev) for k, v in next(it).items()}
        loss = F.cross_entropy(model(b).flatten(0, 1), b["x"][:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 999:
            log(f"  step {step + 1}: train {loss.item():.3f} ({time.time() - t0:.0f}s)")
    del it, dl
    res = dict(dev=round(evaluate(model, devb, dev), 4), core=model.core_params(), cfg=cfg,
               experts=spec.desc() if spec else None, train_s=round(time.time() - t0))
    if with_val:
        res["val"] = round(evaluate(model, batches("val", 256, xpath), dev), 4)
    return res, model


# ------------------------------------------------------------------ signals
def categorize(tok_ids):
    v = vocab()
    cats = []
    for t in tok_ids:
        s = v[t] if 0 <= t < len(v) else ""
        if s.strip() == "":
            c = "whitespace/newline"
        elif s.strip().isdigit():
            c = "digits"
        elif not any(ch.isalnum() for ch in s):
            c = "punct/symbol"
        elif s.startswith(" ") or s.startswith("\n"):
            c = "word start"
        else:
            c = "word piece (continues a word)"
        cats.append(c)
    return np.array(cats)


def item_sources(b, spec):
    """(B,T,Mall) token ids of every item the model can read, and a source name per item column."""
    C = len(KS)
    cols, names = [b["cont"].flatten(2)], [f"menu k={k}" for k in KS for _ in range(NCONT)]
    cp = copy_props(b["x"][:, :-1]); cols.append(cp); names += ["in-context copy"] * cp.shape[-1]
    if "xt" in b and spec is not None:
        cols.append(b["xt"])
        names += [n for n, m in zip(spec.names, spec.items) for _ in range(m)]
    return torch.cat(cols, -1), names


def diagnose(ws: Path, model_path: Path, xpath: Path | None, out: Path):
    dev = torch.device("mps")
    X = XState.get(xpath)
    spec = X.spec if X else None
    st = json.loads((ws / "state.json").read_text())
    model = make_model(spec, st["cfg_full"]).to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    bs = batches("sel", 64, xpath)
    mean, nll, mass = evaluate(model, bs, dev, per_token=True)
    nll = torch.cat(nll).numpy().reshape(-1)
    y = torch.cat([b["x"][:, 1:] for b in bs]).numpy().reshape(-1)
    np.save(ws / "sel_nll.npy", nll)
    cats = categorize(y)
    L = [f"# Where the current best model loses (training text, {len(y)} tokens; mean loss {mean:.3f} nats/token)\n"]
    L.append("## By token category\n| category | share of tokens | mean loss | share of total loss |\n|---|---|---|---|")
    for c in sorted(set(cats), key=lambda c: -nll[cats == c].sum()):
        m = cats == c
        L.append(f"| {c} | {m.mean():.1%} | {nll[m].mean():.2f} | {nll[m].sum() / nll.sum():.1%} |")
    items, names = [], None
    for b in bs:
        it, names = item_sources(b, spec)
        items.append(it)
    items = torch.cat(items).flatten(0, 1).numpy()
    hit = items == y[:, None]
    L.append("\n## Coverage: is the true next token among an expert's items?\n"
             "| source | covers | covers and no other source does | best-model loss on those uniquely covered |\n|---|---|---|---|")
    srcs = list(dict.fromkeys(names))
    names = np.array(names)
    by = {s: hit[:, names == s].any(1) for s in srcs}
    anyhit = hit.any(1)
    for s in srcs:
        others = np.zeros_like(anyhit)
        for s2 in srcs:
            if s2 != s:
                others |= by[s2]
        u = by[s] & ~others
        L.append(f"| {s} | {by[s].mean():.1%} | {u.mean():.1%} | {nll[u].sum() / len(y):.3f} nats/token |")
    L.append(f"\nCovered by at least one item: {anyhit.mean():.1%} of tokens, mean loss there {nll[anyhit].mean():.2f}; "
             f"NOT covered: {(~anyhit).mean():.1%} of tokens, mean loss {nll[~anyhit].mean():.2f}, "
             f"= {nll[~anyhit].sum() / nll.sum():.1%} of all loss.")
    w1 = torch.cat([m.flatten(0, 1) for m in mass]).numpy()  # (N, M1)
    m1names = list(names[:len(KS) * NCONT]) + ([n for n, m in zip(spec.names, spec.items) for _ in range(m)] if spec and spec.M else [])
    m1names = np.array(m1names)
    L.append("\n## Pass-1 read mass (how much attention the core gives each source; the rest goes to 'ignore')\n")
    L.append(", ".join(f"{s}: {w1[:, m1names == s].sum(1).mean():.3f}" for s in dict.fromkeys(m1names)))
    v = vocab()
    toks = torch.cat([b["x"][:, :-1] for b in bs]).numpy().reshape(-1)
    order = np.argsort(-nll * (~anyhit))[:40]
    L.append("\n## 40 highest-loss tokens no expert covers (context = last 24 tokens, `|` marks the target)\n")
    for i in order:
        w0 = (i // SEQ) * SEQ
        ctx = "".join(v[t] for t in toks[max(w0, i - 23):i + 1])
        L.append(f"- loss {nll[i]:.1f} [{cats[i]}]: `{ctx[-160:]!r}` | `{v[y[i]]!r}`")
    out.write_text("\n".join(L) + "\n")
    return mean


def hits(ws: Path):
    """Coverage signal for the candidate experts.py against the kept ones."""
    xp = ws / "experts.py"
    t0 = time.time()
    s, st = windows("sel", 64)
    ke, kt = load_experts(ws / "kept_experts.py")
    kspec, ktab = XSpec(ke), build_tables(ws, kt)
    ce, ct = load_experts(xp)
    cspec, ctab = XSpec(ce), build_tables(ws, ct)
    nll = np.load(ws / "sel_nll.npy")
    ys, base, cand, times = [], [], [], []
    for i in range(0, 64, 16):
        b = lm9.make_batch(s, st[i:i + 16], SEQ)
        B = len(st[i:i + 16])
        p = (st[i:i + 16, None] + np.arange(SEQ)[None]).reshape(-1)
        menu = dict(cont=b["cont"].view(B * SEQ, len(KS), NCONT).numpy(), cnt=b["cnt"].view(B * SEQ, len(KS)).numpy())
        if kspec.M:
            kt_, kf, ke_, ks_ = run_experts(ke, ktab, kspec, s, p, menu, int(st[i]))
            b.update(xt=torch.from_numpy(kt_).view(B, SEQ, -1), xe=torch.from_numpy(ke_).view(B, SEQ, -1))
        it, _ = item_sources(b, kspec if kspec.M else None)
        base.append(it.flatten(0, 1).numpy())
        t1 = time.time()
        tok, _, eid, _ = run_experts(ce, ctab, cspec, s, p, menu, int(st[i]))
        times.append(time.time() - t1)
        cand.append(tok)
        ys.append(b["x"][:, 1:].reshape(-1).numpy())
    y, base, cand = np.concatenate(ys), np.concatenate(base), np.concatenate(cand)
    covered = (base == y[:, None]).any(1)
    out = [f"hits on {len(y)} training tokens (best-model mean loss {nll.mean():.3f}); "
           f"existing sources cover {covered.mean():.1%} of next tokens"]
    m0 = 0
    for name, m in zip(cspec.names, cspec.items):
        h = (cand[:, m0:m0 + m] == y[:, None]).any(1)
        new = h & ~covered
        filled = (cand[:, m0:m0 + m] >= 0).mean()
        out.append(f"  {name}: items filled {filled:.1%}, covers {h.mean():.1%}, NEW coverage {new.mean():.2%}, "
                   f"oracle gap {nll[new].sum() / len(y):.4f} nats/token (best-model loss on newly covered), "
                   f"already-covered overlap {(h & covered).mean():.1%}")
        m0 += m
    out.append(f"  expert time: {np.mean(times):.2f}s per 4096 positions (limit {MAX_BATCH_S}s); tool {time.time() - t0:.0f}s")
    print("\n".join(out))


def check(ws: Path):
    xp = ws / "experts.py"
    ex, tb = load_experts(xp)
    print(f"sandbox ok; experts: {list(ex)}; tables: {list(tb)}")
    tabs = build_tables(ws, tb)
    spec = XSpec(ex)
    s = np.load(D / "train_tokens.npy", mmap_mode="r")
    st = np.random.default_rng(7).integers(TRAIN_LO + IG_CTX, DEV_LO - SEQ - 1, 16)
    b = lm9.make_batch(s, st, SEQ)
    p = (st[:, None] + np.arange(SEQ)[None]).reshape(-1)
    menu = dict(cont=b["cont"].view(-1, len(KS), NCONT).numpy(), cnt=b["cnt"].view(-1, len(KS)).numpy())
    t0 = time.time()
    tok, fid, eid, sid = run_experts(ex, tabs, spec, s, p, menu, 1)
    dt = time.time() - t0
    print(f"shapes ok: {spec.M} items/position, {dt:.2f}s per 4096 positions (limit {MAX_BATCH_S}s)"
          + ("  <-- TOO SLOW" if dt > MAX_BATCH_S else ""))
    # causality: each probe position alone, with every later token scrambled
    bad = []
    rng = np.random.default_rng(3)
    for q in rng.choice(p, 24, replace=False):
        a = run_experts(ex, tabs, spec, s, np.array([q]), {k: v[p == q] for k, v in menu.items()}, 5)
        s2 = np.array(s[q - 4000:q + 300]).astype(np.int64)
        s2[4001:] = rng.integers(0, 50000, len(s2) - 4001)
        b2 = run_experts(ex, tabs, spec, s2, np.array([4000]), {k: v[p == q] for k, v in menu.items()}, 5)
        if not (np.array_equal(a[0], b2[0]) and np.array_equal(a[1], b2[1])):
            bad.append(int(q))
    print("causality ok" if not bad else f"CAUSALITY VIOLATION at {len(bad)}/24 probes: outputs change when FUTURE tokens change")
    v = vocab()
    for i in rng.choice(len(p), 3, replace=False):
        ctx = "".join(v[t] for t in s[p[i] - 12:p[i] + 1])
        print(f"  sample: ...{ctx!r} -> next {v[s[p[i] + 1]]!r}; items: " +
              ", ".join(f"{spec.names[eid[i, j]]}:{v[tok[i, j]]!r}" for j in range(spec.M) if tok[i, j] >= 0))
    if dt > MAX_BATCH_S or bad:
        sys.exit(1)


def score(ws: Path):
    st = json.loads((ws / "state.json").read_text())
    kh = hashlib.sha1((ws / "kept_experts.py").read_bytes()).hexdigest()[:12]
    bp = ws / f"score_base_{kh}.json"
    if not bp.exists():
        r, _ = trial(ws, SCORE, ws / "kept_experts.py", log=lambda *a: None, with_val=False)
        bp.write_text(json.dumps(r))
    base = json.loads(bp.read_text())
    tmp = ws / "cache" / f"cand_{time.time_ns()}.py"
    shutil.copy(ws / "experts.py", tmp)
    r, _ = trial(ws, SCORE, tmp, log=lambda *a: None, with_val=False)
    tmp.unlink()
    print(f"score (1-block core, {SCORE['steps']} steps, dev loss): kept experts {base['dev']:.4f} -> experts.py {r['dev']:.4f} "
          f"(delta {r['dev'] - base['dev']:+.4f}; noise ~0.01)")


# ------------------------------------------------------------------ driver
PROMPT_FILE = """# Task: write callable experts for a tiny looped language model (Pile text, GPT-NeoX BPE tokens)

Goal of the project: a minimal neural "reasoning core" with as much capability as possible moved OUT of the
weights into code and data. The core (a 2-block transformer applied 3 times) predicts the next token. Between
passes it READS items produced by experts: plain code (+ data tables built from a 30M-token Pile corpus) that,
for every position, returns a few candidate tokens with small integer features. The core learns which items to
trust where; an expert does not need to help everywhere, only to be right and readable where it fires.

Already built in (you cannot change them): a suffix-array "menu" (for k in 1,2,3,4,5,6,8: the 4 tokens that
followed one corpus occurrence of the last k tokens), a "verify" call (for each menu guess y, how often y follows
the context, and what followed y), and in-context copy (the token after the latest earlier occurrence of the
last 1/2/3 tokens). Your experts add to these.

Edit ONLY experts.py in this directory. Do not read or edit anything outside this directory.
Allowed imports: numpy, math, re, collections, itertools, functools, heapq, bisect, string, llmc.xapi.

## API
{api}

## Tools (run from this directory)
- diagnose.md : where the current best model loses, which sources cover the true next token, read mass,
  and decoded high-loss examples that NO source covers. Read it first.
- ./tool check : sandbox, shapes, causality and speed test, sample items.
- ./tool hits  : coverage of your experts on training text: NEW coverage (true token offered by your expert
  and by no existing source) and its oracle gap (an upper bound on the nats/token it could save). Fast (~1 min).
- ./tool score : short training run with your experts vs the kept ones; dev-loss delta (~15-25 min, noise ~0.01).
  Use sparingly: at most 2 times.
Items that duplicate existing sources add little; precise items where nothing else is right add a lot.
Features matter: they let the core know WHEN to trust an item (e.g. a confidence or a count bucket).

## State
{state}

## Workflow
1. Read diagnose.md and history.json (earlier attempts and their results).
2. Look for patterns in what is not covered (structure, not memorized strings: e.g. closing brackets/quotes,
   numbers and counting, list continuation, word completion from a prefix, syntax, dates, names seen earlier).
   Write experts (and tables if useful), ./tool check, ./tool hits, iterate. Keep what helps; delete what doesn't.
3. Leave experts.py in its final state (it is trained from scratch in a full trial afterwards; check must pass).
4. Reply with 3-5 sentences: what experts you wrote, why, and what hits/score showed.
"""


def write_tool(ws: Path):
    t = ws / "tool"
    t.write_text(f"#!/bin/bash\ncd {REPO} && exec uv run python -m llmc.discover_call \"$1\" --ws {ws}\n")
    t.chmod(0o755)


def prompt(ws: Path):
    st = json.loads((ws / "state.json").read_text())
    state = (f"Iteration {st['iter'] + 1}. Best dev loss so far {st['best_dev']:.4f} (full trial, 2-block core x3 loops). "
             f"A new set of experts is kept if the full trial improves dev loss by at least {MARGIN}. "
             f"Kept experts so far: {st.get('kept_names', [])} (they are in experts.py now; you may improve or replace them).")
    api = xapi.__doc__.split("\n\n", 1)[1]
    (ws / "PROMPT.md").write_text(PROMPT_FILE.format(api=api, state=state))
    return ws / "PROMPT.md"


def init(ws: Path):
    ws = ws.resolve(); ws.mkdir(parents=True, exist_ok=True)
    (ws / "experts.py").write_text(TEMPLATE)
    shutil.copy(ws / "experts.py", ws / "kept_experts.py")
    write_tool(ws)
    st = dict(iter=0, cfg_full=FULL)
    (ws / "state.json").write_text(json.dumps(st))
    res, model = trial(ws, FULL, ws / "kept_experts.py")
    torch.save(model.state_dict(), ws / "best.pt")
    st.update(best_dev=res["dev"], best_val=res["val"], kept_names=[])
    (ws / "state.json").write_text(json.dumps(st))
    (ws / "history_full.json").write_text(json.dumps([dict(iter=0, kept=True, summary="baseline: no agent experts", **res)], indent=2))
    (ws / "history.json").write_text(json.dumps([dict(iter=0, kept=True, summary="baseline: no agent experts", dev=res["dev"])], indent=2))
    diagnose(ws, ws / "best.pt", ws / "kept_experts.py", ws / "diagnose.md")
    prompt(ws)
    print(json.dumps(res))


def finish(ws: Path, summary: str):
    ws = ws.resolve()
    st = json.loads((ws / "state.json").read_text())
    hist = json.loads((ws / "history_full.json").read_text())
    it = st["iter"] + 1
    cand = ws / "cache" / f"trial_{it}.py"
    shutil.copy(ws / "experts.py", cand)
    try:
        res, model = trial(ws, FULL, cand)
        kept = res["dev"] <= st["best_dev"] - MARGIN
    except Exception:
        res, model, kept = dict(dev=99.0, error=traceback.format_exc()[-1500:]), None, False
    if kept:
        shutil.copy(cand, ws / "kept_experts.py")
        torch.save(model.state_dict(), ws / "best.pt")
        st.update(best_dev=res["dev"], best_val=res["val"], kept_names=res["experts"]["names"])
        XState.path = None
        diagnose(ws, ws / "best.pt", ws / "kept_experts.py", ws / "diagnose.md")
    else:
        shutil.copy(ws / "kept_experts.py", ws / "experts.py")
    st["iter"] = it
    hist.append(dict(iter=it, kept=kept, summary=summary, **res))
    (ws / "state.json").write_text(json.dumps(st))
    (ws / "history_full.json").write_text(json.dumps(hist, indent=2))
    (ws / "history.json").write_text(json.dumps([dict(iter=h["iter"], kept=h["kept"], summary=h["summary"], dev=h.get("dev"),
                                                      experts=(h.get("experts") or {}).get("names")) for h in hist], indent=2))
    prompt(ws)
    print(json.dumps(dict(iter=it, kept=kept, dev=res.get("dev"), val=res.get("val"), best_dev=st["best_dev"],
                          error=res.get("error"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "finish", "check", "hits", "score", "diagnose", "prompt"])
    ap.add_argument("--ws", required=True)
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    ws = Path(a.ws).resolve()
    if a.cmd == "init":
        init(ws)
    elif a.cmd == "finish":
        finish(ws, a.summary)
    elif a.cmd == "check":
        check(ws)
    elif a.cmd == "hits":
        hits(ws)
    elif a.cmd == "score":
        score(ws)
    elif a.cmd == "diagnose":
        diagnose(ws, ws / "best.pt", ws / "kept_experts.py", ws / "diagnose.md")
    elif a.cmd == "prompt":
        print(prompt(ws))


if __name__ == "__main__":
    main()
