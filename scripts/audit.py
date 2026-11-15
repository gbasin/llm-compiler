"""Direction 1 audit: how much of a real LM's next-token competence is procedure-able?

For TinyStories-{1M..33M} on held-out validation stories, compare the model with
code predictors that need no learned weights:
  * infini-gram over the model's *entire* training corpus (longest-match counts),
  * in-context suffix match (copy from earlier in the same story).
Reports per model: losses, fraction of tokens where code >= model, top-1 stats,
oracle per-token min, and a learned mixture (fit on half the stories, evaluated
on the other half) = a realistic estimate of what code could take over.

    uv run python scripts/audit.py --n_stories 400
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmc.audit.infinigram import InfiniGram

D = Path("data/tinystories")
LEN_BUCKETS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 1000]


def bucket(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def ctx_suffix(toks):
    """in-context longest-suffix-match continuation (latest on ties), per position:
    M_t[j] = [toks_j == toks_t] * (M_{t-1}[j-1] + 1) for j < t."""
    T = len(toks)
    nxt = np.full(T, -1)
    ln = np.zeros(T, dtype=np.int64)
    M = np.zeros(T, dtype=np.int64)
    for t in range(T):
        prevM = np.zeros(T, dtype=np.int64)
        prevM[1:] = M[:-1]
        Mt = np.where(toks == toks[t], prevM + 1, 0)
        Mt[t:] = 0
        if Mt.max() > 0:
            j = int(np.flatnonzero(Mt == Mt.max())[-1])
            ln[t], nxt[t] = Mt[j], toks[j + 1]
        M = Mt
    return nxt, ln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_stories", type=int, default=400)
    ap.add_argument("--models", default="1M,3M,8M,28M,33M")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--out", default="runs/audit_ts")
    ap.add_argument("--tokenizer", default="roneneldan/TinyStories-33M")
    ap.add_argument("--model_fmt", default="roneneldan/TinyStories-{}")
    ap.add_argument("--rev", default=str(D / "rev.npy"))
    ap.add_argument("--sa", default=str(D / "sa.npy"))
    ap.add_argument("--eval_json", default=str(D / "valid_stories.json"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    V, EOS = len(tok), tok.eos_token_id
    stories = json.loads(Path(a.eval_json).read_text())[: a.n_stories]
    seqs = [np.asarray(tok(s)["input_ids"][: a.max_tokens], dtype=np.int64) for s in stories]
    ig = InfiniGram(np.load(a.rev), np.load(a.sa), V)
    cnt = np.bincount(ig.rev, minlength=V).astype(np.float64)
    ig.uni = (cnt + 1) / (cnt.sum() + V)
    print(f"index loaded ({ig.n} tokens) {time.time() - t0:.0f}s", flush=True)

    # ---------- code predictors (model-independent), per predicted position
    rows = []
    cache = out / "code_rows.npy"
    for si, s in enumerate([] if cache.exists() else seqs):
        full = np.concatenate([[EOS], s])  # story starts after EOS in the corpus
        T = len(s)
        # reversed contexts for predicting s[t+1] from s[..t] (t = 0..T-2)
        L = 64
        ctx = np.full((T - 1, L), -1)
        for t in range(T - 1):
            c = full[: t + 2][::-1][:L]  # current token s[t] first, then back to EOS
            ctx[t, : len(c)] = c
        mlen, lo, hi = ig.longest_match(ctx, max_len=L)
        counts = ig.next_counts(lo, hi)
        cn, cl = ctx_suffix(s)
        for t in range(T - 1):
            y = s[t + 1]
            u, c = counts[t]
            n = c.sum()
            cy = c[u == y].sum() if len(u) else 0.0
            p_ig = (cy + ig.uni[y]) / (n + 1.0)
            top = int(u[c.argmax()]) if len(u) else -1
            pmax = (c.max() / n) if n else 0.0
            rows.append((si, t, y, mlen[t], n, p_ig, top, pmax, cn[t], cl[t], ig.uni[y]))
        if si % 50 == 0:
            print(f"code preds story {si} {time.time() - t0:.0f}s", flush=True)
    R = np.load(cache) if cache.exists() else np.array(rows, dtype=np.float64)
    cols = dict(si=0, t=1, y=2, mlen=3, n=4, p_ig=5, top=6, pmax=7, cn=8, cl=9, p_uni=10)
    np.save(out / "code_rows.npy", R)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    for name in a.models.split(","):
        model = AutoModelForCausalLM.from_pretrained(a.model_fmt.format(name)).to(dev).eval()
        nparams = sum(p.numel() for p in model.parameters())
        p_model, top_model = [], []
        with torch.no_grad():
            for s in seqs:
                x = torch.tensor(s[None], device=dev)
                lp = F.log_softmax(model(x).logits[0, :-1].float(), -1)
                p_model.append(lp.gather(1, x[0, 1:, None])[:, 0].double().cpu().numpy())
                top_model.append(lp.argmax(-1).cpu().numpy())
        lpm = np.concatenate(p_model)  # log p_model(y)
        pm = np.exp(lpm)
        tm = np.concatenate(top_model)
        y = R[:, 2].astype(int)
        p_ig, p_uni = R[:, 5], R[:, 10]
        Lm, Lig, Lu = -lpm, -np.log(p_ig), -np.log(p_uni)
        ig_top, ctx_top = R[:, 6].astype(int), R[:, 8].astype(int)
        res = dict(
            params=nparams, tokens=len(y),
            loss_model=Lm.mean(), loss_infinigram=Lig.mean(), loss_unigram=Lu.mean(),
            frac_tokens_code_ge_model=float((Lig <= Lm).mean()),
            frac_competence_by_code=float((Lu.mean() - Lig.mean()) / (Lu.mean() - Lm.mean())),
            oracle_min_loss=float(np.minimum(Lm, Lig).mean()),
            acc_model=float((tm == y).mean()), acc_infinigram=float((ig_top == y).mean()),
            acc_ctxcopy=float((ctx_top == y).mean()),
            acc_code_right_model_wrong=float(((ig_top == y) & (tm != y)).mean()),
            acc_model_right_code_wrong=float(((ig_top != y) & (tm == y)).mean()),
        )
        # learned mixture: w = sigmoid(features . theta); p = w p_ig + (1-w) p_model
        feats = [np.eye(len(LEN_BUCKETS) - 1)[bucket(R[:, 3], LEN_BUCKETS)],
                 np.eye(len(LEN_BUCKETS) - 1)[bucket(R[:, 9], LEN_BUCKETS)],
                 np.log1p(R[:, 4])[:, None], R[:, 7][:, None],
                 (ctx_top == ig_top)[:, None].astype(float),
                 lpm[:, None], np.ones((len(y), 1))]
        X = torch.tensor(np.concatenate(feats, 1), dtype=torch.float32)
        LP = torch.tensor(np.stack([np.log(p_ig), lpm], 1), dtype=torch.float64)
        tr = torch.tensor(R[:, 0] % 2 == 0)
        th = torch.zeros(X.shape[1], requires_grad=True)
        opt = torch.optim.Adam([th], lr=0.05)
        for _ in range(600):
            lw = F.logsigmoid(X[tr] @ th).double()
            nll = -torch.logaddexp(lw + LP[tr, 0], torch.log1p(-lw.exp().clamp(max=1 - 1e-12)) + LP[tr, 1]).mean()
            opt.zero_grad(); nll.backward(); opt.step()
        with torch.no_grad():
            lw = F.logsigmoid(X[~tr] @ th).double()
            w = lw.exp()
            mix = -torch.logaddexp(lw + LP[~tr, 0], torch.log1p(-w.clamp(max=1 - 1e-12)) + LP[~tr, 1]).mean().item()
        base = float(Lm[~tr.numpy()].mean())
        res.update(mix_heldout_loss=mix, model_heldout_loss=base,
                   mix_gain_nats=base - mix, mix_gain_pct=100 * (base - mix) / base,
                   mean_code_weight=float(w.mean()))
        res = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}
        results[name] = res
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
        del model
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
