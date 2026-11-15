"""Direction 2 on our own trained char-level model (Tiny Shakespeare, Hybrid core, no code).

Same programs and metrics as decompile_heads.py, hooking our GBlock attention directly.

    uv run python scripts/decompile_own.py runs/m4h/h_4x4x128_none_s0/model.pt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from decompile_heads import PROGS, program_patterns  # noqa: E402

from llmc.hybrid import GBlock, Hybrid  # noqa: E402
from llmc.lmdata import CharSource  # noqa: E402

ST = dict(override={}, ablate={}, pats={}, zmean={}, capture=False, P=None)


def gblock_forward(self, x):
    B, T, D = x.shape
    H = self.n_heads
    if H:
        h = self.ln1(x)
        sh = lambda W: (h @ W.T).view(B, T, H, self.hd).transpose(1, 2)
        q, k, v = sh(self.Wq), sh(self.Wk), sh(self.Wv)
        s = (q @ k.transpose(-1, -2)) / self.hd ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        w = s.softmax(-1)
        l = self.layer_idx
        if ST["capture"]:
            ST["pats"][l] = w
        if l in ST["override"]:
            w = w.clone()
            for hh, prog in ST["override"][l]:
                w[:, hh] = ST["P"][prog]
        z = w @ v
        if ST["capture"]:
            ST["zmean"][l] = z.mean((0, 2), keepdim=True)
        if l in ST["ablate"]:
            z = z.clone()
            for hh in ST["ablate"][l]:
                z[:, hh] = ST["zmean"][l][:, hh]
        x = x + z.transpose(1, 2).reshape(B, T, H * self.hd) @ self.Wo.T
    if self.n_mlp:
        x = x + F.gelu(self.ln2(x) @ self.W1.T + self.b1) @ self.W2.T
    return x + self.b2


def main(path, n=64, T=128, tau=0.5):
    torch.set_grad_enabled(False)
    src = CharSource("data/tinyshakespeare.txt")
    ck = torch.load(path)
    model = Hybrid(src.vocab, 32, ck["arch"], 8, V=src.vocab)
    model.load_state_dict(ck["state"])
    model.eval()
    GBlock.forward = gblock_forward
    for i, b in enumerate(model.blocks):
        b.layer_idx = i
    X = src.sample(n, T + 1, torch.Generator().manual_seed(999), split="val")
    tokens = X[:, :-1]
    punct = torch.tensor([src.stoi[c] for c in ".,;:!?\n" if c in src.stoi])
    ST["P"] = program_patterns(tokens, punct)

    def loss(override=None, ablate=None):
        ST["override"], ST["ablate"] = override or {}, ablate or {}
        lg = model(tokens)
        ST["override"], ST["ablate"] = {}, {}
        return F.cross_entropy(lg.flatten(0, 1), X[:, 1:].flatten()).item()

    ST["capture"] = True
    base = loss()
    ST["capture"] = False

    def group(d):
        g = {}
        for (l, h), v in d.items():
            g.setdefault(l, []).append((h, v) if v is not None else h)
        return g

    results = []
    for l, blk in enumerate(model.blocks):
        P = ST["pats"].get(l)
        if P is None:
            continue
        for h in range(blk.n_heads):
            match = {p: float(1 - 0.5 * (P[:, h] - ST["P"][p]).abs().sum(-1)[:, 1:].mean()) for p in PROGS}
            best = max(match, key=match.get)
            results.append(dict(layer=l, head=h, prog=best, match=round(match[best], 3),
                                d_replace=round(loss(override=group({(l, h): best})) - base, 4),
                                d_ablate=round(loss(ablate=group({(l, h): None})) - base, 4)))
    progs = {(r["layer"], r["head"]): r["prog"] for r in results}
    good = [k for k in progs if next(r for r in results if (r["layer"], r["head"]) == k)["match"] >= tau]
    summary = dict(model=path, base_loss=round(base, 4), heads=len(results), tau=tau, n_matched=len(good),
                   matched_by_prog={p: sum(progs[k] == p for k in good) for p in PROGS},
                   replace_matched_dloss=round(loss(override=group({k: progs[k] for k in good})) - base, 4) if good else 0,
                   ablate_matched_dloss=round(loss(ablate=group({k: None for k in good})) - base, 4) if good else 0,
                   replace_all_dloss=round(loss(override=group(progs)) - base, 4),
                   ablate_all_dloss=round(loss(ablate=group({k: None for k in progs})) - base, 4),
                   frac_heads_small_damage=round(float(np.mean([r["d_replace"] < 0.01 for r in results])), 3),
                   frac_heads_replace_beats_ablate=round(float(np.mean([r["d_replace"] < r["d_ablate"] for r in results])), 3))
    out = Path("runs/decompile")
    out.mkdir(parents=True, exist_ok=True)
    (out / ("own_" + Path(path).parent.name + ".json")).write_text(json.dumps(dict(summary=summary, heads=results), indent=2))
    print(json.dumps(summary))
    for r in sorted(results, key=lambda r: -r["match"]):
        print(r)


if __name__ == "__main__":
    main(sys.argv[1])
