"""Split LM loss on the HMM+copy source into copy-span tokens vs grammar tokens.

    uv run python scripts/copy_split.py runs/m4/copy_nocode runs/m4/copy_code:code
"""
import json
import sys

import torch
import torch.nn.functional as F

from llmc.hybrid import Hybrid
from llmc.ladder import apply_plan
from llmc.lmdata import HMMSource

PLAN = [{"op": "suffix_next", "inputs": ["tok"], "out": "sn"}, {"op": "suffix_len", "inputs": ["tok"], "out": "sl"},
        {"reader": "sn", "site": 0}, {"reader": "sl", "site": 0}]

PLAN2 = PLAN + [{"op": "since", "inputs": ["tok"], "out": "sc", "kwargs": {"value": 16}}, {"reader": "sc", "site": 0}]

src = HMMSource(vocab=16, states=16, branch=2, emit_alpha=0.1, seed=0, p_copy=0.5)
X = src.sample(1024, 65, torch.Generator().manual_seed(999))
is_copy = torch.zeros_like(X, dtype=torch.bool)
for b, t in (X == src.COPY).nonzero().tolist():
    is_copy[b, t + 2:t + 9] = True  # copied tokens after the first are predictable
c = is_copy[:, 1:]
for arg in sys.argv[1:]:
    path, _, flag = arg.partition(":")
    ck = torch.load(f"{path}/model.pt")
    m = Hybrid(src.vocab, 16, ck["arch"], 8, V=src.vocab)
    apply_plan(m, {"code": PLAN, "code2": PLAN2}.get(flag, []), [])
    m.load_state_dict(ck["state"])
    m.eval()
    with torch.no_grad():
        nll = F.cross_entropy(m(X[:, :-1]).flatten(0, 1), X[:, 1:].flatten(), reduction="none").view(len(X), -1)
    print(f"{path:24s} core {m.param_counts()['core']:5d}  all {nll.mean():.3f}  copy-span {nll[c].mean():.3f}"
          f"  grammar {nll[~c].mean():.3f}  (copy frac {c.float().mean():.3f})")

# Oracle check: how often is the code's suffix_next correct on copy-span targets, and the model loss there.
from llmc.oplib import _suffix_match  # noqa: E402

sn, sl = _suffix_match(X[:, :-1])
right = (sn == X[:, 1:]) & c
print(f"suffix_next correct on {right.sum() / c.sum():.3f} of copy-span targets "
      f"(len>=3 there: {(sl >= 3)[c].float().mean():.3f})")
for arg in sys.argv[1:]:
    path, _, flag = arg.partition(":")
    ck = torch.load(f"{path}/model.pt")
    m = Hybrid(src.vocab, 16, ck["arch"], 8, V=src.vocab)
    apply_plan(m, {"code": PLAN, "code2": PLAN2}.get(flag, []), [])
    m.load_state_dict(ck["state"])
    m.eval()
    with torch.no_grad():
        nll = F.cross_entropy(m(X[:, :-1]).flatten(0, 1), X[:, 1:].flatten(), reduction="none").view(len(X), -1)
    print(f"{path:24s} loss where code is right: {nll[right].mean():.3f}   where wrong: {nll[c & ~right].mean():.3f}")
