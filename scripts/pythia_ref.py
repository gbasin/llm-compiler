"""Pythia reference loss on rung 7's exact val windows (256 windows x 256 tokens of Pile val docs)."""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from llmc.lm7 import val_batches

out = {}
vb = val_batches(256, 256, False)
for name in sys.argv[1:]:
    m = AutoModelForCausalLM.from_pretrained(name).eval()
    tot = n = 0
    with torch.no_grad():
        for b in vb:
            x = b["x"]
            lg = m(x[:, :-1]).logits
            tot += F.cross_entropy(lg.flatten(0, 1).float(), x[:, 1:].flatten(), reduction="sum").item()
            n += x[:, 1:].numel()
    nonemb = sum(p.numel() for k, p in m.named_parameters() if "embed" not in k)
    out[name] = dict(val=round(tot / n, 4), non_embedding_params=nonemb)
    print(name, out[name], flush=True)
json.dump(out, open("runs/lm7/pythia_ref.json", "w"), indent=1)
