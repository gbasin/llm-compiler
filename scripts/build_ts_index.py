"""Tokenize TinyStories train with the GPT-Neo tokenizer and build the reversed suffix array.

    uv run python scripts/build_ts_index.py
Writes data/tinystories/{train_tokens.npy, rev.npy, sa.npy, valid_stories.json}.
"""
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

D = Path("data/tinystories")
tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
EOS = tok.eos_token_id


def stories(path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [s.strip() for s in txt.split("<|endoftext|>") if s.strip()]


t0 = time.time()
if not (D / "train_tokens.npy").exists():
    st = stories(D / "TinyStories-train.txt")
    print(len(st), "train stories", time.time() - t0, flush=True)
    parts = []
    for i in range(0, len(st), 20000):
        enc = tok(st[i:i + 20000])["input_ids"]
        for e in enc:
            parts.append(np.asarray(e + [EOS], dtype=np.uint16))
        if i % 200000 == 0:
            print(i, time.time() - t0, flush=True)
    toks = np.concatenate(parts)
    np.save(D / "train_tokens.npy", toks)
toks = np.load(D / "train_tokens.npy")
print("train tokens", len(toks), time.time() - t0, flush=True)
vs = stories(D / "TinyStories-valid.txt")
(D / "valid_stories.json").write_text(json.dumps(vs))

from llmc.audit.infinigram import build_index  # noqa: E402

rev, sa = build_index(toks)
np.save(D / "rev.npy", rev)
np.save(D / "sa.npy", sa)
print("index built", len(sa), time.time() - t0, flush=True)
