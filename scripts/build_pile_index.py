"""Stream a Pile training sample (monology/pile-uncopyrighted, shard 00), tokenize with the
Pythia tokenizer, build nested reversed suffix arrays (10M/30M/100M tokens), and save
validation docs for evaluation.

    uv run python scripts/build_pile_index.py
"""
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from llmc.audit.infinigram import build_index

D = Path("data/pile")
D.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
EOS = tok.eos_token_id
t0 = time.time()
if not (D / "train_tokens.npy").exists():
    ds = load_dataset("monology/pile-uncopyrighted", data_files="train/00.jsonl.zst", split="train", streaming=True)
    parts, n, buf = [], 0, []
    for ex in ds:
        buf.append(ex["text"])
        if len(buf) == 2000:
            for ids in tok(buf)["input_ids"]:
                parts.append(np.asarray(ids + [EOS], dtype=np.uint16))
                n += len(ids) + 1
            buf = []
            print(n, f"{time.time() - t0:.0f}s", flush=True)
            if n >= 100_000_000:
                break
    np.save(D / "train_tokens.npy", np.concatenate(parts))
toks = np.load(D / "train_tokens.npy")
if not (D / "valid_docs.json").exists():
    ds = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train", streaming=True)
    docs = []
    for ex in ds:
        if len(ex["text"]) > 1000:
            docs.append(ex["text"][:4000])
        if len(docs) >= 400:
            break
    (D / "valid_docs.json").write_text(json.dumps(docs))
for size in [10_000_000, 30_000_000, 100_000_000]:
    rev, sa = build_index(toks[:size])
    np.save(D / f"rev_{size // 1_000_000}M.npy", rev)
    np.save(D / f"sa_{size // 1_000_000}M.npy", sa)
    print("index", size, f"{time.time() - t0:.0f}s", flush=True)
