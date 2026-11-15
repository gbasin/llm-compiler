"""Rungs 2-3: a looped core that CALLS a code expert between loop steps.

Task: k-hop pointer chasing (llmc.loop.PointerTask). Expert (code): lookup(key) = the value paired
with `key` in the context ("x P(x)" pairs), computed exactly from the tokens. Each loop step:
  1. block runs on the sequence (input injection of e0 and of the result register at the last position)
  2. query head on the last position -> distribution over N keys + NOOP
  3. expert executes; result -> register (embedding of the returned symbol); NOOP keeps the register
Query training:
  labeled : CE(query, true current symbol) for t < k, NOOP for t >= k (training wheels); hard argmax used
  soft    : no labels; register = p_noop * reg + sum_i p_i * E[lookup(i)] (candidate method: the expert
            answers every candidate key, the core weights them)
  st      : no labels; hard argmax forward, soft gradient (straight-through)
Readout at the final step. Compare tiny cores (+expert) with llmc.loop's plain looped/stacked cores.

    uv run python -m llmc.loop_expert --query soft --name r3_soft
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid import GBlock
from .loop import PointerTask


def expert_table(tok, n):
    """Code: parse the 'x P(x)' pairs from the context -> table[b, x] = P(x)."""
    B = tok.shape[0]
    pairs = tok[:, :2 * n].view(B, n, 2)
    table = torch.zeros(B, n, dtype=torch.long)
    table.scatter_(1, pairs[..., 0], pairs[..., 1])
    return table


class CallerCore(nn.Module):
    def __init__(self, vocab, length, n, d=32, heads=1, mlp=64, step_emb=True, max_steps=64):
        super().__init__()
        self.n = n
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(length, d) * 0.02)
        self.block = GBlock(d, heads, max(d // max(heads, 1), 1), mlp)
        self.inject = nn.Linear(3 * d, d, bias=False)
        with torch.no_grad():
            self.inject.weight.copy_(torch.cat([torch.eye(d)] * 3, 1) / 3)
        self.step = nn.Parameter(torch.zeros(max_steps, d)) if step_emb else None
        self.res_emb = nn.Embedding(n, d)  # result port: embedding of the returned symbol
        self.query = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n + 1))  # query port: keys + NOOP
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def core_params(self):
        """Core = block + injection + step emb. Lexicon (emb/pos/head) and ports (res_emb/query) excluded."""
        skip = ("emb.", "pos", "head.", "ln.", "res_emb.", "query.")
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith(skip))

    def port_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith(("res_emb.", "query.")))

    def forward(self, tok, T, mode="soft", use_expert=True, all_steps=False):
        B, L = tok.shape
        e0 = self.emb(tok) + self.pos[:L]
        table = expert_table(tok, self.n)  # (B, n)
        cand = self.res_emb(table)  # (B, n, d): the expert's answer for every candidate key
        reg = torch.zeros(B, e0.shape[-1])
        x = e0
        qlogits, outs = [], []
        for t in range(T):
            r = torch.zeros_like(e0)
            r[:, -1] = reg
            x = self.inject(torch.cat([x, e0, r], -1))
            if self.step is not None:
                x = x + self.step[min(t, len(self.step) - 1)]
            x = self.block(x)
            if use_expert:
                ql = self.query(x[:, -1])
                qlogits.append(ql)
                p = ql.softmax(-1)
                if mode == "soft":
                    reg = p[:, -1:] * reg + torch.einsum("bn,bnd->bd", p[:, :-1], cand)
                else:  # labeled (hard) or straight-through
                    hard = F.one_hot(ql.argmax(-1), self.n + 1).float()
                    w = hard + (p - p.detach()) if mode == "st" else hard
                    reg = w[:, -1:] * reg + torch.einsum("bn,bnd->bd", w[:, :-1], cand)
            if all_steps:
                outs.append(self.head(self.ln(x[:, -1])))
        logits = self.head(self.ln(x[:, -1]))
        return (outs if all_steps else logits), qlogits


def chain(tok, n, kmax):
    """Ground-truth symbols after 0..kmax hops (for query labels)."""
    table = expert_table(tok, n)
    s = tok[:, 2 * n + 1]  # QUERY at 2n, start symbol at 2n+1
    cur, out = s, [s]
    for _ in range(kmax):
        cur = table.gather(1, cur[:, None])[:, 0]
        out.append(cur)
    return torch.stack(out, 1)  # (B, kmax+1)


@torch.no_grad()
def eval_grid(model, task, Ts, ks, mode, use_expert, n=512, seed=123):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    grid = {}
    for k in ks:
        tok, y, _ = task.sample(n, g, kmin=k, kmax=k)
        outs, _ = model(tok, T=max(Ts), mode="labeled" if mode == "labeled" else mode, use_expert=use_expert, all_steps=True)
        for T in Ts:
            grid[(k, T)] = (outs[T - 1].argmax(-1) == y).float().mean().item()
    model.train()
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--query", choices=["labeled", "soft", "st", "none"], required=True)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--mlp", type=int, default=64)
    ap.add_argument("--tmin", type=int, default=8)
    ap.add_argument("--tmax", type=int, default=12)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k_train", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out_dir", default="runs/loop")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    task = PointerTask(a.n, kmax=12)
    model = CallerCore(task.vocab, task.length, a.n, a.d, a.heads, a.mlp)
    use_expert = a.query != "none"
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500))
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    f = (out / "log.jsonl").open("w")
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    ks, Ts = list(range(1, 13)), [1, 2, 4, 6, 8, 12, 16, 24]
    for step in range(a.steps + 1):
        if step % 2000 == 0 or step == a.steps:
            grid = eval_grid(model, task, Ts, ks, a.query, use_expert)
            rec = dict(step=step, query=a.query, core=model.core_params(), ports=model.port_params(),
                       time=round(time.time() - t0), acc_by_k={k: round(grid[(k, a.tmax)], 3) for k in ks},
                       grid={f"k{k}_T{T}": round(v, 3) for (k, T), v in grid.items()})
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps({x: rec[x] for x in ("step", "core", "ports", "time", "acc_by_k")}), flush=True)
        if step == a.steps:
            break
        tok, y, k = task.sample(a.bs, g, kmin=1, kmax=a.k_train)
        T = int(torch.randint(a.tmin, a.tmax + 1, (1,), generator=g))
        logits, ql = model(tok, T=T, mode=a.query if use_expert else "soft", use_expert=use_expert)
        loss = F.cross_entropy(logits, y)
        if a.query == "labeled":
            ch = chain(tok, a.n, T)
            for t, q in enumerate(ql):
                lab = torch.where(k > t, ch[:, t], torch.full_like(k, a.n))  # NOOP once k hops are done
                loss = loss + F.cross_entropy(q, lab) / len(ql)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()


if __name__ == "__main__":
    main()
