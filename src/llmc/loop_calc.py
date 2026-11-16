"""Rung 5: chained calls to a calculator expert from a looped core.

Task (generated): "LOAD x0  op1 a1  op2 a2 ... opk ak  PAD.. =" with ops in {+, -, *} mod M, evaluated
left to right; target = the result. Each loop step the core picks WHICH step to execute (or NOOP);
the calculator (code) applies it to the current value. The register is a distribution over the M values,
so the call is exact AND differentiable: the expert pushes the distribution through
v -> calc(v, op_i, a_i) for every candidate step i, and the core's soft query mixes the candidates.
  labeled : query label = step t at loop t (training wheels); soft : no labels.
Readout: head(LN(h_last)) + w * log(register) on the value tokens (logit port).
Plain control: llmc.loop.Looped (stacked) on the same task.

    uv run python -m llmc.loop_calc --arm soft --name r5_soft
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
from .loop import Looped


class CalcTask:
    def __init__(self, M=50, kmax=8):
        self.M, self.kmax = M, kmax
        self.LOAD, self.ADD, self.SUB, self.MUL, self.PAD, self.EQ = range(M, M + 6)
        self.vocab = M + 6
        self.S = kmax + 1  # step slots (slot 0 = LOAD)
        self.length = 2 * self.S + 1

    def sample(self, B, g, kmin=1, kmax=None):
        M, S = self.M, self.S
        kmax = kmax or self.kmax
        k = torch.randint(kmin, kmax + 1, (B,), generator=g)
        ops = torch.randint(0, 3, (B, S), generator=g) + self.ADD
        args = torch.randint(0, M, (B, S), generator=g)
        ops[:, 0] = self.LOAD
        live = torch.arange(S)[None] <= k[:, None]
        acc = args[:, 0].clone()
        for i in range(1, S):
            nxt = calc(acc, ops[:, i], args[:, i], self)
            acc = torch.where(live[:, i], nxt, acc)
        ops = torch.where(live, ops, torch.full_like(ops, self.PAD))
        args = torch.where(live, args, torch.full_like(args, self.PAD))
        tok = torch.cat([torch.stack([ops, args], -1).view(B, 2 * S), torch.full((B, 1), self.EQ)], 1)
        return tok, acc, k


def calc(v, op, a, task):
    M = task.M
    return torch.where(op == task.ADD, (v + a) % M, torch.where(op == task.SUB, (v - a) % M, (v * a) % M))


class CalcCaller(nn.Module):
    def __init__(self, task, d=32, heads=1, mlp=64, counter=True, max_steps=64, pc=False):
        super().__init__()
        self.t, self.pc = task, pc
        self.emb = nn.Embedding(task.vocab, d)
        self.pos = nn.Parameter(torch.randn(task.length, d) * 0.02)
        self.block = GBlock(d, heads, d // heads, mlp)
        self.inject = nn.Linear(3 * d, d, bias=False)
        with torch.no_grad():
            self.inject.weight.copy_(torch.cat([torch.eye(d)] * 3, 1) / 3)
        self.reg_emb = nn.Linear(task.M, d, bias=False)  # result port: register distribution -> state
        self.counter = nn.Embedding(3, d) if counter else None  # code sensor: steps remaining 0/1/2+
        # slot addressing: pick a step slot (absolute index) or NOOP; pc addressing: NEXT or NOOP
        self.query = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 if pc else task.S + 1))
        self.w = nn.Parameter(torch.ones(()))  # logit port weight
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, task.vocab, bias=False)

    def core_params(self):
        skip = ("emb.", "pos", "head.", "ln.", "reg_emb.", "query.", "counter.", "w")
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith(skip))

    def candidates(self, tok, reg):
        """Code: for every step slot i, the register pushed through v -> calc(v, op_i, a_i)."""
        t, M, S = self.t, self.t.M, self.t.S
        B = tok.shape[0]
        ops, args = tok[:, 0:2 * S:2], tok[:, 1:2 * S:2]
        v = torch.arange(M)
        idx = calc(v[None, None], ops[..., None], args[..., None].clamp(max=M - 1), t)  # (B,S,M)
        out = torch.zeros(B, S, M).scatter_add_(2, idx, reg[:, None, :].expand(B, S, M))
        load = F.one_hot(args[:, 0].clamp(max=M - 1), M).float()
        out[:, 0] = load
        pad = ops == t.PAD
        return torch.where(pad[..., None], reg[:, None, :].expand(B, S, M), out)

    def forward(self, tok, T, mode="soft"):
        B, L = tok.shape
        e0 = self.emb(tok) + self.pos[:L]
        reg = torch.full((B, self.t.M), 1.0 / self.t.M)
        nsteps = (tok[:, 0:2 * self.t.S:2] != self.t.PAD).sum(1)
        x, ql = e0, []
        pcd = F.one_hot(torch.zeros(B, dtype=torch.long), self.t.S).float()  # expert's program counter (dist)
        for step in range(T):
            r = torch.zeros_like(e0)
            r[:, -1] = self.reg_emb(reg)
            if self.counter is not None:
                r[:, -1] = r[:, -1] + self.counter((nsteps - step).clamp(0, 2))
            x = self.block(self.inject(torch.cat([x, e0, r], -1)))
            q = self.query(x[:, -1])
            ql.append(q)
            p = q.softmax(-1)
            if mode == "labeled":
                p = F.one_hot(q.argmax(-1), q.shape[-1]).float()
            cand = self.candidates(tok, reg)
            if self.pc:  # p = [NEXT, NOOP]; the expert executes the step under its program counter
                ran = torch.einsum("bs,bsm->bm", pcd, cand)
                reg = p[:, 1:] * reg + p[:, :1] * ran
                pcd = p[:, 1:] * pcd + p[:, :1] * torch.roll(pcd, 1, dims=1)
            else:
                reg = p[:, -1:] * reg + torch.einsum("bs,bsm->bm", p[:, :-1], cand)
        logits = self.head(self.ln(x[:, -1]))
        logits = torch.cat([logits[:, :self.t.M] + self.w * torch.log(reg + 1e-6), logits[:, self.t.M:]], -1)
        return logits, ql


@torch.no_grad()
def evaluate(model, task, ks, T, arm, n=512, seed=123):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in ks:
        tok, y, _ = task.sample(n, g, kmin=k, kmax=k)
        lg = model(tok, T=T)[0] if arm != "plain" else model(tok, T=1)
        out[k] = round((lg.argmax(-1) == y).float().mean().item(), 3)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", choices=["soft", "labeled", "plain"], required=True)
    ap.add_argument("--blocks", type=int, default=8, help="plain arm depth")
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--k_train", type=int, default=4)
    ap.add_argument("--tmin", type=int, default=6)
    ap.add_argument("--tmax", type=int, default=8)
    ap.add_argument("--no_counter", action="store_true")
    ap.add_argument("--pc", action="store_true", help="relative addressing: expert program counter, query NEXT/NOOP")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out_dir", default="runs/loop")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    task = CalcTask(a.M, kmax=8)
    if a.arm == "plain":
        model = Looped(task.vocab, task.length, 64, 4, 128, a.blocks)
    else:
        model = CalcCaller(task, counter=not a.no_counter, pc=a.pc)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500))
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    f = (out / "log.jsonl").open("w")
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    ks = list(range(1, 9))
    for step in range(a.steps + 1):
        if step % 2000 == 0 or step == a.steps:
            rec = dict(step=step, arm=a.arm, core=model.core_params(), time=round(time.time() - t0),
                       acc_T10=evaluate(model, task, ks, 10, a.arm), acc_T16=evaluate(model, task, ks, 16, a.arm))
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps(rec), flush=True)
        if step == a.steps:
            break
        tok, y, k = task.sample(a.bs, g, kmin=1, kmax=a.k_train)
        if a.arm == "plain":
            loss = F.cross_entropy(model(tok, T=1), y)
        else:
            T = int(torch.randint(a.tmin, a.tmax + 1, (1,), generator=g))
            lg, ql = model(tok, T=T, mode=a.arm)
            loss = F.cross_entropy(lg, y)
            if a.arm == "labeled":
                for t_, q in enumerate(ql):
                    if a.pc:
                        lab = torch.where(k >= t_, torch.zeros_like(k), torch.ones_like(k))
                    else:
                        lab = torch.where(k >= t_, torch.full_like(k, t_), torch.full_like(k, task.S))
                    loss = loss + F.cross_entropy(q, lab) / len(ql)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()


if __name__ == "__main__":
    main()
