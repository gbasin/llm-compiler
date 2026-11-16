"""Rung 6: multi-step word problems. The looped core does the LANGUAGE part; code does control
flow, number parsing and exact arithmetic.

  "Mia has seven apples. At 3 o'clock she buys 12 more. Tom is 9 years old. Then she gives away two."
  answer = ((7 + 12) - 2) mod M

Code (experts/sensors):
  - sentence cursor: loop step t works on sentence t (marked on the sentence's chars each loop)
  - number parser: every number mention (digits or words) with its value and char span
  - calculator over a register that is a distribution over M values (exact + differentiable)
Core, each loop: op in {LOAD, ADD, SUB, MUL, SKIP} for the current sentence, and WHICH number mention
in it is the operand (attention: query from the core's state, keys = core's hidden state at each
mention, so selection is by context, e.g. "o'clock" vs "more").
  labeled: CE on op and operand per sentence (training wheels); soft: answer loss only.
Plain control: a transformer reading the text, predicting the answer class directly.
Splits: train 1..3 steps; test 1..6 steps; 'novel' = held-out phrasings.

    uv run python -m llmc.loop_words --arm soft --name r6_soft
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid import GBlock

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
NAMES = ["Mia", "Tom", "Sam", "Lily", "Ben", "Zoe", "Max", "Ava"]
OBJS = ["apples", "pens", "cards", "stones", "cups", "books", "coins", "shells"]
LOAD, ADD, SUB, MUL, SKIP = range(5)
START = ["{N} has {a} {o}.", "There are {a} {o} in a box.", "{N} starts with {a} {o}."]
STEP = {
    ADD: ["{P} buys {b} more.", "At {d} o'clock {P} finds {b} more.", "Then {P} gets {b} more {o}.",
          "{P} adds {b} to the pile."],
    SUB: ["{P} gives away {b}.", "At {d} o'clock {P} loses {b}.", "Then {P} drops {b} of them.",
          "{P} eats {b}."],
    MUL: ["{P} now has {b} times as many.", "On day {d} the pile grows {b} times.", "Then the {o} multiply by {b}."],
}
DISTRACT = ["{P} is {d} years old.", "It is {d} o'clock.", "The room has {d} chairs.", "{P} counts to {d}."]
NOVEL = {
    ADD: ["Later {P} receives {b} extra.", "In room {d} {P} picks up {b} more."],
    SUB: ["Later {P} sells {b}.", "In room {d} {P} breaks {b}."],
    MUL: ["Somehow the {o} become {b} times more.", "By week {d} there are {b} times as many."],
}
NOVEL_DISTRACT = ["The bus has {d} wheels.", "It rained for {d} days."]
CHARS = sorted(set("".join(WORDS + NAMES + OBJS) + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,'"))
STOI = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = pad
VOCAB = len(CHARS) + 1


def say(n, rng):
    return WORDS[n] if n <= 20 and rng.random() < 0.5 else str(n)


def make(rng, k, M, novel=False, p_distract=0.5):
    N, o = rng.choice(NAMES), rng.choice(OBJS)
    P = rng.choice([N, "she", "he"])
    a = rng.randint(1, 20)
    sents = [(rng.choice(START).format(N=N, o=o, a=say(a, rng)), LOAD, a)]
    acc = a
    for _ in range(k):
        if rng.random() < p_distract:
            d = rng.randint(1, 20)
            sents.append((rng.choice(NOVEL_DISTRACT if novel else DISTRACT).format(P=P, d=say(d, rng)), SKIP, None))
        op = rng.choice([ADD, SUB, MUL])
        b = rng.randint(2, 5) if op == MUL else rng.randint(1, 20)
        tmpl = rng.choice((NOVEL if novel else STEP)[op])
        d = rng.randint(1, 20)
        sents.append((tmpl.format(P=P, o=o, b=say(b, rng), d=say(d, rng)), op, b))
        acc = (acc + b) % M if op == ADD else (acc - b) % M if op == SUB else (acc * b) % M
    return sents, acc


def mentions(sent):
    """Code: number mentions in a sentence -> list of (value, start, end) char offsets."""
    out, i = [], 0
    for w in sent.replace(".", " ").split(" "):
        j = sent.find(w, i) if w else i
        if w.isdigit():
            out.append((int(w), j, j + len(w)))
        elif w in WORDS:
            out.append((WORDS.index(w), j, j + len(w)))
        i = j + len(w)
    return out


class Batch:
    def __init__(self, problems, M, max_s=16, max_m=3):
        B = len(problems)
        texts, self.y = [], torch.tensor([acc for _, acc in problems])
        L = max(sum(len(s) + 1 for s, _, _ in sents) for sents, _ in problems)
        self.tok = torch.zeros(B, L, dtype=torch.long)
        self.sid = torch.full((B, L), -1, dtype=torch.long)  # sentence id per char
        self.ment_pos = torch.zeros(B, max_s, max_m, dtype=torch.long)  # last char of each mention
        self.ment_val = torch.zeros(B, max_s, max_m, dtype=torch.long)
        self.ment_ok = torch.zeros(B, max_s, max_m, dtype=torch.bool)
        self.op = torch.full((B, max_s), SKIP, dtype=torch.long)
        self.arg = torch.zeros(B, max_s, dtype=torch.long)  # index of operand mention (labels)
        self.nsent = torch.zeros(B, dtype=torch.long)
        for b, (sents, _) in enumerate(problems):
            pos = 0
            self.nsent[b] = len(sents)
            for si, (s, op, val) in enumerate(sents):
                for c_i, c in enumerate(s + " "):
                    self.tok[b, pos + c_i] = STOI[c]
                    self.sid[b, pos + c_i] = si
                for mi, (v, st, en) in enumerate(mentions(s)[:max_m]):
                    self.ment_pos[b, si, mi] = pos + en - 1
                    self.ment_val[b, si, mi] = v % M
                    self.ment_ok[b, si, mi] = True
                    if val is not None and v == val and (op in (LOAD, MUL) or True):
                        self.arg[b, si] = mi  # last matching mention = operand (templates put b last)
                self.op[b, si] = op
                pos += len(s) + 1


def sample(B, rng, kmin, kmax, M, novel=False):
    return Batch([make(rng, rng.randint(kmin, kmax), M, novel) for _ in range(B)], M)


class WordCaller(nn.Module):
    def __init__(self, M, d=48, heads=2, mlp=96, max_len=512):
        super().__init__()
        self.M = M
        self.emb = nn.Embedding(VOCAB, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-torch.log(torch.tensor(10000.0)) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe * 0.5)
        self.block = GBlock(d, heads, d // heads, mlp)
        self.inject = nn.Linear(2 * d, d, bias=False)
        with torch.no_grad():
            self.inject.weight.copy_(torch.cat([torch.eye(d)] * 2, 1) / 2)
        self.cursor = nn.Parameter(torch.randn(d) * 0.1)  # sensor: marks the current sentence's chars
        self.op_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 5))  # query port: op
        self.q = nn.Linear(d, d, bias=False)  # query port: operand selection
        self.k = nn.Linear(d, d, bias=False)

    def core_params(self):
        skip = ("emb.", "op_head.", "q.", "k.", "cursor")
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith(skip))

    def forward(self, bt: Batch, T, mode="soft"):
        tok = bt.tok
        B, L = tok.shape
        M = self.M
        e0 = self.emb(tok) + self.pe[:L]
        x = e0
        reg = torch.full((B, M), 1.0 / M)
        v = torch.arange(M)
        op_logits, arg_logits = [], []
        for t in range(T):
            mark = (bt.sid == t).float()[..., None] * self.cursor
            x = self.block(self.inject(torch.cat([x, e0 + mark], -1)))
            # summary of the current sentence = mean hidden state over its chars
            m = (bt.sid == t).float()
            hs = (x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            ol = self.op_head(hs)
            ts = min(t, bt.ment_pos.shape[1] - 1)
            keys = self.k(x.gather(1, bt.ment_pos[:, ts, :, None].expand(-1, -1, x.shape[-1])))  # (B, m, d)
            al = torch.einsum("bd,bmd->bm", self.q(hs), keys) / keys.shape[-1] ** 0.5
            al = al.masked_fill(~bt.ment_ok[:, ts], -1e9)
            op_logits.append(ol); arg_logits.append(al)
            po, pa = ol.softmax(-1), al.softmax(-1)
            if mode == "labeled":
                po = F.one_hot(ol.argmax(-1), 5).float()
                pa = F.one_hot(al.argmax(-1), al.shape[-1]).float()
            valid = t < bt.nsent
            po = torch.where(valid[:, None], po, F.one_hot(torch.full((B,), SKIP), 5).float())
            # operand distribution over values
            pv = torch.zeros(B, M).scatter_add_(1, bt.ment_val[:, ts], pa * bt.ment_ok[:, ts])
            pv = pv / pv.sum(1, keepdim=True).clamp(min=1e-9)
            # calculator: pushforward of reg under each op, mixed by po and pv (outer over operand values)
            def push(fn):
                idx = fn(v[None, :, None], v[None, None, :])  # (1, M_reg, M_arg)
                w = reg[:, :, None] * pv[:, None, :]  # (B, M, M)
                return torch.zeros(B, M).scatter_add_(1, idx.expand(B, M, M).reshape(B, -1), w.reshape(B, -1))
            new = (po[:, LOAD:LOAD + 1] * pv + po[:, ADD:ADD + 1] * push(lambda r, a: (r + a) % M)
                   + po[:, SUB:SUB + 1] * push(lambda r, a: (r - a) % M) + po[:, MUL:MUL + 1] * push(lambda r, a: (r * a) % M)
                   + po[:, SKIP:SKIP + 1] * reg)
            reg = new
        return torch.log(reg + 1e-6), op_logits, arg_logits


class Plain(nn.Module):
    def __init__(self, M, d=64, heads=4, mlp=128, n_blocks=4, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.blocks = nn.ModuleList(GBlock(d, heads, d // heads, mlp) for _ in range(n_blocks))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, M)

    def core_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith("blocks"))

    def forward(self, bt, T=None, mode=None):
        x = self.emb(bt.tok) + self.pos[:bt.tok.shape[1]]
        for b in self.blocks:
            x = b(x)
        last = (bt.tok > 0).sum(1) - 1
        return self.head(self.ln(x[torch.arange(len(x)), last])), [], []


@torch.no_grad()
def evaluate(model, M, arm, n=256):
    model.eval()
    out = {}
    for novel in (False, True):
        for k in (1, 2, 3, 4, 6):
            rng = random.Random(1000 + k + 100 * novel)
            bt = sample(n, rng, k, k, M, novel)
            T = int(bt.nsent.max()) + 1
            lp = model(bt, T=T, mode="labeled" if arm == "labeled" else "soft")[0]
            out[f"{'novel' if novel else 'id'}_k{k}"] = round((lp.argmax(-1) == bt.y).float().mean().item(), 3)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", choices=["soft", "labeled", "plain"], required=True)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--k_train", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out_dir", default="runs/loop")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = random.Random(a.seed)
    model = Plain(a.M, n_blocks=a.blocks) if a.arm == "plain" else WordCaller(a.M)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300))
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    f = (out / "log.jsonl").open("w")
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % 1000 == 0 or step == a.steps:
            rec = dict(step=step, arm=a.arm, core=model.core_params(), time=round(time.time() - t0),
                       acc=evaluate(model, a.M, a.arm))
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps(rec), flush=True)
        if step == a.steps:
            break
        bt = sample(a.bs, rng, 1, a.k_train, a.M)
        T = int(bt.nsent.max())
        lp, ol, al = model(bt, T=T, mode=a.arm)
        loss = F.cross_entropy(lp, bt.y) if a.arm == "plain" else F.nll_loss(lp, bt.y)
        if a.arm == "labeled":
            for t, (o, g) in enumerate(zip(ol, al)):
                valid = t < bt.nsent
                loss = loss + F.cross_entropy(o[valid], bt.op[valid, t]) / len(ol)
                hasnum = valid & bt.ment_ok[:, t].any(1) & (bt.op[:, t] != SKIP)
                if hasnum.any():
                    loss = loss + F.cross_entropy(g[hasnum], bt.arg[hasnum, t]) / len(ol)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()


if __name__ == "__main__":
    main()
