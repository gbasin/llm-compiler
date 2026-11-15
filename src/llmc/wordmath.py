"""Direction 4: messy-language input, exact computation.

Char-level word problems. Operands are written as digits or words ("seven",
"a dozen"), phrasing varies, and distractor sentences may contain numbers that
must be ignored. The answer is digits after "A: ".

    Mia has seven apples. It is 3 o'clock. She buys 12 more. Q: how many apples now? A: 19

Batches carry metadata (operands, op, answer, char spans of each number mention,
which mentions are operands) so a teacher can define interface contracts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen", "twenty"]
NAMES = ["Mia", "Tom", "Sam", "Lily", "Ben", "Zoe", "Max", "Ava"]
OBJS = ["apples", "pens", "cards", "stones", "cups", "books", "coins", "shells"]
OPS = ["add", "sub", "mul"]

TEMPLATES = {
    "add": ["{N} has {a} {o}. {P} buys {b} more.", "{N} had {a} {o} and found {b} more.",
            "There are {a} {o} in a box. {N} puts in {b} more."],
    "sub": ["{N} has {a} {o}. {P} gives away {b}.", "{N} had {a} {o} but lost {b}.",
            "There are {a} {o} in a box. {N} takes out {b}."],
    "mul": ["{N} has {a} bags with {b} {o} each.", "There are {a} rows of {b} {o}.",
            "{N} buys {a} packs of {o}. Each pack has {b}."],
}
# v2: templates with an embedded distractor number {d} anywhere, incl. between operands
TEMPLATES_V2 = {
    "add": ["{N} is {d} years old and has {a} {o}. {P} buys {b} more.",
            "{N} had {a} {o}. At {d} o'clock {P} found {b} more.",
            "There are {a} {o} in box {d}. {N} puts in {b} more."],
    "sub": ["{N} is {d} years old and has {a} {o}. {P} gives away {b}.",
            "{N} had {a} {o}. At {d} o'clock {P} lost {b}.",
            "There are {a} {o} in box {d}. {N} takes out {b}."],
    "mul": ["{N} has {a} bags in room {d} with {b} {o} each.",
            "There are {a} rows of {b} {o} on shelf {d}.",
            "On day {d} {N} buys {a} packs of {o}. Each pack has {b}."],
}
QUESTIONS = {"add": "Q: how many {o} now?", "sub": "Q: how many {o} are left?", "mul": "Q: how many {o} in total?"}
DISTRACT = ["It is {d} o'clock.", "{P} is {d} years old.", "The sky is blue.", "{P} likes cats.", ""]


def say(n: int, rng: random.Random) -> str:
    r = rng.random()
    if n == 12 and r < 0.15:
        return "a dozen"
    if r < 0.5 and n <= 20:
        return WORDS[n]
    return str(n)


@dataclass
class Problem:
    text: str  # full text including answer
    prompt_len: int  # index where the answer digits start
    a: int
    b: int
    op: str
    ans: int
    spans: list  # (start, end, value, role) role in {"a", "b", "distractor"}


def make_problem(rng: random.Random, max_n: int = 20, p_distract: float = 0.6, v2: bool = False) -> Problem:
    op = rng.choice(OPS)
    a, b = rng.randint(0, max_n), rng.randint(0, max_n)
    if op == "sub" and b > a:
        a, b = b, a
    ans = {"add": a + b, "sub": a - b, "mul": a * b}[op]
    N, o = rng.choice(NAMES), rng.choice(OBJS)
    P = "She" if N in ("Mia", "Lily", "Zoe", "Ava") else "He"
    tmpl = rng.choice(TEMPLATES[op])
    sa, sb = say(a, rng), say(b, rng)
    spans = []
    if v2 and rng.random() < 0.5:
        tmpl = rng.choice(TEMPLATES_V2[op])
    d_in = rng.randint(1, 12)
    fills = {"a": (sa, a, "a"), "b": (sb, b, "b"), "d": (say(d_in, rng), d_in, "distractor")}
    body = tmpl.format(N=N, P=P, o=o, a="\0a", b="\0b", d="\0d")
    sent, op_spans, i = "", [], 0
    while i < len(body):
        if body[i] == "\0":
            surf, val, role = fills[body[i + 1]]
            op_spans.append((len(sent), len(sent) + len(surf), val, role))
            sent += surf
            i += 2
        else:
            sent += body[i]
            i += 1
    pre, post, dspan = "", "", None
    if rng.random() < p_distract:
        dt = rng.choice(DISTRACT)
        d = rng.randint(1, 12)
        dsurf = say(d, rng)
        ds = dt.format(d=dsurf, P=P)
        if ds:
            k = ds.index(dsurf) if "{d}" in dt else None
            if rng.random() < 0.5:
                pre = ds + " "
                if k is not None:
                    dspan = (k, k + len(dsurf), d, "distractor")
            else:
                post = " " + ds
                if k is not None:
                    off = len(sent) + 1 + k
                    dspan = (off, off + len(dsurf), d, "distractor")
    shift = len(pre)
    spans = [(s0 + shift, e0 + shift, v, r) for s0, e0, v, r in op_spans]
    if dspan:
        spans.append((dspan[0] + (0 if pre else shift), dspan[1] + (0 if pre else shift), dspan[2], dspan[3]))
    sent = pre + sent + post
    pre = ""
    q = QUESTIONS[op].format(o=o)
    prompt = sent + " " + q + " A: "
    text = prompt + str(ans) + "\n"
    for s0, e0, v, r in spans:
        assert prompt[s0:e0] in (str(v), WORDS[v] if v <= 20 else "", "a dozen"), (prompt, s0, e0, v)
    return Problem(text, len(prompt), a, b, op, ans, sorted(spans))


class WordMath:
    def __init__(self, max_n: int = 20, p_distract: float = 0.6, v2: bool = False):
        self.v2 = v2
        chars = sorted(set("".join(WORDS + NAMES + OBJS) + "".join(sum(TEMPLATES.values(), []))
                           + "".join(QUESTIONS.values()) + "".join(sum(TEMPLATES_V2.values(), [])) + "".join(DISTRACT) + "0123456789 .:'?\nQAHSTI{}abcdefghijklmnopqrstuvwxyz"))
        chars = [c for c in chars if c not in "{}"]
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.PAD = len(chars)
        self.vocab = len(chars) + 1
        self.max_n, self.p_distract = max_n, p_distract

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def batch(self, B: int, rng: random.Random, **kw):
        probs = [make_problem(rng, kw.get("max_n", self.max_n), kw.get("p_distract", self.p_distract), self.v2)
                 for _ in range(B)]
        T = max(len(p.text) for p in probs)
        tokens = torch.full((B, T), self.PAD, dtype=torch.long)
        targets = torch.full((B, T), -100, dtype=torch.long)
        for i, p in enumerate(probs):
            ids = self.encode(p.text)
            tokens[i, :len(ids)] = torch.tensor(ids)
            # predict answer chars (incl. final newline): target at position t is token t+1
            for t in range(p.prompt_len - 1, len(ids) - 1):
                targets[i, t] = ids[t + 1]
        return tokens, targets, probs

    @torch.no_grad()
    def exact_match(self, model, probs, device="cpu"):
        """Greedy-decode answers; return fraction exactly correct."""
        ok = 0
        for p in probs:
            ids = self.encode(p.text[:p.prompt_len])
            for _ in range(5):
                x = torch.tensor([ids], device=device)
                nxt = int(model(x)[0, -1].argmax())
                if nxt == self.stoi["\n"] or nxt == self.PAD:
                    break
                ids.append(nxt)
            ok += "".join(self.itos[i] for i in ids[p.prompt_len:]) == str(p.ans)
        return ok / len(probs)
