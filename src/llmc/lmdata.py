"""Language-model data sources for grow-on-demand experiments.

HMMSource: random hidden Markov model with *secret* parameters. The teacher
can't write it as code; the network must learn it. Exact Bayes-optimal
next-token loss comes from the forward algorithm, so the gap to optimum tells
us whether the core is still short of capacity.

Planted structure (optional, codeable): with probability `p_copy` per sequence,
a random earlier span is repeated after a COPY marker (induction-solvable).

CharSource: random crops of a text file, char-level.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


class HMMSource:
    def __init__(self, vocab: int = 16, states: int = 8, branch: int = 3, emit_alpha: float = 0.2,
                 seed: int = 0, p_copy: float = 0.0, copy_len: tuple[int, int] = (4, 8)):
        with torch.random.fork_rng():
            torch.manual_seed(seed)  # Dirichlet draws use the global RNG; pin it
            self._init(vocab, states, branch, emit_alpha, seed, p_copy, copy_len)

    def _init(self, vocab, states, branch, emit_alpha, seed, p_copy, copy_len):
        g = torch.Generator().manual_seed(seed)
        self.states, self.p_copy, self.copy_len = states, p_copy, copy_len
        self.n_base = vocab
        self.COPY = vocab if p_copy > 0 else None
        self.vocab = vocab + (1 if p_copy > 0 else 0)
        # sparse transitions: each state moves to `branch` random states
        T = torch.zeros(states, states)
        for s in range(states):
            nxt = torch.randperm(states, generator=g)[:branch]
            T[s, nxt] = torch.distributions.Dirichlet(torch.ones(branch)).sample()
        self.T = T
        self.E = torch.distributions.Dirichlet(torch.full((vocab,), emit_alpha)).sample((states,))
        # stationary distribution as initial belief
        pi = torch.ones(states) / states
        for _ in range(500):
            pi = pi @ T
        self.pi = pi
        self.g = torch.Generator().manual_seed(seed + 1)

    def _hmm(self, B, L, g):
        z = torch.multinomial(self.pi.expand(B, -1), 1, generator=g)[:, 0]
        out = torch.empty(B, L, dtype=torch.long)
        for t in range(L):
            out[:, t] = torch.multinomial(self.E[z], 1, generator=g)[:, 0]
            z = torch.multinomial(self.T[z], 1, generator=g)[:, 0]
        return out

    def sample(self, B: int, L: int, g=None):
        g = g or self.g
        x = self._hmm(B, L, g)
        if self.p_copy > 0:
            for b in range(B):
                if torch.rand(1, generator=g).item() < self.p_copy:
                    n = int(torch.randint(self.copy_len[0], self.copy_len[1] + 1, (1,), generator=g))
                    start = int(torch.randint(0, L // 2 - n, (1,), generator=g))
                    at = int(torch.randint(L // 2, L - n - 1, (1,), generator=g))
                    x[b, at] = self.COPY
                    x[b, at + 1: at + 1 + n] = x[b, start: start + n]
        return x

    @torch.no_grad()
    def bayes_loss(self, x: torch.Tensor) -> float:
        """Mean -log p(x_t | x_<t) under the true HMM, for t >= 1 (pure HMM only)."""
        assert self.p_copy == 0, "exact optimum only for the pure HMM"
        B, L = x.shape
        belief = self.pi.expand(B, -1).clone()  # p(z_t | x_<t)
        nll = []
        for t in range(L):
            p_x = belief @ self.E  # (B, V)
            px = p_x.gather(1, x[:, t:t + 1])[:, 0]
            if t >= 1:
                nll.append(-torch.log(px))
            post = belief * self.E[:, x[:, t]].T
            post = post / post.sum(1, keepdim=True)
            belief = post @ self.T
        return torch.stack(nll, 1).mean().item()

    @torch.no_grad()
    def unigram_loss(self, x):
        p = (self.pi @ self.E)
        return (-torch.log(p[x[:, 1:]])).mean().item()


class CharSource:
    def __init__(self, path: str, val_frac: float = 0.1, seed: int = 0):
        text = Path(path).read_text()
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.train, self.val = data[:n], data[n:]
        self.g = torch.Generator().manual_seed(seed)

    def sample(self, B, L, g=None, split="train"):
        g = g or self.g
        d = self.train if split == "train" else self.val
        ix = torch.randint(0, len(d) - L, (B,), generator=g)
        return torch.stack([d[i:i + L] for i in ix])
