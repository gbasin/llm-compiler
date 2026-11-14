"""Toy algorithmic tasks as causal-LM sequences.

Each task produces (tokens, targets, loss_mask) batches. Targets are aligned to
positions: targets[t] is what the model should predict *at* position t (i.e. the
next token). Loss/accuracy are computed only where loss_mask is True.

Scan tasks (running max, running sum mod V). Default format is
    BOS x1 x2 ... xn          (predict y_i at the position of x_i)
with interleave=True it is
    BOS x1 y1 x2 y2 ... xn yn (y_{i-1} in context -> local shortcut; much easier)
x tokens are values in [0, V); y tokens are V + scan(x_1..x_i).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    tokens: torch.Tensor  # (B, T) long
    targets: torch.Tensor  # (B, T) long
    mask: torch.Tensor  # (B, T) bool
    lengths: torch.Tensor  # (B,) long, number of items n per sequence

    def to(self, device):
        return Batch(*(t.to(device) for t in (self.tokens, self.targets, self.mask, self.lengths)))


class InterleavedScan:
    """BOS x1 y1 ... xn yn with y_i = scan(x_1..x_i). Subclasses define scan()."""

    name = "scan"

    def scan(self, x: torch.Tensor) -> torch.Tensor:  # (B, n) -> (B, n)
        raise NotImplementedError

    def __init__(self, V: int = 64, interleave: bool = False):
        self.V = V
        self.interleave = interleave
        self.BOS = 2 * V
        self.PAD = 2 * V + 1
        self.vocab_size = 2 * V + 2

    def max_len(self, n_max: int) -> int:
        return (2 * n_max if self.interleave else n_max) + 1

    def sample(self, lengths: list[int] | torch.Tensor, g: torch.Generator | None = None) -> Batch:
        lengths = torch.as_tensor(lengths, dtype=torch.long)
        B, n_max = len(lengths), int(lengths.max())
        T = self.max_len(n_max)
        x = torch.randint(0, self.V, (B, n_max), generator=g)
        y = self.scan(x)
        xs = slice(1, None, 2) if self.interleave else slice(1, None)
        tokens = torch.full((B, T), self.PAD, dtype=torch.long)
        tokens[:, 0] = self.BOS
        tokens[:, xs] = x
        if self.interleave:
            tokens[:, 2::2] = y + self.V
        targets = torch.full((B, T), -100, dtype=torch.long)
        targets[:, xs] = y + self.V  # at x_i predict y_i
        item = torch.arange(n_max)[None, :] < lengths[:, None]  # (B, n_max)
        mask = torch.zeros((B, T), dtype=torch.bool)
        mask[:, xs] = item
        # pad beyond length
        valid = torch.arange(T)[None, :] < (self.max_len(1) - 1) * lengths[:, None] + 1
        tokens = torch.where(valid, tokens, torch.full_like(tokens, self.PAD))
        targets = torch.where(mask, targets, torch.full_like(targets, -100))
        return Batch(tokens, targets, mask, lengths)

    def code_inputs(self, batch: Batch) -> torch.Tensor:
        """Labels a scan program should read per position: x at item
        positions, 0 (identity for max and sum) at BOS, -1 (ignore) at PAD."""
        tok = batch.tokens
        lab = torch.where(tok < self.V, tok, torch.full_like(tok, -1))
        return torch.where(tok == self.BOS, torch.zeros_like(tok), lab)

    # ground-truth intermediate quantities, used by probes ---------------------
    def probe_targets(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Per-position integer labels (or -1 where undefined)."""
        tok = batch.tokens
        x_val = torch.where(tok < self.V, tok, torch.full_like(tok, -1))
        is_x = tok < self.V
        ans = torch.where(batch.mask, batch.targets - self.V, torch.full_like(tok, -1))
        return {"current_x": x_val, "answer": ans}


class RunningMax(InterleavedScan):
    name = "running_max"

    def scan(self, x):
        return torch.cummax(x, dim=1).values


class RunningSumMod(InterleavedScan):
    """y_i = (x_1 + ... + x_i) mod V. Hard for transformers (parity-like)."""

    name = "running_sum_mod"

    def scan(self, x):
        return torch.cumsum(x, dim=1) % self.V


TASKS = {"running_max": RunningMax, "running_sum_mod": RunningSumMod}


def make_task(name: str, **kw):
    return TASKS[name](**kw)
