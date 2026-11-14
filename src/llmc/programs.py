"""Program library: the code the teacher writes.

A program maps (B, T, in_dim) -> (B, T, out_dim). It sits between learned
projections inside a CodeModule:

    h_write += gate * W_out( P( W_in(LN(h_read)) ) )

Tier 1: differentiable tensor programs. `ScanProgram` decodes a categorical
value per position (softmax over V), runs an exact *distributional* causal scan,
and returns a distribution over V. Gradients reach W_in through every position.

Tier 2: `PyProgram` runs arbitrary Python on argmax-decoded integers, with a
straight-through gradient.

Programs hold no learned parameters: all numbers live in the module's
W_in / W_out / gate, so code size and neural size stay separately accountable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _decode(logits: torch.Tensor, hard: bool) -> torch.Tensor:
    p = logits.float().softmax(-1)
    if hard:  # forward = exact one-hot, backward = softmax gradient
        one = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
        p = one + p - p.detach()
    return p


def dist_cummax(p: torch.Tensor) -> torch.Tensor:
    """p: (B,T,V) independent per-position distributions -> distribution of the
    running max. CDF of a max is the product of CDFs."""
    cdf = torch.cumprod(p.cumsum(-1).clamp(0, 1), dim=1)
    return torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1])).clamp_min(0)


def dist_cumsum_mod(p: torch.Tensor) -> torch.Tensor:
    """Distribution of the running sum mod V: circular convolution over time,
    i.e. a cumulative product in the Fourier domain."""
    f = torch.fft.fft(p.to(torch.complex64), dim=-1)
    out = torch.fft.ifft(torch.cumprod(f, dim=1), dim=-1).real
    return out.clamp_min(0)


SCANS = {"max": dist_cummax, "summod": dist_cumsum_mod}


class ScanProgram(nn.Module):
    def __init__(self, V: int, op: str, hard: bool = False):
        super().__init__()
        self.V, self.op, self.hard = V, op, hard
        self.in_dim = self.out_dim = V

    def forward(self, logits):
        return SCANS[self.op](_decode(logits, self.hard)).to(logits.dtype)


class _PySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, fn, V_out):
        ints = p.argmax(-1).cpu().numpy()
        out = np.asarray(fn(ints), dtype=np.int64)
        assert out.shape == ints.shape, "PyProgram fn must map (B,T) ints -> (B,T) ints"
        return F.one_hot(torch.from_numpy(out).to(p.device), V_out).to(p.dtype)

    @staticmethod
    def backward(ctx, g):
        # straight-through: treat the program as identity on the distribution.
        return g, None, None


class PyProgram(nn.Module):
    """Tier 2: arbitrary Python over integer sequences.

    fn: np.ndarray[int] (B, T) -> np.ndarray[int] (B, T), values in [0, V_out).
    The STE gradient ignores what fn does, so the interface only learns well when
    fn's output is closely tied to its input (or W_in is already sensible).
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], V_in: int, V_out: int | None = None):
        super().__init__()
        self.fn = fn
        self.in_dim, self.out_dim = V_in, V_out or V_in

    def forward(self, logits):
        return _PySTE.apply(logits.float().softmax(-1), self.fn, self.out_dim).to(logits.dtype)


class NeuralControl(nn.Module):
    """Control: same interface dims, but P is a pointwise nonlinearity, so the
    module adds the same learned parameters and no algorithm."""

    def __init__(self, dim: int):
        super().__init__()
        self.in_dim = self.out_dim = dim

    def forward(self, z):
        return F.gelu(z)


def build_program(name: str, V: int) -> nn.Module:
    """Names: soft_<op>, hard_<op>, py_<op> for op in {max, summod}; control."""
    if name == "control":
        return NeuralControl(V)
    kind, op = name.split("_", 1)
    if kind in ("soft", "hard"):
        return ScanProgram(V, op, hard=(kind == "hard"))
    if kind == "py":
        fns = {
            "max": lambda a: np.maximum.accumulate(a, axis=1),
            "summod": lambda a: np.cumsum(a, axis=1) % V,
        }
        return PyProgram(fns[op], V)
    raise KeyError(name)
