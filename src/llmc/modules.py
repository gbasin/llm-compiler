"""Gated code module: h_write += gate * W_out(P(W_in(LN(h_read))))."""

from __future__ import annotations

import torch
import torch.nn as nn


class CodeModule(nn.Module):
    def __init__(self, d_model: int, program: nn.Module, read: int, write: int, gate_init: float = 0.0, name: str = ""):
        super().__init__()
        self.read, self.write = read, write
        self.name = name or type(program).__name__
        self.ln = nn.LayerNorm(d_model)
        self.W_in = nn.Linear(d_model, program.in_dim)
        self.program = program
        self.W_out = nn.Linear(program.out_dim, d_model, bias=False)
        nn.init.normal_(self.W_out.weight, std=0.02)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.enabled = True

    def forward(self, h):
        return self.gate * self.W_out(self.program(self.W_in(self.ln(h))))

    def describe(self):
        return dict(name=self.name, read=self.read, write=self.write, gate=float(self.gate.detach()),
                    params=sum(p.numel() for p in self.parameters()))
