"""Gated code module.

    z      = W_out(P(W_in(LN(h_read))))            computed at the read site
    h_write += gate(h_write) * z                   added at the write site

gate_mode="scalar": gate is one learned scalar (doubleAI-style), init gate_init.
gate_mode="routed": gate = g * sigmoid(r . LN(h_write) + c), a per-position
router, so the module can be on for some tasks/positions and off for others.
The router reads the *write* site, where attention has already mixed in
context (e.g. the task token).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CodeModule(nn.Module):
    def __init__(self, d_model: int, program: nn.Module, read: int, write: int, gate_init: float = 0.0,
                 name: str = "", gate_mode: str = "scalar"):
        super().__init__()
        self.read, self.write = read, write
        self.name = name or type(program).__name__
        self.gate_mode = gate_mode
        self.ln = nn.LayerNorm(d_model)
        self.W_in = nn.Linear(d_model, program.in_dim)
        self.program = program
        self.W_out = nn.Linear(program.out_dim, d_model, bias=False)
        nn.init.normal_(self.W_out.weight, std=0.02)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        if gate_mode == "routed":
            self.router_ln = nn.LayerNorm(d_model)
            self.router = nn.Linear(d_model, 1)
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        self.enabled = True

    def compute(self, h_read):
        return self.W_out(self.program(self.W_in(self.ln(h_read))))

    def gate_value(self, h_write):
        if self.gate_mode == "routed":
            return self.gate * torch.sigmoid(self.router(self.router_ln(h_write)))
        return self.gate

    def describe(self):
        return dict(name=self.name, read=self.read, write=self.write, gate=float(self.gate.detach()),
                    gate_mode=self.gate_mode, params=sum(p.numel() for p in self.parameters()))
