"""Multitask scan suite for the capacity experiments (milestone 2).

Sequence: TASK x1 x2 ... xn, predict y_i = f_task(x)_i at the position of x_i.
Vocab: x in [0,V), y in [V,2V), task tokens [2V, 2V+K), PAD = 2V+K.
All tasks are causal per-position functions of the prefix.

Skill set A (code library has these):   summod, max
Skill set B, disjoint (no module helps): induct, count_eq, lag2
Skill set B, compositional (A's primitive applies to a learned transform):
    summod_even:  sum of x_j where x_j is even, mod V
    summod_pair:  sum of x_j * x_{j-1}, mod V
"""

from __future__ import annotations

import torch

from .tasks import Batch


def _summod(x, V):
    return torch.cumsum(x, 1) % V


def _max(x, V):
    return torch.cummax(x, 1).values


def _induct(x, V):
    # y_i = token that followed the most recent earlier occurrence of x_i, else 0
    B, n = x.shape
    y = torch.zeros_like(x)
    for i in range(1, n):
        prev = x[:, :i] == x[:, i:i + 1]  # (B, i)
        idx = torch.where(prev, torch.arange(i).expand(B, i), torch.full((B, i), -1)).max(1).values
        nxt = x.gather(1, (idx + 1).clamp(max=n - 1)[:, None])[:, 0]
        y[:, i] = torch.where(idx >= 0, nxt, torch.zeros_like(nxt))
    return y


def _count_eq(x, V):
    # number of earlier-or-equal occurrences of x_i, mod V
    oh = torch.nn.functional.one_hot(x, V).cumsum(1)
    return oh.gather(2, x[..., None])[..., 0] % V


def _lag2(x, V):
    y = torch.zeros_like(x)
    y[:, 2:] = x[:, :-2]
    return y


def _summod_even(x, V):
    return torch.cumsum(torch.where(x % 2 == 0, x, 0), 1) % V


def _summod_pair(x, V):
    prev = torch.zeros_like(x)
    prev[:, 1:] = x[:, :-1]
    return torch.cumsum(x * prev, 1) % V


FNS = dict(summod=_summod, max=_max, induct=_induct, count_eq=_count_eq, lag2=_lag2,
           summod_even=_summod_even, summod_pair=_summod_pair)


class MultiScan:
    name = "multi"

    def __init__(self, V: int = 5, tasks: tuple[str, ...] = tuple(FNS)):
        self.V = V
        self.tasks = list(tasks)
        self.tid = {t: 2 * V + k for k, t in enumerate(self.tasks)}
        self.PAD = 2 * V + len(self.tasks)
        self.BOS = -999  # no plain BOS; the task token plays that role
        self.vocab_size = self.PAD + 1
        self.interleave = False

    def max_len(self, n_max: int) -> int:
        return n_max + 1

    def sample(self, lengths, g=None, task: str | list[str] | None = None) -> Batch:
        """task: a single name, or a list to mix uniformly (default: all)."""
        lengths = torch.as_tensor(lengths, dtype=torch.long)
        B, n = len(lengths), int(lengths.max())
        names = [task] if isinstance(task, str) else (task or self.tasks)
        which = torch.randint(0, len(names), (B,), generator=g)
        x = torch.randint(0, self.V, (B, n), generator=g)
        y = torch.zeros_like(x)
        for k, t in enumerate(names):
            sel = which == k
            if sel.any():
                y[sel] = FNS[t](x[sel], self.V)
        tokens = torch.full((B, n + 1), self.PAD, dtype=torch.long)
        tokens[:, 0] = torch.tensor([self.tid[names[k]] for k in which.tolist()])
        tokens[:, 1:] = x
        item = torch.arange(n)[None] < lengths[:, None]
        mask = torch.zeros((B, n + 1), dtype=torch.bool)
        mask[:, 1:] = item
        tokens[:, 1:] = torch.where(item, x, torch.full_like(x, self.PAD))
        targets = torch.full((B, n + 1), -100, dtype=torch.long)
        targets[:, 1:] = torch.where(item, y + self.V, torch.full_like(y, -100))
        return Batch(tokens, targets, mask, lengths)

    def code_inputs(self, batch: Batch) -> torch.Tensor:
        tok = batch.tokens
        lab = torch.where(tok < self.V, tok, torch.full_like(tok, -1))
        is_task = (tok >= 2 * self.V) & (tok < self.PAD)
        return torch.where(is_task, torch.zeros_like(tok), lab)

    def comp_inputs(self, batch: Batch) -> torch.Tensor:
        """Teacher-proposed contract for the compositional B tasks: the per-item
        value whose running sum mod V is the answer (task token -> 0, other
        tasks -> -1 = don't care)."""
        tok = batch.tokens
        x = tok[:, 1:]
        prev = torch.zeros_like(x)
        prev[:, 1:] = x[:, :-1]
        lab = torch.full_like(tok, -1)
        for name, f in (("summod_even", lambda: torch.where(x % 2 == 0, x, 0)),
                        ("summod_pair", lambda: (x * prev) % self.V)):
            if name not in self.tid:
                continue
            sel = tok[:, 0] == self.tid[name]
            lab[sel, 0] = 0
            lab[sel, 1:] = f()[sel]
        valid = (tok < self.V) | ((tok >= 2 * self.V) & (tok < self.PAD))
        return torch.where(valid, lab, torch.full_like(lab, -1))

    def probe_targets(self, batch: Batch) -> dict[str, torch.Tensor]:
        tok = batch.tokens
        return {"current_x": torch.where(tok < self.V, tok, torch.full_like(tok, -1)),
                "answer": torch.where(batch.mask, batch.targets - self.V, torch.full_like(tok, -1))}
