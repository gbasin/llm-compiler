"""Corpus n-gram tables as code: next-token statistics externalized from weights.

Interpolated (Witten-Bell) n-gram model of orders 0..K built from the training
split by counting. At each position it gives log p(next | last k tokens), which
the hybrid adds to the output logits with a learned, per-position weight.

Leave-one-out: when training on the same text the table was counted from, the
table "remembers" the token being predicted. With `targets`, we subtract that
occurrence from the counts at every order before interpolating, so the network
sees held-out-quality table predictions during training too.

Lookup is vectorized: contexts are hashed as base-V integers, mapped to dense
rows with searchsorted.
"""

from __future__ import annotations

import torch


class NGramTable:
    def __init__(self, data: torch.Tensor, vocab: int, K: int = 4):
        self.V, self.K = vocab, K
        data = data.long()
        n = len(data)
        self.uni = torch.bincount(data, minlength=vocab).float()
        self.orders = []  # per k>=1: (keys sorted, counts (R,V), N (R,), n1 (R,))
        for k in range(1, K + 1):
            h = torch.zeros(n - k, dtype=torch.long)
            for j in range(k):
                h = h * vocab + data[j:n - k + j]
            nxt = data[k:]
            keys, inv = torch.unique(h, return_inverse=True)
            counts = torch.zeros(len(keys), vocab)
            counts.index_put_((inv, nxt), torch.ones(len(nxt)), accumulate=True)
            self.orders.append((keys, counts, counts.sum(1), (counts > 0).sum(1).float()))

    def _ctx_hash(self, tok, k):
        B, T = tok.shape
        h = torch.zeros(B, T, dtype=torch.long)
        valid = torch.arange(T)[None].expand(B, T) >= k - 1
        for j in range(k):
            sh = torch.zeros_like(tok)
            off = k - 1 - j
            sh[:, off:] = tok[:, :T - off] if off else tok
            h = h * self.V + sh
        return h, valid

    @torch.no_grad()
    def logprobs(self, tok: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """tok (B,T) -> (B,T,V) log p(next | context ending at t). With targets
        (B,T) = the true next tokens, applies leave-one-out."""
        tok = tok.cpu()
        B, T = tok.shape
        V = self.V
        uni = self.uni.expand(B, T, V).clone()
        if targets is not None:
            tg = targets.cpu()
            uni.scatter_add_(2, tg[..., None], -torch.ones(B, T, 1))
        p = (uni.clamp_min(0) + 0.5) / (uni.clamp_min(0) + 0.5).sum(-1, keepdim=True)
        for k, (keys, counts, N, n1) in enumerate(self.orders, start=1):
            h, valid = self._ctx_hash(tok, k)
            pos = torch.searchsorted(keys, h.flatten()).clamp(max=len(keys) - 1)
            found = (keys[pos] == h.flatten()).view(B, T) & valid
            c = counts[pos].view(B, T, V)
            Nk, n1k = N[pos].view(B, T), n1[pos].view(B, T)
            if targets is not None:
                ct = c.gather(2, tg[..., None])[..., 0]
                c = c.scatter_add(2, tg[..., None], -torch.ones(B, T, 1))
                Nk = Nk - 1
                n1k = n1k - (ct == 1).float()
                found = found & (Nk > 0)
            denom = (Nk + n1k).clamp_min(1e-9)
            pk = (c.clamp_min(0) + n1k[..., None] * p) / denom[..., None]
            p = torch.where(found[..., None], pk, p)
        return torch.log(p.clamp_min(1e-9))
