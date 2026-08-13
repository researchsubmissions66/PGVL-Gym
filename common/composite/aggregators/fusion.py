"""Aggregator fusion.

Runs every enabled aggregator in parallel and combines them in one of
two modes:

  - "logit_ensemble" (Level 1):
        each aggregator produces (C,) logits via `forward_logits`.
        Fuse logits via mean / weighted_sum / max / voting.
        Each aggregator has its own classifier head; fully decoupled.

  - "vector_fusion" (Level 2):
        each aggregator produces a slide vector via `forward_vector`.
        Vectors are combined via concat / mean / weighted_sum / gated;
        the result is fed through a SHARED classifier head to produce
        (C,) logits. Tighter coupling, fewer parameters in the head.

Both modes accept any subset of aggregators; the user can enable just
one (= no fusion) or all of them (= full ensemble).
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import Aggregator, PromptBank


# =====================================================================
# Logit-level (Level 1)
# =====================================================================
class LogitEnsemble(nn.Module):
    """Combine each aggregator's (C,) logits."""

    def __init__(self, aggregators: List[Aggregator],
                 mode: str = "mean", n_classes: int = 2):
        super().__init__()
        if not aggregators:
            raise ValueError("Need at least one aggregator.")
        self.aggregators = nn.ModuleList(aggregators)
        self.mode = mode
        if mode == "weighted_sum":
            self.weights = nn.Parameter(
                torch.ones(len(aggregators)) / len(aggregators))

    def forward(self, patches: torch.Tensor, bank: PromptBank) -> torch.Tensor:
        logits = torch.stack([a.forward_logits(patches, bank)
                              for a in self.aggregators])      # (K, C)
        if self.mode == "mean":
            return logits.mean(0)
        if self.mode == "weighted_sum":
            w = torch.softmax(self.weights, dim=0)
            return (logits * w.view(-1, 1)).sum(0)
        if self.mode == "max":
            return logits.max(0).values
        if self.mode == "voting":
            # soft voting: average of softmax probs, then convert back
            probs = F.softmax(logits, dim=-1).mean(0)
            # avoid log(0) by adding eps
            return torch.log(probs.clamp(min=1e-12))
        raise KeyError(f"Unknown logit-fusion mode '{self.mode}'.")


# =====================================================================
# Vector-level (Level 2)
# =====================================================================
class VectorFusion(nn.Module):
    """Combine each aggregator's (D,) slide vector, then classify."""

    def __init__(self, aggregators: List[Aggregator],
                 mode: str = "concat", n_classes: int = 2):
        super().__init__()
        if not aggregators:
            raise ValueError("Need at least one aggregator.")
        self.aggregators = nn.ModuleList(aggregators)
        self.mode = mode

        dims = [a.out_dim for a in aggregators]
        if any(d == 0 for d in dims):
            raise ValueError(
                "VectorFusion requires every aggregator to set `out_dim`. "
                "Use LogitEnsemble for aggregators that only support "
                "`forward_logits`.")

        if mode == "concat":
            self.head = nn.Linear(sum(dims), n_classes)
        elif mode == "mean":
            assert len(set(dims)) == 1, \
                "mean-fusion requires identical out_dim across aggregators"
            self.head = nn.Linear(dims[0], n_classes)
        elif mode == "weighted_sum":
            assert len(set(dims)) == 1, \
                "weighted_sum requires identical out_dim across aggregators"
            self.weights = nn.Parameter(
                torch.ones(len(aggregators)) / len(aggregators))
            self.head = nn.Linear(dims[0], n_classes)
        elif mode == "gated":
            # per-element learnable gate per aggregator
            assert len(set(dims)) == 1, \
                "gated fusion requires identical out_dim across aggregators"
            self.gates = nn.ParameterList([
                nn.Parameter(torch.ones(dims[0])) for _ in aggregators])
            self.head = nn.Linear(dims[0], n_classes)
        elif mode == "cross_attention":
            # vectors as a length-K sequence, aggregate via 1-head MHA
            assert len(set(dims)) == 1
            self.attn = nn.MultiheadAttention(
                embed_dim=dims[0], num_heads=1, batch_first=False)
            self.cls = nn.Parameter(torch.randn(1, 1, dims[0]) * 0.02)
            self.head = nn.Linear(dims[0], n_classes)
        else:
            raise KeyError(f"Unknown vector-fusion mode '{mode}'.")

    def forward(self, patches: torch.Tensor, bank: PromptBank) -> torch.Tensor:
        vecs = [a.forward_vector(patches, bank) for a in self.aggregators]

        if self.mode == "concat":
            v = torch.cat(vecs, dim=-1)
        elif self.mode == "mean":
            v = torch.stack(vecs).mean(0)
        elif self.mode == "weighted_sum":
            w = torch.softmax(self.weights, dim=0)
            v = (torch.stack(vecs) * w.view(-1, 1)).sum(0)
        elif self.mode == "gated":
            stacked = torch.stack(vecs)                     # (K, D)
            gates = torch.stack([torch.sigmoid(g) for g in self.gates])  # (K, D)
            v = (stacked * gates).sum(0) / gates.sum(0).clamp(min=1e-6)
        elif self.mode == "cross_attention":
            seq = torch.stack(vecs).unsqueeze(1)             # (K, 1, D)
            cls = self.cls
            out, _ = self.attn(cls, seq, seq)
            v = out.squeeze(0).squeeze(0)
        else:
            raise KeyError(f"Unknown vector-fusion mode '{self.mode}'.")

        return self.head(v)
