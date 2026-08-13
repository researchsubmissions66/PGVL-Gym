"""Fuse multiple PromptBanks into one.

When several prompt modules are enabled, each emits its own (C, D)
text features. We combine them via:

  - "average":         element-wise mean across modules
  - "weighted_sum":    learnable per-module scalar weights
  - "concat":          channel-wise concatenation, then a Linear D' -> D
                       projection so downstream aggregators still see (C, D)
  - "first":           use only the first enabled module's features
                       (degenerate; useful for ablations)

`aux` dicts from all modules are merged with a name prefix so keys
don't collide.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

from common.composite.interfaces import PromptModule, PromptBank


class PromptFusion(nn.Module):
    """Run each enabled PromptModule, then fuse to a single PromptBank."""

    def __init__(self,
                 modules: List[PromptModule],
                 mode: str = "average",
                 dim: int = 512,
                 n_classes: int = 2):
        super().__init__()
        if not modules:
            raise ValueError("PromptFusion requires at least one PromptModule.")
        self.modules_ = nn.ModuleList(modules)
        self.mode = mode

        if mode == "weighted_sum":
            self.weights = nn.Parameter(torch.ones(len(modules)) / len(modules))
        elif mode == "concat":
            self.proj = nn.Linear(dim * len(modules), dim)

    def forward(self) -> PromptBank:
        banks: List[PromptBank] = [m() for m in self.modules_]

        # ---- text_features fusion ----
        feats = [b.text_features for b in banks]    # each (C, D)
        if self.mode == "average":
            fused = torch.stack(feats).mean(dim=0)
        elif self.mode == "weighted_sum":
            w = torch.softmax(self.weights, dim=0)
            fused = (torch.stack(feats) * w.view(-1, 1, 1)).sum(dim=0)
        elif self.mode == "concat":
            fused = self.proj(torch.cat(feats, dim=-1))
        elif self.mode == "first":
            fused = feats[0]
        else:
            raise KeyError(f"Unknown fusion mode '{self.mode}'.")

        # ---- aux fusion: merge with module-name prefixes ----
        aux = {}
        for m, b in zip(self.modules_, banks):
            prefix = m.name
            for k, v in b.aux.items():
                aux[f"{prefix}.{k}"] = v
                # also expose the LAST one without a prefix for back-compat
                aux[k] = v

        return PromptBank(text_features=fused, aux=aux)
