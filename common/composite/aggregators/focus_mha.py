"""FOCUS-style cross-attention aggregator.

Text features as queries, patches as keys/values; mean-pool over
attended outputs; linear classifier.

Lifted from `methods/focus/model.py::FOCUS.cross_attention`.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from common.composite.interfaces import Aggregator, PromptBank


class FocusMHAAggregator(Aggregator):
    name = "focus_mha"

    def __init__(self, in_dim: int, n_classes: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Linear(in_dim, n_classes)
        self.out_dim = in_dim

    def forward_vector(self, patches, bank: PromptBank):
        # text_features: (C, D) as queries; patches: (N, D) as K/V
        q = bank.text_features.unsqueeze(0)        # (1, C, D)
        kv = patches.unsqueeze(0)                  # (1, N, D)
        attended, _ = self.attn(q, kv, kv)
        return attended.squeeze(0).mean(0)         # (D,)

    def forward_logits(self, patches, bank: PromptBank):
        v = self.forward_vector(patches, bank)
        return self.classifier(v)
