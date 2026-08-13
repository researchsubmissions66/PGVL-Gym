"""Gated-attention pooling aggregator (CLAM-style).

A clean baseline: gated attention over patches → slide vector → linear
classifier. Implements both `forward_vector` and `forward_logits`.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import Aggregator, PromptBank
from common.models import Attn_Net_Gated


class AttnPoolAggregator(Aggregator):
    name = "attn_pool"

    def __init__(self, in_dim: int, hidden_dim: int = 192,
                 n_classes: int = 2, dropout: bool = True):
        super().__init__()
        self.attn = Attn_Net_Gated(L=in_dim, D=hidden_dim,
                                   dropout=dropout, n_classes=1)
        self.classifier = nn.Linear(in_dim, n_classes)
        self.out_dim = in_dim

    def forward_vector(self, patches, bank: PromptBank) -> torch.Tensor:
        A, h = self.attn(patches)              # A: (N, 1), h: (N, D)
        A = torch.transpose(A, 1, 0)           # (1, N)
        A = F.softmax(A, dim=1)
        slide_vec = torch.mm(A, h).squeeze(0)  # (D,)
        return slide_vec

    def forward_logits(self, patches, bank: PromptBank) -> torch.Tensor:
        v = self.forward_vector(patches, bank)
        return self.classifier(v)
