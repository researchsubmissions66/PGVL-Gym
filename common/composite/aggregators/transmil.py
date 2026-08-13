"""TransMIL aggregator.

Re-uses the `TransLayer` (NyströmAttention) from
`common.models.transmil`. Two layers + class token + linear head.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from common.composite.interfaces import Aggregator, PromptBank
from common.models.transmil import TransLayer


class TransMILAggregator(Aggregator):
    name = "transmil"

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim) \
            if in_dim != hidden_dim else nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.layer1 = TransLayer(dim=hidden_dim)
        self.layer2 = TransLayer(dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.out_dim = hidden_dim

    def forward_vector(self, patches, bank: PromptBank):
        h = self.proj(patches).unsqueeze(0)        # (1, N, D)
        cls = self.cls_token.expand(1, -1, -1)
        h = torch.cat([cls, h], dim=1)             # (1, N+1, D)
        # pad so length is divisible by num_landmarks (Nystrom requirement)
        n = h.size(1)
        landmarks = h.size(-1) // 2 or 1
        pad = (landmarks - n % landmarks) % landmarks
        if pad > 0:
            h = torch.cat([h, h[:, :pad]], dim=1)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)
        return h[0, 0]                             # cls-token output, (D,)

    def forward_logits(self, patches, bank: PromptBank):
        v = self.forward_vector(patches, bank)
        return self.classifier(v)
