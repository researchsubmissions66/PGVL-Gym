"""PathPT spatial-aware Conv1D aggregator.

The 'vision_grad' branch of PathPT applies a multi-kernel 1D
convolution to the patch sequence before averaging. Here we expose
the same idea as a small standalone aggregator.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import Aggregator, PromptBank


class PathPTConv1DAggregator(Aggregator):
    name = "pathpt_conv1d"

    def __init__(self, in_dim: int, out_dim: int = 768,
                 n_classes: int = 2, kernels=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, kernel_size=k, padding=k // 2)
            for k in kernels])
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)
        self.classifier = nn.Linear(out_dim, n_classes)
        self.out_dim = out_dim

    def forward_vector(self, patches, bank: PromptBank):
        # patches: (N, D) → (1, D, N) for Conv1d
        x = patches.t().unsqueeze(0)
        h_sum = sum(self.act(c(x)) for c in self.convs) / len(self.convs)
        h = h_sum.squeeze(0).t()      # (N, out_dim)
        h = self.norm(h)
        return h.mean(0)              # global average pool

    def forward_logits(self, patches, bank: PromptBank):
        v = self.forward_vector(patches, bank)
        return self.classifier(v)
