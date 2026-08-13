"""SLIP tissue-routing aggregator.

Lifted from `methods/slip/methods/slidecoop.py::SLIP.forward`.
Requires `bank.aux["tissue"]` (a (T, D) matrix of frozen tissue-prompt
features). If unavailable (e.g. only `coop_flat` prompts are enabled),
falls back to plain text-cosine pooling.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import Aggregator, PromptBank


class SLIPRoutingAggregator(Aggregator):
    name = "slip_routing"

    def __init__(self, in_dim: int, n_classes: int,
                 temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.out_dim = in_dim
        self.fallback_classifier = nn.Linear(in_dim, n_classes)

    def forward_vector(self, patches, bank: PromptBank):
        # SLIP works at logit level; the "vector" is the
        # class-conditional slide feature of the predicted class.
        logits = self.forward_logits(patches, bank)
        c_hat = logits.argmax()
        # Recompute the class slide vector for the argmax class.
        slide_features = F.normalize(patches, dim=1)
        slide_weights = F.normalize(bank.text_features, dim=1).T
        tissue = bank.aux.get("tissue")
        if tissue is None:
            # Fall back: use text-similarity-weighted pool
            sim = slide_features @ slide_weights / self.temperature
            attn = F.softmax(sim, dim=1)                  # (N, C)
            return F.normalize(slide_features.T @ attn, dim=0)[:, c_hat]

        with torch.no_grad():
            s_pt = F.softmax(slide_features @ tissue.T.to(patches.device)
                             / self.temperature, dim=1)        # (N, T)
            s_st = F.softmax(tissue.to(patches.device) @ slide_weights
                             / self.temperature, dim=0)        # (T, C)
            s_attn = s_pt @ s_st                               # (N, C)
            class_vecs = F.normalize(
                slide_features.T @ s_attn, dim=0)              # (D, C)
        return class_vecs[:, c_hat]

    def forward_logits(self, patches, bank: PromptBank):
        slide_features = F.normalize(patches, dim=1)
        slide_weights = F.normalize(bank.text_features, dim=1).T   # (D, C)
        tissue = bank.aux.get("tissue")

        if tissue is None:
            # No tissue prompts available -> fallback: top-K-style pooling
            sim = slide_features @ slide_weights / self.temperature
            return sim.mean(0)

        with torch.no_grad():
            s_pt = F.softmax(slide_features @ tissue.T.to(patches.device)
                             / self.temperature, dim=1)
            s_st = F.softmax(tissue.to(patches.device) @ slide_weights
                             / self.temperature, dim=0)
            s_attn = s_pt @ s_st
            class_vecs = F.normalize(
                slide_features.T @ s_attn, dim=0)              # (D, C)

        cross_corr = class_vecs.T @ slide_weights              # (C, C)
        return torch.diag(cross_corr) / self.temperature
