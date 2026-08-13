"""ViLa-MIL learnable-prototype aggregator.

Lifted from `methods/vila_mil/model.py::ViLa_MIL_Model.forward`. The
key innovation: K learnable image prototype tokens cross-attend to
the patch sequence, then a gated-attention pool produces the slide
vector.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import Aggregator, PromptBank


def _trunc_normal_(t, std=0.02):
    nn.init.trunc_normal_(t, std=std)


class ViLaPrototypeAggregator(Aggregator):
    name = "vila_prototypes"

    def __init__(self, in_dim: int, hidden_dim: int = 192,
                 prototype_number: int = 16, n_classes: int = 2):
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.empty(prototype_number, 1, in_dim))
        _trunc_normal_(self.prototypes, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=1, batch_first=False)
        self.norm = nn.LayerNorm(in_dim)

        self.attn_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(in_dim, n_classes)
        self.out_dim = in_dim

    def forward_vector(self, patches, bank: PromptBank):
        # patches: (N, D); turn into (N, 1, D) for nn.MHA
        M = patches.unsqueeze(1)
        protos = self.prototypes      # (P, 1, D)
        comp, _ = self.cross_attn(protos, M, M)
        comp = self.norm(comp + protos)
        H = comp.squeeze(1)            # (P, D)

        A = self.attn_w(self.attn_V(H) * self.attn_U(H))   # (P, 1)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        slide_vec = torch.mm(A, H).squeeze(0)              # (D,)
        return slide_vec

    def forward_logits(self, patches, bank: PromptBank):
        v = self.forward_vector(patches, bank)
        # ViLa-MIL's original head also dot-products with text features
        # for a similarity-based logit. We blend both:
        #   plain classifier + (slide_vec · text_features)
        clf = self.classifier(v)
        if bank.text_features.size(-1) == v.size(-1):
            sim = bank.text_features @ v          # (C,)
            return clf + sim
        return clf
