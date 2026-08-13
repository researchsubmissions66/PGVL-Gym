"""MSCPT context-focused top-K patch selector.

Keep only the K patches with highest summed text-similarity. Lifted
from MSCPT's `select_5X_pic.py` logic, adapted for in-line use.

Reference:
    Han et al., "MSCPT: Few-shot WSI Classification with Multi-scale
    and Context-focused Prompt Tuning", IEEE TMI 2025.
"""
from __future__ import annotations
import torch

from common.composite.interfaces import PatchSelector


class MSCPTTopK(PatchSelector):
    name = "mscpt_topk"

    def __init__(self, k: int = 100):
        super().__init__()
        self.k = k

    def forward(self, patches, text_features, coords=None):
        n = patches.size(0)
        if n <= self.k or patches.size(-1) != text_features.size(-1):
            return patches
        scores = (patches @ text_features.T).sum(dim=1)   # (N,)
        _, top = torch.topk(scores, self.k)
        top, _ = torch.sort(top)
        return patches[top]
