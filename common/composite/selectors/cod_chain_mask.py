"""CoD-MIL chain-of-diagnosis mask selector.

Drops patches that score in the bottom `topk_ratio` fraction by
maximum text similarity against the slide-class prompts. Mirrors the
patch-masking step used in `methods/cod_mil/model.py` (the lines that
build `mask_id` via `torch.topk(logits_text_low, ...)`).

Reference:
    Shi et al., "Chain-of-Diagnosis Prompting MIL", IEEE TMI 2024.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from common.composite.interfaces import PatchSelector


class CoDChainMask(PatchSelector):
    name = "cod_chain_mask"

    def __init__(self, topk_ratio: float = 0.1):
        """
        Args:
            topk_ratio: fraction of LOWEST-scoring patches to drop.
        """
        super().__init__()
        assert 0.0 <= topk_ratio < 1.0
        self.topk_ratio = topk_ratio

    def forward(self, patches, text_features, coords=None):
        if patches.size(-1) != text_features.size(-1) or self.topk_ratio == 0.0:
            return patches
        sims = patches @ text_features.T                  # (N, C)
        scores = sims.max(dim=1).values                   # (N,)
        n = patches.size(0)
        n_drop = int(n * self.topk_ratio)
        if n_drop <= 0:
            return patches
        # keep the (n - n_drop) highest-scoring patches
        _, top_idx = torch.topk(scores, n - n_drop)
        top_idx, _ = torch.sort(top_idx)
        return patches[top_idx]
