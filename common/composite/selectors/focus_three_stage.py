"""FOCUS three-stage adaptive visual compression, lifted from
`methods/focus/model.py` into a standalone `PatchSelector`.

Stages:
  1. Window-wise redundancy elimination via intra-window cosine
     similarity and adaptive threshold (mean + std).
  2. Top-K text-relevance selection: keep the most semantically
     relevant patches up to `max_context`.
  3. Neighbor-aware compression: drop a patch if its cosine similarity
     to its immediate predecessor exceeds `sim_threshold`.

This is the entire "secret sauce" of FOCUS, free of its CONCH-loading
boilerplate so it can run on any backbone's features.

References:
    Guo et al., "FOCUS: Knowledge-enhanced Adaptive Visual Compression",
    CVPR 2025.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import PatchSelector


class FocusThreeStage(PatchSelector):
    name = "focus_three_stage"

    def __init__(self,
                 window_size: int = 7,
                 sim_threshold: float = 0.7,
                 max_context: int = 4096):
        super().__init__()
        self.window_size = window_size
        self.sim_threshold = sim_threshold
        self.max_context = max_context

    # -- Stage 1: window-wise redundancy ----------------------------------
    def _stage1_redundancy(self, x: torch.Tensor):
        """Returns the indices of patches that survive stage 1."""
        N, _ = x.shape
        x_norm = F.normalize(x, p=2, dim=-1)
        keep_idx_chunks = []

        for i in range(0, N, self.window_size):
            window = x_norm[i:i + self.window_size]
            if window.size(0) < 2:
                keep_idx_chunks.append(
                    torch.arange(i, min(i + self.window_size, N), device=x.device))
                continue
            window_sim = window @ window.t()
            if window_sim.numel() > 1:
                threshold = window_sim.mean() + window_sim.std(unbiased=False)
            else:
                threshold = window_sim.mean()
            redundant = window_sim.mean(1) > threshold
            keep = torch.where(~redundant)[0] + i
            if keep.numel() == 0:
                keep = torch.tensor([i], device=x.device)
            keep_idx_chunks.append(keep)

        if not keep_idx_chunks:
            return torch.arange(N, device=x.device)
        return torch.cat(keep_idx_chunks)

    # -- Stage 2: text-guided top-K ----------------------------------------
    def _stage2_text_relevance(self, x: torch.Tensor,
                               text_features: torch.Tensor,
                               survivors: torch.Tensor):
        """Among the survivor indices, keep top max_context by text relevance."""
        N = x.size(0)
        # Compute relevance for *all* patches (the original code does this).
        if x.size(-1) != text_features.size(-1):
            # Skip relevance scoring if dims mismatch -- keep survivors unchanged.
            return survivors
        relevance_all = (x @ text_features.T).mean(-1)        # (N,)
        importance_mask = torch.zeros(N, device=x.device)
        importance_mask[survivors] = relevance_all[survivors]
        k = min(self.max_context, N)
        _, top_idx = torch.topk(importance_mask, k)
        top_idx, _ = torch.sort(top_idx)
        return top_idx

    # -- Stage 3: neighbor-aware compression -------------------------------
    def _stage3_spatial_compression(self, x: torch.Tensor):
        """Within consecutive 8-patch chunks, drop patches whose cos-sim
        with their predecessor exceeds sim_threshold."""
        N, _ = x.shape
        chunk_size = 8
        kept_chunks = []
        for i in range(0, N, chunk_size):
            chunk = x[i:i + chunk_size]
            if chunk.size(0) == 1:
                kept_chunks.append(chunk)
                continue
            chunk_n = F.normalize(chunk, p=2, dim=-1)
            sim = F.cosine_similarity(chunk_n[:-1], chunk_n[1:], dim=-1)
            keep_mask = sim < self.sim_threshold
            kept = torch.cat([chunk[:1], chunk[1:][keep_mask]])
            kept_chunks.append(kept)
        out = torch.cat(kept_chunks)
        if out.size(0) > self.max_context:
            out = out[: self.max_context]
        return out

    # -- forward ----------------------------------------------------------
    def forward(self, patches, text_features, coords=None):
        survivors = self._stage1_redundancy(patches)
        idx = self._stage2_text_relevance(patches, text_features, survivors)
        selected = patches[idx]
        return self._stage3_spatial_compression(selected)
