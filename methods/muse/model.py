"""MUSE: text-retrieval, mixture-of-experts WSI aggregation.

MUSE consumes registered static patch embeddings and class-specific LLM descriptions.
The descriptions first retrieve a compact, slide-relevant query set; a sparse
MoE transforms these queries before cross-attention aggregates the patch bag.
This is the feature-space portion of the official implementation and avoids
its hard-coded absolute paths and CUDA-only text initialization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SparseMoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0:
            raise ValueError("num_selected must be positive")
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(num_experts)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.router(tokens), dim=-1)
        values, indices = weights.topk(self.top_k, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        result = torch.zeros_like(tokens)
        for expert_index, expert in enumerate(self.experts):
            selected = indices.eq(expert_index)
            if not selected.any():
                continue
            token_rows, route_cols = selected.nonzero(as_tuple=True)
            result[token_rows] += expert(tokens[token_rows]) * values[token_rows, route_cols].unsqueeze(-1)
        return result


class MUSEModel(nn.Module):
    """Official-MUSE-compatible slide classifier over one feature bag.

    ``prompt_bank`` has shape ``(classes, descriptions, dim)`` and contains
    frozen prompt-encoder embeddings for the published class descriptions. The bank
    is intentionally an input artifact: it makes the exact text prior
    explicit and avoids silently substituting an unrelated text encoder.
    """
    def __init__(self, input_dim: int, n_classes: int, prompt_bank: torch.Tensor,
                 embed_dim: int = 512, num_heads: int = 8, num_experts: int = 8,
                 num_selected: int = 2, retrieval_k: int = 8, dropout: float = 0.25):
        super().__init__()
        if prompt_bank.ndim != 3 or prompt_bank.shape[0] != n_classes:
            raise ValueError("prompt_bank must have shape [n_classes, n_descriptions, dim]")
        if prompt_bank.shape[1] == 0:
            raise ValueError("prompt_bank must contain at least one description per class")
        if not torch.isfinite(prompt_bank).all():
            raise ValueError("prompt_bank contains NaN or infinity")
        if prompt_bank.shape[-1] != embed_dim:
            raise ValueError(
                f"prompt_bank dim {prompt_bank.shape[-1]} does not match embed_dim {embed_dim}")
        if input_dim <= 0 or embed_dim <= 0:
            raise ValueError("input_dim and embed_dim must be positive")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by positive num_heads")
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_classes = n_classes
        self.input_dim = int(input_dim)
        self.retrieval_k = retrieval_k
        self.register_buffer("prompt_bank", F.normalize(prompt_bank.float(), dim=-1))
        self.visual_adapter = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.class_queries = nn.Parameter(self.prompt_bank.mean(dim=1).clone())
        self.moe = _SparseMoE(embed_dim, num_experts, num_selected)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads,
                                                      dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.bag_norm = nn.LayerNorm(embed_dim)
        self.class_gate = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def _retrieve(self, patch_features: torch.Tensor) -> torch.Tensor:
        """Select the most slide-relevant descriptions for every class."""
        slide_query = F.normalize(patch_features.mean(dim=0), dim=-1)
        scores = torch.einsum("cpd,d->cp", self.prompt_bank, slide_query)
        k = min(self.retrieval_k, self.prompt_bank.shape[1])
        values, indices = scores.topk(k, dim=1)
        selected = torch.gather(
            self.prompt_bank, 1,
            indices.unsqueeze(-1).expand(-1, -1, self.prompt_bank.shape[-1]))
        return (selected * F.softmax(values, dim=1).unsqueeze(-1)).sum(dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            if features.shape[0] != 1:
                raise ValueError("MUSE requires batch_size=1 for variable-length WSI bags")
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError("MUSE expects patch features with shape [patches, dim]")
        if features.shape[0] == 0 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"MUSE expects a non-empty patch bag with width "
                f"{self.input_dim}, got {list(features.shape)}")
        patches = self.visual_adapter(features.float())
        retrieved = self._retrieve(patches)
        queries = self.query_norm(self.class_queries + retrieved)
        queries = self.moe(queries)
        # One text-conditioned query per class attends to the same patch bag.
        key_values = self.bag_norm(patches).unsqueeze(0).expand(self.n_classes, -1, -1)
        attended, _ = self.cross_attention(queries.unsqueeze(1), key_values, key_values,
                                           need_weights=False)
        class_features = attended.squeeze(1)
        mixture = F.softmax(self.class_gate(class_features).squeeze(-1), dim=0)
        slide_feature = (mixture.unsqueeze(-1) * class_features).sum(dim=0, keepdim=True)
        return self.classifier(slide_feature)
