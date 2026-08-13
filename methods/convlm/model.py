"""Attribute-conditioned token-pruning ViT used by ConVLM.

The official ConVLM release is a patch-level zero-shot pipeline.  Its key
mechanisms are retained here: pathology-patch tokens are conditioned on the
Quilt-LLaVA/QuiltNet attribute of the observed class and pruned at three
transformer depths; the resulting visual representation is aligned to the
same attribute space for zero-shot prediction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConditionedBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float, attr_dim: int,
                 drop: float, prune_rate: float | None):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=drop,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
            nn.Linear(int(width * mlp_ratio), width), nn.Dropout(drop),
        )
        self.prune_rate = prune_rate
        self.attr_to_vis = (nn.Sequential(nn.Linear(attr_dim, width), nn.GELU(),
                                          nn.Linear(width, width))
                            if prune_rate is not None else None)
        self.token_to_attr = (nn.Linear(width, attr_dim)
                              if prune_rate is not None else None)

    def forward(self, x: torch.Tensor, attributes: torch.Tensor | None):
        aux_vis = aux_attr = None
        if self.attr_to_vis is not None and attributes is not None:
            aux_vis = self.attr_to_vis(attributes)
            # The released implementation blends the class-conditioned
            # attribute token into the query stream before token selection.
            x = x.clone()
            x[:, 0] = 0.9 * x[:, 0] + 0.1 * aux_vis
        update, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                              need_weights=False)
        x = x + update
        x = x + self.mlp(self.norm2(x))
        if self.prune_rate is None:
            return x, aux_vis, aux_attr

        patches = x[:, 1:]
        n_keep = max(1, min(patches.shape[1], math.ceil(patches.shape[1] * self.prune_rate)))
        scores = (F.normalize(patches, dim=-1) * F.normalize(x[:, :1], dim=-1)).sum(-1)
        keep = scores.topk(n_keep, dim=1).indices
        gathered = torch.gather(patches, 1, keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        # A per-stage semantic reconstruction head is the sparse-regression
        # term used by the official training objective.
        aux_attr = self.token_to_attr(gathered.mean(dim=1))
        return torch.cat([x[:, :1], gathered], dim=1), aux_vis, aux_attr


class AttributeConVLM(nn.Module):
    """ConVLM patch encoder returning an attribute-space visual embedding."""
    def __init__(self, attr_dim: int, image_size: int = 448, patch_size: int = 16,
                 width: int = 768, depth: int = 12, heads: int = 12,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 prune_layers: tuple[int, ...] = (3, 6, 9),
                 keep_rate: float = 0.7):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.grid_size = image_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size ** 2 + 1, width))
        self.pos_drop = nn.Dropout(drop)
        blocks = []
        for index in range(depth):
            blocks.append(_ConditionedBlock(
                width, heads, mlp_ratio, attr_dim, drop,
                keep_rate if index + 1 in prune_layers else None))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(width)
        self.to_attributes = nn.Linear(width, attr_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        if (height, width) == (self.grid_size, self.grid_size):
            return self.pos_embed
        patch_pos = self.pos_embed[:, 1:].transpose(1, 2).reshape(
            1, -1, self.grid_size, self.grid_size)
        patch_pos = F.interpolate(patch_pos, size=(height, width), mode="bicubic",
                                  align_corners=False)
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return torch.cat([self.pos_embed[:, :1], patch_pos], dim=1)

    def forward(self, images: torch.Tensor,
                condition_attributes: torch.Tensor | None = None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        x = self.patch_embed(images)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x + self._position_embedding(height, width))
        aux_vis: list[torch.Tensor] = []
        aux_attr: list[torch.Tensor] = []
        for block in self.blocks:
            x, visual, semantic = block(x, condition_attributes)
            if visual is not None:
                aux_vis.append(visual)
            if semantic is not None:
                aux_attr.append(semantic)
        pooled = self.norm(x[:, 0])
        return {"embedding": F.normalize(self.to_attributes(pooled), dim=-1),
                "global": pooled, "aux_vis": aux_vis, "aux_attr": aux_attr}
