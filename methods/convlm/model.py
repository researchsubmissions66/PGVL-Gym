"""Local attribute-conditioned token-pruning transformer inspired by ConVLM.

The released training path feeds RGB images to its own patch-embedding ViT.
PGVL-Gym instead consumes already embedded WSI patches and reconstructs the
reported attribute conditioning and token-pruning ideas in feature space.
That distinction is recorded as partial implementation provenance.
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
    """Local ConVLM-like encoder over precomputed patch features.

    The release contains a separate UNI feature-extraction utility, but its
    ``train.py`` constructs the raw-image ViT in ``convlm.py`` and does not
    consume those extracted bags. This class therefore defines PGVL's own
    feature-bag adaptation boundary rather than claiming input equivalence.

    The reconstructed mechanisms inject attributes into the token stream,
    prune tokens at three depths, and align the output to attribute space for
    zero-shot comparison.
    """

    def __init__(self, attr_dim: int, feature_dim: int = 1024,
                 max_patches: int = 4096,
                 width: int = 768, depth: int = 12, heads: int = 12,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 prune_layers: tuple[int, ...] = (3, 6, 9),
                 keep_rate: float = 0.7):
        super().__init__()
        if attr_dim <= 0 or feature_dim <= 0 or width <= 0:
            raise ValueError("ConVLM dimensions must be positive")
        if max_patches <= 0 or depth <= 0:
            raise ValueError("ConVLM max_patches and depth must be positive")
        if heads <= 0 or width % heads != 0:
            raise ValueError("ConVLM width must be divisible by positive heads")
        if not 0.0 <= drop < 1.0:
            raise ValueError("ConVLM drop must be in [0, 1)")
        if not 0.0 < keep_rate <= 1.0:
            raise ValueError("ConVLM keep_rate must be in (0, 1]")
        self.feature_dim = int(feature_dim)
        # Upstream's configurable "embedding dimensions": the offline encoder's
        # width enters here, it is not baked into the architecture.
        self.patch_embed = nn.Linear(self.feature_dim, width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, width))
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

    def _position_embedding(self, patches: int) -> torch.Tensor:
        """Return class + patch positions for a bag of ``patches`` features."""
        if patches > self.pos_embed.shape[1] - 1:
            raise ValueError(
                f"ConVLM received {patches} patches but max_patches is "
                f"{self.pos_embed.shape[1] - 1}")
        return torch.cat(
            [self.pos_embed[:, :1], self.pos_embed[:, 1:patches + 1]], dim=1)

    def forward(self, features: torch.Tensor,
                condition_attributes: torch.Tensor | None = None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Encode a bag of precomputed patch features.

        Args:
            features: ``[batch, patches, feature_dim]`` from the offline encoder.
            condition_attributes: ``[batch, attr_dim]`` attribute vector of the
                observed class during training; ``None`` at inference, where the
                embedding is compared against every class attribute instead.
        """
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"ConVLM expects [batch, patches, {self.feature_dim}], got "
                f"{list(features.shape)}")
        if features.shape[1] == 0:
            raise ValueError("ConVLM requires at least one patch per slide")
        x = self.patch_embed(features.float())
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x + self._position_embedding(features.shape[1]))
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
