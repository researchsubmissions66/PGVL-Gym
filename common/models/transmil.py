"""TransMIL building blocks (NyströmAttention).

The `TransLayer` class below appears byte-for-byte identical in
SLIP's `networks/transmil.py` and PathPT's `model_utils.py`.  Methods
that wrap or extend TransMIL should `import` from here instead of
duplicating it.
"""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    from nystrom_attention import NystromAttention
except ImportError as e:                                      # pragma: no cover
    NystromAttention = None
    _IMPORT_ERROR = e


class TransLayer(nn.Module):
    """Single Nyström-approximated transformer layer (TransMIL block)."""

    def __init__(self, norm_layer=nn.LayerNorm, dim: int = 512):
        super().__init__()
        if NystromAttention is None:
            raise ImportError(
                "TransLayer requires `nystrom_attention`. "
                "Install it via `pip install nystrom-attention`.\n"
                f"Original error: {_IMPORT_ERROR}")
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator from TransMIL."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat \
            + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
