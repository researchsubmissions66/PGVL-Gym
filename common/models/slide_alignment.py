"""Reusable alignment layers for precomputed slide embeddings."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideEmbeddingAdapter(nn.Module):
    """Map an arbitrary registered slide-vector width to a model space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mode: str = "linear",
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Slide-adapter dimensions must be positive")
        if mode == "linear":
            self.network = nn.Linear(input_dim, output_dim, bias=False)
        elif mode == "mlp":
            width = int(hidden_dim or max(input_dim, output_dim))
            if width <= 0:
                raise ValueError("Slide-adapter hidden_dim must be positive")
            self.network = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, output_dim, bias=False),
            )
        else:
            raise ValueError("Slide-adapter mode must be linear or mlp")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                "Slide adapter expects embeddings shaped "
                f"[batch, {self.input_dim}], got {list(features.shape)}")
        parameter = next(self.parameters())
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        return F.normalize(self.network(features).float(), dim=-1)


def build_slide_embedding_adapter(
    projection: Mapping[str, Any],
    *,
    input_dim: int,
    output_dim: int,
) -> SlideEmbeddingAdapter | None:
    """Build an explicit learned adapter; ``native`` remains model-owned."""
    mode = str(projection.get("mode", "native")).lower()
    if mode == "native":
        return None
    if mode not in {"linear", "mlp"}:
        raise ValueError("Slide projection mode must be native, linear, or mlp")
    declared_input = int(projection.get("input_dim", input_dim))
    declared_output = int(projection.get("output_dim", output_dim))
    if declared_input != input_dim:
        raise ValueError(
            f"Slide projection input_dim {declared_input} != {input_dim}")
    if declared_output != output_dim:
        raise ValueError(
            f"Slide projection output_dim {declared_output} != {output_dim}")
    return SlideEmbeddingAdapter(
        input_dim,
        output_dim,
        mode=mode,
        hidden_dim=projection.get("hidden_dim"),
        dropout=float(projection.get("dropout", 0.0)),
    )


# Backward-compatible name used by the original SLDPC integration.
TrainableSlideAdapter = SlideEmbeddingAdapter


__all__ = [
    "SlideEmbeddingAdapter",
    "TrainableSlideAdapter",
    "build_slide_embedding_adapter",
]
