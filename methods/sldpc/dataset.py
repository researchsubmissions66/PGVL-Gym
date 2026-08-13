"""Compatibility imports for the framework-level slide-embedding loader."""
from __future__ import annotations

from common.datasets.slide_embeddings import (
    SlideEmbeddingDataset,
    SlideEmbeddingSource,
    build_slide_embedding_loader,
    infer_slide_embedding_source_type,
    normalise_slide_id,
)


def build_sldpc_loader(cfg: dict, split: str, fold: int, shuffle: bool = True):
    """Deprecated method-owned alias; use ``build_slide_embedding_loader``."""
    return build_slide_embedding_loader(cfg, split, fold, shuffle=shuffle)


__all__ = [
    "SlideEmbeddingDataset",
    "SlideEmbeddingSource",
    "build_sldpc_loader",
    "build_slide_embedding_loader",
    "infer_slide_embedding_source_type",
    "normalise_slide_id",
]
