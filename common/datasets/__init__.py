"""Shared data contracts used across method adapters."""

from .slide_embeddings import (  # noqa: F401
    SlideEmbeddingDataset,
    SlideEmbeddingSource,
    build_slide_embedding_loader,
    infer_slide_embedding_source_type,
)

__all__ = [
    "SlideEmbeddingDataset",
    "SlideEmbeddingSource",
    "build_slide_embedding_loader",
    "infer_slide_embedding_source_type",
]
