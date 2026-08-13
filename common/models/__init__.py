"""Shared model building blocks.

Re-exports:
    Attn_Net, Attn_Net_Gated      -- gated attention pooling (CLAM)
    MIL_fc, MIL_fc_mc             -- vanilla MIL baselines
    PromptLearner, TextEncoder    -- canonical CoOp blocks
    TransLayer, PPEG              -- TransMIL Nyström-attention block
"""
from ._clam_blocks import Attn_Net, Attn_Net_Gated         # noqa: F401
from .mil_baselines import MIL_fc, MIL_fc_mc               # noqa: F401
from .coop import PromptLearner, TextEncoder               # noqa: F401
from .transmil import TransLayer, PPEG                     # noqa: F401
from .slide_alignment import (  # noqa: F401
    SlideEmbeddingAdapter,
    TrainableSlideAdapter,
    build_slide_embedding_adapter,
)
