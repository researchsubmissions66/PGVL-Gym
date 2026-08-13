"""Common building blocks shared by all WSI/VLM methods.

This package contains de-duplicated code that almost every paper in the
unified codebase pulls from:

  - common.wsi_core      : CLAM-derived WSI segmentation / patching primitives
  - common.datasets      : CLAM-derived bag/slide dataset wrappers
  - common.utils         : training-loop helpers (EarlyStopping, collate, etc.)
  - common.models        : shared model building blocks (Attn_Net_Gated,
                           Nystrom-attention TransLayer, vanilla MIL_fc,
                           and the canonical CoOp PromptLearner / TextEncoder)
  - common.backbones     : a single factory that loads CLIP / PLIP / CONCH /
                           MUSK / KEEP into a uniform interface
  - (top-level `clip/` is the shared OpenAI CLIP source (used by methods that
                           tune CLIP-RN50 / CLIP-ViT-B/16)
"""
