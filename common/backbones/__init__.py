"""Public backbone API.

Use ``build_encoder`` for new integrations. ``build_backbone`` remains for
the vendored method implementations that still expect a raw tuple.
"""

from .factory import (  # noqa: F401
    BackboneInfo,
    build_backbone,
    build_backbone_handle,
    build_encoder,
    get_info,
    get_spec,
    list_backbones,
    register_backbone,
    unregister_backbone,
)
from .interfaces import (  # noqa: F401
    BackboneCapability,
    BackboneCompatibilityError,
    BackboneSpec,
    EncoderBundle,
    FeatureLevel,
    MethodBackboneContract,
    PromptableTextEncoder,
    SlideProjector,
    SwapPolicy,
    TextEncoder,
    TileEncoder,
    TokenBatch,
    canonical_backbone_name,
)

__all__ = [
    "BackboneCapability", "BackboneCompatibilityError", "BackboneInfo",
    "BackboneSpec", "EncoderBundle", "FeatureLevel", "MethodBackboneContract",
    "PromptableTextEncoder", "SlideProjector", "SwapPolicy", "TextEncoder",
    "TileEncoder", "TokenBatch", "build_backbone", "build_backbone_handle",
    "build_encoder", "canonical_backbone_name", "get_info", "get_spec",
    "list_backbones", "register_backbone", "unregister_backbone",
]
