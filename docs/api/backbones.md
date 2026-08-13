# Backbone API

## Interfaces and contracts

::: common.backbones.interfaces
    options:
      members:
        - BackboneCompatibilityError
        - BackboneCapability
        - FeatureLevel
        - SwapPolicy
        - TokenBatch
        - BackboneSpec
        - TextEncoder
        - PromptableTextEncoder
        - TileEncoder
        - SlideProjector
        - EncoderBundle
        - MethodBackboneContract
        - canonical_backbone_name
        - normalize_features

## Registry and builders

::: common.backbones.factory
    options:
      members:
        - BackboneInfo
        - TitanPromptableText
        - list_backbones
        - get_spec
        - get_info
        - register_backbone
        - unregister_backbone
        - build_encoder
        - build_backbone
