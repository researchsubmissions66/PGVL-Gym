"""Capability-based interfaces for vision-language backbones.

The papers in :mod:`methods` do not all consume the same kind of encoder.
Some need a black-box text encoder, some inject learnable tokens into a text
transformer, and SLDPC needs a *paired* slide projector and promptable text
tower.  A single ``(model, tokenizer)`` tuple cannot express those
differences safely.

This module deliberately keeps the interfaces narrow.  Vendor-specific
wrappers implement them while method code keeps ownership of the published
architecture.  In particular, compatibility never implies that an
unpublished alignment projection may be inserted between unrelated models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneCompatibilityError(ValueError):
    """Raised before model construction when an encoder contract is unmet."""


class BackboneCapability(str, Enum):
    """Small, independently testable encoder capabilities."""

    TEXT_ENCODE = "text_encode"
    SOFT_PROMPT = "soft_prompt"
    DEEP_TEXT_PROMPT = "deep_text_prompt"
    TILE_ENCODE = "tile_encode"
    DEEP_VISION_PROMPT = "deep_vision_prompt"
    PATCH_PROJECT = "patch_project"
    SLIDE_PROJECT = "slide_project"
    PAIRED_TILE_TEXT = "paired_tile_text"
    PAIRED_SLIDE_TEXT = "paired_slide_text"
    REPORT_TEXT = "report_text"


class FeatureLevel(str, Enum):
    """The tensor level at which a method meets its encoder."""

    RAW_TILE = "raw_tile"
    PATCH_BAG = "patch_bag"
    DUAL_SCALE_PATCH_BAG = "dual_scale_patch_bag"
    SLIDE_EMBEDDING = "slide_embedding"
    PATCH_SEQUENCE_WITH_REPORT = "patch_sequence_with_report"
    ATTRIBUTE_CONDITIONED_TILE = "attribute_conditioned_tile"
    COMPOSITE = "composite"


class SwapPolicy(str, Enum):
    """How broadly a method can accept another encoder implementation."""

    CAPABILITY = "capability"       # any registered bundle satisfying the hooks
    ALLOWLIST = "allowlist"         # native code has one branch per listed family
    FIXED = "fixed"                 # the encoder is part of the paper architecture
    PRECOMPUTED = "precomputed"     # no runtime encoder; aligned files are required


@dataclass(frozen=True)
class TokenBatch:
    """Tokenizer output with explicit masks and pooling positions.

    ``extras`` preserves vendor fields such as ``token_type_ids`` without
    making every consumer depend on a Hugging Face ``BatchEncoding``.
    """

    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    eot_indices: Optional[torch.Tensor] = None
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: Any) -> "TokenBatch":
        """Return a copy with every tensor moved to ``device``."""
        return TokenBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=(self.attention_mask.to(device)
                            if self.attention_mask is not None else None),
            eot_indices=(self.eot_indices.to(device)
                         if self.eot_indices is not None else None),
            extras={key: value.to(device) for key, value in self.extras.items()},
        )

    def as_kwargs(self) -> Dict[str, torch.Tensor]:
        """Convert the batch to keyword arguments accepted by text models."""
        output = {"input_ids": self.input_ids, **dict(self.extras)}
        if self.attention_mask is not None:
            output["attention_mask"] = self.attention_mask
        return output


@dataclass(frozen=True)
class BackboneSpec:
    """Record dimensions and provenance needed to judge a backbone swap.

    Capabilities describe operations exposed by the runtime bundle. Dimension
    fields refer to native encoder boundaries; ``shared_dim`` is the paired
    comparison space when one exists.
    """

    name: str
    family: str
    feature_space_id: str
    capabilities: FrozenSet[BackboneCapability]
    revision: Optional[str] = None
    tile_dim: Optional[int] = None
    slide_input_dim: Optional[int] = None
    vision_token_dim: Optional[int] = None
    text_token_dim: Optional[int] = None
    shared_dim: Optional[int] = None
    context_length: Optional[int] = None
    image_size: Optional[int] = None
    aliases: Tuple[str, ...] = ()

    def has(self, *capabilities: BackboneCapability) -> bool:
        """Return whether the spec declares every requested capability."""
        return all(capability in self.capabilities for capability in capabilities)


class TextEncoder(Protocol):
    """Black-box text encoding in the model's native shared space."""

    def tokenize(self, texts: Sequence[str]) -> TokenBatch:
        """Tokenize ordered strings without changing their order."""
        ...

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        """Encode strings or tokens into the native shared feature space."""
        ...


class PromptableTextEncoder(TextEncoder, Protocol):
    """Text tower that supports differentiable embedded soft prompts."""

    @property
    def token_width(self) -> int:
        """Return the width of native token embeddings."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Return the computation dtype of the promptable text tower."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the device hosting the promptable text tower."""
        ...

    @property
    def tokenizer(self) -> Any:
        """Return the native tokenizer used by this text tower."""
        ...

    def embed_tokens(self, tokens: TokenBatch) -> torch.Tensor:
        """Map token IDs to differentiable native token embeddings."""
        ...

    def encode_embedded(self, embeddings: torch.Tensor, tokens: TokenBatch,
                        normalize: bool = True) -> torch.Tensor:
        """Encode pre-embedded prompt tokens into the paired shared space."""
        ...


class TileEncoder(Protocol):
    """Black-box image-tile encoder in the model's native feature space."""

    def encode_tiles(self, images: torch.Tensor,
                     normalize: bool = True) -> torch.Tensor:
        """Encode a batch of image tiles, optionally normalizing each row."""
        ...


class SlideProjector(Protocol):
    """Native projection from raw slide vectors to a paired text space."""

    def project_slide(self, raw_embeddings: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
        """Project raw slide embeddings through the paired model head."""
        ...


@dataclass
class EncoderBundle:
    """Uniform runtime handle while retaining access to the native objects.

    ``raw_model`` and ``raw_tokenizer`` are intentional escape hatches for
    paper implementations that traverse a specific transformer's layers.
    Capability validation must happen before those objects are passed on.
    """

    raw_model: nn.Module
    spec: BackboneSpec
    raw_tokenizer: Any = None
    text: Optional[TextEncoder] = None
    tile: Optional[TileEncoder] = None
    slide: Optional[SlideProjector] = None
    preprocess: Optional[Callable[..., Any]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # Friendly aliases for callers migrating from the legacy tuple API.
    @property
    def model(self) -> nn.Module:
        """Return the native model (migration alias for ``raw_model``)."""
        return self.raw_model

    @property
    def tokenizer(self) -> Any:
        """Return the native tokenizer (migration alias for ``raw_tokenizer``)."""
        return self.raw_tokenizer

    def require(self, *capabilities: BackboneCapability,
                consumer: Optional[str] = None) -> "EncoderBundle":
        """Assert that this bundle advertises the requested operations.

        Args:
            *capabilities: Operations required by the caller.
            consumer: Optional name included in validation errors.

        Returns:
            This bundle, enabling fluent validation.

        Raises:
            BackboneCompatibilityError: If any capability is absent.
        """
        missing = [item.value for item in capabilities
                   if item not in self.spec.capabilities]
        if missing:
            prefix = f"{consumer} requires" if consumer else "Encoder requires"
            raise BackboneCompatibilityError(
                f"{prefix} {missing}, but backbone '{self.spec.name}' provides "
                f"{sorted(item.value for item in self.spec.capabilities)}.")
        return self

    def freeze(self) -> "EncoderBundle":
        """Put the native model in evaluation mode and disable gradients."""
        self.raw_model.eval()
        self.raw_model.requires_grad_(False)
        return self

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        """Encode text using the bundle's validated text wrapper.

        Raises:
            BackboneCompatibilityError: If text encoding is unavailable or
                the spec declares it without providing a wrapper.
        """
        self.require(BackboneCapability.TEXT_ENCODE)
        if self.text is None:
            raise BackboneCompatibilityError(
                f"Backbone '{self.spec.name}' declares text encoding but has no text wrapper.")
        return self.text.encode_text(texts_or_tokens, normalize=normalize)

    def project_slide(self, raw_embeddings: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
        """Apply the bundle's native slide projector.

        Raises:
            BackboneCompatibilityError: If native slide projection is absent.
        """
        self.require(BackboneCapability.SLIDE_PROJECT)
        if self.slide is None:
            raise BackboneCompatibilityError(
                f"Backbone '{self.spec.name}' declares slide projection but has no projector.")
        return self.slide.project_slide(raw_embeddings, normalize=normalize)

    def assert_feature_space(self, *, feature_space_id: Optional[str] = None,
                             dimension: Optional[int] = None,
                             level: FeatureLevel = FeatureLevel.PATCH_BAG) -> None:
        """Validate cached feature provenance and width against the spec.

        Args:
            feature_space_id: Exact producer/checkpoint identity, when known.
            dimension: Last dimension of the cached feature tensor.
            level: Tensor level used to select tile versus slide dimensions.

        Raises:
            BackboneCompatibilityError: If provenance or width differs.
        """
        if feature_space_id and feature_space_id != self.spec.feature_space_id:
            raise BackboneCompatibilityError(
                f"Feature space '{feature_space_id}' does not match backbone "
                f"'{self.spec.name}' ({self.spec.feature_space_id}).")
        native_dim = (self.spec.slide_input_dim
                      if level == FeatureLevel.SLIDE_EMBEDDING
                      else self.spec.tile_dim)
        if dimension is not None and native_dim is not None and dimension != native_dim:
            raise BackboneCompatibilityError(
                f"{level.value} width {dimension} does not match backbone "
                f"'{self.spec.name}' width {native_dim}.")


@dataclass(frozen=True)
class MethodBackboneContract:
    """Declare the encoder and feature boundary accepted by a method.

    ``swap_policy`` distinguishes architectural compatibility from mere tensor
    shape compatibility. Allowlisted, fixed, and precomputed methods must name
    a supported native family; capability-based methods validate operations on
    the returned bundle.
    """

    method: str
    feature_level: FeatureLevel
    swap_policy: SwapPolicy
    required_capabilities: FrozenSet[BackboneCapability] = frozenset()
    config_key: Optional[str] = "backbone"
    default_backbone: Optional[str] = None
    supported_backbones: Tuple[str, ...] = ()
    name_aliases: Mapping[str, str] = field(default_factory=dict)
    feature_dims: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    feature_dim_key: Optional[str] = "feature_dim"
    feature_space_key: Optional[str] = "feature_space_id"
    enforce_native_dimension: bool = False
    require_feature_space: bool = False
    rationale: str = ""

    def resolve_name(self, cfg: Mapping[str, Any]) -> Optional[str]:
        """Resolve and canonicalize the backbone selected by a run config."""
        if self.config_key is None:
            return self.default_backbone
        value = cfg.get(self.config_key, self.default_backbone)
        if value is None:
            return None
        raw = str(value).strip().lower().replace("_", "-")
        return canonical_backbone_name(self.name_aliases.get(raw, raw))

    def validate_config(self, cfg: Mapping[str, Any]) -> Optional[str]:
        """Validate config-only constraints before allocating model weights.

        Returns:
            The canonical selected backbone, or ``None`` for a contract with
            no runtime encoder name.

        Raises:
            BackboneCompatibilityError: If the name, width, provenance, or
                required config fields violate this contract.
        """
        selected = self.resolve_name(cfg)
        allowed = tuple(canonical_backbone_name(item)
                        for item in self.supported_backbones)
        if self.swap_policy in {SwapPolicy.ALLOWLIST, SwapPolicy.FIXED,
                                SwapPolicy.PRECOMPUTED}:
            if selected is None or selected not in allowed:
                raise BackboneCompatibilityError(
                    f"Method '{self.method}' cannot use backbone {selected!r}. "
                    f"Supported: {list(allowed)}. {self.rationale}".strip())
        if (self.require_feature_space and self.feature_space_key and
                not cfg.get(self.feature_space_key)):
            raise BackboneCompatibilityError(
                f"Method '{self.method}' requires '{self.feature_space_key}' "
                "to identify the offline encoder checkpoint exactly.")
        if (self.enforce_native_dimension and self.feature_dim_key and
                self.feature_dim_key not in cfg):
            raise BackboneCompatibilityError(
                f"Method '{self.method}' requires '{self.feature_dim_key}' "
                f"for its {self.feature_level.value} input.")
        if self.feature_dim_key and self.feature_dim_key in cfg and selected is not None:
            expected = self.feature_dims.get(selected)
            actual = int(cfg[self.feature_dim_key])
            if expected and actual not in expected:
                raise BackboneCompatibilityError(
                    f"Method '{self.method}' expects {selected} "
                    f"{self.feature_level.value} width in {list(expected)}, got {actual}.")
        return selected

    def validate_bundle(self, cfg: Mapping[str, Any],
                        bundle: EncoderBundle) -> EncoderBundle:
        """Validate a loaded bundle against this contract and run config.

        Returns:
            The validated bundle.

        Raises:
            BackboneCompatibilityError: If capabilities, identity, dimensions,
                or feature provenance are incompatible.
        """
        selected = self.validate_config(cfg)
        if self.swap_policy != SwapPolicy.CAPABILITY and selected != bundle.spec.name:
            raise BackboneCompatibilityError(
                f"Method '{self.method}' selected '{selected}', but loader returned "
                f"'{bundle.spec.name}'.")
        bundle.require(*self.required_capabilities, consumer=f"Method '{self.method}'")
        feature_space = (cfg.get(self.feature_space_key)
                         if self.feature_space_key else None)
        if self.require_feature_space and not feature_space:
            raise BackboneCompatibilityError(
                f"Method '{self.method}' requires '{self.feature_space_key}' "
                "to identify the offline encoder checkpoint exactly.")
        dimension = (int(cfg[self.feature_dim_key])
                     if self.feature_dim_key and self.feature_dim_key in cfg else None)
        # A learned input adapter is part of some native methods (e.g. MUSE),
        # so only enforce native width when this contract declares exact dims.
        if selected in self.feature_dims or self.enforce_native_dimension:
            bundle.assert_feature_space(feature_space_id=feature_space,
                                        dimension=dimension,
                                        level=self.feature_level)
        elif feature_space:
            bundle.assert_feature_space(feature_space_id=feature_space,
                                        level=self.feature_level)
        return bundle

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the contract to JSON-compatible registry metadata."""
        return {
            "method": self.method,
            "feature_level": self.feature_level.value,
            "swap_policy": self.swap_policy.value,
            "config_key": self.config_key,
            "default_backbone": self.default_backbone,
            "supported_backbones": list(self.supported_backbones),
            "name_aliases": dict(self.name_aliases),
            "required_capabilities": sorted(item.value for item in self.required_capabilities),
            "feature_dims": {key: list(value) for key, value in self.feature_dims.items()},
            "enforce_native_dimension": self.enforce_native_dimension,
            "require_feature_space": self.require_feature_space,
            "rationale": self.rationale,
        }


_ALIASES = {
    "clip": "clip-vitb",
    "clip-vit-b": "clip-vitb",
    "clip-vit-b-16": "clip-vitb",
    "vit-b/16": "clip-vitb",
    "rn50": "clip-rn50",
    "clip-rn-50": "clip-rn50",
    "clip-rn50": "clip-rn50",
    "biomedclip": "biomedclip",
    "mahmoodlab/titan": "titan",
    "titan": "titan",
}


def canonical_backbone_name(name: Any) -> str:
    """Normalize config spellings without erasing meaningful model families.

    Args:
        name: User- or config-provided encoder name.

    Returns:
        The canonical registry key when an alias is known.
    """
    value = str(name).strip().lower().replace("_", "-")
    return _ALIASES.get(value, value)


def normalize_features(features: torch.Tensor, normalize: bool) -> torch.Tensor:
    """Convert features to float and optionally L2-normalize the last axis."""
    return F.normalize(features.float(), dim=-1) if normalize else features
