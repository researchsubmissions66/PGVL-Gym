"""Backbone registry and vendor adapters.

``build_backbone`` retains the historical ``(model, tokenizer, info)``
return value.  New code should use :func:`build_encoder`, whose
:class:`~common.backbones.interfaces.EncoderBundle` exposes explicit
capabilities and feature-space provenance.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import (
    BackboneCapability as Cap,
    BackboneCompatibilityError,
    BackboneSpec,
    EncoderBundle,
    TokenBatch,
    canonical_backbone_name,
)


@dataclass(frozen=True)
class BackboneInfo:
    """Backward-compatible metadata returned by ``build_backbone``.

    ``patch_dim`` and ``text_dim`` are the projected comparison-space
    dimensions.  Use :class:`BackboneSpec` when token widths or raw slide
    dimensions matter.
    """

    name: str
    patch_dim: int
    text_dim: int
    image_size: int = 224
    is_clip_compatible: bool = True


_TEXT_TILE = frozenset({Cap.TEXT_ENCODE, Cap.SOFT_PROMPT, Cap.TILE_ENCODE,
                        Cap.PAIRED_TILE_TEXT})
_DEEP_PROMPT_TILE = frozenset(set(_TEXT_TILE) |
                              {Cap.DEEP_TEXT_PROMPT, Cap.DEEP_VISION_PROMPT})
_NATIVE_PATHPT_TILE = frozenset({Cap.SOFT_PROMPT, Cap.PAIRED_TILE_TEXT})
# KEEP and MUSK are dual-tower models whose text side is reachable through a
# wrapper (_NativeText and _MuskText respectively), so they additionally
# provide black-box text encoding. The bundle above is named for what PathPT
# requires of them, not for the limit of what they can do.
_NATIVE_TILE_WITH_TEXT = frozenset(set(_NATIVE_PATHPT_TILE) | {Cap.TEXT_ENCODE})

_SPECS: Dict[str, BackboneSpec] = {
    "clip-rn50": BackboneSpec(
        name="clip-rn50", family="openai_clip", revision="official",
        feature_space_id="openai/clip-rn50@official", capabilities=_TEXT_TILE,
        tile_dim=1024, text_token_dim=512, shared_dim=1024,
        context_length=77, image_size=224, aliases=("RN50", "CLIP-RN50")),
    "clip-vitb": BackboneSpec(
        name="clip-vitb", family="openai_clip", revision="official",
        feature_space_id="openai/clip-vit-b-16@official", capabilities=_TEXT_TILE,
        tile_dim=512, vision_token_dim=768, text_token_dim=512, shared_dim=512,
        context_length=77, image_size=224, aliases=("CLIP", "ViT-B/16")),
    "plip": BackboneSpec(
        name="plip", family="hf_clip", revision=None,
        feature_space_id="hf:vinid/plip", capabilities=_DEEP_PROMPT_TILE,
        tile_dim=512, vision_token_dim=768, text_token_dim=512, shared_dim=512,
        context_length=77, image_size=224, aliases=("PLIP",)),
    "hf-clip-vitb": BackboneSpec(
        name="hf-clip-vitb", family="hf_clip", revision=None,
        feature_space_id="hf:openai/clip-vit-base-patch16",
        capabilities=_DEEP_PROMPT_TILE,
        tile_dim=512, vision_token_dim=768, text_token_dim=512, shared_dim=512,
        context_length=77, image_size=224, aliases=("HF-CLIP",)),
    "conch": BackboneSpec(
        name="conch", family="conch", revision=None,
        feature_space_id="hf:MahmoodLab/conch", capabilities=_DEEP_PROMPT_TILE,
        tile_dim=512, vision_token_dim=768, text_token_dim=768, shared_dim=512,
        context_length=128, image_size=448, aliases=("CONCH",)),
    "musk": BackboneSpec(
        name="musk", family="musk", revision=None,
        feature_space_id="hf:xiangjx/musk", capabilities=_NATIVE_TILE_WITH_TEXT,
        tile_dim=1024, vision_token_dim=1024, text_token_dim=768, shared_dim=1024,
        context_length=100, image_size=384, aliases=("MUSK",)),
    "keep": BackboneSpec(
        name="keep", family="keep", revision=None,
        feature_space_id="hf:Astaxanthin/KEEP", capabilities=_NATIVE_TILE_WITH_TEXT,
        tile_dim=768, vision_token_dim=768, text_token_dim=768, shared_dim=768,
        context_length=77, image_size=224, aliases=("KEEP",)),
    "biomedclip": BackboneSpec(
        name="biomedclip", family="open_clip", revision=None,
        feature_space_id=("hf:microsoft/"
                          "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
        capabilities=_TEXT_TILE, tile_dim=512, vision_token_dim=768,
        text_token_dim=768, shared_dim=512, context_length=256,
        image_size=224, aliases=("BiomedCLIP",)),
    "quiltnet": BackboneSpec(
        name="quiltnet", family="open_clip", revision=None,
        feature_space_id="hf:wisdomik/QuiltNet-B-32", capabilities=_TEXT_TILE,
        tile_dim=512, vision_token_dim=768, text_token_dim=512, shared_dim=512,
        context_length=77, image_size=224,
        aliases=("QuiltNet", "QuiltNet-B-32", "wisdomik/QuiltNet-B-32")),
    "titan": BackboneSpec(
        name="titan", family="titan", revision=None,
        feature_space_id="hf:MahmoodLab/TITAN",
        capabilities=frozenset({Cap.TEXT_ENCODE, Cap.SOFT_PROMPT,
                                Cap.SLIDE_PROJECT, Cap.PAIRED_SLIDE_TEXT}),
        slide_input_dim=768, text_token_dim=768, shared_dim=768,
        context_length=128, aliases=("MahmoodLab/TITAN", "TITAN")),
}


def _legacy_info(spec: BackboneSpec) -> BackboneInfo:
    projected = spec.shared_dim or spec.tile_dim or spec.slide_input_dim
    if projected is None:
        raise BackboneCompatibilityError(
            f"Backbone '{spec.name}' has no projected embedding dimension.")
    return BackboneInfo(
        name=spec.name,
        patch_dim=int(spec.tile_dim or spec.slide_input_dim or projected),
        text_dim=int(projected),
        image_size=int(spec.image_size or 224),
        is_clip_compatible=spec.family == "openai_clip",
    )


def _cached_hf_file(repo_id: str, filename: str) -> Optional[str]:
    """Return an already-cached Hugging Face file without network access."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id, filename=filename, local_files_only=True)
    except Exception:
        return None


@contextmanager
def _hf_offline(enabled: bool):
    """Make ``local_files_only`` reach tokenizer loads inside remote code."""
    if not enabled:
        yield
        return
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    # huggingface_hub snapshots these environment variables when its constants
    # module is imported.  The factory itself may already have imported
    # transformers, so changing os.environ alone is not sufficient here.
    import huggingface_hub.constants as hub_constants
    previous_hub_offline = hub_constants.HF_HUB_OFFLINE
    os.environ.update({key: "1" for key in keys})
    hub_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hub_constants.HF_HUB_OFFLINE = previous_hub_offline
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def list_backbones() -> list[str]:
    """Return canonical encoder names in registry order."""
    return list(_SPECS)


def get_spec(name: str) -> BackboneSpec:
    """Return immutable capability and provenance metadata for an encoder.

    Args:
        name: Canonical encoder name or registered alias.

    Raises:
        KeyError: If no matching encoder is registered.
    """
    canonical = canonical_backbone_name(name)
    if canonical not in _SPECS:
        raise KeyError(f"Unknown backbone '{name}'. Available: {list_backbones()}")
    return _SPECS[canonical]


def get_info(name: str) -> BackboneInfo:
    """Return legacy projected dimensions for a registered encoder."""
    return _legacy_info(get_spec(name))


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _token_batch(output: Any, *, eot_by_argmax: bool = False) -> TokenBatch:
    if isinstance(output, TokenBatch):
        return output
    if torch.is_tensor(output):
        ids = output
        mask = None
        extras: Mapping[str, torch.Tensor] = {}
    elif isinstance(output, Mapping):
        ids = output["input_ids"]
        mask = output.get("attention_mask")
        extras = {key: value for key, value in output.items()
                  if key not in {"input_ids", "attention_mask"}
                  and torch.is_tensor(value)}
    else:
        raise TypeError(f"Unsupported tokenizer output: {type(output)!r}")
    if eot_by_argmax:
        eot = ids.argmax(dim=-1)
    elif mask is not None:
        eot = mask.long().sum(dim=-1).sub(1).clamp_min(0)
    else:
        eot = ids.ne(0).long().sum(dim=-1).sub(1).clamp_min(0)
    return TokenBatch(ids, mask, eot, extras)


class _OpenAIClipText:
    def __init__(self, model: nn.Module, tokenizer: Callable[..., torch.Tensor]):
        self.model, self.tokenizer = model, tokenizer

    def tokenize(self, texts: Sequence[str]) -> TokenBatch:
        return _token_batch(self.tokenizer(list(texts)), eot_by_argmax=True)

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        tokens = (self.tokenize(texts_or_tokens) if isinstance(texts_or_tokens, (list, tuple))
                  else _token_batch(texts_or_tokens, eot_by_argmax=True))
        result = self.model.encode_text(tokens.input_ids.to(_module_device(self.model)))
        return F.normalize(result.float(), dim=-1) if normalize else result


class _HFClipText:
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model, self.tokenizer = model, tokenizer

    def tokenize(self, texts: Sequence[str]) -> TokenBatch:
        return _token_batch(self.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"))

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        tokens = (self.tokenize(texts_or_tokens) if isinstance(texts_or_tokens, (list, tuple))
                  else _token_batch(texts_or_tokens))
        kwargs = tokens.to(_module_device(self.model)).as_kwargs()
        result = self.model.get_text_features(**kwargs)
        return F.normalize(result.float(), dim=-1) if normalize else result


class _ConchText:
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model, self.tokenizer = model, tokenizer

    def tokenize(self, texts: Sequence[str]) -> TokenBatch:
        from conch.open_clip_custom import tokenize
        ids = tokenize(self.tokenizer, list(texts))
        # CONCH reserves the last position as a placeholder for its appended
        # CLS embedding.  EOT is therefore the final non-padding input token.
        pad = getattr(self.tokenizer, "pad_token_id", 0) or 0
        mask = ids.ne(int(pad)).long()
        eot = mask.sum(dim=-1).sub(1).clamp_min(0)
        return TokenBatch(ids, mask, eot)

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        tokens = (self.tokenize(texts_or_tokens) if isinstance(texts_or_tokens, (list, tuple))
                  else _token_batch(texts_or_tokens))
        result = self.model.encode_text(
            tokens.input_ids.to(_module_device(self.model)), normalize=normalize)
        return result.float()


class _NativeText:
    """Conservative wrapper for KEEP/MUSK-style native ``encode_text`` APIs."""

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model, self.tokenizer = model, tokenizer

    def tokenize(self, texts: Sequence[str]) -> TokenBatch:
        try:
            output = self.tokenizer(
                list(texts), padding=True, truncation=True, return_tensors="pt")
        except TypeError:
            output = self.tokenizer(list(texts))
        return _token_batch(output)

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        tokens = (self.tokenize(texts_or_tokens) if isinstance(texts_or_tokens, (list, tuple))
                  else _token_batch(texts_or_tokens))
        device_tokens = tokens.to(_module_device(self.model))
        if hasattr(self.model, "encode_text"):
            # Native APIs differ in how they take tokens: CLIP-style models want a
            # bare id tensor, some accept the token fields as keywords, and KEEP
            # takes the whole tokenizer mapping as one positional argument.
            try:
                result = self.model.encode_text(device_tokens.input_ids)
            except TypeError:
                try:
                    result = self.model.encode_text(**device_tokens.as_kwargs())
                except TypeError:
                    result = self.model.encode_text(device_tokens.as_kwargs())
        elif hasattr(self.model, "get_text_features"):
            result = self.model.get_text_features(**device_tokens.as_kwargs())
        else:
            raise BackboneCompatibilityError(
                f"{type(self.model).__name__} has no black-box text encoding API; "
                "use its method-specific soft-prompt adapter.")
        return F.normalize(result.float(), dim=-1) if normalize else result


class _MuskText:
    """Text wrapper for MUSK's unified vision-language forward.

    MUSK exposes neither ``encode_text`` nor ``get_text_features``. Its
    ``ModelWrapper.forward`` takes both modalities and returns
    ``(vision_cls, language_cls)``, so the text tower is reached by passing
    ``text_description`` alone. ``with_head=True`` applies ``language_head``,
    which is what projects the 768-wide text tokens into the 1024-d space the
    vision tower shares; without it the two towers would not be comparable.

    Tokenisation follows the released ``musk.utils.xlm_tokenizer``: strip the
    tokenizer's own BOS/EOS, re-add MUSK's, pad to ``context_length``, and carry
    a padding mask where 1 marks padding.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, max_len: int = 100):
        self.model, self.tokenizer, self.max_len = model, tokenizer, max_len

    def _tokenize_one(self, text: str) -> tuple[list[int], list[int]]:
        from musk import utils as musk_utils
        return musk_utils.xlm_tokenizer(text, self.tokenizer, max_len=self.max_len)

    def tokenize(self, texts: Sequence[str]) -> TokenBatch:
        pairs = [self._tokenize_one(str(text)) for text in texts]
        ids = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        padding = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        # TokenBatch.attention_mask marks real tokens; MUSK's padding_mask is
        # the complement, so keep both and convert at the call site.
        return TokenBatch(ids, (1 - padding), ids.ne(0).long().sum(-1).sub(1).clamp_min(0),
                          {"padding_mask": padding})

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        if isinstance(texts_or_tokens, (list, tuple)):
            tokens = self.tokenize(texts_or_tokens)
        else:
            tokens = _token_batch(texts_or_tokens)
        device = _module_device(self.model)
        ids = tokens.input_ids.to(device)
        padding_mask = tokens.extras.get("padding_mask")
        if padding_mask is None:
            padding_mask = (1 - tokens.attention_mask)
        _, language_cls = self.model(
            image=None, text_description=ids,
            padding_mask=padding_mask.to(device),
            return_global=True, with_head=True, out_norm=bool(normalize))
        if language_cls is None:
            raise BackboneCompatibilityError(
                "MUSK returned no language embedding for the given text.")
        return language_cls.float()


class _NativeTile:
    def __init__(self, model: nn.Module):
        self.model = model

    def encode_tiles(self, images: torch.Tensor,
                     normalize: bool = True) -> torch.Tensor:
        images = images.to(_module_device(self.model))
        if hasattr(self.model, "encode_image"):
            try:
                result = self.model.encode_image(images, normalize=normalize)
            except TypeError:
                result = self.model.encode_image(images)
        elif hasattr(self.model, "get_image_features"):
            result = self.model.get_image_features(pixel_values=images)
        else:
            raise BackboneCompatibilityError(
                f"{type(self.model).__name__} has no tile encoding API.")
        return F.normalize(result.float(), dim=-1) if normalize else result


def _titan_tokenize(tokenizer: Any, texts: Sequence[str]) -> TokenBatch:
    """Preserve TITAN's 127-token plus reserved-CLS-placeholder layout."""
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        context_length = int(getattr(tokenizer, "context_length", 128))
        output = inner(
            list(texts), max_length=context_length - 1, add_special_tokens=True,
            return_token_type_ids=False, truncation=True, padding="max_length",
            return_tensors="pt")
        pad = getattr(inner, "pad_token_id", 0) or 0
        ids = F.pad(output["input_ids"], (0, 1), value=int(pad))
        mask = F.pad(output.get("attention_mask", output["input_ids"].ne(pad).long()),
                     (0, 1), value=0)
    else:
        output = tokenizer(list(texts))
        batch = _token_batch(output)
        ids, mask = batch.input_ids, batch.attention_mask
        if mask is None:
            mask = ids.ne(0).long()
    # The tokenizer mask identifies EOS as its last non-padding position.
    eot = mask.sum(dim=-1).sub(1).clamp_min(0)
    return TokenBatch(ids, mask, eot)


class TitanPromptableText:
    """Expose TITAN's text tower through the promptable-text protocol.

    The wrapper preserves TITAN's native tokenizer, end-of-text pooling, token
    width, dtype, and paired projection. SLDPC uses it to optimize context
    embeddings without replacing the native text tower.
    """

    def __init__(self, titan: nn.Module):
        self.model = titan
        self._text = titan.text_encoder

    @property
    def tokenizer(self) -> Any:
        return self._text.tokenizer

    @property
    def token_width(self) -> int:
        return int(self._text.ln_final.weight.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        return self._text.ln_final.weight.dtype

    @property
    def device(self) -> torch.device:
        return _module_device(self._text)

    def tokenize(self, texts: Sequence[str]) -> TokenBatch:
        return _titan_tokenize(self.tokenizer, texts)

    def embed_tokens(self, tokens: TokenBatch) -> torch.Tensor:
        return self._text.token_embedding(tokens.input_ids.to(self.device)).to(self.dtype)

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        tokens = (self.tokenize(texts_or_tokens) if isinstance(texts_or_tokens, (list, tuple))
                  else _token_batch(texts_or_tokens))
        return self.model.encode_text(tokens.input_ids.to(self.device), normalize=normalize).float()

    def encode_embedded(self, embeddings: torch.Tensor, tokens: TokenBatch,
                        normalize: bool = True) -> torch.Tensor:
        embeddings = embeddings.to(device=self.device, dtype=self.dtype)
        position = self._text.positional_embedding[:embeddings.shape[1]].to(
            device=self.device, dtype=self.dtype)
        encoded = self._text.transformer(embeddings + position.unsqueeze(0))
        encoded = self._text.ln_final(encoded)
        indices = tokens.eot_indices
        if indices is None:
            indices = tokens.input_ids.argmax(dim=-1)
        indices = indices.to(self.device)
        pooled = encoded[torch.arange(encoded.shape[0], device=self.device), indices]
        projection = self._text.text_projection
        pooled = projection(pooled) if isinstance(projection, nn.Linear) else pooled @ projection
        return F.normalize(pooled.float(), dim=-1) if normalize else pooled


class _TitanSlide:
    def __init__(self, titan: nn.Module):
        self.model = titan

    def project_slide(self, raw_embeddings: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
        projection = self.model.vision_encoder.proj
        raw_embeddings = raw_embeddings.to(
            device=projection.device, dtype=projection.dtype)
        result = raw_embeddings @ projection
        return F.normalize(result.float(), dim=-1) if normalize else result


EncoderBuilder = Callable[..., EncoderBundle]
_CUSTOM_BUILDERS: Dict[str, EncoderBuilder] = {}


def register_backbone(spec: BackboneSpec, builder: EncoderBuilder,
                      *, overwrite: bool = False) -> None:
    """Register an encoder bundle without modifying any method architecture.

    A builder receives the same keyword arguments as :func:`build_encoder`
    and must return a bundle whose ``spec.name`` matches ``spec.name``.

    Args:
        spec: Immutable metadata for the new encoder.
        builder: Callable returning a compatible :class:`EncoderBundle`.
        overwrite: Whether to replace an existing registration deliberately.

    Raises:
        KeyError: If the name already exists and ``overwrite`` is false.
        ValueError: If ``spec.name`` is not already canonical.
    """
    name = canonical_backbone_name(spec.name)
    if name in _SPECS and not overwrite:
        raise KeyError(f"Backbone '{name}' is already registered")
    if name != spec.name:
        raise ValueError("BackboneSpec.name must already be canonical")
    _SPECS[name] = spec
    _CUSTOM_BUILDERS[name] = builder


def unregister_backbone(name: str) -> None:
    """Remove a process-local custom registration.

    Raises:
        KeyError: If ``name`` is built in or is not currently registered as a
            custom encoder.
    """
    canonical = canonical_backbone_name(name)
    if canonical not in _CUSTOM_BUILDERS:
        raise KeyError(f"'{name}' is not a custom backbone")
    _CUSTOM_BUILDERS.pop(canonical)
    _SPECS.pop(canonical)


def _load_titan(model_id: str, device: str, *, revision: Optional[str],
                local_files_only: bool) -> EncoderBundle:
    try:
        import transformers
        from transformers import AutoModel, PreTrainedModel
    except ImportError as error:
        raise ImportError("TITAN requires transformers") from error
    # Cached TITAN remote code predates Transformers 5 and has no tied weights.
    # Keep the compatibility attribute scoped to this load so constructing
    # TITAN cannot change how unrelated Transformers models behave later in
    # the same training process.
    needs_tied_weight_compat = int(transformers.__version__.split(".", 1)[0]) >= 5
    missing = object()
    previous_tied_keys = getattr(
        PreTrainedModel, "all_tied_weights_keys", missing)
    if needs_tied_weight_compat:
        PreTrainedModel.all_tied_weights_keys = {}
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if revision:
        kwargs["revision"] = revision
    # TITAN's remote constructor instantiates its tokenizer with a second
    # from_pretrained() call that does not forward local_files_only. Propagate
    # offline intent for the duration of construction so a complete cache is
    # genuinely sufficient and no metadata request is attempted.
    try:
        with _hf_offline(local_files_only):
            model = AutoModel.from_pretrained(model_id, **kwargs).to(device)
    finally:
        if needs_tied_weight_compat:
            if previous_tied_keys is missing:
                delattr(PreTrainedModel, "all_tied_weights_keys")
            else:
                PreTrainedModel.all_tied_weights_keys = previous_tied_keys
    resolved_revision = revision or getattr(model.config, "_commit_hash", None)
    base = _SPECS["titan"]
    spec = BackboneSpec(**{
        **base.__dict__,
        "revision": resolved_revision,
        "feature_space_id": (f"hf:{model_id}" +
                             (f"@{resolved_revision}"
                              if resolved_revision else "")),
    })
    text = TitanPromptableText(model)
    return EncoderBundle(model, spec, raw_tokenizer=text.tokenizer,
                         text=text, slide=_TitanSlide(model))


def _load_builtin(name: str, weights_path: Optional[str], device: str
                  ) -> Tuple[nn.Module, Any, Any]:
    preprocess = None
    if name == "clip-rn50":
        from clip import clip
        model, _, preprocess = clip.load(
            weights_path or "RN50", device=device, jit=False)
        return model, clip.tokenize, preprocess
    if name == "clip-vitb":
        from clip import clip
        model, _, preprocess = clip.load(
            weights_path or "ViT-B/16", device=device, jit=False)
        return model, clip.tokenize, preprocess
    if name in {"plip", "hf-clip-vitb"}:
        from transformers import AutoTokenizer, CLIPModel
        # transformers refuses to `torch.load` a .bin checkpoint below torch 2.6
        # (CVE-2025-32434), and vinid/plip's default revision ships only
        # pytorch_model.bin. Point these at a directory holding the same weights
        # as model.safetensors -- transformers prefers safetensors and that path
        # never calls torch.load -- rather than pinning a revision, because the
        # safetensors-only revision carries no config or tokenizer files.
        env_key = "PLIP_CKPT" if name == "plip" else "HF_CLIP_CKPT"
        repo = (weights_path or os.environ.get(env_key)
                or ("vinid/plip" if name == "plip"
                    else "openai/clip-vit-base-patch16"))
        return (CLIPModel.from_pretrained(repo).to(device),
                AutoTokenizer.from_pretrained(repo, use_fast=True), None)
    if name == "conch":
        try:
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        except ImportError as error:
            raise ImportError(
                "CONCH requires the Mahmood-Lab/CONCH package and access to "
                "MahmoodLab/conch.") from error
        checkpoint = weights_path or os.environ.get("CONCH_CKPT")
        if not checkpoint:
            checkpoint = (_cached_hf_file("MahmoodLab/conch", "pytorch_model.bin")
                          or "hf_hub:MahmoodLab/conch")
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint)
        return model.to(device), get_tokenizer(), preprocess
    if name == "musk":
        try:
            from musk import modeling as musk_modeling  # noqa: F401
            from musk import utils as musk_utils
            from timm.models import create_model
            from transformers import XLMRobertaTokenizer
        except ImportError as error:
            raise ImportError(
                "MUSK requires the MUSK package from the PathPT distribution.") from error
        checkpoint = weights_path or os.environ.get("MUSK_CKPT")
        if not checkpoint:
            cached = os.path.join(os.path.expanduser("~"), ".cache", "model.safetensors")
            if os.path.isfile(cached):
                checkpoint = cached
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                "MUSK requires its model.safetensors checkpoint. Set "
                "backbone_weights or MUSK_CKPT to the local file.")
        model = create_model("musk_large_patch16_384")
        musk_utils.load_model_and_may_interpolate(
            checkpoint, model, "model|module", "")
        # The released pip package does not always ship tokenizer.spm, but the
        # published Hugging Face snapshot does, so look beside the checkpoint
        # before giving up.
        candidates = [
            os.path.join(os.path.dirname(musk_modeling.__file__), "models", "tokenizer.spm"),
            os.path.join(os.path.dirname(checkpoint), "tokenizer.spm"),
            os.environ.get("MUSK_TOKENIZER", ""),
        ]
        tokenizer_path = next(
            (c for c in candidates if c and os.path.isfile(c)), candidates[0])
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(
                f"MUSK tokenizer model is missing: {tokenizer_path}")
        tokenizer = XLMRobertaTokenizer(tokenizer_path)
        return model.to(device), tokenizer, None
    if name == "keep":
        from transformers import AutoModel, AutoTokenizer
        repo = weights_path or "Astaxanthin/KEEP"
        model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        return model, tokenizer, None
    if name == "quiltnet":
        import open_clip

        # PGVL's ConVLM compatibility path can encode attribute text with
        # QuiltNet, so this is a text tower in its own right, not only a tile
        # encoder. The upstream attribute matrix itself is not released.
        repo = weights_path or "hf-hub:wisdomik/QuiltNet-B-32"
        model, _, preprocess = open_clip.create_model_and_transforms(repo)
        return model.to(device), open_clip.get_tokenizer(repo), preprocess

    if name == "biomedclip":
        try:
            import open_clip
        except ImportError as error:
            raise ImportError(
                "BiomedCLIP requires `pip install open-clip-torch`.") from error
        identifier = (weights_path or
                      "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model, _, preprocess = open_clip.create_model_and_transforms(identifier)
        tokenizer = open_clip.get_tokenizer(identifier)
        return model.to(device), tokenizer, preprocess
    raise KeyError(f"Backbone '{name}' has no built-in loader implementation")


def build_encoder(name: str, weights_path: Optional[str] = None,
                  device: str = "cuda", **kwargs: Any) -> EncoderBundle:
    """Load a capability-aware encoder bundle.

    Args:
        name: Canonical encoder name or alias.
        weights_path: Optional local checkpoint, snapshot, or model identifier.
        device: PyTorch device on which to construct the native model.
        **kwargs: Family-specific loader options. TITAN accepts ``model_id``,
            ``revision``, and ``local_files_only``.

    Returns:
        The native model, tokenizer, provenance spec, and supported wrappers.

    Raises:
        KeyError: If the encoder is unknown.
        TypeError: If loader options are unsupported for the selected family.
        BackboneCompatibilityError: If a custom builder returns the wrong spec.
    """
    canonical = canonical_backbone_name(name)
    spec = get_spec(canonical)
    if canonical in _CUSTOM_BUILDERS:
        bundle = _CUSTOM_BUILDERS[canonical](
            weights_path=weights_path, device=device, **kwargs)
        if bundle.spec.name != spec.name:
            raise BackboneCompatibilityError(
                f"Builder registered as '{spec.name}' returned '{bundle.spec.name}'.")
        return bundle
    if canonical == "titan":
        model_id = kwargs.pop(
            "model_id", weights_path or "MahmoodLab/TITAN")
        revision = kwargs.pop("revision", None)
        local_files_only = kwargs.pop("local_files_only", False)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected loader options for '{canonical}': {unknown}")
        return _load_titan(
            model_id, device, revision=revision,
            local_files_only=local_files_only)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected loader options for '{canonical}': {unknown}")
    model, tokenizer, preprocess = _load_builtin(canonical, weights_path, device)
    if spec.family == "openai_clip":
        text = _OpenAIClipText(model, tokenizer)
    elif spec.family == "hf_clip":
        text = _HFClipText(model, tokenizer)
    elif spec.family == "conch":
        text = _ConchText(model, tokenizer)
    elif spec.family == "musk":
        text = _MuskText(model, tokenizer, max_len=spec.context_length or 100)
    else:
        text = _NativeText(model, tokenizer)
    return EncoderBundle(model, spec, raw_tokenizer=tokenizer,
                         text=text, tile=_NativeTile(model), preprocess=preprocess)


# Explicit alias requested by callers that prefer the old terminology.
build_backbone_handle = build_encoder


def build_backbone(name: str, weights_path: Optional[str] = None,
                   device: str = "cuda") -> Tuple[nn.Module, Callable, BackboneInfo]:
    """Load the legacy ``(model, tokenizer, info)`` tuple.

    New framework integrations should call :func:`build_encoder`; this wrapper
    exists for vendored implementations that still consume native objects.
    """
    bundle = build_encoder(name, weights_path=weights_path, device=device)
    return bundle.raw_model, bundle.raw_tokenizer, _legacy_info(bundle.spec)
