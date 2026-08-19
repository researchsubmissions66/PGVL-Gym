"""Unit tests for the capability-aware backbone API.

These tests intentionally use tiny in-memory modules.  They must never load a
checkpoint or contact a model hub: the public interfaces should be testable
independently of any vendor implementation.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.backbones import (
    BackboneCapability as Cap,
    BackboneCompatibilityError,
    BackboneSpec,
    EncoderBundle,
    FeatureLevel,
    TokenBatch,
    build_encoder,
    canonical_backbone_name,
    get_spec,
    list_backbones,
    register_backbone,
    unregister_backbone,
)
from common.backbones.factory import _hf_offline


class _DummyNativeModel(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.projection = nn.Parameter(torch.eye(width))


class _DummyTextEncoder:
    def __init__(self, model: _DummyNativeModel):
        self.model = model

    def tokenize(self, texts: Sequence[str]) -> TokenBatch:
        lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
        ids = torch.stack((lengths, lengths + 1), dim=1)
        return TokenBatch(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            eot_indices=torch.ones(len(texts), dtype=torch.long),
        )

    def encode_text(self, texts_or_tokens: Any,
                    normalize: bool = True) -> torch.Tensor:
        tokens = (self.tokenize(texts_or_tokens)
                  if isinstance(texts_or_tokens, (list, tuple))
                  else texts_or_tokens)
        values = tokens.input_ids[:, :1].float().repeat(1, 4)
        return F.normalize(values, dim=-1) if normalize else values


class _DummyTileEncoder:
    def encode_tiles(self, images: torch.Tensor,
                     normalize: bool = True) -> torch.Tensor:
        output = images.flatten(1)[:, :4].float()
        return F.normalize(output, dim=-1) if normalize else output


class _DummySlideProjector:
    def project_slide(self, raw_embeddings: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
        output = raw_embeddings.float()[:, :4]
        return F.normalize(output, dim=-1) if normalize else output


@pytest.fixture
def registered_dummy_encoder():
    """Register a complete paired tile/text encoder for one test at a time."""
    spec = BackboneSpec(
        name="test-paired-encoder",
        family="test",
        feature_space_id="test:paired-v1",
        capabilities=frozenset({
            Cap.TEXT_ENCODE,
            Cap.SOFT_PROMPT,
            Cap.TILE_ENCODE,
            Cap.PAIRED_TILE_TEXT,
        }),
        tile_dim=4,
        text_token_dim=4,
        shared_dim=4,
        context_length=2,
        image_size=2,
    )
    calls = []

    def builder(*, weights_path=None, device="cpu", **options):
        calls.append({
            "weights_path": weights_path,
            "device": device,
            "options": options,
        })
        model = _DummyNativeModel().to(device)
        return EncoderBundle(
            raw_model=model,
            raw_tokenizer="dummy-tokenizer",
            spec=spec,
            text=_DummyTextEncoder(model),
            tile=_DummyTileEncoder(),
            metadata={"source": "unit-test"},
        )

    register_backbone(spec, builder)
    try:
        yield spec, calls
    finally:
        unregister_backbone(spec.name)


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("CLIP", "clip-vitb"),
        ("ViT-B/16", "clip-vitb"),
        ("RN50", "clip-rn50"),
        ("CLIP_RN50", "clip-rn50"),
        ("MahmoodLab/TITAN", "titan"),
    ],
)
def test_builtin_aliases_resolve_without_loading(spelling, canonical):
    assert canonical_backbone_name(spelling) == canonical
    assert get_spec(spelling).name == canonical


def test_titan_rejects_unknown_loader_options_before_loading():
    with pytest.raises(TypeError, match="Unexpected loader options"):
        build_encoder("titan", device="cpu", unsupported_option=True)


def test_custom_registration_builds_bundle_without_builtin_loader(
        registered_dummy_encoder):
    spec, calls = registered_dummy_encoder

    bundle = build_encoder(
        spec.name,
        weights_path="not-a-real-checkpoint",
        device="cpu",
        test_option=17,
    )

    assert bundle.spec is spec
    assert bundle.model is bundle.raw_model
    assert bundle.tokenizer == "dummy-tokenizer"
    assert bundle.metadata["source"] == "unit-test"
    assert calls == [{
        "weights_path": "not-a-real-checkpoint",
        "device": "cpu",
        "options": {"test_option": 17},
    }]
    assert spec.name in list_backbones()

    encoded = bundle.encode_text(["a", "abcd"])
    assert encoded.shape == (2, 4)
    torch.testing.assert_close(encoded.norm(dim=-1), torch.ones(2))


def test_custom_registration_rejects_duplicates(registered_dummy_encoder):
    spec, _ = registered_dummy_encoder

    with pytest.raises(KeyError, match="already registered"):
        register_backbone(
            spec,
            lambda **_: EncoderBundle(_DummyNativeModel(), spec),
        )


def test_bundle_capabilities_and_feature_provenance_are_enforced(
        registered_dummy_encoder):
    spec, _ = registered_dummy_encoder
    bundle = build_encoder(spec.name, device="cpu")

    assert bundle.require(Cap.TEXT_ENCODE, Cap.PAIRED_TILE_TEXT) is bundle
    with pytest.raises(BackboneCompatibilityError, match="slide_project"):
        bundle.require(Cap.SLIDE_PROJECT, consumer="test consumer")

    bundle.assert_feature_space(
        feature_space_id="test:paired-v1",
        dimension=4,
        level=FeatureLevel.PATCH_BAG,
    )
    with pytest.raises(BackboneCompatibilityError, match="Feature space"):
        bundle.assert_feature_space(feature_space_id="test:different-space")
    with pytest.raises(BackboneCompatibilityError, match="width 8"):
        bundle.assert_feature_space(dimension=8)


def test_bundle_freeze_is_explicit_and_chainable(registered_dummy_encoder):
    spec, _ = registered_dummy_encoder
    bundle = build_encoder(spec.name, device="cpu")

    assert bundle.raw_model.training
    assert bundle.freeze() is bundle
    assert not bundle.raw_model.training
    assert all(not parameter.requires_grad
               for parameter in bundle.raw_model.parameters())


def test_token_batch_moves_all_tensor_fields_and_exposes_model_kwargs():
    batch = TokenBatch(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.tensor([[1, 1]]),
        eot_indices=torch.tensor([1]),
        extras={"token_type_ids": torch.tensor([[0, 0]])},
    ).to("cpu")

    assert batch.eot_indices.tolist() == [1]
    assert set(batch.as_kwargs()) == {
        "input_ids", "attention_mask", "token_type_ids",
    }
    assert "eot_indices" not in batch.as_kwargs()


def test_slide_projector_delegation_and_missing_wrapper_error():
    slide_spec = BackboneSpec(
        name="test-slide-projector",
        family="test",
        feature_space_id="test:slide-v1",
        capabilities=frozenset({Cap.SLIDE_PROJECT}),
        slide_input_dim=8,
        shared_dim=4,
    )
    bundle = EncoderBundle(
        raw_model=_DummyNativeModel(),
        spec=slide_spec,
        slide=_DummySlideProjector(),
    )
    projected = bundle.project_slide(torch.arange(16).reshape(2, 8))
    assert projected.shape == (2, 4)
    torch.testing.assert_close(projected.norm(dim=-1), torch.ones(2))

    broken = EncoderBundle(raw_model=_DummyNativeModel(), spec=slide_spec)
    with pytest.raises(BackboneCompatibilityError, match="no projector"):
        broken.project_slide(torch.ones(1, 8))


def test_hf_offline_updates_imported_constant_and_restores_all_state(
        monkeypatch):
    """Remote-code tokenizer loads see offline mode, without leaking it."""
    import huggingface_hub.constants as hub_constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "preexisting-value")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)

    with pytest.raises(RuntimeError, match="exercise finally"):
        with _hf_offline(True):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
            assert hub_constants.HF_HUB_OFFLINE is True
            raise RuntimeError("exercise finally")

    assert os.environ["HF_HUB_OFFLINE"] == "preexisting-value"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert hub_constants.HF_HUB_OFFLINE is False
