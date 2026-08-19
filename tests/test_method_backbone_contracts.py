"""Contract tests for all method adapters.

The tests stop at config/bundle validation; none calls ``build_model`` or
loads a foundation-model checkpoint.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

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
    MethodBackboneContract,
    SwapPolicy,
    TokenBatch,
    get_spec,
)
from methods import get_method, list_methods
from methods.base import BaseMethod, probabilities_to_logits


def _bundle_for_builtin(name: str) -> EncoderBundle:
    """Create metadata-only bundle carrying a built-in spec."""
    return EncoderBundle(raw_model=nn.Identity(), spec=get_spec(name))


@pytest.mark.parametrize("method_name", list_methods())
def test_every_registered_adapter_declares_a_contract(method_name):
    adapter = get_method(method_name)
    contract = adapter.get_backbone_contract()

    assert issubclass(adapter, BaseMethod)
    assert isinstance(contract, MethodBackboneContract)
    assert contract.method == adapter.name
    assert isinstance(contract.feature_level, FeatureLevel)
    assert isinstance(contract.swap_policy, SwapPolicy)
    assert contract.rationale.strip()


@pytest.mark.parametrize(
    ("method_name", "cfg", "expected"),
    [
        ("top", {"clip_arch": "RN50"}, "clip-rn50"),
        ("maple", {"backbone": "CLIP"}, "hf-clip-vitb"),
        ("mscpt", {"backbone": "clip_vitb"}, "hf-clip-vitb"),
        ("slip", {"backbone": "CLIP-RN50"}, "clip-rn50"),
        ("sldpc", {
            "backbone": "MahmoodLab/TITAN", "feature_dim": 768,
            "feature_space_id": "hf:MahmoodLab/TITAN",
            "prompt_feature_space_id": "hf:MahmoodLab/TITAN",
        }, "titan"),
    ],
)
def test_method_specific_aliases_resolve(method_name, cfg, expected):
    contract = get_method(method_name).get_backbone_contract()
    assert contract.resolve_name(cfg) == expected
    assert contract.validate_config(cfg) == expected


@pytest.mark.parametrize(
    ("method_name", "cfg"),
    [
        ("top", {"clip_arch": "plip"}),
    ],
)
def test_fixed_architectures_reject_incompatible_swaps(method_name, cfg):
    contract = get_method(method_name).get_backbone_contract()
    assert contract.swap_policy is SwapPolicy.FIXED

    with pytest.raises(BackboneCompatibilityError, match="cannot use backbone"):
        contract.validate_config(cfg)


def test_precomputed_cod_mil_accepts_upstream_spaces_and_rejects_unaligned_backbone():
    contract = get_method("cod_mil").get_backbone_contract()
    assert contract.swap_policy is SwapPolicy.PRECOMPUTED
    assert contract.validate_config({
        "backbone": "plip", "feature_dim": 512,
        "feature_space_id": "hf:vinid/plip",
    }) == "plip"
    assert contract.validate_config({
        "backbone": "quiltnet", "feature_dim": 512,
        "feature_space_id": "hf:wisdomik/QuiltNet-B-32",
    }) == "quiltnet"

    with pytest.raises(BackboneCompatibilityError, match="cannot use backbone"):
        contract.validate_config({
            "backbone": "conch", "feature_dim": 512,
            "feature_space_id": "hf:MahmoodLab/conch",
        })


def test_convlm_is_precomputed_not_encoder_owning():
    """ConVLM trains on precomputed patch features, so it owns no vision tower.

    Upstream extracts them with UNI (`ConVLM_visual_feature_extraction.py`) and
    exposes the embedding width as a config value; the contract is therefore
    ``PRECOMPUTED``. Rejection of an unaligned backbone is unchanged.
    """
    contract = get_method("convlm").get_backbone_contract()
    assert contract.swap_policy is SwapPolicy.PRECOMPUTED
    assert contract.feature_level is FeatureLevel.PATCH_BAG

    with pytest.raises(BackboneCompatibilityError, match="cannot use backbone"):
        contract.validate_config({"backbone": "conch"})


def test_wsi_five_is_precomputed_not_encoder_owning():
    """WSI-FiVE reads a patch bag from disk; it owns no vision tower.

    Upstream sets ``self.visual = nn.Identity()`` under its shipped
    ``IS_IMG_PTH: True`` and the paper uses DSMIL features, so the contract is
    ``PRECOMPUTED`` rather than ``FIXED``. Rejection of an unaligned backbone is
    unchanged -- only the reason recorded for it.
    """
    contract = get_method("wsi_five").get_backbone_contract()
    assert contract.swap_policy is SwapPolicy.PRECOMPUTED

    with pytest.raises(BackboneCompatibilityError, match="cannot use backbone"):
        contract.validate_config({"backbone": "clip-vitb"})


def test_focus_declares_the_single_bag_it_actually_consumes():
    contract = get_method("focus").get_backbone_contract()

    assert contract.feature_level is FeatureLevel.PATCH_BAG


def test_probability_outputs_are_converted_without_a_second_softmax():
    probabilities = torch.tensor([[0.7, 0.2, 0.1]])

    logits = probabilities_to_logits(probabilities)

    torch.testing.assert_close(logits.softmax(dim=1), probabilities)


@pytest.mark.parametrize(
    ("method_name", "valid_cfg", "invalid_cfg", "expected_fragment"),
    [
        (
            "vila_mil",
            {"backbone": "RN50", "feature_dim": 1024},
            {"backbone": "RN50", "feature_dim": 512},
            "width in [1024], got 512",
        ),
        (
            "pathpt",
            {"backbone": "keep", "feature_dim": 768},
            {"backbone": "keep", "feature_dim": 512},
            "width in [768], got 512",
        ),
        (
            "slip",
            {"backbone": "clip-rn50", "feature_dim": 1024},
            {"backbone": "clip-rn50", "feature_dim": 768},
            "width in [1024], got 768",
        ),
    ],
)
def test_declared_feature_dimensions_are_checked_before_loading(
        method_name, valid_cfg, invalid_cfg, expected_fragment):
    contract = get_method(method_name).get_backbone_contract()

    assert contract.validate_config(valid_cfg)
    with pytest.raises(BackboneCompatibilityError, match=expected_fragment.replace("[", r"\[").replace("]", r"\]")):
        contract.validate_config(invalid_cfg)


def test_pathpt_aggregates_native_patch_probabilities_to_slide_logits():
    from methods.pathpt.adapter import PathPTMethod

    method = PathPTMethod({
        "backbone": "conch",
        "feature_dim": 512,
        "n_classes": 2,
    }, device="cpu")
    patch_probabilities = torch.tensor([
        [0.8, 0.2],
        [0.6, 0.4],
    ])

    logits = method._slide_logits((None, patch_probabilities))

    assert logits.shape == (1, 2)
    torch.testing.assert_close(logits.exp(), torch.tensor([[0.7, 0.3]]))


def test_pathpt_warmup_does_not_zero_the_first_epoch_learning_rate():
    from methods.pathpt.adapter import PathPTMethod

    method = PathPTMethod({
        "backbone": "conch", "feature_dim": 512,
        "n_classes": 2, "epochs": 20,
    }, device="cpu")
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)

    method.build_scheduler(optimizer)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_pathpt_plip_uses_pretrained_token_embeddings_with_random_context():
    from methods.pathpt.pathpt_models.PathPT_model_PLIP import PromptLearnerPLIP

    class TokenBatch(dict):
        def to(self, device):
            return TokenBatch({key: value.to(device)
                               for key, value in self.items()})

    class Tokenizer:
        def __call__(self, texts, **_kwargs):
            ids = torch.tensor([[1, 2, 3, 4, 0] for _ in texts])
            return TokenBatch({
                "input_ids": ids,
                "attention_mask": ids.ne(0).long(),
            })

    native_embedding = nn.Embedding(8, 4)
    with torch.no_grad():
        native_embedding.weight.copy_(torch.arange(32).reshape(8, 4))
    model = nn.Module()
    model.text_model = nn.Module()
    model.text_model.embeddings = nn.Module()
    model.text_model.embeddings.token_embedding = native_embedding
    cfg = SimpleNamespace(n_ctx=2, ctx_init=None, token_embedding_size=4)

    learner = PromptLearnerPLIP(
        cfg, [["class a"], ["class b"]], model, Tokenizer(), "cpu")

    expected_prefix = native_embedding(torch.tensor([1, 1])).unsqueeze(1)
    expected_suffix = native_embedding(
        torch.tensor([[4, 0], [4, 0]]))
    torch.testing.assert_close(learner.token_prefix, expected_prefix)
    torch.testing.assert_close(learner.token_suffix, expected_suffix)


def test_pathpt_vision_only_aggregation_preserves_single_patch_axis():
    from methods.pathpt._model_utils import ConvTransAttentionAgg

    model = ConvTransAttentionAgg(dim=8, cls_num=3).eval()

    output = model(torch.randn(1, 1, 8))

    assert output.shape == (1, 3)


def test_top_averages_prototype_probabilities_like_upstream():
    """TOP pools prototypes as upstream does: mean *after* softmax.

    ``train_TCGAFeat_MIL_CLIP.py`` takes ``bag_prediction.mean(0)``, so the
    slide probability is the mean of the per-prototype probability rows, not a
    reduction of the raw scores. The adapter returns the log of that mean, which
    makes ``softmax(logits)`` recover upstream's probabilities exactly while
    still being a valid input to cross-entropy.
    """
    from methods.top.adapter import TOPMethod

    method = TOPMethod({"clip_arch": "RN50", "n_classes": 3}, device="cpu")
    prototype_scores = torch.tensor([
        [3.0, 0.2, 0.1],
        [0.3, 4.0, 0.4],
        [0.5, 0.6, 5.0],
    ])
    attention = torch.tensor([
        [1.0, 0.2],
        [0.3, 2.0],
        [0.7, 0.4],
    ])

    logits, auxiliary = method._slide_logits((prototype_scores, attention))

    expected = prototype_scores.softmax(dim=1).mean(dim=0, keepdim=True)
    assert logits.shape == (1, 3)
    torch.testing.assert_close(logits.softmax(dim=1), expected)
    # The diagonal of the raw scores is what a naive reduction would return;
    # upstream's averaging must not reproduce it.
    assert not torch.allclose(logits, torch.tensor([[3.0, 4.0, 5.0]]))

    normed = torch.softmax(attention, dim=0)
    torch.testing.assert_close(
        auxiliary, torch.triu(normed.T @ normed, diagonal=1).mean())
    assert torch.isfinite(auxiliary)


def test_top_auxiliary_is_absent_without_a_2d_attention_matrix():
    """LossA needs the instance attention matrix; a vector carries no pairs."""
    from methods.top.adapter import TOPMethod

    method = TOPMethod({"clip_arch": "RN50", "n_classes": 2}, device="cpu")
    logits, auxiliary = method._slide_logits(
        (torch.tensor([[1.0, 2.0], [0.5, 1.5]]), torch.ones(2)))

    assert logits.shape == (1, 2)
    assert auxiliary is None


def test_vila_rejects_multi_slide_variable_length_batches():
    from methods.vila_mil.adapter import ViLaMILMethod

    assert ViLaMILMethod._slide_bag(torch.ones(1, 4, 1024)).shape == (4, 1024)
    with pytest.raises(ValueError, match="batch_size=1"):
        ViLaMILMethod._slide_bag(torch.ones(2, 4, 1024))


def test_focus_removes_only_the_loader_batch_axis():
    from methods.focus.adapter import FOCUSMethod

    features, labels = FOCUSMethod._features_and_label((
        torch.ones(1, 4, 512), torch.tensor([0])))

    assert features.shape == (4, 512)
    assert labels.shape == (1,)
    with pytest.raises(ValueError, match="batch_size=1"):
        FOCUSMethod._features_and_label((
            torch.ones(2, 4, 512), torch.tensor([0, 1])))


def test_maple_removes_only_the_loader_batch_axis():
    from methods.maple.adapter import MAPLEMethod

    assert MAPLEMethod._slide_bag(torch.ones(1, 4, 512)).shape == (4, 512)
    with pytest.raises(ValueError, match="batch_size=1"):
        MAPLEMethod._slide_bag(torch.ones(2, 4, 512))


def test_maple_preserves_entity_major_attribute_prompt_order():
    from methods.maple.maple_model.model import (
        _entity_major_attribute_view,
    )

    # PromptLearner emits [entity0/class0, entity0/class1, entity0/class2,
    # entity1/class0, ...]. Each row must remain attached to that pair.
    attributes = torch.arange(6).reshape(6, 1)
    restored = _entity_major_attribute_view(
        attributes, n_cls=3, num_entities=2)

    assert restored[:, :, 0].tolist() == [[0, 1, 2], [3, 4, 5]]


def test_mscpt_restores_native_dropped_batch_axis():
    from methods.mscpt.adapter import MSCPTMethod

    logits = MSCPTMethod._batch_logits(torch.tensor([1.0, 2.0, 3.0]))

    assert logits.shape == (1, 3)


def test_convlm_averages_tile_embeddings_per_slide():
    from methods.convlm.adapter import ConVLMMethod

    tiles = F.normalize(torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [1.0, 0.0],
    ]), dim=-1)
    slides = ConVLMMethod._slide_embeddings(tiles, batch_size=2, tile_count=2)

    assert slides.shape == (2, 2)
    torch.testing.assert_close(slides.norm(dim=-1), torch.ones(2))


def test_allowlisted_contract_rejects_bundle_from_another_family():
    contract = get_method("pathpt").get_backbone_contract()
    cfg = {"backbone": "plip", "feature_dim": 512}

    with pytest.raises(BackboneCompatibilityError, match="loader returned 'conch'"):
        contract.validate_bundle(cfg, _bundle_for_builtin("conch"))


def test_bundle_feature_space_is_checked_when_declared():
    contract = get_method("slip").get_backbone_contract()
    cfg = {
        "backbone": "clip-vitb",
        "feature_dim": 512,
        "feature_space_id": "some-other-model",
    }

    with pytest.raises(BackboneCompatibilityError, match="Feature space"):
        contract.validate_bundle(cfg, _bundle_for_builtin("clip-vitb"))


def test_sldpc_rejects_a_slide_only_encoder_bundle():
    """Projection alone cannot preserve SLDPC's prompted text alignment."""
    slide_only_spec = BackboneSpec(
        name="test-slide-only",
        family="test",
        feature_space_id="test:slide-only-v1",
        capabilities=frozenset({Cap.SLIDE_PROJECT}),
        slide_input_dim=768,
        shared_dim=768,
    )
    slide_only = EncoderBundle(raw_model=nn.Identity(), spec=slide_only_spec)
    contract = get_method("sldpc").get_backbone_contract()

    with pytest.raises(BackboneCompatibilityError) as captured:
        contract.validate_bundle({
            "backbone": slide_only_spec.name,
            "feature_dim": 768,
            "feature_space_id": slide_only_spec.feature_space_id,
            "prompt_feature_space_id": slide_only_spec.feature_space_id,
        }, slide_only)

    message = str(captured.value)
    assert "soft_prompt" in message
    assert "text_encode" in message


def test_sldpc_accepts_a_capability_compatible_custom_bundle():
    paired_spec = BackboneSpec(
        name="test-slide-text",
        family="test",
        feature_space_id="test:slide-text-v1",
        capabilities=frozenset({
            Cap.TEXT_ENCODE,
            Cap.SOFT_PROMPT,
            Cap.SLIDE_PROJECT,
            Cap.PAIRED_SLIDE_TEXT,
        }),
        slide_input_dim=768,
        text_token_dim=32,
        shared_dim=64,
    )
    paired = EncoderBundle(raw_model=nn.Identity(), spec=paired_spec)
    contract = get_method("sldpc").get_backbone_contract()

    cfg = {
        "backbone": paired_spec.name,
        "feature_dim": 768,
        "feature_space_id": paired_spec.feature_space_id,
        "prompt_feature_space_id": paired_spec.feature_space_id,
    }
    assert contract.validate_bundle(cfg, paired) is paired


def test_capability_policy_does_not_claim_arbitrary_slide_encoder_support():
    contract = get_method("sldpc").get_backbone_contract()

    assert contract.swap_policy is SwapPolicy.CAPABILITY
    assert contract.required_capabilities.issuperset({
        Cap.TEXT_ENCODE,
        Cap.SOFT_PROMPT,
    })


class _TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    cls_token_id = 1
    eos_token_id = 2
    sep_token_id = 2
    model_max_length = 16


class _TinyPairedEncoder(nn.Module):
    """Small non-TITAN foundation model owned by both runtime wrappers."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(64, 6)
        self.text_projection = nn.Linear(6, 4, bias=False)
        self.slide_projection = nn.Linear(5, 4, bias=False)


class _TinyPromptableText:
    def __init__(self, model: _TinyPairedEncoder):
        self.model = model
        self.tokenizer = _TinyTokenizer()

    @property
    def token_width(self) -> int:
        return self.model.token_embedding.embedding_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.model.token_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.model.token_embedding.weight.device

    def tokenize(self, texts) -> TokenBatch:
        words = [str(text).split() for text in texts]
        width = max(len(row) for row in words) + 2
        ids = torch.full((len(words), width), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        eot = torch.empty(len(words), dtype=torch.long, device=self.device)
        for row_index, row in enumerate(words):
            ids[row_index, 0] = self.tokenizer.bos_token_id
            for token_index, word in enumerate(row, start=1):
                ids[row_index, token_index] = 3 + sum(map(ord, word)) % 61
            eot[row_index] = len(row) + 1
            ids[row_index, eot[row_index]] = self.tokenizer.eos_token_id
            mask[row_index, :eot[row_index] + 1] = 1
        return TokenBatch(ids, mask, eot)

    def embed_tokens(self, tokens: TokenBatch) -> torch.Tensor:
        return self.model.token_embedding(tokens.input_ids.to(self.device))

    def encode_embedded(self, embeddings: torch.Tensor, tokens: TokenBatch,
                        normalize: bool = True) -> torch.Tensor:
        mask = tokens.attention_mask.to(
            device=embeddings.device, dtype=embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        output = self.model.text_projection(pooled)
        return F.normalize(output.float(), dim=-1) if normalize else output

    def encode_text(self, texts_or_tokens,
                    normalize: bool = True) -> torch.Tensor:
        tokens = (self.tokenize(texts_or_tokens)
                  if isinstance(texts_or_tokens, (list, tuple))
                  else texts_or_tokens)
        return self.encode_embedded(
            self.embed_tokens(tokens), tokens, normalize=normalize)


class _TinySlideProjector:
    def __init__(self, model: _TinyPairedEncoder):
        self.model = model

    def project_slide(self, raw_embeddings: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
        output = self.model.slide_projection(raw_embeddings)
        return F.normalize(output.float(), dim=-1) if normalize else output


class _FixedLengthTinyPromptableText(_TinyPromptableText):
    """Mimic TITAN by padding every tokenized sequence to context length."""

    def tokenize(self, texts) -> TokenBatch:
        batch = super().tokenize(texts)
        width = self.tokenizer.model_max_length
        ids = F.pad(batch.input_ids, (0, width - batch.input_ids.shape[1]))
        mask = F.pad(
            batch.attention_mask, (0, width - batch.attention_mask.shape[1]))
        return TokenBatch(ids, mask, batch.eot_indices)


def test_non_titan_paired_bundle_runs_sldpc_prompt_model_end_to_end():
    """SLDPC depends on interfaces, not TITAN's concrete class hierarchy."""
    from methods.sldpc.model import SLDPCPromptLearner, PromptedSlideTextModel

    raw_encoder = _TinyPairedEncoder()
    text_encoder = _TinyPromptableText(raw_encoder)
    spec = BackboneSpec(
        name="test-tiny-slide-text",
        family="test-transformer",
        feature_space_id="test:tiny-slide-text-v1",
        capabilities=frozenset({
            Cap.TEXT_ENCODE,
            Cap.SOFT_PROMPT,
            Cap.SLIDE_PROJECT,
            Cap.PAIRED_SLIDE_TEXT,
        }),
        slide_input_dim=5,
        text_token_dim=6,
        shared_dim=4,
        context_length=16,
    )
    bundle = EncoderBundle(
        raw_model=raw_encoder,
        raw_tokenizer=text_encoder.tokenizer,
        spec=spec,
        text=text_encoder,
        slide=_TinySlideProjector(raw_encoder),
    )
    contract = get_method("sldpc").get_backbone_contract()
    contract.validate_bundle({
        "backbone": spec.name,
        "feature_dim": 5,
        "feature_space_id": spec.feature_space_id,
        "prompt_feature_space_id": spec.feature_space_id,
    }, bundle)

    prompt = SLDPCPromptLearner(
        ["lung_adenocarcinoma", "squamous", "normal"],
        text_encoder,
        n_ctx=2,
    )
    model = PromptedSlideTextModel(bundle, prompt)
    features = torch.tensor([
        [1.0, 0.5, -0.5, 0.25, 0.75],
        [-0.5, 1.0, 0.5, 0.75, -0.25],
    ])
    logits = model(features)

    assert spec.family != "titan"
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert not raw_encoder.training
    assert all(not parameter.requires_grad
               for parameter in raw_encoder.parameters())
    assert prompt.ctx_learnable.requires_grad

    logits.square().mean().backward()
    assert prompt.ctx_learnable.grad is not None
    assert torch.isfinite(prompt.ctx_learnable.grad).all()
    assert all(parameter.grad is None for parameter in raw_encoder.parameters())


def test_sldpc_learned_adapter_accepts_an_arbitrary_slide_feature_width():
    from common.models.slide_alignment import TrainableSlideAdapter
    from methods.sldpc.model import (
        PromptedSlideTextModel,
        SLDPCPromptLearner,
    )

    raw_encoder = _TinyPairedEncoder()
    text_encoder = _TinyPromptableText(raw_encoder)
    spec = BackboneSpec(
        name="test-prompt-only",
        family="test-transformer",
        feature_space_id="test:prompt-space-v1",
        capabilities=frozenset({Cap.TEXT_ENCODE, Cap.SOFT_PROMPT}),
        text_token_dim=6,
        shared_dim=4,
        context_length=16,
    )
    bundle = EncoderBundle(
        raw_model=raw_encoder,
        raw_tokenizer=text_encoder.tokenizer,
        spec=spec,
        text=text_encoder,
    )
    prompt = SLDPCPromptLearner(["class one", "class two"], text_encoder, n_ctx=2)
    adapter = TrainableSlideAdapter(11, 4, mode="mlp", hidden_dim=7)
    model = PromptedSlideTextModel(
        bundle, prompt, slide_adapter=adapter, slide_input_dim=11)

    logits = model(torch.randn(3, 11))

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())
    assert prompt.ctx_learnable.grad is not None


def test_sldpc_ctx_init_uses_eot_instead_of_fixed_padding_width():
    from methods.sldpc.model import SLDPCPromptLearner

    raw_encoder = _TinyPairedEncoder()
    text_encoder = _FixedLengthTinyPromptableText(raw_encoder)
    prompt = SLDPCPromptLearner(
        ["squamous"], text_encoder, n_ctx=2,
        ctx_init="a histology image")

    assert prompt.n_ctx == 3
    assert prompt.ctx_learnable.shape == (3, text_encoder.token_width)
    assert prompt(mode="task").shape == (1, 4)


def test_sldpc_requires_exact_prompt_backbone_provenance():
    paired_spec = BackboneSpec(
        name="test-provenance-slide-text",
        family="test",
        feature_space_id="test:paired@revision-1",
        capabilities=frozenset({
            Cap.TEXT_ENCODE, Cap.SOFT_PROMPT, Cap.SLIDE_PROJECT,
            Cap.PAIRED_SLIDE_TEXT,
        }),
        slide_input_dim=5,
        shared_dim=4,
    )
    paired = EncoderBundle(raw_model=nn.Identity(), spec=paired_spec)
    contract = get_method("sldpc").get_backbone_contract()

    with pytest.raises(
            BackboneCompatibilityError,
            match="requires 'prompt_feature_space_id'"):
        contract.validate_bundle({
            "backbone": paired_spec.name,
            "feature_dim": 5,
            "feature_space_id": "test:offline-slide-encoder",
        }, paired)
    with pytest.raises(BackboneCompatibilityError, match="Feature space"):
        contract.validate_bundle({
            "backbone": paired_spec.name,
            "feature_dim": 5,
            "feature_space_id": "test:offline-slide-encoder",
            "prompt_feature_space_id": "test:unrelated@revision-1",
        }, paired)


def test_sldpc_dhno_excludes_other_positive_classes_from_hard_retrieval():
    """Retain upstream DHNO's mini-batch false-negative guard."""
    from methods.sldpc.adapter import SLDPCMethod

    method = SLDPCMethod({
        "backbone": "titan",
        "feature_dim": 768,
        "feature_space_id": "hf:MahmoodLab/TITAN",
        "prompt_feature_space_id": "hf:MahmoodLab/TITAN",
        "n_classes": 4,
        "classnames": ["zero", "one", "two", "three"],
        "topk": 2,
    }, device="cpu")
    method._base_text = torch.eye(4)
    method._bank_by_class = {
        index: torch.eye(4)[index:index + 1] for index in range(4)
    }
    method._stage2_rng = random.Random(0)
    projected = torch.tensor([
        [0.1, 0.9, 0.8, 0.0],  # class 1 is highest but is another positive
        [0.9, 0.1, 0.0, 0.8],  # class 0 is highest but is another positive
    ])
    labels = torch.tensor([0, 1])

    _, extended_labels = method._hard_negative_extension(
        projected, labels, task_text=torch.eye(4))

    assert extended_labels.tolist() == [0, 2, 1, 3]


def test_sldpc_restores_each_stages_best_prompt_before_handoff_and_test():
    from methods.sldpc.adapter import SLDPCMethod

    class _Prompt(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx_learnable = nn.Parameter(torch.tensor([[0.0]]))
            self.ctx_frozen = nn.Parameter(
                torch.tensor([[0.0]]), requires_grad=False)

        @torch.no_grad()
        def clone_learnable_to_frozen(self):
            self.ctx_frozen.copy_(self.ctx_learnable)

        @torch.no_grad()
        def reinit_learnable_from_frozen(self):
            self.ctx_learnable.copy_(self.ctx_frozen)

    class _Model:
        prompt_learner = _Prompt()

    method = SLDPCMethod({
            "backbone": "titan", "n_classes": 2,
            "feature_dim": 768,
            "feature_space_id": "hf:MahmoodLab/TITAN",
            "prompt_feature_space_id": "hf:MahmoodLab/TITAN",
        "stage1_epochs": 2, "stage2_epochs": 2, "epochs": 4,
        "early_stopping": False, "monitor_metric": "F1",
    }, device="cpu")
    model = _Model()
    method._last_model = model
    method._phase = "stage1"
    method._validation_logits = []
    method._validation_labels = []
    method._best_scores = {"stage1": float("-inf"),
                           "stage2": float("-inf")}
    method._best_contexts = {"stage1": None, "stage2": None}
    method._optimizer = torch.optim.AdamW(
        [model.prompt_learner.ctx_learnable], lr=1e-3)
    method._prepare_stage2 = lambda _model: None

    labels = torch.tensor([0, 1])
    good_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    bad_logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])

    with torch.no_grad():
        model.prompt_learner.ctx_learnable.fill_(1.0)
    method._validation_logits.append(good_logits)
    method._validation_labels.append(labels)
    method.on_epoch_end(0, {"val_loss": 0.2})

    with torch.no_grad():
        model.prompt_learner.ctx_learnable.fill_(2.0)
    method._validation_logits.append(bad_logits)
    method._validation_labels.append(labels)
    method.on_epoch_end(1, {"val_loss": 0.3})

    assert method._phase == "stage2"
    torch.testing.assert_close(
        model.prompt_learner.ctx_learnable, torch.tensor([[1.0]]))
    torch.testing.assert_close(
        model.prompt_learner.ctx_frozen, torch.tensor([[1.0]]))

    with torch.no_grad():
        model.prompt_learner.ctx_learnable.fill_(3.0)
    method._validation_logits.append(good_logits)
    method._validation_labels.append(labels)
    method.on_epoch_end(2, {"val_loss": 0.2})

    with torch.no_grad():
        model.prompt_learner.ctx_learnable.fill_(4.0)
    method._validation_logits.append(bad_logits)
    method._validation_labels.append(labels)
    method.on_epoch_end(3, {"val_loss": 0.3})

    torch.testing.assert_close(
        model.prompt_learner.ctx_learnable, torch.tensor([[3.0]]))


def test_sldpc_rejects_outer_loop_schedule_changes_before_loading():
    from methods.sldpc.adapter import SLDPCMethod

    base = {
            "backbone": "titan", "feature_dim": 768,
            "feature_space_id": "hf:MahmoodLab/TITAN", "n_classes": 2,
            "prompt_feature_space_id": "hf:MahmoodLab/TITAN",
        "classnames": ["a", "b"], "stage1_epochs": 2,
        "stage2_epochs": 2, "early_stopping": False,
    }
    with pytest.raises(ValueError, match="match both stages exactly"):
        SLDPCMethod({**base, "epochs": 5}, device="cpu").build_model()
    with pytest.raises(ValueError, match="early_stopping: false"):
        SLDPCMethod({**base, "epochs": 4, "early_stopping": True},
                    device="cpu").build_model()


def test_sldpc_final_checkpoint_restores_stage2_adapter_state():
    from methods.sldpc.adapter import SLDPCMethod

    method = object.__new__(SLDPCMethod)
    model = object()

    method.on_checkpoint_loaded(model, "final", fold=0)

    assert method._phase == "stage2"
    assert method._last_model is model


def test_muse_precomputed_prompt_bank_keeps_encoder_provenance(tmp_path):
    from methods.muse.adapter import MUSEMethod

    wrong_width = tmp_path / "wrong_width.pt"
    torch.save(torch.zeros(2, 3, 7), wrong_width)
    method = MUSEMethod({
        "backbone": "conch", "n_classes": 2,
        "classnames": ["a", "b"], "prompt_features": str(wrong_width),
        "prompt_feature_space_id": "hf:MahmoodLab/conch",
    }, device="cpu")
    with pytest.raises(BackboneCompatibilityError, match="prompt bank width 7"):
        method._load_prompt_bank()

    incorrect_space = tmp_path / "incorrect_space.pt"
    torch.save({
        "embeddings": torch.zeros(2, 3, 512),
        "feature_space_id": "test:unrelated",
    }, incorrect_space)
    method = MUSEMethod({
        "backbone": "conch", "n_classes": 2,
        "classnames": ["a", "b"], "prompt_features": str(incorrect_space),
        "prompt_feature_space_id": "hf:MahmoodLab/conch",
    }, device="cpu")
    with pytest.raises(BackboneCompatibilityError, match="prompt artifact space"):
        method._load_prompt_bank()


@pytest.mark.parametrize("input_dim", [512, 768, 1024])
def test_muse_projects_registered_patch_width_into_prompt_space(input_dim):
    from methods.muse.model import MUSEModel

    model = MUSEModel(
        input_dim=input_dim,
        n_classes=3,
        prompt_bank=torch.randn(3, 4, 512),
        embed_dim=512,
        num_heads=8,
        num_experts=2,
        num_selected=1,
        retrieval_k=2,
        dropout=0.0,
    )

    logits = model(torch.randn(1, 7, input_dim))

    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert model.visual_adapter[0].weight.grad is not None
