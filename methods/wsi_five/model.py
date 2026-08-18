"""WSI-FiVE: fine-grained visual-semantic interaction over a WSI patch bag.

This follows the published architecture rather than approximating it. With the
release's shipped default (``IS_IMG_PTH: True``) FiVE has **no vision tower** --
``self.visual`` is ``nn.Identity()`` and precomputed patch features are read from
disk. The paper states it directly: *"we employed ResNet following [15] as image
encoder to extract image features, while pre-trained BioClinicalBERT from [30] as
text encoder. LoRA was adopted for fine-tuning the text encoder"*, where [15] is
DSMIL. No MedCLIP weights are loaded anywhere upstream: ``MedCLIPTextModel`` --
the only MedCLIP class instantiated -- loads ``Bio_ClinicalBERT``, and every
checkpoint-loading path in ``MedCLIPModel.py`` belongs to a vision class that is
never constructed.

The method's contribution is the *interaction*, and it has four parts, all
reproduced here from the vendored upstream modules:

1. ``PatchFusionTransformer`` (X-CLIP-derived) aggregates the bag through a
   self-attention branch and a cross-attention branch, fused by concatenation.
2. The cross-attention uses **encoded clinical questions as queries**, so text
   conditions the aggregation instead of only being compared to it. Those
   questions are the release's ``PROMPT_LIST``.
3. Sixteen learnable soft prompts are injected into BioClinicalBERT's embedding
   layer and concatenated to the encoded questions.
4. Positional information comes from each patch's index within the slide, and
   padded positions are masked per slide via ``sample_range``.

Deviations from the release are listed in ``docs/design-decisions.md``.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wsi_five_models.MedCLIPModel import MedCLIPTextModel
from .wsi_five_models.lora_wrap import LoraWrap
from .wsi_five_models.patch_fusion import PatchFusionTransformer

# The six clinical questions the release ships in every configuration under
# `_configs/wsi/*.yaml`. They are the text the patch aggregation cross-attends
# to, so they are part of the architecture, not a tunable prompt choice.
PROMPT_LIST: tuple[str, ...] = (
    "What is the differentiation of the lesion? (maybe: Well-differentiated; "
    "Moderately differentiated; Poorly differentiated; Moderately to poorly "
    "differentiated; Mixed differentiation. or others.)",
    "Is there any indication of spread through air spaces around the lesion?",
    "Is there any indication of vascular invasion by the lesion?",
    "Is there any indication of pleural invasion by the lesion?",
    "Is there any evidence of the lesion invading adjacent tissues or organs "
    "(excluding the current lung organ)?",
    "Are the margins of the excised tissue clear of disease? (note that: R0 "
    "means negative; R1 R2 are both mean positive; Rx means Unknown.",
)


class WSIFiVEModel(nn.Module):
    """Reproduce FiVE's patch-fusion and BioClinicalBERT prompt interaction.

    Args:
        classnames: Diagnostic class names, embedded as the comparison text.
        clinicalbert_path: Local BioClinicalBERT directory or model id.
        feature_dim: Width of the precomputed patch features (512 upstream).
        num_frames: Maximum patches per slide; sizes the fusion transformer.
        context_length: Token length of a learnable soft prompt.
        learnable_prompts: Number of soft prompts (16 upstream).
        lora_targets: Comma-separated BERT submodules LoRA adapts.
        logit_scale: Upstream hardcodes 300 rather than learning this.
        prompt_list: Clinical questions cross-attended during aggregation.
    """

    def __init__(self, classnames: Sequence[str], clinicalbert_path: str,
                 feature_dim: int = 512, num_frames: int = 2048,
                 context_length: int = 308, learnable_prompts: int = 16,
                 lora_targets: str = "query,key,value,dense",
                 logit_scale: float = 300.0,
                 prompt_list: Sequence[str] | None = None):
        super().__init__()
        self.classnames = list(classnames)
        self.feature_dim = int(feature_dim)
        self.logit_scale = float(logit_scale)
        self.prompt_list = self._resolve_questions(prompt_list)

        self.text = MedCLIPTextModel(bert_type=clinicalbert_path,
                                     proj_dim=self.feature_dim)
        hidden = int(self.text.model.config.hidden_size)
        # LoRA on the text tower, then re-enable the projection head, exactly as
        # the release does after wrapping (mark_only_lora_as_trainable freezes
        # everything else).
        self.text.model = LoraWrap(self.text.model, lora_targets)
        self.text.projection_head.weight.requires_grad = True

        self.mit = PatchFusionTransformer(
            num_frames, embed_dim=self.feature_dim, layers_sa=1, layers_ca=1)

        self.prompt_learn_param = nn.Parameter(
            torch.empty(learnable_prompts, context_length, hidden))
        nn.init.normal_(self.prompt_learn_param, std=0.01)

    @staticmethod
    def _resolve_questions(prompt_list) -> list[str]:
        """Accept an inline list, a path to a question set, or the default.

        The released question set is lung-specific -- "spread through air
        spaces", "pleural invasion" and "excluding the current lung organ" have
        no meaning outside lung -- so a cohort may declare its own. Sets that
        are not the authors' record `_provenance: generated`.
        """
        if prompt_list is None:
            return list(PROMPT_LIST)
        if isinstance(prompt_list, (str, pathlib.Path)):
            with open(prompt_list, encoding="utf-8") as handle:
                payload = json.load(handle)
            questions = (payload.get("questions")
                         if isinstance(payload, dict) else payload)
            if not questions:
                raise ValueError(f"{prompt_list} declares no questions")
            return [str(item) for item in questions]
        return [str(item) for item in prompt_list]

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.prompt_learn_param.device

    def _tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        tokens = self.text.tokenizer(
            list(texts), padding=True, truncation=True, max_length=256,
            return_tensors="pt")
        return {key: value.to(self.device) for key, value in tokens.items()}

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode strings through the LoRA-adapted BioClinicalBERT tower."""
        tokens = self._tokenize(texts)
        return self.text(tokens["input_ids"], tokens["attention_mask"])

    def encode_prompt_embed(self, prompt_embed: torch.Tensor) -> torch.Tensor:
        """Encode learnable soft prompts through BERT's own embedding stack.

        The release injects these *after* token embedding, so the soft prompts
        occupy the same space as embedded tokens and pass through the encoder
        unchanged otherwise.
        """
        bert = self.text.model.model
        batch, sequence, _ = prompt_embed.shape
        embeddings_layer = bert.embeddings
        position_ids = embeddings_layer.position_ids[:, :sequence]
        token_type_ids = torch.zeros(
            (batch, sequence), dtype=torch.long, device=prompt_embed.device)

        embeddings = prompt_embed + embeddings_layer.token_type_embeddings(
            token_type_ids)
        if embeddings_layer.position_embedding_type == "absolute":
            embeddings = embeddings + embeddings_layer.position_embeddings(
                position_ids)
        embeddings = embeddings_layer.dropout(
            embeddings_layer.LayerNorm(embeddings))

        attention_mask = torch.ones((batch, sequence), device=prompt_embed.device)
        extended = bert.get_extended_attention_mask(
            attention_mask, (batch, sequence))
        encoded = bert.encoder(embeddings, attention_mask=extended)[0]
        return self.text.projection_head(encoded.mean(dim=1))

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor,
                patch_info: dict[str, Any]) -> torch.Tensor:
        """Return class logits for one slide bag.

        Args:
            features: ``[batch, patches, feature_dim]`` precomputed patch bag.
            patch_info: ``sample_range`` (real patch count per slide),
                ``patch_inds`` (each patch's index within the slide) and
                ``patch_pub_cnt`` (total patches in the slide).
        """
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"WSI-FiVE expects [batch, patches, {self.feature_dim}], got "
                f"{list(features.shape)}")
        for key in ("sample_range", "patch_inds", "patch_pub_cnt"):
            if key not in patch_info:
                raise KeyError(f"WSI-FiVE patch_info is missing '{key}'")

        question_features = self.encode_text(self.prompt_list)
        learned_features = self.encode_prompt_embed(self.prompt_learn_param)
        prompts = torch.cat([question_features, learned_features], dim=0)
        prompts = F.normalize(prompts, dim=-1)

        slide = self.mit(features.float(), prompts, patch_info)
        slide = F.normalize(slide, dim=-1)

        class_features = F.normalize(self.encode_text(self.classnames), dim=-1)
        return self.logit_scale * slide @ class_features.t()
