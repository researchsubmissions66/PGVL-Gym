"""SLIP frozen tissue prompts.

ChatGPT-generated tissue type names are encoded once via the frozen
text encoder and stored as `aux["tissue"]`. They are *not* learnable.

For aggregators that don't use them (e.g. ViLa-MIL, simple attention),
the aux entry is harmlessly ignored.
"""
from __future__ import annotations
from typing import List
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import PromptModule, PromptBank


class SLIPTissuePrompts(PromptModule):
    name = "slip_tissue"

    def __init__(self,
                 classnames: List[str],
                 backbone: nn.Module,
                 tokenizer,
                 tissue_classnames: List[str] = None,
                 tissue_classnames_path: str = None,
                 templates: List[str] = None,
                 n_ctx: int = 16):
        super().__init__()

        # Tissue list comes from explicit list or a JSON file
        if tissue_classnames is None and tissue_classnames_path is not None \
                and os.path.isfile(tissue_classnames_path):
            with open(tissue_classnames_path) as f:
                tissue_classnames = json.load(f)
        if tissue_classnames is None:
            # Sensible default (the user should override per dataset)
            tissue_classnames = [
                "tumor cells", "stromal cells", "lymphocytes",
                "necrotic tissue", "blood vessels"]
        self.tissue_classnames = tissue_classnames
        self.templates = templates or ["a histopathology image of {}."]

        # Encode tissue prompts ONCE, frozen
        with torch.no_grad():
            tissue_embeds = self._encode_tissue(backbone, tokenizer)
        self.register_buffer("tissue_features", tissue_embeds)

        # Slide-class side: a learnable CoOp head, just like coop_flat
        from common.models.coop import PromptLearner, TextEncoder
        self.prompt_learner = PromptLearner(
            classnames=classnames, clip_model=backbone,
            n_ctx=n_ctx, ctx_init="", csc=False, tokenizer_fn=tokenizer)
        self.text_encoder = TextEncoder(backbone)

    def _encode_tissue(self, backbone, tokenizer):
        all_embeds = []
        for tissue in self.tissue_classnames:
            texts = [t.format(tissue) for t in self.templates]
            tokenized = tokenizer(texts) if callable(tokenizer) \
                else tokenizer(texts, return_tensors="pt", padding=True)["input_ids"]
            if isinstance(tokenized, dict):
                tokenized = tokenized["input_ids"]
            embed = backbone.encode_text(tokenized.to(
                next(backbone.parameters()).device))
            embed = F.normalize(embed, dim=-1).mean(dim=0)
            embed = F.normalize(embed, dim=-1)
            all_embeds.append(embed)
        return torch.stack(all_embeds)            # (T, D)

    def forward(self) -> PromptBank:
        prompts = self.prompt_learner()
        text_features = self.text_encoder(
            prompts, self.prompt_learner.tokenized_prompts)
        return PromptBank(
            text_features=text_features,
            aux={"tissue": self.tissue_features})
