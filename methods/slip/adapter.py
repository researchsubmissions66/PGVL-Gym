"""SLIP adapter (ISBI 2025).

The vendored SLIP class signature is
    SLIP(model, tokenizer, templates, slide_classnames,
         tissue_classnames, experiment_dir, context_size,
         temperature, imgsize, context_init, cls_token_position,
         context_gain, arch).

We instantiate via the SlideCoOp constructor and then upcast to SLIP.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)


class SLIPMethod(BaseMethod):
    """Adapt SLIP's tissue-routed prompt learner for supported CLIP families."""
    name = "slip"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_BAG,
        swap_policy=SwapPolicy.ALLOWLIST, default_backbone="clip-vitb",
        supported_backbones=("clip-vitb", "clip-rn50", "plip", "biomedclip"),
        feature_dims={"clip-vitb": (512,), "clip-rn50": (1024,),
                      "plip": (512,), "biomedclip": (512,)},
        required_capabilities=frozenset({Cap.TEXT_ENCODE, Cap.SOFT_PROMPT,
                                        Cap.PAIRED_TILE_TEXT}),
        rationale="SLIP needs paired patch/text features plus a native soft-prompt branch.")

    def build_model(self) -> nn.Module:
        from .methods.slidecoop import SLIP

        # Keep SLIP's native family-specific prompt classes; the bundle only
        # standardizes loading, dimensions, and compatibility validation.
        arch_map = {"clip-vitb": "CLIP", "clip-rn50": "CLIP-RN50",
                    "plip": "PLIP", "biomedclip": "BiomedCLIP"}
        arch_slip = arch_map[self.backbone_name]
        encoder = self.load_encoder(
            weights_path=self.cfg.get("backbone_weights")).freeze()

        templates = self.cfg.get("text_templates", ["a photo of a {}."])
        tissues = self.cfg.get("tissue_classnames", self.cfg["classnames"])
        tissues = [item if isinstance(item, (list, tuple)) else [item]
                   for item in tissues]
        model = SLIP(
            model=encoder.raw_model,
            tokenizer=encoder.raw_tokenizer,
            templates=templates,
            slide_classnames=self.cfg["classnames"],
            tissue_classnames=tissues,
            experiment_dir=self.cfg.get("results_dir", "./results/slip"),
            context_size=self.cfg.get("context_size", 1),
            temperature=self.cfg.get("temp", 0.01),
            imgsize=self.cfg.get("image_size", 224),
            context_init=self.cfg.get("context_init", None),
            cls_token_position="end",
            context_gain=self.cfg.get("context_gain", 0.01),
            arch=arch_slip,
        )
        return model.to(self.device)

    def train_step(self, batch, model, optimizer, loss_fn):
        feats = batch[0]
        label = batch[-1]
        optimizer.zero_grad()
        # SLIP exposes its own contrastive loss via compute_loss
        loss, logits = model.compute_loss(feats, label)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(),
                "logits": logits.detach(),
                "label": label.to(self.device)}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        feats = batch[0].to(self.device).squeeze(0)
        label = batch[-1].to(self.device).view(-1)
        # The unified loop does not call SLIP's legacy ``test`` helper that
        # materializes ``slide_weights``. Recompute weights from the current
        # prompt parameters so validation follows the trained model.
        out = model(feats, eval=False)
        logits = out[0] if isinstance(out, tuple) else out
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        loss = 0.0  # SLIP eval uses argmax on cross-correlation
        return {"loss": loss, "logits": logits, "label": label}
