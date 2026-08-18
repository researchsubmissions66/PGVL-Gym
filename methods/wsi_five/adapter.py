"""WSI-FiVE adapter (CVPR 2024).

FiVE consumes a *precomputed* patch bag, not raw tiles: with the release's
shipped `IS_IMG_PTH: True` its vision tower is `nn.Identity()` and features are
read from disk. What the method owns is the text side -- a LoRA-adapted
BioClinicalBERT tower whose encoded clinical questions condition patch
aggregation through cross-attention.

The contract therefore declares a patch bag with a fixed native width, not an
encoder-owning architecture. See `docs/design-decisions.md` for the evidence and
for the deviations that remain.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import FeatureLevel, MethodBackboneContract, SwapPolicy


class WSIFiVEMethod(BaseMethod):
    """Adapt WSI-FiVE patch bags, clinical questions and pathology reports."""

    name = "wsi_five"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_SEQUENCE_WITH_REPORT,
        swap_policy=SwapPolicy.PRECOMPUTED, default_backbone="wsi-five-vit",
        supported_backbones=("wsi-five-vit",), feature_dims={"wsi-five-vit": (512,)},
        rationale=(
            "FiVE reads a precomputed 512-wide patch bag; its own tower is the "
            "LoRA-adapted BioClinicalBERT text encoder, not a vision encoder."))

    def build_model(self) -> nn.Module:
        from .model import WSIFiVEModel

        model = WSIFiVEModel(
            self.cfg["classnames"],
            self.cfg["clinicalbert_weights"],
            feature_dim=self.cfg.get("feature_dim", 512),
            num_frames=self.cfg.get("num_frames", 2048),
            context_length=self.cfg.get("prompt_context_length", 308),
            learnable_prompts=self.cfg.get("learnable_prompts", 16),
            lora_targets=self.cfg.get("lora_targets", "query,key,value,dense"),
            logit_scale=self.cfg.get("logit_scale", 300.0),
            prompt_list=self.cfg.get("clinical_questions"),
        )
        return model.to(self.device)

    @staticmethod
    def _unpack(batch):
        """Return (features, patch_info, label) from either batch shape."""
        features, _reports, patch_info = batch[0], batch[1], batch[2]
        return features, patch_info, batch[-1]

    def _to_device(self, patch_info: dict) -> dict:
        return {
            "patch_inds": patch_info["patch_inds"].to(self.device),
            "patch_pub_cnt": patch_info["patch_pub_cnt"].to(self.device),
            "sample_range": patch_info["sample_range"],
        }

    def train_step(self, batch, model, optimizer, loss_fn):
        features, patch_info, label = self._unpack(batch)
        features, label = features.to(self.device), label.to(self.device)
        optimizer.zero_grad()
        logits = model(features, self._to_device(patch_info))
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        features, patch_info, label = self._unpack(batch)
        features, label = features.to(self.device), label.to(self.device)
        logits = model(features, self._to_device(patch_info))
        loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": label}
