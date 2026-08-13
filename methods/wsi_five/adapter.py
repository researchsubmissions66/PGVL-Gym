"""WSI-FiVE adapter (CVPR 2024).

The FiVE class is a subclass of OpenAI CLIP (lots of constructor args
mirroring CLIP). We construct it via the upstream `build_model`
helper that the original repo ships in `models/build.py`.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import FeatureLevel, MethodBackboneContract, SwapPolicy


class WSIFiVEMethod(BaseMethod):
    """Adapt WSI-FiVE patch sequences and pathology reports."""
    name = "wsi_five"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_SEQUENCE_WITH_REPORT,
        swap_policy=SwapPolicy.FIXED, default_backbone="wsi-five-vit",
        supported_backbones=("wsi-five-vit",), feature_dims={"wsi-five-vit": (512,)},
        rationale="FiVE's tile/video transformer and ClinicalBERT prompt tower are method-owned architecture.")

    def build_model(self) -> nn.Module:
        from .model import WSIFiVEClassifier
        model = WSIFiVEClassifier(
            self.cfg["classnames"], self.cfg["clinicalbert_weights"],
            feature_dim=self.cfg.get("feature_dim", 512),
            num_heads=self.cfg.get("num_heads", 8),
            max_frames=self.cfg.get("num_frames", 2048),
            freeze_text_base=self.cfg.get("freeze_text_base", True),
        )
        return model.to(self.device)

    def train_step(self, batch, model, optimizer, loss_fn):
        feats = batch[0].to(self.device)
        label = batch[-1].to(self.device)
        reports = batch[1]
        optimizer.zero_grad()
        logits = model(feats, reports)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        feats = batch[0].to(self.device)
        label = batch[-1].to(self.device)
        reports = batch[1]
        logits = model(feats, reports)
        loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": label}
