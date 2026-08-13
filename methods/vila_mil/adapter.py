"""ViLa-MIL adapter (CVPR 2024).

ViLa_MIL_Model takes a config object with attributes:
    input_size, hidden_size, prototype_number, text_prompt
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
import torch.nn as nn
import pandas as pd

from methods.base import BaseMethod
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)


def _build_config(cfg):
    text_prompt = None
    if cfg.get("text_prompt_path"):
        frame = pd.read_csv(cfg["text_prompt_path"])
        if {"low_res_prompt", "high_res_prompt"}.issubset(frame.columns):
            text_prompt = (frame["low_res_prompt"].astype(str).tolist() +
                           frame["high_res_prompt"].astype(str).tolist())
        else:
            text_prompt = [str(item) for item in frame.values.reshape(-1)
                           if pd.notna(item)]
    return SimpleNamespace(
        input_size=cfg.get("feature_dim", 1024),
        hidden_size=cfg.get("hidden_size", 192),
        prototype_number=cfg.get("prototype_number", 16),
        text_prompt=text_prompt,
    )


class ViLaMILMethod(BaseMethod):
    """Adapt ViLa-MIL's CLIP-RN50 dual-scale patch-bag workflow."""
    name = "vila_mil"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.DUAL_SCALE_PATCH_BAG,
        swap_policy=SwapPolicy.ALLOWLIST, default_backbone="clip-rn50",
        supported_backbones=("clip-rn50",), feature_dims={"clip-rn50": (1024,)},
        required_capabilities=frozenset({Cap.SOFT_PROMPT, Cap.PAIRED_TILE_TEXT}),
        rationale="The published model compares 1024-wide RN50 patch and prompted text features directly.")

    def build_model(self) -> nn.Module:
        from .model import ViLa_MIL_Model
        config = _build_config(self.cfg)
        encoder = self.load_encoder(
            weights_path=self.cfg.get("backbone_weights")).freeze()
        model = ViLa_MIL_Model(
            config, num_classes=self.cfg["n_classes"],
            clip_model=encoder.raw_model)
        return model.to(self.device)

    @staticmethod
    def _slide_bag(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3 and value.shape[0] == 1:
            return value.squeeze(0)
        if value.ndim != 2:
            raise ValueError(
                "ViLa-MIL requires batch_size=1 variable-length bags; got "
                f"{list(value.shape)}")
        return value

    def train_step(self, batch, model, optimizer, loss_fn):
        # ViLa-MIL forward: (x_s, coord_s, x_l, coords_l, label)
        # but our default loader yields (x_s, x_l, label); we synthesise
        # zero coordinates if missing.
        x_s, x_l, label = batch[0], batch[1], batch[-1]
        x_s = self._slide_bag(x_s.to(self.device))
        x_l = self._slide_bag(x_l.to(self.device))
        label = label.to(self.device)
        coord_s = torch.zeros(x_s.shape[0], 2, device=self.device)
        coord_l = torch.zeros(x_l.shape[0], 2, device=self.device)

        optimizer.zero_grad()
        out = model(x_s, coord_s, x_l, coord_l, label)
        logits = out[0] if isinstance(out, tuple) else out
        loss = (out[2] if isinstance(out, tuple) and len(out) > 2
                and torch.is_tensor(out[2]) else loss_fn(logits, label))
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        x_s, x_l, label = batch[0], batch[1], batch[-1]
        x_s = self._slide_bag(x_s.to(self.device))
        x_l = self._slide_bag(x_l.to(self.device))
        label = label.to(self.device)
        coord_s = torch.zeros(x_s.shape[0], 2, device=self.device)
        coord_l = torch.zeros(x_l.shape[0], 2, device=self.device)
        out = model(x_s, coord_s, x_l, coord_l, label)
        logits = out[0] if isinstance(out, tuple) else out
        loss = (out[2].item() if isinstance(out, tuple) and len(out) > 2
                and torch.is_tensor(out[2]) else
                (loss_fn(logits, label).item() if loss_fn is not None else 0.0))
        return {"loss": loss, "logits": logits, "label": label}
