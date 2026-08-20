"""ViLa-MIL adapter (CVPR 2024).

ViLa_MIL_Model takes a config object with attributes:
    input_size, hidden_size, prototype_number, text_prompt
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
import torch.nn as nn

from methods.base import BaseMethod, probabilities_to_logits
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)
from common.prompts import VILA_PROMPT_FORMAT, load_vila_prompt_bank


def _build_config(cfg):
    if not cfg.get("text_prompt_path"):
        raise ValueError("ViLa-MIL requires text_prompt_path")
    if cfg.get("vila_prompt_format") != VILA_PROMPT_FORMAT:
        raise ValueError(
            "ViLa-MIL requires vila_prompt_format=" + VILA_PROMPT_FORMAT)
    for key in (
        "vila_prompt_file_classnames",
        "vila_prompt_file_sha256",
        "vila_prompt_bank_sha256",
        "prompt_provenance",
        "prompt_source",
    ):
        if not cfg.get(key):
            raise ValueError(f"ViLa-MIL requires {key}")
    label_dict = cfg.get("label_dict")
    if not isinstance(label_dict, dict) or not label_dict:
        raise ValueError("ViLa-MIL requires an ordered label_dict")
    if (any(not isinstance(index, int) or isinstance(index, bool)
            for index in label_dict.values())
            or sorted(label_dict.values()) != list(range(len(label_dict)))):
        raise ValueError(
            "ViLa-MIL label_dict indices must be contiguous from zero")
    labels = [
        str(label) for label, _index in sorted(
            label_dict.items(), key=lambda item: item[1])]
    if int(cfg.get("n_classes", -1)) != len(labels):
        raise ValueError("ViLa-MIL n_classes does not match label_dict")
    bank = load_vila_prompt_bank(
        cfg["text_prompt_path"],
        class_names=labels,
        file_class_names=cfg.get("vila_prompt_file_classnames"),
        expected_provenance=cfg.get("prompt_provenance"),
        expected_file_sha256=cfg.get("vila_prompt_file_sha256"),
        expected_ordered_prompt_bank_sha256=cfg.get("vila_prompt_bank_sha256"),
    )
    expected_source = {
        "upstream": "vila_mil_upstream_native_two_scale_csv",
        "derived": "vila_mil_derived_native_two_scale_csv",
        "generated": "vila_mil_generated_native_two_scale_csv",
    }[bank.provenance]
    if cfg["prompt_source"] != expected_source:
        raise ValueError(
            "ViLa-MIL prompt_source does not match prompt provenance")
    return SimpleNamespace(
        input_size=cfg.get("feature_dim", 1024),
        hidden_size=cfg.get("hidden_size", 192),
        prototype_number=cfg.get("prototype_number", 16),
        text_prompt=list(bank.prompts),
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
        logits = (probabilities_to_logits(out[0])
                  if isinstance(out, tuple) else out)
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
        logits = (probabilities_to_logits(out[0])
                  if isinstance(out, tuple) else out)
        loss = (out[2].item() if isinstance(out, tuple) and len(out) > 2
                and torch.is_tensor(out[2]) else
                (loss_fn(logits, label).item() if loss_fn is not None else 0.0))
        return {"loss": loss, "logits": logits, "label": label}
