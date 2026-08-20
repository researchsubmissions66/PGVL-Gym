"""FOCUS adapter.

Paper: Guo et al., "FOCUS: Knowledge-enhanced Adaptive Visual
Compression for Few-shot Whole Slide Image Classification", CVPR 2025.

The FOCUS class (`methods/focus/model.py::FOCUS`) takes a `config`
object with attributes:
    input_size, hidden_size, prototype_number, text_prompt,
    window_size, sim_threshold, max_context_length
and loads CONCH internally from `ckpts/conch.pth`.
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
import torch.nn as nn

from methods.base import BaseMethod, probabilities_to_logits
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)
from common.prompts import FOCUS_PROMPT_FORMAT, load_focus_prompt_bank


def _build_config(cfg):
    if not cfg.get("text_prompt_path"):
        raise ValueError("FOCUS requires text_prompt_path")
    if cfg.get("focus_prompt_format") != FOCUS_PROMPT_FORMAT:
        raise ValueError(
            "FOCUS requires focus_prompt_format=" + FOCUS_PROMPT_FORMAT)
    for key in (
        "focus_prompt_file_classnames",
        "focus_prompt_file_sha256",
        "focus_prompt_bank_sha256",
        "prompt_provenance",
        "prompt_source",
    ):
        if not cfg.get(key):
            raise ValueError(f"FOCUS requires {key}")
    label_dict = cfg.get("label_dict")
    if not isinstance(label_dict, dict) or not label_dict:
        raise ValueError("FOCUS requires an ordered label_dict")
    if (any(not isinstance(index, int) or isinstance(index, bool)
            for index in label_dict.values())
            or sorted(label_dict.values()) != list(range(len(label_dict)))):
        raise ValueError("FOCUS label_dict indices must be contiguous from zero")
    labels = [
        str(label) for label, _index in sorted(
            label_dict.items(), key=lambda item: item[1])]
    if int(cfg.get("n_classes", -1)) != len(labels):
        raise ValueError("FOCUS n_classes does not match label_dict")
    bank = load_focus_prompt_bank(
        cfg["text_prompt_path"],
        class_names=labels,
        file_class_names=cfg.get("focus_prompt_file_classnames"),
        expected_provenance=cfg.get("prompt_provenance"),
        expected_file_sha256=cfg.get("focus_prompt_file_sha256"),
        expected_ordered_prompt_bank_sha256=cfg.get(
            "focus_prompt_bank_sha256"),
    )
    expected_source = {
        "upstream": "focus_upstream_native_two_scale_csv",
        "derived": "focus_derived_native_two_scale_csv",
        "generated": "focus_generated_native_two_scale_csv",
    }[bank.provenance]
    if cfg["prompt_source"] != expected_source:
        raise ValueError("FOCUS prompt_source does not match prompt provenance")
    return SimpleNamespace(
        input_size=cfg.get("feature_dim", 1024),
        hidden_size=cfg.get("hidden_size", 192),
        prototype_number=cfg.get("prototype_number", 16),
        text_prompt=list(bank.prompts),
        window_size=cfg.get("window_size", 7),
        sim_threshold=cfg.get("sim_threshold", 0.7),
        max_context_length=cfg.get("max_context_length", 4096),
    )


class FOCUSMethod(BaseMethod):
    """Adapt FOCUS's single high-resolution bag and CONCH soft prompts."""
    name = "focus"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_BAG,
        swap_policy=SwapPolicy.ALLOWLIST, default_backbone="conch",
        supported_backbones=("conch",),
        required_capabilities=frozenset({Cap.SOFT_PROMPT}),
        rationale="FOCUS injects prompts into a CONCH-style text tower and its forward path consumes only the high-resolution patch bag.")

    @staticmethod
    def _features_and_label(batch):
        """Accept the fixed single-bag loader and legacy paired-bag batches."""
        label = batch[-1]
        if (len(batch) >= 3 and torch.is_tensor(batch[1])):
            features = batch[1]
        else:
            features = batch[0]
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError(
                "FOCUS requires batch_size=1 variable-length bags; got "
                f"{list(features.shape)}")
        return features, label

    def build_model(self) -> nn.Module:
        from .model import FOCUS
        config = _build_config(self.cfg)
        encoder = self.load_encoder(weights_path=self.cfg.get("conch_ckpt"))
        model = FOCUS(config, num_classes=self.cfg["n_classes"],
                      conch_model=encoder.raw_model)
        return model.to(self.device)

    def train_step(self, batch, model, optimizer, loss_fn):
        x_l, label = self._features_and_label(batch)
        x_l = x_l.to(self.device)
        label = label.to(self.device)

        optimizer.zero_grad()
        out = model(x_l, x_l, label)
        logits = (probabilities_to_logits(out[0])
                  if isinstance(out, tuple) else out)
        loss = (out[2] if isinstance(out, tuple) and len(out) > 2
                and torch.is_tensor(out[2]) else loss_fn(logits, label))
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        x_l, label = self._features_and_label(batch)
        x_l = x_l.to(self.device)
        label = label.to(self.device)
        out = model(x_l, x_l, label)
        logits = (probabilities_to_logits(out[0])
                  if isinstance(out, tuple) else out)
        loss = (out[2].item() if isinstance(out, tuple) and len(out) > 2
                and torch.is_tensor(out[2]) else
                (loss_fn(logits, label).item() if loss_fn is not None else 0.0))
        return {"loss": loss, "logits": logits, "label": label}
