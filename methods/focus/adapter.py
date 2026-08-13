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
from pathlib import Path
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
            # Native learner indexes [all low classes, all high classes].
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
        window_size=cfg.get("window_size", 7),
        sim_threshold=cfg.get("sim_threshold", 0.7),
        max_context_length=cfg.get("max_context_length", 4096),
    )


def _set_trainable_scope(model: nn.Module, scope: str) -> None:
    """Restrict second-stage FOCUS optimization to its learned context."""
    if scope == "all":
        return
    if scope != "soft_context":
        raise ValueError(
            "FOCUS trainable_scope must be 'all' or 'soft_context', "
            f"got {scope!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.prompt_learner.ctx.requires_grad_(True)


class FOCUSMethod(BaseMethod):
    """Adapt FOCUS dual-scale bags and CONCH soft prompts to the unified loop."""
    name = "focus"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.DUAL_SCALE_PATCH_BAG,
        swap_policy=SwapPolicy.ALLOWLIST, default_backbone="conch",
        supported_backbones=("conch",),
        required_capabilities=frozenset({Cap.SOFT_PROMPT}),
        rationale="FOCUS injects prompts into a CONCH-style text tower; its native input adapter may accept other patch widths.")

    def build_model(self) -> nn.Module:
        from .model import FOCUS
        config = _build_config(self.cfg)
        encoder = self.load_encoder(weights_path=self.cfg.get("conch_ckpt"))
        model = FOCUS(config, num_classes=self.cfg["n_classes"],
                      conch_model=encoder.raw_model)
        initial_checkpoint = self.cfg.get("initial_checkpoint")
        if initial_checkpoint:
            checkpoint_path = Path(initial_checkpoint).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"FOCUS initial checkpoint does not exist: {checkpoint_path}")
            try:
                state = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:  # torch < 2.0
                state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
            # Prompt buffers are part of the checkpoint. Retokenize only after
            # loading so the configured GEPA text replaces the old seed text
            # while the learned context vectors remain intact.
            model.set_class_prompts(config.text_prompt)

        _set_trainable_scope(
            model, self.cfg.get("trainable_scope", "all"))
        return model.to(self.device)

    def train_step(self, batch, model, optimizer, loss_fn):
        x_s, x_l, label = batch[0], batch[1], batch[-1]
        x_s = x_s.to(self.device)
        x_l = x_l.to(self.device)
        label = label.to(self.device)

        optimizer.zero_grad()
        out = model(x_s, x_l, label)
        logits = out[0] if isinstance(out, tuple) else out
        loss = (out[2] if isinstance(out, tuple) and len(out) > 2
                and torch.is_tensor(out[2]) else loss_fn(logits, label))
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        x_s, x_l, label = batch[0], batch[1], batch[-1]
        x_s = x_s.to(self.device)
        x_l = x_l.to(self.device)
        label = label.to(self.device)
        out = model(x_s, x_l, label)
        logits = out[0] if isinstance(out, tuple) else out
        loss = (out[2].item() if isinstance(out, tuple) and len(out) > 2
                and torch.is_tensor(out[2]) else
                (loss_fn(logits, label).item() if loss_fn is not None else 0.0))
        return {"loss": loss, "logits": logits, "label": label}
