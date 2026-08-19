"""MAPLE adapter (NeurIPS 2025).

The MAPLE class takes a single `args` namespace with attributes:
    base_model, text_path, attr_n_ctx, attr_edge_topk,
    all_ctx_trainable, csc, p_drop_out, p_bag_drop_out,
    weight, pos_ratio, neg_ratio
and loads CLIP / PLIP internally.
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
import torch.nn as nn

from methods.base import BaseMethod, probabilities_to_logits
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)


def _build_args(cfg):
    configured = str(cfg.get("backbone", "PLIP")).lower()
    base_model = "CLIP" if configured in {"clip", "clip-vitb", "hf-clip-vitb"} else "PLIP"
    return SimpleNamespace(
        base_model=base_model,
        text_path=cfg["text_prompt_path"],
        attr_n_ctx=cfg.get("n_ctx", 0),
        bagLevel_n_ctx=cfg.get("bag_n_ctx", cfg.get("n_ctx", 0)),
        attr_edge_topk=cfg.get("attr_edge_topk", 7),
        all_ctx_trainable=cfg.get("all_ctx_trainable", False),
        csc=cfg.get("csc", False),
        p_drop_out=cfg.get("p_drop_out", 0.0),
        p_bag_drop_out=cfg.get("p_bag_drop_out", 0.0),
        weight=cfg.get("entity_weight", 0.3),
        pos_ratio=cfg.get("pos_ratio", 0.8),
        neg_ratio=cfg.get("neg_ratio", 0.2),
    )


class MAPLEMethod(BaseMethod):
    """Adapt MAPLE's multiscale entity prompts and graph aggregation."""
    name = "maple"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.DUAL_SCALE_PATCH_BAG,
        swap_policy=SwapPolicy.ALLOWLIST, default_backbone="plip",
        supported_backbones=("plip", "hf-clip-vitb"),
        name_aliases={"clip": "hf-clip-vitb", "clip-vitb": "hf-clip-vitb"},
        feature_dims={"plip": (512,), "hf-clip-vitb": (512,)},
        required_capabilities=frozenset({Cap.SOFT_PROMPT, Cap.PAIRED_TILE_TEXT}),
        rationale="MAPLE traverses Hugging Face CLIP/PLIP text layers and expects projected 512-wide bags.")

    def build_model(self) -> nn.Module:
        from .maple_model.model import MAPLE
        from common.prompts.maple import load_maple_prompt_bank
        bank = load_maple_prompt_bank(
            self.cfg["text_prompt_path"],
            classnames=self.cfg["classnames"],
        )
        declared = self.cfg.get("prompt_provenance")
        if (declared and bank.provenance != "unknown"
                and declared != bank.provenance):
            raise ValueError(
                "MAPLE prompt_provenance contradicts the active bank: "
                f"declared {declared!r}, expected {bank.provenance!r}")
        args = _build_args(self.cfg)
        encoder = self.load_encoder(weights_path=self.cfg.get("backbone_weights"))
        model = MAPLE(args, clip_model=encoder.raw_model,
                      tokenizer=encoder.raw_tokenizer)
        return model.to(self.device)

    @staticmethod
    def _slide_bag(value: torch.Tensor) -> torch.Tensor:
        """Remove only the loader's singleton slide-batch dimension."""
        if value.ndim == 3 and value.shape[0] == 1:
            return value.squeeze(0)
        if value.ndim != 2:
            raise ValueError(
                "MAPLE requires batch_size=1 variable-length bags; got "
                f"{list(value.shape)}")
        return value

    def train_step(self, batch, model, optimizer, loss_fn):
        x_s, x_l, label = batch[0], batch[1], batch[-1]
        x_s = self._slide_bag(x_s.to(self.device))
        x_l = self._slide_bag(x_l.to(self.device))
        label = label.to(self.device)
        optimizer.zero_grad()
        out = model(x_s, None, x_l, None, label)
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
        out = model(x_s, None, x_l, None, label)
        logits = (probabilities_to_logits(out[0])
                  if isinstance(out, tuple) else out)
        loss = (out[2].item() if isinstance(out, tuple) and len(out) > 2
                and torch.is_tensor(out[2]) else
                (loss_fn(logits, label).item() if loss_fn is not None else 0.0))
        return {"loss": loss, "logits": logits, "label": label}
