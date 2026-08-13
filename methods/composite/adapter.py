"""Composite method adapter.

Builds a `CompositeModel` from a single YAML config and plugs into the
unified `train.py` like any other method.

Use it via:
    python train.py --method composite --config configs/composite/<name>.yaml
"""
from __future__ import annotations
import torch

from methods.base import BaseMethod
from common.composite import CompositeModel
from common.composite.recipes import build_recipe
from common.composite.losses import composite_loss
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy,
    get_info)


class CompositeMethod(BaseMethod):
    """Adapt the configurable selector/prompt/aggregator composition model."""
    name = "composite"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.COMPOSITE,
        swap_policy=SwapPolicy.CAPABILITY, default_backbone="clip-vitb",
        required_capabilities=frozenset({Cap.TEXT_ENCODE, Cap.PAIRED_TILE_TEXT}),
        rationale="Enabled prompt modules may impose additional soft-prompt hooks.")

    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        self._recipe = None

    # --------------------------------------------------------------
    def build_model(self):
        encoder = self.load_encoder().freeze()
        backbone, tokenizer = encoder.raw_model, encoder.raw_tokenizer
        info = get_info(encoder.spec.name)

        self.cfg.setdefault("feature_dim", info.patch_dim)
        model = CompositeModel(self.cfg, backbone, tokenizer, info,
                               encoder_bundle=encoder)
        return model.to(self.device)

    def build_optimizer(self, model):
        rcfg = self.cfg.get("recipe", {"type": "focus"})
        kw = {k: v for k, v in rcfg.items() if k != "type"}
        self._recipe = build_recipe(rcfg.get("type", "focus"), **kw)
        return self._recipe.build_optimizer(model)

    def build_scheduler(self, optimizer):
        if self._recipe is None:
            return None
        return self._recipe.build_scheduler(optimizer)

    # --------------------------------------------------------------
    def _loss_kwargs(self):
        loss_cfg = self.cfg.get("recipe", {}).get("loss", {})
        return {
            "ce_weight":               loss_cfg.get("ce_weight", 1.0),
            "slip_contrastive_weight": loss_cfg.get("slip_contrastive_weight", 0.0),
            "aux_attribute_weight":    loss_cfg.get("aux_attribute_weight", 0.0),
        }

    def _unpack_batch(self, batch):
        # Common shapes: (feats, label) or (feats, coords, label) or (lo, hi, label)
        if len(batch) == 2:
            feats, label = batch
            coords = None
        elif (len(batch) == 3 and torch.is_tensor(batch[1])
              and batch[1].dim() == 3 and batch[1].size(-1) == 2):
            feats, coords, label = batch
        else:
            # lo, hi, label  ->  use lo as the patch input, ignore hi
            feats = batch[0]
            label = batch[-1]
            coords = None
        return (feats.to(self.device),
                coords.to(self.device) if coords is not None else None,
                label.to(self.device))

    def train_step(self, batch, model, optimizer, loss_fn):
        feats, coords, label = self._unpack_batch(batch)
        optimizer.zero_grad()
        logits, extras = model(feats, coords=coords, return_extras=True)
        kwargs = self._loss_kwargs()
        if "slip_cross_corr" in extras:
            kwargs["slip_extras"] = (extras["slip_cross_corr"],
                                     extras.get("slip_temperature", 0.01))
        if "maple_attributes" in extras:
            kwargs["maple_extras"] = extras["maple_attributes"]
        loss = composite_loss(logits, label, **kwargs)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(),
                "logits": logits.detach().unsqueeze(0)
                          if logits.dim() == 1 else logits.detach(),
                "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        feats, coords, label = self._unpack_batch(batch)
        logits, _ = model(feats, coords=coords, return_extras=False)
        loss = 0.0
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return {"loss": loss, "logits": logits, "label": label}
