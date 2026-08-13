"""MSCPT adapter (TMI 2025).

Class names in the vendored repo:
    methods/mscpt/mscpt_model/mscpt.py        :: Mscpt        (CLIP / PLIP)
    methods/mscpt/mscpt_model/mscpt_conch.py  :: MscptConch   (CONCH)

The original code uses PyTorch-Lightning. We bypass Lightning here and
drive the model from our unified loop.
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)


def _build_args(cfg):
    return SimpleNamespace(
        n_classes=cfg["n_classes"],
        n_tpro=cfg.get("n_tpro", 2),
        n_vpro=cfg.get("n_vpro", 2),
        n_set=cfg.get("n_set", 5),
        base_model=cfg.get("backbone", "plip"),
        dataset_name=cfg.get("dataset_name", "UBC-OCEAN"),
        gpt_dir=cfg.get("gpt_dir", "./train_data/gpt"),
        num_k=cfg.get("num_k", 100),
        target_size=cfg.get("target_size", 224),
    )


class MSCPTMethod(BaseMethod):
    """Adapt MSCPT's paired deep text/vision prompting implementation."""
    name = "mscpt"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.DUAL_SCALE_PATCH_BAG,
        swap_policy=SwapPolicy.ALLOWLIST, default_backbone="plip",
        supported_backbones=("plip", "hf-clip-vitb", "conch"),
        name_aliases={"clip": "hf-clip-vitb", "clip-vitb": "hf-clip-vitb"},
        feature_dims={"plip": (512,), "hf-clip-vitb": (512,), "conch": (512,)},
        required_capabilities=frozenset({Cap.DEEP_TEXT_PROMPT, Cap.DEEP_VISION_PROMPT,
                                        Cap.PAIRED_TILE_TEXT}),
        rationale="MSCPT needs paired deep prompt hooks in both text and tile transformers.")

    def build_model(self) -> nn.Module:
        backbone = self.backbone_name
        encoder = self.load_encoder(weights_path=self.cfg.get("backbone_weights"))
        if backbone == "conch":
            from .mscpt_model.mscpt_conch import MscptConch as _Cls
            native_name = "conch"
        else:
            from .mscpt_model.mscpt import Mscpt as _Cls
            native_name = "clip" if backbone == "hf-clip-vitb" else "plip"
        # Native MSCPT keys its released GPT JSON by dataset label code.
        label_dicts = self.cfg["label_dict"]
        model = _Cls(
            base_model=native_name,
            base_pretrain_path=self.cfg.get("backbone_weights", ""),
            trainer_perc=self.cfg.get("precision", "fp16"),
            dataset_name=self.cfg.get("dataset_name", "RCC"),
            gpt_dir=self.cfg.get("gpt_dir", "./train_data/gpt"),
            label_dicts=label_dicts, n_set=self.cfg.get("n_set", 5),
            n_tpro=self.cfg.get("n_tpro", 2), n_vpro=self.cfg.get("n_vpro", 2),
            n_high=self.cfg.get("n_high", 10), n_topk=self.cfg.get("num_k", 5),
            clip_model=encoder.raw_model, tokenizer=encoder.raw_tokenizer)
        if self.cfg.get("input_mode") == "precomputed_shared_features":
            # The benchmark supplies both scales after the paired visual
            # projection. The native raw-image vision prompt branch is not in
            # this computation graph and must not appear as trainable.
            for module_name in ("vision_prompt_learner", "image_encoder"):
                module = getattr(model.Custom_model, module_name)
                module.requires_grad_(False)
        return model.to(self.device)

    def train_step(self, batch, model, optimizer, loss_fn):
        # MSCPT batches: (lo_feats, hi_feats, label) where lo/hi are the
        # two scales; some configs also pass selected-5x patches.
        feats_lo = batch[0].to(self.device)
        feats_hi = batch[1].to(self.device) if len(batch) > 2 else feats_lo
        label = batch[-1].to(self.device)

        optimizer.zero_grad()
        out = model((feats_hi, feats_lo), train=True)
        logits = out[0] if isinstance(out, tuple) else out
        loss = loss_fn(logits, label)
        if isinstance(out, tuple):
            loss = sum(loss_fn(branch, label) for branch in out[:3])
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        feats_lo = batch[0].to(self.device)
        feats_hi = batch[1].to(self.device) if len(batch) > 2 else feats_lo
        label = batch[-1].to(self.device)
        out = model((feats_hi, feats_lo), train=False)
        logits = out[0] if isinstance(out, tuple) else out
        loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": label}
