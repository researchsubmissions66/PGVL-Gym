"""CoD-MIL adapter (TMI 2024).

The original `CoT` class needs `(x_s, coord_s, x_l, coords_l,
patch_label, label, text_prompt_feature, slide_id)`. The
`text_prompt_feature` is precomputed CLIP text embeddings of the
chain-of-diagnosis prompts; we cache them on first forward pass.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import FeatureLevel, MethodBackboneContract, SwapPolicy


class CoDMILMethod(BaseMethod):
    """Adapt CoD-MIL's precomputed prompts and cross-scale correspondence."""
    name = "cod_mil"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.DUAL_SCALE_PATCH_BAG,
        swap_policy=SwapPolicy.PRECOMPUTED, default_backbone="clip-rn50",
        supported_backbones=("clip-rn50",), feature_dims={"clip-rn50": (1024,)},
        required_capabilities=frozenset(),
        rationale="CoD-MIL consumes aligned 1024-wide patch and precomputed prompt tensors; it has no runtime encoder.")

    def build_model(self) -> nn.Module:
        from .model import CoT
        config = SimpleNamespace(
            input_size=self.cfg.get("feature_dim", 1024),
            hidden_size=self.cfg.get("hidden_size", 192),
        )
        model = CoT(config, num_classes=self.cfg["n_classes"])
        return model.to(self.device)

    def _prepare_text_features(self):
        """Load the text embeddings matching the patch-feature backbone."""
        if hasattr(self, "_txt_feats"):
            return self._txt_feats

        path = self.cfg.get("text_prompt_features")
        if path:
            payload = torch.load(
                path, map_location=self.device, weights_only=True)
        elif self.cfg.get("prompt_encoding") == "runtime_cached":
            chain_path = Path(self.cfg["text_prompt_path"])
            with chain_path.open(encoding="utf-8") as handle:
                chain = json.load(handle)
            classnames = list(self.cfg["classnames"])
            low = [chain[name]["broad"][0] for name in classnames]
            high = [chain[name]["specific"][0] for name in classnames]
            background = [
                f"non-diagnostic background tissue adjacent to {name}"
                for name in classnames
            ]
            prompts = (
                low + high + background
                + ["non-neoplastic background tissue"])
            bundle = self.load_encoder(
                weights_path=self.cfg.get("backbone_weights"))
            bundle.freeze()
            payload = bundle.encode_text(prompts, normalize=True).detach()
        else:
            raise KeyError(
                "CoD-MIL requires text_prompt_features or "
                "prompt_encoding=runtime_cached with text_prompt_path.")
        embedded_space = None
        if isinstance(payload, dict):
            embedded_space = payload.get("feature_space_id")
            payload = payload.get("embeddings", payload.get("features"))
        if not isinstance(payload, torch.Tensor) or payload.ndim != 2:
            raise ValueError(
                "CoD-MIL text_prompt_features must be a rank-2 tensor or a "
                "dict containing 'embeddings'.")
        expected_dim = self.cfg.get("feature_dim", 1024)
        if payload.shape[-1] != expected_dim:
            raise ValueError(
                f"CoD-MIL prompt width {payload.shape[-1]} does not match "
                f"patch feature_dim {expected_dim}; no alignment layer is inserted.")
        configured_space = self.cfg.get("feature_space_id")
        text_space = self.cfg.get("text_feature_space_id", embedded_space)
        if configured_space and text_space and configured_space != text_space:
            raise ValueError(
                f"CoD-MIL patch space '{configured_space}' and prompt space "
                f"'{text_space}' differ.")
        self._txt_feats = payload.to(self.device)
        return self._txt_feats

    def train_step(self, batch, model, optimizer, loss_fn):
        x_s, x_l, cross_map, label = batch
        x_s = x_s.to(self.device)
        x_l = x_l.to(self.device)
        cross_map = cross_map.to(self.device)
        label = label.to(self.device)
        patch_label = torch.zeros(x_s.shape[0], dtype=torch.long, device=self.device)
        text_features = self._prepare_text_features()

        optimizer.zero_grad()
        out = model(x_s, None, x_l, None, patch_label, label,
                    text_features, cross_map)
        logits = out[0] if isinstance(out, tuple) else out
        # The original CoD-MIL objective combines slide classification with
        # its chain-of-diagnosis masking loss and is returned as the last
        # output.  Preserve it when the vendored model exposes it.
        loss = out[-1] if isinstance(out, tuple) and torch.is_tensor(out[-1]) \
            and out[-1].ndim == 0 else loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        x_s, x_l, cross_map, label = batch
        x_s = x_s.to(self.device)
        x_l = x_l.to(self.device)
        cross_map = cross_map.to(self.device)
        label = label.to(self.device)
        patch_label = torch.zeros(x_s.shape[0], dtype=torch.long, device=self.device)
        text_features = self._prepare_text_features()
        out = model(x_s, None, x_l, None, patch_label, label,
                    text_features, cross_map)
        logits = out[0] if isinstance(out, tuple) else out
        if isinstance(out, tuple) and torch.is_tensor(out[-1]) and out[-1].ndim == 0:
            loss = out[-1].item()
        else:
            loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": label}
