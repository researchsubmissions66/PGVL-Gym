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
from common.prompts import (
    load_prompt_bank_csv,
    validate_prompt_feature_metadata,
)


class CoDMILMethod(BaseMethod):
    """Adapt CoD-MIL's precomputed prompts and cross-scale correspondence."""
    name = "cod_mil"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.DUAL_SCALE_PATCH_BAG,
        swap_policy=SwapPolicy.PRECOMPUTED, default_backbone="clip-rn50",
        supported_backbones=("clip-rn50", "plip", "quiltnet"),
        feature_dims={
            "clip-rn50": (1024,), "plip": (512,), "quiltnet": (512,),
        },
        require_feature_space=True,
        required_capabilities=frozenset(),
        rationale=(
            "CoD-MIL consumes aligned dual-scale patch and prompt tensors. "
            "Upstream released CLIP-RN50, PLIP, and QuiltNet prompt banks; "
            "every selected bank must match its patch feature space."))

    _GENERIC_BACKGROUND = Path(
        "text_prompts/cod_mil/background_tissue_generic.json")

    def _background_prompts(self) -> list[str]:
        """Return the normal-tissue bank the auxiliary contrastive branch needs.

        Cohort-specific normal structures come first, mirroring the released
        bank's ordering (organ anatomy, then organ-independent phenotypes).
        """
        prompts: list[str] = []
        for key in ("normal_structures_json", None):
            source = self.cfg.get(key) if key else self._GENERIC_BACKGROUND
            if not source:
                continue
            path = Path(source)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            values = payload.get("prompts") if isinstance(payload, dict) else payload
            prompts.extend(str(item) for item in values if str(item).strip())
        return prompts

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
            raw_payload = torch.load(
                path, map_location=self.device, weights_only=True)
            source = self.cfg.get("text_prompt_bank_csv")
            if not source:
                raise ValueError(
                    "Precomputed CoD-MIL prompts require "
                    "'text_prompt_bank_csv' so positional rows can be verified.")
            prompts = load_prompt_bank_csv(source)
            payload = validate_prompt_feature_metadata(
                raw_payload,
                prompts=prompts,
                n_classes=int(self.cfg["n_classes"]),
                source_path=source,
                context=path,
            )
            embedded_space = raw_payload.get("feature_space_id")
        elif self.cfg.get("prompt_encoding") == "runtime_cached":
            chain_path = Path(self.cfg["text_prompt_path"])
            with chain_path.open(encoding="utf-8") as handle:
                chain = json.load(handle)
            # Underscore keys carry provenance, not classes.
            chain = {k: v for k, v in chain.items() if not str(k).startswith("_")}
            classnames = list(self.cfg["classnames"])
            low = [chain[name]["broad"][0] for name in classnames]
            high = [chain[name]["specific"][0] for name in classnames]
            # When a complete bank CSV exists, it is canonical and can be
            # encoded by any supported paired text tower. Other cohorts compile
            # the same layout from their chain plus normal-tissue assets.
            bank_source = self.cfg.get("text_prompt_bank_csv")
            if bank_source:
                prompts = load_prompt_bank_csv(bank_source)
                class_rows = low + high
                if prompts[:len(class_rows)] != class_rows:
                    raise ValueError(
                        "CoD-MIL source bank's leading rows do not match its "
                        "configured low/high diagnosis chain.")
            else:
                prompts = []
            # CoD-MIL's auxiliary branch pushes non-diagnostic patches toward a
            # bank of *normal tissue phenotypes*: the released kidney bank is
            # 3 low + 3 high + 21 background. Templated strings naming the
            # tumour ("background adjacent to <class>") are not a substitute --
            # they carry the diagnosis the branch is meant to contrast against.
            # The 15 organ-independent rows are reused verbatim; a cohort may
            # add its own normal structures via `normal_structures_json`.
            if not prompts:
                background = self._background_prompts()
                if not background:
                    raise ValueError(
                        "CoD-MIL needs a normal-tissue background bank. Expected "
                        "text_prompts/cod_mil/background_tissue_generic.json or a "
                        "'normal_structures_json' entry in the config.")
                prompts = low + high + background
            bundle = self.load_encoder(
                weights_path=self.cfg.get("backbone_weights"))
            bundle.freeze()
            payload = bundle.encode_text(prompts, normalize=True).detach()
            embedded_space = getattr(
                getattr(bundle, "spec", None), "feature_space_id", None)
        else:
            raise KeyError(
                "CoD-MIL requires text_prompt_features or "
                "prompt_encoding=runtime_cached with text_prompt_path.")
        if not isinstance(payload, torch.Tensor) or payload.ndim != 2:
            raise ValueError(
                "CoD-MIL text_prompt_features embeddings must be a rank-2 tensor.")
        if path and payload.shape[0] != len(prompts):
            raise ValueError(
                f"CoD-MIL prompt tensor has {payload.shape[0]} rows but its "
                f"verified source bank has {len(prompts)} prompts.")
        if not torch.isfinite(payload).all():
            raise ValueError("CoD-MIL prompt embeddings contain non-finite values.")
        expected_dim = self.cfg.get("feature_dim", 1024)
        if payload.shape[-1] != expected_dim:
            raise ValueError(
                f"CoD-MIL prompt width {payload.shape[-1]} does not match "
                f"patch feature_dim {expected_dim}; no alignment layer is inserted.")
        configured_space = self.cfg.get("feature_space_id")
        declared_text_space = self.cfg.get("text_feature_space_id")
        if (declared_text_space and embedded_space
                and declared_text_space != embedded_space):
            raise ValueError(
                f"CoD-MIL configured prompt space '{declared_text_space}' "
                f"does not match embedded space '{embedded_space}'.")
        text_space = embedded_space or declared_text_space
        if configured_space and text_space and configured_space != text_space:
            raise ValueError(
                f"CoD-MIL patch space '{configured_space}' and prompt space "
                f"'{text_space}' differ.")
        self._txt_feats = payload.to(self.device)
        return self._txt_feats

    def train_step(self, batch, model, optimizer, loss_fn):
        x_s, x_l, cross_map, label = batch[0], batch[1], batch[2], batch[-1]
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
        x_s, x_l, cross_map, label = batch[0], batch[1], batch[2], batch[-1]
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
