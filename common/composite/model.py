"""The composite WSI/VLM model.

Wires together:

    Frozen backbone
        |
        v
    Selector pipeline      (zero or more stages, applied sequentially)
        |
        v
    Prompt bank             (one or more prompt modules, fused)
        |
        v
    Aggregator fusion       (one or more aggregators, fused at logit OR
                             vector level)
        |
        v
    Logits

Constructed from a YAML config; see `configs/composite/*.yaml` for
examples ranging from "vanilla baseline" to "kitchen sink".
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn

from common.composite.interfaces import PromptBank, PatchSelector
from common.composite.selectors import build_selector
from common.composite.prompts import build_prompt_module
from common.composite.prompts.fusion import PromptFusion
from common.composite.aggregators import build_aggregator
from common.composite.aggregators.fusion import LogitEnsemble, VectorFusion
from common.backbones import EncoderBundle


class CompositeModel(nn.Module):
    """Compose patch selectors, prompt modules, and slide aggregators.

    Args:
        cfg: Composite configuration containing class metadata and component
            registries.
        backbone: Native vision-language backbone used by prompt modules.
        tokenizer: Tokenizer paired with ``backbone``.
        info: Legacy backbone dimension metadata.
        encoder_bundle: Validated capability-aware bundle for components that
            use black-box encoder operations.

    The model accepts one slide at a time as ``[patches, dim]`` or
    ``[1, patches, dim]`` and returns class logits plus optional auxiliary loss
    inputs.
    """
    def __init__(self, cfg: Dict[str, Any], backbone: nn.Module,
                 tokenizer: Any, info: Any,
                 encoder_bundle: Optional[EncoderBundle] = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.info = info
        self.encoder_bundle = encoder_bundle
        self.n_classes = cfg["n_classes"]
        self.classnames = cfg["classnames"]

        if (encoder_bundle is not None and encoder_bundle.spec.shared_dim is not None
                and cfg.get("feature_dim", encoder_bundle.spec.shared_dim)
                != encoder_bundle.spec.shared_dim):
            raise ValueError(
                "Composite selectors compare patch and text vectors directly: "
                f"feature_dim={cfg.get('feature_dim')} but "
                f"{encoder_bundle.spec.name} shared_dim={encoder_bundle.spec.shared_dim}. "
                "Provide aligned features; no implicit projection is inserted.")

        # ---- 1. Selectors (stack) -------------------------------------
        self.selectors = nn.ModuleList(self._build_selectors(cfg))

        # ---- 2. Prompts (bank) ----------------------------------------
        self.prompt_fusion = self._build_prompt_bank(cfg, backbone, tokenizer)

        # ---- 3. Aggregators (fused) -----------------------------------
        in_dim = cfg.get("feature_dim", info.patch_dim)
        agg_cfg = cfg["aggregators"]
        self.fusion_mode = agg_cfg.get("fusion", "logit_ensemble")
        aggs = self._build_aggregators(agg_cfg, in_dim)
        if self.fusion_mode == "logit_ensemble":
            self.aggregator_block = LogitEnsemble(
                aggs, mode=agg_cfg.get("logit_mode", "mean"),
                n_classes=self.n_classes)
        elif self.fusion_mode == "vector_fusion":
            self.aggregator_block = VectorFusion(
                aggs, mode=agg_cfg.get("vector_mode", "concat"),
                n_classes=self.n_classes)
        else:
            raise KeyError(
                f"Unknown aggregator fusion '{self.fusion_mode}'. "
                f"Use 'logit_ensemble' or 'vector_fusion'.")

    # ------------------------------------------------------------------
    def _build_selectors(self, cfg) -> List[PatchSelector]:
        """Build the selector pipeline. Empty list = identity."""
        out = []
        for s in cfg.get("selectors", []):
            if not s.get("enabled", True):
                continue
            kw = {k: v for k, v in s.items() if k not in ("type", "enabled")}
            out.append(build_selector(s["type"], **kw))
        return out

    def _build_prompt_bank(self, cfg, backbone, tokenizer):
        modules = []
        prompts_cfg = cfg.get("prompts", {})
        for name, sub in prompts_cfg.items():
            if name == "fusion":
                continue
            if not isinstance(sub, dict) or not sub.get("enabled", False):
                continue
            kw = {k: v for k, v in sub.items() if k != "enabled"}
            kw.setdefault("classnames", self.classnames)
            kw.setdefault("backbone", backbone)
            kw.setdefault("tokenizer", tokenizer)
            if name in {"maple_graph", "cod_chain"}:
                kw.setdefault("encoder", self.encoder_bundle)
            modules.append(build_prompt_module(name, **kw))
        if not modules:
            # Default: plain CoOp on the slide classes.
            modules.append(build_prompt_module(
                "coop_flat",
                classnames=self.classnames,
                backbone=backbone, tokenizer=tokenizer))
        fmode = prompts_cfg.get("fusion", {}).get("mode", "average")
        return PromptFusion(modules, mode=fmode,
                            dim=cfg.get("feature_dim", self.info.patch_dim),
                            n_classes=self.n_classes)

    def _build_aggregators(self, agg_cfg, in_dim):
        aggs = []
        for name, sub in agg_cfg.items():
            if name in ("fusion", "logit_mode", "vector_mode"):
                continue
            if not isinstance(sub, dict) or not sub.get("enabled", False):
                continue
            kw = {k: v for k, v in sub.items() if k != "enabled"}
            kw.setdefault("in_dim", in_dim)
            kw.setdefault("n_classes", self.n_classes)
            aggs.append(build_aggregator(name, **kw))
        if not aggs:
            # default fallback: simple attention pool
            aggs.append(build_aggregator(
                "attn_pool", in_dim=in_dim, n_classes=self.n_classes))
        return aggs

    # ------------------------------------------------------------------
    def _apply_selectors(self, patches, text_features, coords=None):
        for sel in self.selectors:
            patches = sel(patches, text_features, coords)
        return patches

    def forward(self, patches: torch.Tensor,
                coords: torch.Tensor = None,
                return_extras: bool = False
                ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run the configured selector, prompt, and aggregation pipeline.

        Args:
            patches: One variable-length patch bag.
            coords: Optional patch coordinates aligned with ``patches``.
            return_extras: Include tensors used by optional composite losses.

        Returns:
            A pair containing class logits and an auxiliary tensor mapping.
        """
        # Strip batch dim if present (every method assumes one slide)
        if patches.dim() == 3:
            patches = patches.squeeze(0)
        if coords is not None and coords.dim() == 3:
            coords = coords.squeeze(0)

        # 1. prompt bank (don't depend on patches)
        bank: PromptBank = self.prompt_fusion()

        # 2. selectors (text-aware)
        patches = self._apply_selectors(patches, bank.text_features, coords)

        # 3. fused aggregators -> logits
        logits = self.aggregator_block(patches, bank)

        if not return_extras:
            return logits, {}

        # Extras for composite loss: SLIP cross-corr if available, MAPLE
        # attributes if available.
        extras: Dict[str, Any] = {}
        slip = next((a for a in self._iter_aggregators()
                     if a.__class__.__name__ == "SLIPRoutingAggregator"), None)
        if slip is not None and "tissue" in bank.aux:
            # cheap recomputation of cross_corr for the contrastive loss
            extras["slip_cross_corr"] = self._slip_cross_corr(slip, patches, bank)
            extras["slip_temperature"] = slip.temperature

        if "attributes" in bank.aux and "attribute_class_index" in bank.aux:
            # use the last vector aggregator's slide vector if available
            sv = self._slide_vector_for_aux(patches, bank)
            extras["maple_attributes"] = (
                sv, bank.aux["attributes"], bank.aux["attribute_class_index"])
        return logits, extras

    # ------------------------------------------------------------------
    def _iter_aggregators(self):
        block = self.aggregator_block
        return list(block.aggregators) if hasattr(block, "aggregators") else []

    def _slip_cross_corr(self, slip, patches, bank):
        """Recompute SLIP's (C, C) cross-correlation matrix for the loss."""
        import torch.nn.functional as F
        slide_features = F.normalize(patches, dim=1)
        slide_weights = F.normalize(bank.text_features, dim=1).T
        tissue = bank.aux["tissue"].to(patches.device)
        with torch.no_grad():
            s_pt = F.softmax(slide_features @ tissue.T / slip.temperature, dim=1)
            s_st = F.softmax(tissue @ slide_weights / slip.temperature, dim=0)
            s_attn = s_pt @ s_st
            class_vecs = F.normalize(slide_features.T @ s_attn, dim=0)
        return class_vecs.T @ slide_weights        # (C, C)

    def _slide_vector_for_aux(self, patches, bank):
        for a in self._iter_aggregators():
            try:
                return a.forward_vector(patches, bank)
            except NotImplementedError:
                continue
        return patches.mean(0)
