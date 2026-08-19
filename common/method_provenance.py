"""Authoritative implementation provenance for registered method adapters.

Encoder compatibility and prompt provenance do not establish that a local
training objective reproduces an upstream paper.  This registry keeps that
independent claim explicit and serializable in every generated result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MethodProvenance:
    implementation: str
    upstream_fidelity: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


METHOD_PROVENANCE: dict[str, MethodProvenance] = {
    "composite": MethodProvenance(
        "local", "local_baseline",
        "PGVL-Gym baseline; it does not claim to reproduce one upstream paper."),
    "focus": MethodProvenance(
        "vendored", "upstream",
        "Vendored FOCUS model using its actual single high-resolution bag."),
    "vila_mil": MethodProvenance(
        "vendored", "upstream", "Vendored ViLa-MIL architecture and objective."),
    "cod_mil": MethodProvenance(
        "vendored", "upstream",
        "Vendored CoD-MIL; prompt-asset provenance remains a separate axis."),
    "maple": MethodProvenance(
        "vendored_with_attribute_alignment_fix", "partial",
        "Vendored MAPLE architecture and objective with the released "
        "entity-major/class-major attribute reshape defect corrected."),
    "mscpt": MethodProvenance(
        "vendored_feature_only", "partial",
        "Generated runs use precomputed high-scale features and bypass the "
        "upstream selected-5x raw-image visual-prompt branch."),
    "pathpt": MethodProvenance(
        "vendored_model_simplified_objective", "partial",
        "Local training uses slide labels and mean patch probabilities rather "
        "than upstream patch selection, PatchSSLoss, and vote-based inference."),
    "top": MethodProvenance(
        "vendored", "upstream", "Vendored TOP architecture and pooling rule."),
    "slip": MethodProvenance(
        "mixed", "partial",
        "Mostly vendored SLIP with a local unified-training integration."),
    "wsi_five": MethodProvenance(
        "rebuilt_from_vendored_components", "partial",
        "Explicit simplified classname baseline; it does not use the upstream "
        "answer-bank objective or privileged per-slide report conditioning."),
    "muse": MethodProvenance(
        "reimplemented_feature_space", "partial",
        "Local feature-space reconstruction omits upstream stochastic "
        "multi-view semantic optimization."),
    "convlm": MethodProvenance(
        "reimplemented", "partial",
        "Local implementation consumes upstream-style offline features but is "
        "not vendored upstream training code."),
    "sldpc": MethodProvenance(
        "reimplemented", "partial",
        "CPI, DHNO, and SICL are local implementations of the published method."),
}


def method_provenance(
    method: str, cfg: Mapping[str, Any] | None = None,
) -> MethodProvenance:
    canonical = str(method).strip().lower().replace("-", "_")
    aliases = {
        "vila": "vila_mil", "vilamil": "vila_mil",
        "codmil": "cod_mil", "five": "wsi_five", "wsifive": "wsi_five",
    }
    canonical = aliases.get(canonical, canonical)
    cod_backbone = (
        str(cfg.get("backbone", "clip-rn50")).strip().lower().replace("_", "-")
        if cfg is not None else "clip-rn50")
    if canonical == "cod_mil" and cod_backbone not in {"clip-rn50", "rn50"}:
        return MethodProvenance(
            "vendored_width_parameterized_feature_space_extension", "partial",
            "Upstream released PLIP/QuiltNet RCC prompt artifacts, but its "
            "vendored model hardcodes the RN50 width. The local model "
            "parameterizes that width; prompt and patch spaces remain "
            "strictly paired and prompt provenance is reported separately.")
    if canonical == "pathpt" and cfg is not None \
            and cfg.get("training_mode") == "upstream_patch_ssl":
        task = str(cfg.get("task", "")).lower().replace("-", "_")
        if task in {"camelyon", "camelyon16", "cam16"}:
            return MethodProvenance(
                "vendored_native_objective_binary_adaptation", "partial",
                "Prompt selection, PatchSSLoss, and patch voting are restored, "
                "but CAMELYON normal-vs-tumour WSI classification adapts "
                "PathPT's tumour-subtyping label and voting semantics.")
        return MethodProvenance(
            "vendored_native_objective", "upstream",
            "Uses training-fold prompt selection, a synthetic Normal patch "
            "class, upstream PatchSSLoss, and Normal-excluding WSI voting; "
            "task prompt origin is recorded separately.")
    if canonical == "wsi_five" and cfg is not None \
            and cfg.get("training_mode") == "upstream_answer_bank":
        return MethodProvenance(
            "rebuilt_from_vendored_components_native_text_objective", "partial",
            "Restores training-fold answer candidates, question dropout, label "
            "hashing, and the released evaluation descriptions without "
            "test-time report conditioning. The orchestrator remains rebuilt, "
            "and feature/frame provenance is reported separately.")
    try:
        return METHOD_PROVENANCE[canonical]
    except KeyError as error:
        raise KeyError(f"No implementation provenance for method {method!r}") \
            from error
