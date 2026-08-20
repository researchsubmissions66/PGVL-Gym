#!/usr/bin/env python3
"""Generate audited ConVLM attributes and verify CoD-MIL prompt banks.

CoD-MIL prompt features are intentionally built by
``build_cod_mil_prompt_features.py`` from a declared source CSV.  This legacy
multi-asset utility no longer invents background prompts or silently preserves
an unverifiable tensor. ConVLM attributes are encoded from the selected,
provenance-checked JSON and saved with their prompt and encoder bindings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _encode(model: Any, tokenizer: Any, texts: list[str]) -> Any:
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        tokens = tokenizer(texts)
        return F.normalize(model.encode_text(tokens).float(), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--quilt-weights", type=Path,
        help="Required only when the protocol requests ConVLM attributes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn.functional as F
        import yaml
    except ModuleNotFoundError as error:
        raise SystemExit(
            "TCGA text-feature generation requires the PGVL-Gym core "
            f"environment (missing {error.name!r}).") from error
    protocol_path = args.protocol.expanduser().resolve()
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)

    attribute_requested = any(
        cfg.get("convlm_attribute_embeddings")
        for cfg in protocol.get("cohorts", {}).values())
    quilt = quilt_tokenizer = None
    if attribute_requested:
        if args.quilt_weights is None or not args.quilt_weights.is_file():
            raise FileNotFoundError(
                "--quilt-weights is required for ConVLM attribute generation")
        import open_clip
        quilt = open_clip.create_model(
            "ViT-B-32", pretrained=str(args.quilt_weights), device="cpu").eval()
        quilt_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    for cohort, cfg in protocol["cohorts"].items():
        if cfg.get("cod_prompt_features"):
            from common.prompts import (
                load_prompt_bank_csv,
                validate_prompt_feature_metadata,
            )

            if not cfg.get("cod_prompt_bank_csv"):
                raise ValueError(
                    f"{cohort}: cod_prompt_features requires "
                    "cod_prompt_bank_csv")
            cod_output = _repo_path(cfg["cod_prompt_features"])
            bank_path = _repo_path(cfg["cod_prompt_bank_csv"])
            if not cod_output.is_file():
                raise FileNotFoundError(
                    f"{cod_output} is missing; build it with "
                    "scripts/build_cod_mil_prompt_features.py")
            raw = torch.load(cod_output, map_location="cpu", weights_only=True)
            prompts = load_prompt_bank_csv(bank_path)
            validate_prompt_feature_metadata(
                raw, prompts=prompts, n_classes=len(cfg["classnames"]),
                source_path=bank_path, context=cod_output)
            print(f"verified {cod_output}: {len(prompts)} ordered prompts")

        if not cfg.get("convlm_attribute_embeddings"):
            continue
        from common.prompts import (
            ATTRIBUTE_EMBEDDING_SCHEMA,
            file_sha256,
            load_convlm_prompt_bank,
        )

        declared_prompts = cfg.get("prompts", {})
        prompt_value = (
            declared_prompts.get("convlm")
            if isinstance(declared_prompts, dict) else None)
        prompt_value = prompt_value or cfg.get("convlm_attribute_prompt_json")
        if not prompt_value:
            raise ValueError(
                f"{cohort}: ConVLM attribute generation requires an audited "
                "prompts.convlm or convlm_attribute_prompt_json bank")
        prompt_path = _repo_path(prompt_value)
        prompt_bank = load_convlm_prompt_bank(
            prompt_path, classnames=cfg["classnames"])
        attribute_space = "hf:wisdomik/QuiltNet-B-32"
        declared_space = cfg.get("convlm_attribute_feature_space_id")
        if declared_space is not None and declared_space != attribute_space:
            raise ValueError(
                f"{cohort}: --quilt-weights produces {attribute_space!r}, "
                f"not declared space {declared_space!r}")
        quilt_rows = []
        for descriptions in prompt_bank.prompts:
            encoded = _encode(quilt, quilt_tokenizer, list(descriptions))
            quilt_rows.append(F.normalize(encoded.mean(dim=0), dim=0))
        attributes = torch.stack(quilt_rows).cpu()
        attribute_output = _repo_path(cfg["convlm_attribute_embeddings"])
        attribute_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": ATTRIBUTE_EMBEDDING_SCHEMA,
            "embeddings": attributes,
            "feature_space_id": attribute_space,
            "classnames": list(cfg["classnames"]),
            "prompt_bank_sha256": prompt_bank.prompt_bank_sha256,
            "source_prompt_file_sha256": prompt_bank.file_sha256,
            "prompt_provenance": prompt_bank.provenance,
            "encoder": {
                "model_name": "ViT-B-32",
                "weights": str(args.quilt_weights.expanduser().resolve()),
                "feature_space_id": attribute_space,
                "checkpoint_sha256": file_sha256(args.quilt_weights),
            },
        }, attribute_output)
        print(
            f"wrote {attribute_output}: {tuple(attributes.shape)}, "
            f"source={prompt_path}, provenance={prompt_bank.provenance}")


if __name__ == "__main__":
    main()
