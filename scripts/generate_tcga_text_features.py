#!/usr/bin/env python3
"""Generate deterministic TCGA text tensors from locally cached encoders."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
PROTOCOL = REPO_ROOT / "benchmarks" / "tcga" / "protocol.yaml"
RN50_WEIGHTS = Path("/path/to/model-cache/clip/RN50.pt")
QUILT_WEIGHTS = Path(
    "/path/to/model-cache/huggingface/hub/"
    "models--wisdomik--QuiltNet-B-32/snapshots/"
    "8ce77289ce35a90b2f1db1137dfa4bc2df175e33/"
    "open_clip_pytorch_model.bin"
)


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _encode(model, tokenizer, texts: list[str]) -> torch.Tensor:
    with torch.inference_mode():
        tokens = tokenizer(texts)
        return F.normalize(model.encode_text(tokens).float(), dim=-1)


def main() -> None:
    if not RN50_WEIGHTS.is_file() or not QUILT_WEIGHTS.is_file():
        raise FileNotFoundError("The cached CLIP-RN50 and QuiltNet checkpoints are required")
    with PROTOCOL.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)

    from common.backbones import build_encoder
    rn50 = build_encoder(
        "clip-rn50", weights_path=str(RN50_WEIGHTS), device="cpu")

    import open_clip
    quilt = open_clip.create_model(
        "ViT-B-32", pretrained=str(QUILT_WEIGHTS), device="cpu").eval()
    quilt_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    for cohort, cfg in protocol["cohorts"].items():
        cod_output = _repo_path(cfg["cod_prompt_features"])
        # Preserve the imported upstream RCC tensor. Generate only missing
        # task-specific tensors using the same declared CLIP-RN50 space.
        if not cod_output.is_file():
            with _repo_path(cfg["cod_prompt_json"]).open(encoding="utf-8") as handle:
                chain = json.load(handle)
            classnames = list(cfg["classnames"])
            low = [chain[name]["broad"][0] for name in classnames]
            high = [chain[name]["specific"][0] for name in classnames]
            background = [
                f"non-diagnostic background tissue adjacent to {name}"
                for name in classnames
            ]
            prompts = low + high + background + ["non-neoplastic background tissue"]
            embeddings = rn50.encode_text(prompts, normalize=True).cpu()
            cod_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embeddings": embeddings,
                "feature_space_id": "openai/clip-rn50@official",
                "prompt_order": {
                    "low": low, "high": high,
                    "background": background,
                    "sentinel": prompts[-1],
                },
            }, cod_output)
            print(f"wrote {cod_output}: {tuple(embeddings.shape)}")

        focus = pd.read_csv(_repo_path(cfg["focus_prompt_csv"]))
        quilt_rows = []
        for index, classname in enumerate(cfg["classnames"]):
            descriptions = [
                f"a histopathology image of {classname}",
                str(focus.iloc[index]["low_res_prompt"]),
                str(focus.iloc[index]["high_res_prompt"]),
            ]
            encoded = _encode(quilt, quilt_tokenizer, descriptions)
            quilt_rows.append(F.normalize(encoded.mean(dim=0), dim=0))
        attributes = torch.stack(quilt_rows).cpu()
        attribute_output = _repo_path(cfg["convlm_attribute_embeddings"])
        attribute_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embeddings": attributes,
            "feature_space_id": "hf:wisdomik/QuiltNet-B-32",
            "classnames": list(cfg["classnames"]),
        }, attribute_output)
        print(f"wrote {attribute_output}: {tuple(attributes.shape)}")


if __name__ == "__main__":
    main()
