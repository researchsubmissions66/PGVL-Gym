#!/usr/bin/env python3
"""Encode an ordered CoD-MIL prompt CSV with verifiable row metadata."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.prompts import (  # noqa: E402
    file_sha256,
    load_prompt_bank_csv,
    prompt_feature_metadata,
)


DEFAULT_SOURCE = REPO_ROOT / "text_prompts/cod_mil/rcc_chain_of_diagnosis.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "text_prompts/cod_mil"


def _default_weights(encoder: str) -> Path | None:
    env_key = encoder.upper().replace("-", "_") + "_CKPT"
    configured = os.environ.get(env_key)
    if configured:
        return Path(configured).expanduser()
    if encoder == "clip-rn50":
        return Path.home() / ".cache/clip/RN50.pt"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--encoder", choices=("clip-rn50", "plip", "quiltnet"),
                        default="clip-rn50")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument(
        "--expected-rows", type=int, default=27,
        help="Fail if the source bank changed unexpectedly (0 disables).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.prompt_csv.expanduser().resolve()
    weights_value = args.weights or _default_weights(args.encoder)
    if weights_value is None:
        env_key = args.encoder.upper().replace("-", "_") + "_CKPT"
        raise ValueError(
            f"--weights or {env_key} is required for {args.encoder}")
    weights = weights_value.expanduser().resolve()
    output_value = args.output or (
        DEFAULT_OUTPUT_DIR
        / f"rcc_text_prompt_features_{args.encoder.replace('-', '_')}_verified.pt"
    )
    output = output_value.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"prompt CSV does not exist: {source}")
    if not weights.exists() or (not weights.is_file() and not weights.is_dir()):
        raise FileNotFoundError(f"encoder checkpoint does not exist: {weights}")

    prompts = load_prompt_bank_csv(source)
    if args.expected_rows and len(prompts) != args.expected_rows:
        raise ValueError(
            f"{source}: expected {args.expected_rows} rows, found {len(prompts)}")

    try:
        import torch
        from common.backbones import build_encoder
    except ModuleNotFoundError as error:
        raise SystemExit(
            "CoD-MIL prompt generation requires the PGVL-Gym core "
            f"environment (missing {error.name!r}).") from error
    bundle = build_encoder(
        args.encoder, weights_path=str(weights), device="cpu").freeze()
    with torch.inference_mode():
        embeddings = bundle.encode_text(prompts, normalize=True).cpu().half()
    expected_dim = bundle.spec.shared_dim
    if expected_dim is None or embeddings.shape != (len(prompts), expected_dim):
        raise ValueError(
            f"unexpected {args.encoder} tensor shape {tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{args.encoder} produced non-finite prompt embeddings")
    norms = embeddings.float().norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=0):
        raise ValueError(f"{args.encoder} prompt embeddings are not normalized")

    try:
        source_label = str(source.relative_to(REPO_ROOT))
    except ValueError:
        source_label = str(source)
    payload = {
        "embeddings": embeddings,
        **prompt_feature_metadata(
            prompts,
            n_classes=args.n_classes,
            source_path=source,
            feature_space_id=bundle.spec.feature_space_id,
            encoder=bundle.spec.name,
            checkpoint_sha256=file_sha256(weights),
        ),
        "source_prompt_path": source_label,
        "derivation": (
            f"Exact source CSV rows encoded in order with the {args.encoder} "
            "text tower; no prompt insertion, rewrite, or truncation."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"wrote {output} shape={tuple(embeddings.shape)} "
        f"source_sha256={payload['source_prompt_sha256']} "
        f"checkpoint_sha256={payload['encoder_checkpoint_sha256']}")


if __name__ == "__main__":
    main()
