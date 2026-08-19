"""Integrity helpers for CoD-MIL's ordered prompt-feature banks.

CoD-MIL assigns meaning by row position: the first ``C`` rows are low-power
class prompts, the next ``C`` are high-power class prompts, and the remaining
rows are normal-tissue contrasts.  Shape checks alone therefore cannot detect
an inserted or reordered prompt.  The helpers here bind a tensor to the exact
CSV rows that produced it.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROMPT_FEATURE_SCHEMA = "pgvl.cod_mil.prompt_features.v1"
MIN_BACKGROUND_PROMPTS = 15


def load_prompt_bank_csv(path: str | Path) -> list[str]:
    """Read a one-prompt-per-row CSV without normalizing published text."""
    source = Path(path)
    prompts: list[str] = []
    with source.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 1 or not row[0].strip():
                raise ValueError(
                    f"{source}: row {line_number} must contain exactly one "
                    "non-empty prompt")
            prompts.append(row[0])
    if not prompts:
        raise ValueError(f"{source}: prompt bank is empty")
    return prompts


def prompt_bank_sha256(prompts: Sequence[str]) -> str:
    """Hash an ordered prompt list using a stable, unambiguous encoding."""
    encoded = json.dumps(
        list(prompts), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a stable SHA-256 for a file or checkpoint directory."""
    source = Path(path)
    digest = hashlib.sha256()

    def update_file(item: Path) -> None:
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)

    if source.is_dir():
        files = sorted(item for item in source.rglob("*") if item.is_file())
        if not files:
            raise ValueError(f"checkpoint directory is empty: {source}")
        for item in files:
            digest.update(item.relative_to(source).as_posix().encode("utf-8"))
            digest.update(b"\0")
            update_file(item)
    else:
        update_file(source)
    return digest.hexdigest()


def expected_row_roles(n_classes: int, n_prompts: int) -> dict[str, list[int]]:
    """Return half-open row spans used by CoD-MIL's positional slicing."""
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    if n_prompts <= 2 * n_classes:
        raise ValueError(
            "CoD-MIL needs low- and high-power class rows followed by at "
            "least one background row")
    return {
        "low": [0, n_classes],
        "high": [n_classes, 2 * n_classes],
        "background": [2 * n_classes, n_prompts],
    }


def prompt_feature_metadata(
    prompts: Sequence[str], *, n_classes: int, source_path: str | Path,
    feature_space_id: str, encoder: str, checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build the metadata embedded beside a derived prompt tensor."""
    prompt_list = list(prompts)
    return {
        "schema": PROMPT_FEATURE_SCHEMA,
        "feature_space_id": feature_space_id,
        "encoder": encoder,
        "encoder_checkpoint_sha256": checkpoint_sha256,
        "source_prompt_path": str(source_path),
        "source_prompt_sha256": file_sha256(source_path),
        "prompt_sha256": prompt_bank_sha256(prompt_list),
        "prompts": prompt_list,
        "row_roles": expected_row_roles(n_classes, len(prompt_list)),
        "normalized": True,
        "provenance": "derived",
    }


def validate_prompt_feature_metadata(
    payload: Any, *, prompts: Sequence[str], n_classes: int,
    source_path: str | Path, context: str | Path,
) -> Any:
    """Validate provenance/order metadata and return its embedding payload.

    Bare tensors are intentionally rejected.  They can be dimensionally valid
    while their positional semantics are wrong, which is exactly the defect in
    the released RCC CLIP-RN50 asset.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"{context}: unverified legacy CoD-MIL prompt tensor; expected a "
            f"{PROMPT_FEATURE_SCHEMA!r} mapping bound to its source CSV")
    if payload.get("schema") != PROMPT_FEATURE_SCHEMA:
        raise ValueError(
            f"{context}: unsupported or missing CoD-MIL prompt schema "
            f"{payload.get('schema')!r}")
    feature_space = payload.get("feature_space_id")
    if not isinstance(feature_space, str) or not feature_space.strip():
        raise ValueError(f"{context}: prompt feature_space_id is missing")
    encoder = payload.get("encoder")
    if not isinstance(encoder, str) or not encoder.strip():
        raise ValueError(f"{context}: prompt encoder identity is missing")
    checkpoint_digest = payload.get("encoder_checkpoint_sha256")
    if (not isinstance(checkpoint_digest, str)
            or len(checkpoint_digest) != 64
            or any(character not in "0123456789abcdef"
                   for character in checkpoint_digest.lower())):
        raise ValueError(
            f"{context}: encoder_checkpoint_sha256 is missing or invalid")
    if payload.get("provenance") != "derived":
        raise ValueError(
            f"{context}: verified prompt features must declare derived provenance")

    expected_prompts = list(prompts)
    minimum_rows = 2 * n_classes + MIN_BACKGROUND_PROMPTS
    if len(expected_prompts) < minimum_rows:
        raise ValueError(
            f"{source_path}: CoD-MIL needs {2 * n_classes} class prompts plus "
            f"at least {MIN_BACKGROUND_PROMPTS} normal-tissue prompts; found "
            f"{len(expected_prompts)} total rows")
    embedded_prompts = payload.get("prompts")
    if embedded_prompts != expected_prompts:
        raise ValueError(
            f"{context}: embedded prompt order does not exactly match "
            f"source bank {source_path}")
    expected_digest = prompt_bank_sha256(expected_prompts)
    if payload.get("prompt_sha256") != expected_digest:
        raise ValueError(
            f"{context}: prompt_sha256 does not match the ordered source bank")
    expected_source_digest = file_sha256(source_path)
    if payload.get("source_prompt_sha256") != expected_source_digest:
        raise ValueError(
            f"{context}: source_prompt_sha256 does not match {source_path}")
    roles = expected_row_roles(n_classes, len(expected_prompts))
    if payload.get("row_roles") != roles:
        raise ValueError(
            f"{context}: row_roles must be {roles}, got "
            f"{payload.get('row_roles')!r}")
    if payload.get("normalized") is not True:
        raise ValueError(f"{context}: CoD-MIL prompt embeddings must be normalized")

    embeddings = payload.get("embeddings")
    if embeddings is None:
        raise ValueError(f"{context}: prompt payload has no 'embeddings' tensor")
    return embeddings
