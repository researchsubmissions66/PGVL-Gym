"""Strict prompt loading for ViLa-MIL's native dual-scale text banks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .positional import (
    POSITIONAL_PROMPT_FORMAT,
    PositionalPromptBank,
    load_positional_prompt_bank,
    positional_prompt_bank_sha256,
)


VILA_PROMPT_FORMAT = POSITIONAL_PROMPT_FORMAT
ViLAPromptBank = PositionalPromptBank


def vila_prompt_bank_sha256(
    class_names: Sequence[str],
    low_resolution: Sequence[str],
    high_resolution: Sequence[str],
) -> str:
    return positional_prompt_bank_sha256(
        class_names, low_resolution, high_resolution, consumer="ViLa-MIL")


def load_vila_prompt_bank(
    path: str | Path,
    *,
    class_names: Sequence[str],
    file_class_names: Sequence[str] | None = None,
    record: Mapping[str, Any] | None = None,
    expected_provenance: str | None = None,
    expected_file_sha256: str | None = None,
    expected_ordered_prompt_bank_sha256: str | None = None,
) -> ViLAPromptBank:
    """Load and bind a native headerless ViLa-MIL prompt CSV."""
    return load_positional_prompt_bank(
        path,
        consumer="ViLa-MIL",
        class_names=class_names,
        file_class_names=file_class_names,
        record=record,
        expected_provenance=expected_provenance,
        expected_file_sha256=expected_file_sha256,
        expected_ordered_prompt_bank_sha256=(
            expected_ordered_prompt_bank_sha256),
    )


__all__ = [
    "VILA_PROMPT_FORMAT", "ViLAPromptBank", "load_vila_prompt_bank",
    "vila_prompt_bank_sha256",
]
