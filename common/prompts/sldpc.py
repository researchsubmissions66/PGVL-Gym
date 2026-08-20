"""Prompt contracts for SLDPC training and its separate TITAN baseline.

SLDPC Stage 1/2 learns context tokens around one fixed class token sequence.
The released synonym YAMLs and 23 templates are consumed only by an optional,
separately reported TITAN zero-shot baseline.  Keeping those roles distinct
prevents an unused YAML from determining the trained run's provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SLDPC_ZERO_SHOT_TEMPLATES = (
    "CLASSNAME.",
    "an image of CLASSNAME.",
    "the image shows CLASSNAME.",
    "the image displays CLASSNAME.",
    "the image exhibits CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "I observe CLASSNAME.",
    "the pathology image shows CLASSNAME.",
    "a pathology image shows CLASSNAME.",
    "the pathology slide shows CLASSNAME.",
    "shows CLASSNAME.",
    "contains CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "CLASSNAME is observed.",
    "the pathology image reveals CLASSNAME.",
    "a microscopic image of showing CLASSNAME.",
    "histology shows CLASSNAME.",
    "CLASSNAME can be seen.",
    "the tissue shows CLASSNAME.",
    "CLASSNAME is identified.",
)
_PROVENANCE = frozenset({"upstream", "derived", "generated"})


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sldpc_prompt_classnames(
    values: Sequence[str], *, n_classes: int | None = None,
) -> tuple[str, ...]:
    """Validate the exact fixed class tokens embedded by Stage 1/2."""
    result = tuple(values)
    if (not result
            or any(not isinstance(value, str) or not value.strip()
                   for value in result)
            or len(set(result)) != len(result)):
        raise ValueError(
            "SLDPC prompt_classnames must be unique non-empty strings")
    if n_classes is not None and len(result) != n_classes:
        raise ValueError(
            f"SLDPC prompt_classnames has {len(result)} entries, expected "
            f"n_classes={n_classes}")
    return result


def sldpc_prompt_classname_sha256(values: Sequence[str]) -> str:
    """Hash the ordered fixed class tokens used by the learned prompts."""
    return _json_sha256(list(sldpc_prompt_classnames(values)))


def sldpc_zero_shot_templates_sha256() -> str:
    """Hash the exact ordered 23-template TITAN baseline ensemble."""
    return _json_sha256(list(SLDPC_ZERO_SHOT_TEMPLATES))


def _manifest_record(path: Path) -> dict[str, Any]:
    prompt_root = Path(__file__).resolve().parents[2] / "text_prompts"
    try:
        key = str(path.resolve().relative_to(prompt_root.resolve()))
    except ValueError:
        return {}
    try:
        manifest = json.loads(
            (prompt_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    record = manifest.get("assets", {}).get(key, {}) \
        if isinstance(manifest, dict) else {}
    return record if isinstance(record, dict) else {}


@dataclass(frozen=True)
class SLDPCZeroShotPromptBank:
    """A validated synonym bank for the separate TITAN zero-shot baseline."""

    path: Path
    class_names: tuple[str, ...]
    prompts: tuple[tuple[str, ...], ...]
    file_sha256: str
    source_prompt_sha256: str
    ordered_prompt_sha256: str
    provenance: str
    source: str

    @property
    def prompt_counts(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self.prompts)


def load_sldpc_zero_shot_prompt_bank(
    path: str | Path,
    *,
    class_names: Sequence[str],
    record: Mapping[str, Any] | None = None,
    expected_file_sha256: str | None = None,
    expected_ordered_prompt_sha256: str | None = None,
) -> SLDPCZeroShotPromptBank:
    """Load a YAML synonym bank without treating it as an SLDPC train input."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, dict):
        raise ValueError(
            f"{source}: SLDPC zero-shot bank needs a 'prompts' mapping")

    ordered_names = sldpc_prompt_classnames(class_names)
    expected = set(ordered_names)
    actual = set(prompts)
    if actual != expected:
        raise ValueError(
            f"{source}: SLDPC zero-shot prompt keys must exactly match "
            f"class names; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}")
    rows: list[tuple[str, ...]] = []
    for name in ordered_names:
        values = prompts[name]
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value.strip()
                       for value in values)):
            raise ValueError(
                f"{source}: SLDPC zero-shot prompts for {name!r} must be a "
                "non-empty string list")
        rows.append(tuple(values))

    file_digest = _file_sha256(source)
    source_digest = _json_sha256({str(key): value for key, value in prompts.items()})
    ordered_payload = {
        name: list(values) for name, values in zip(ordered_names, rows)
    }
    ordered_digest = _json_sha256(ordered_payload)
    audited = dict(record) if record is not None else _manifest_record(source)
    provenance = str(audited.get("provenance", "unknown"))
    if provenance not in _PROVENANCE:
        raise ValueError(
            f"{source}: SLDPC zero-shot bank must have audited provenance")
    if (provenance == "generated"
            and audited.get("copied_from_upstream") not in {None, False}):
        raise ValueError(
            f"{source}: generated SLDPC bank cannot be marked copied upstream")

    declared_file = audited.get("sha256")
    if declared_file is not None and declared_file != file_digest:
        raise ValueError(
            f"{source}: SLDPC zero-shot file sha256 does not match provenance")
    if expected_file_sha256 is not None and expected_file_sha256 != file_digest:
        raise ValueError(
            f"{source}: SLDPC zero-shot file sha256 does not match config")
    declared_source = audited.get("prompt_mapping_sha256")
    if declared_source is not None and declared_source != source_digest:
        raise ValueError(
            f"{source}: SLDPC zero-shot prompt digest does not match provenance")
    if (expected_ordered_prompt_sha256 is not None
            and expected_ordered_prompt_sha256 != ordered_digest):
        raise ValueError(
            f"{source}: SLDPC ordered zero-shot digest does not match config")
    declared_names = audited.get("class_names")
    if (declared_names is not None
            and tuple(declared_names) != tuple(prompts)):
        raise ValueError(
            f"{source}: SLDPC provenance class-name order does not match "
            "the source bank")
    declared_counts = audited.get("prompt_counts_per_class")
    actual_counts = {str(key): len(value) for key, value in prompts.items()}
    if declared_counts is not None and declared_counts != actual_counts:
        raise ValueError(
            f"{source}: SLDPC zero-shot prompt counts do not match provenance")
    declared_templates = audited.get("templates")
    if (declared_templates is not None
            and declared_templates != len(SLDPC_ZERO_SHOT_TEMPLATES)):
        raise ValueError(
            f"{source}: SLDPC provenance template count is not the released "
            "23-template ensemble")
    declared_template_digest = audited.get("templates_sha256")
    if (declared_template_digest is not None
            and declared_template_digest
            != sldpc_zero_shot_templates_sha256()):
        raise ValueError(
            f"{source}: SLDPC provenance template digest does not match the "
            "released ensemble")

    return SLDPCZeroShotPromptBank(
        path=source,
        class_names=ordered_names,
        prompts=tuple(rows),
        file_sha256=file_digest,
        source_prompt_sha256=source_digest,
        ordered_prompt_sha256=ordered_digest,
        provenance=provenance,
        source=str(audited.get("source", source)),
    )


__all__ = [
    "SLDPC_ZERO_SHOT_TEMPLATES", "SLDPCZeroShotPromptBank",
    "load_sldpc_zero_shot_prompt_bank", "sldpc_prompt_classname_sha256",
    "sldpc_prompt_classnames", "sldpc_zero_shot_templates_sha256",
]
