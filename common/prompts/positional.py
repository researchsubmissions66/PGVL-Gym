"""Shared strict loading for positional dual-scale prompt CSVs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


POSITIONAL_PROMPT_FORMAT = "headerless_low_then_high"
_PROVENANCE = frozenset({"upstream", "derived", "generated"})


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class_names(
    values: Sequence[str], *, consumer: str, label: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{consumer} {label} must be a sequence, not one string")
    result = tuple(values)
    if (not result
            or any(not isinstance(value, str) or not value.strip()
                   for value in result)
            or len(set(result)) != len(result)):
        raise ValueError(
            f"{consumer} {label} must be unique non-empty strings")
    return result


def positional_prompt_bank_sha256(
    class_names: Sequence[str],
    low_resolution: Sequence[str],
    high_resolution: Sequence[str],
    *,
    consumer: str,
) -> str:
    """Hash prompts after binding their positional rows to classifier classes."""
    names = _class_names(
        class_names, consumer=consumer, label="class_names")
    low = tuple(low_resolution)
    high = tuple(high_resolution)
    if len(low) != len(names) or len(high) != len(names):
        raise ValueError(
            f"{consumer} needs one low- and high-resolution prompt per class")
    return _json_sha256({
        "class_names": list(names),
        "low_resolution": list(low),
        "high_resolution": list(high),
    })


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
class PositionalPromptBank:
    """A class-bound low/high prompt bank with audited provenance."""

    path: Path
    file_class_names: tuple[str, ...]
    class_names: tuple[str, ...]
    low_resolution: tuple[str, ...]
    high_resolution: tuple[str, ...]
    file_sha256: str
    source_prompt_bank_sha256: str
    ordered_prompt_bank_sha256: str
    provenance: str
    source: str

    @property
    def prompts(self) -> tuple[str, ...]:
        """Return the low-then-high order consumed by native prompt learners."""
        return self.low_resolution + self.high_resolution


def load_positional_prompt_bank(
    path: str | Path,
    *,
    consumer: str,
    class_names: Sequence[str],
    file_class_names: Sequence[str] | None = None,
    record: Mapping[str, Any] | None = None,
    expected_provenance: str | None = None,
    expected_file_sha256: str | None = None,
    expected_ordered_prompt_bank_sha256: str | None = None,
) -> PositionalPromptBank:
    """Load a native headerless, low-then-high prompt bank."""
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.reader(handle))
    if not csv_rows:
        raise ValueError(f"{source}: {consumer} prompt CSV is empty")
    rows: list[str] = []
    for index, row in enumerate(csv_rows, start=1):
        if len(row) != 1:
            raise ValueError(
                f"{source}: {consumer} native row {index} must contain "
                "exactly one prompt and no header columns")
        prompt = row[0].strip()
        if not prompt:
            raise ValueError(f"{source}: {consumer} native row {index} is empty")
        rows.append(prompt)

    audited = dict(record) if record is not None else _manifest_record(source)
    declared_file_names = audited.get("class_names")
    if (file_class_names is not None and declared_file_names is not None
            and tuple(file_class_names) != tuple(declared_file_names)):
        raise ValueError(
            f"{source}: {consumer} configured file class order contradicts "
            "the provenance record")
    raw_file_names = (
        file_class_names if file_class_names is not None
        else declared_file_names if declared_file_names is not None
        else class_names
    )
    source_names = _class_names(
        raw_file_names, consumer=consumer, label="file_class_names")
    requested_names = _class_names(
        class_names, consumer=consumer, label="class_names")
    if set(source_names) != set(requested_names):
        raise ValueError(
            f"{source}: {consumer} file classes must exactly match classifier "
            f"classes; missing={sorted(set(requested_names) - set(source_names))}, "
            f"extra={sorted(set(source_names) - set(requested_names))}")
    expected_rows = 2 * len(source_names)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{source}: {consumer} native bank has {len(rows)} prompts, "
            f"expected 2 x {len(source_names)} = {expected_rows}")

    source_low = tuple(rows[:len(source_names)])
    source_high = tuple(rows[len(source_names):])
    source_digest = positional_prompt_bank_sha256(
        source_names, source_low, source_high, consumer=consumer)
    source_indices = {name: index for index, name in enumerate(source_names)}
    low = tuple(source_low[source_indices[name]] for name in requested_names)
    high = tuple(source_high[source_indices[name]] for name in requested_names)
    ordered_digest = positional_prompt_bank_sha256(
        requested_names, low, high, consumer=consumer)
    file_digest = _file_sha256(source)

    provenance = str(audited.get(
        "provenance", expected_provenance or "unknown"))
    if provenance not in _PROVENANCE:
        raise ValueError(
            f"{source}: {consumer} prompt bank must have audited provenance")
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError(
            f"{source}: {consumer} prompt provenance {provenance!r} does not "
            f"match config {expected_provenance!r}")
    if (provenance == "upstream" and audited
            and audited.get("copied_from_upstream") is not True):
        raise ValueError(
            f"{source}: upstream {consumer} bank is not recorded as an exact copy")
    if (provenance == "generated"
            and audited.get("copied_from_upstream") not in {None, False}):
        raise ValueError(
            f"{source}: generated {consumer} bank cannot be marked copied upstream")

    checks = (
        (audited.get("sha256"), file_digest, "file sha256"),
        (expected_file_sha256, file_digest, "configured file sha256"),
        (audited.get("prompt_bank_sha256"), source_digest,
         "source prompt-bank sha256"),
        (expected_ordered_prompt_bank_sha256, ordered_digest,
         "configured ordered prompt-bank sha256"),
    )
    for declared, actual, label in checks:
        if declared is not None and declared != actual:
            raise ValueError(f"{source}: {consumer} {label} does not match")
    declared_format = audited.get("format")
    if (declared_format is not None
            and declared_format != POSITIONAL_PROMPT_FORMAT):
        raise ValueError(
            f"{source}: {consumer} provenance declares an unsupported format")
    if audited.get("rows") is not None and audited["rows"] != len(rows):
        raise ValueError(
            f"{source}: {consumer} provenance row count does not match")

    return PositionalPromptBank(
        path=source,
        file_class_names=source_names,
        class_names=requested_names,
        low_resolution=low,
        high_resolution=high,
        file_sha256=file_digest,
        source_prompt_bank_sha256=source_digest,
        ordered_prompt_bank_sha256=ordered_digest,
        provenance=provenance,
        source=str(audited.get("source", source)),
    )


__all__ = [
    "POSITIONAL_PROMPT_FORMAT", "PositionalPromptBank",
    "load_positional_prompt_bank", "positional_prompt_bank_sha256",
]
