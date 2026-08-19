"""Validate MUSE's ordered, class-specific description CSV banks."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MUSEPromptCSV:
    """One validated class branch from a MUSE knowledge bank."""

    path: Path
    descriptions: tuple[str, ...]
    file_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.descriptions)


@dataclass(frozen=True)
class MUSEPromptBank:
    """A class-index ordered collection of MUSE description branches."""

    classnames: tuple[str, ...]
    branches: tuple[MUSEPromptCSV, ...]

    @property
    def descriptions(self) -> tuple[tuple[str, ...], ...]:
        return tuple(branch.descriptions for branch in self.branches)


def load_muse_prompt_csv(
    path: str | Path,
    *,
    expected_rows: int | None = None,
    expected_sha256: str | None = None,
) -> MUSEPromptCSV:
    """Load one native MUSE CSV without guessing which cell is its text.

    Released MUSE files are pandas exports with the exact header ``,0`` and
    rows ``index,description``.  Generated task extensions use the same schema.
    Duplicate descriptions are retained because they are present in the
    released knowledge bases and therefore affect stochastic sampling.
    """
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{source}: MUSE prompt CSV is empty")
    if rows[0] != ["", "0"]:
        raise ValueError(
            f"{source}: MUSE prompt CSV header must be exactly ['', '0']"
        )

    descriptions: list[str] = []
    for index, row in enumerate(rows[1:]):
        if len(row) != 2:
            raise ValueError(
                f"{source}: MUSE row {index + 2} must contain exactly "
                "an index and one description"
            )
        if row[0].strip() != str(index):
            raise ValueError(
                f"{source}: MUSE row {index + 2} index {row[0]!r} does "
                f"not match expected index {index}"
            )
        description = row[1].strip()
        if not description:
            raise ValueError(
                f"{source}: MUSE row {index + 2} has an empty description"
            )
        descriptions.append(description)
    if not descriptions:
        raise ValueError(f"{source}: MUSE prompt CSV has no descriptions")
    if expected_rows is not None and len(descriptions) != expected_rows:
        raise ValueError(
            f"{source}: MUSE prompt CSV has {len(descriptions)} descriptions, "
            f"expected {expected_rows}"
        )

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"{source}: MUSE file sha256 does not match its provenance record"
        )
    return MUSEPromptCSV(source, tuple(descriptions), digest)


def load_muse_prompt_bank(
    prompt_csvs: Mapping[str, str | Path],
    *,
    classnames: Sequence[str],
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> MUSEPromptBank:
    """Load MUSE branches in classifier order and validate audited bindings."""
    ordered = tuple(classnames)
    if (not ordered
            or any(not isinstance(name, str) or not name.strip()
                   for name in ordered)
            or len(set(ordered)) != len(ordered)):
        raise ValueError("MUSE classnames must be unique non-empty strings")
    if not isinstance(prompt_csvs, Mapping):
        raise TypeError("MUSE prompt_csvs must map classnames to CSV paths")
    actual = set(prompt_csvs)
    expected = set(ordered)
    if actual != expected:
        raise ValueError(
            "MUSE prompt_csvs keys must exactly match classnames; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    metadata = records or {}
    branches: list[MUSEPromptCSV] = []
    for class_index, classname in enumerate(ordered):
        path = Path(prompt_csvs[classname])
        record = metadata.get(str(path), {})
        recorded_classname = record.get("classname")
        if recorded_classname is not None and recorded_classname != classname:
            raise ValueError(
                f"{path}: MUSE provenance binds this CSV to "
                f"{recorded_classname!r}, not {classname!r}"
            )
        recorded_index = record.get("class_index")
        if recorded_index is not None and recorded_index != class_index:
            raise ValueError(
                f"{path}: MUSE provenance binds this CSV to class index "
                f"{recorded_index}, not {class_index}"
            )
        branches.append(load_muse_prompt_csv(
            path,
            expected_rows=record.get("rows"),
            expected_sha256=record.get("sha256"),
        ))
    return MUSEPromptBank(ordered, tuple(branches))


__all__ = [
    "MUSEPromptBank", "MUSEPromptCSV", "load_muse_prompt_bank",
    "load_muse_prompt_csv",
]
