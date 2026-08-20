"""Lightweight feature-readiness refresh for generated benchmark artifacts.

This module deliberately uses only the standard library plus the repository's
configuration path expansion. Campaign planning must be able to notice feature
files that arrived asynchronously without importing model libraries or
regenerating manifests, splits, prompts, and configs.
"""
from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.configuration import REPO_ROOT, expand_path


@dataclass(frozen=True)
class ReadinessRefresh:
    """Summary of one benchmark's on-disk readiness refresh."""

    feature_sources: int
    matrix_rows: int
    changed_rows: int


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"CSV has no header: {path}")
        if any(not field for field in fields):
            raise ValueError(f"CSV has a blank header field: {path}")
        duplicates = sorted({field for field in fields if fields.count(field) > 1})
        if duplicates:
            raise ValueError(
                f"CSV has duplicate header fields {duplicates}: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values())
           for row in rows):
        raise ValueError(f"CSV has a malformed row: {path}")
    return fields, rows


def _write_csv_atomic(
    path: Path, fields: list[str], rows: list[dict[str, Any]],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, path.stat().st_mode & 0o777)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, lineterminator="\n",
                extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _required(fields: list[str], required: set[str], path: Path) -> None:
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(
            f"CSV is missing required columns {missing}: {path}")


def _truthy(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"expected true or false, got {value!r}")


def _nonnegative(value: Any, field: str) -> int:
    try:
        number = int(str(value or "").strip() or 0)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a non-negative integer") from error
    if number < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return number


def _local_path(value: Any) -> Path:
    path = Path(expand_path(str(value)))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _source_present(
    value: Any, input_kind: str, directory_files: dict[Path, set[str]],
) -> bool:
    path = _local_path(value)
    if input_kind == "raw_tile_directory":
        if not path.is_dir():
            return False
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
        return any(
            item.is_file() and item.suffix.lower() in extensions
            for item in path.iterdir())
    # Feature sources normally place every slide in one shared directory.
    # Listing that directory once avoids thousands of latency-heavy stat calls
    # on a cluster filesystem while retaining exact filename checks.
    if path.parent not in directory_files:
        try:
            with os.scandir(path.parent) as entries:
                directory_files[path.parent] = {
                    entry.name for entry in entries
                    if entry.is_file(follow_symlinks=True)
                }
        except (FileNotFoundError, NotADirectoryError):
            directory_files[path.parent] = set()
    return path.name in directory_files[path.parent]


def _feature_sources(signature: str) -> list[str]:
    sources: list[str] = []
    for binding in signature.split(";"):
        role, separator, source = binding.partition("=")
        if not separator or not role.strip() or not source.strip():
            raise ValueError(f"invalid feature_signature {signature!r}")
        sources.append(source.strip())
    if not sources:
        raise ValueError("feature_signature is empty")
    return sources


def refresh_benchmark_readiness(benchmark_dir: Path) -> ReadinessRefresh:
    """Refresh feature coverage and matrix readiness from existing manifests.

    Only feature-derived cells are changed. Configuration validity, encoder,
    metadata, split, and auxiliary readiness remain owned by full generation
    and validation. A refresh therefore cannot make a row runnable by masking a
    non-feature blocker.
    """
    benchmark_dir = benchmark_dir.resolve()
    coverage_path = benchmark_dir / "feature_coverage.csv"
    matrix_path = benchmark_dir / "run_matrix.csv"
    coverage_fields, coverage_rows = _read_csv(coverage_path)
    matrix_fields, matrix_rows = _read_csv(matrix_path)
    _required(coverage_fields, {
        "cohort", "feature_source", "input_kind", "feature_column",
        "available_slides", "annotated_slides", "coverage", "metadata_ready",
    }, coverage_path)
    _required(matrix_fields, {
        "cohort", "feature_signature", "ready", "missing_feature_files",
        "missing_auxiliary_files", "metadata_ready", "split_ready",
        "encoder_ready", "config_valid", "auxiliary_ready",
    }, matrix_path)

    manifests: dict[str, list[dict[str, str]]] = {}
    manifest_fields_by_cohort: dict[str, set[str]] = {}
    source_missing: dict[tuple[str, str], int] = {}
    directory_files: dict[Path, set[str]] = {}
    for row in coverage_rows:
        cohort = row["cohort"].strip()
        source = row["feature_source"].strip()
        column = row["feature_column"].strip()
        kind = row["input_kind"].strip() or "patch_bag"
        if not cohort or not source or not column:
            raise ValueError(f"coverage row has a blank identity: {coverage_path}")
        key = (cohort, source)
        if key in source_missing:
            raise ValueError(
                f"duplicate feature coverage row for {cohort}/{source}")
        if cohort not in manifests:
            manifest_path = benchmark_dir / "data" / cohort / "manifest.csv"
            manifest_fields, manifest_rows = _read_csv(manifest_path)
            manifests[cohort] = manifest_rows
            manifest_fields_by_cohort[cohort] = set(manifest_fields)
        if column not in manifest_fields_by_cohort[cohort]:
            raise ValueError(
                f"manifest for {cohort} is missing feature column {column!r}")
        manifest = manifests[cohort]
        available = sum(
            _source_present(item[column], kind, directory_files)
            for item in manifest)
        annotated = len(manifest)
        source_missing[key] = annotated - available
        row["available_slides"] = str(available)
        row["annotated_slides"] = str(annotated)
        row["coverage"] = str(available / annotated if annotated else 0.0)
        row["metadata_ready"] = str(bool(annotated))

    changed_rows = 0
    for row in matrix_rows:
        cohort = row["cohort"].strip()
        sources = _feature_sources(row["feature_signature"])
        unknown = [source for source in sources
                   if (cohort, source) not in source_missing]
        if unknown:
            raise ValueError(
                f"run matrix references feature source(s) absent from coverage "
                f"for {cohort}: {', '.join(unknown)}")
        missing_features = sum(
            source_missing[(cohort, source)] for source in sources)
        missing_auxiliary = _nonnegative(
            row["missing_auxiliary_files"], "missing_auxiliary_files")
        readiness = all(_truthy(row[field]) for field in (
            "metadata_ready", "split_ready", "encoder_ready",
            "config_valid", "auxiliary_ready"))
        ready = (
            missing_features == 0 and missing_auxiliary == 0 and readiness)
        old_missing = row["missing_feature_files"]
        old_ready = row["ready"]
        row["missing_feature_files"] = str(missing_features)
        row["ready"] = str(ready)
        if (old_missing != row["missing_feature_files"]
                or old_ready.lower() != row["ready"].lower()):
            changed_rows += 1

    # Compute everything before replacing either artifact. A malformed matrix
    # therefore cannot leave coverage half-refreshed.
    _write_csv_atomic(coverage_path, coverage_fields, coverage_rows)
    _write_csv_atomic(matrix_path, matrix_fields, matrix_rows)
    return ReadinessRefresh(
        feature_sources=len(coverage_rows), matrix_rows=len(matrix_rows),
        changed_rows=changed_rows)
