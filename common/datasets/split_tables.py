"""Resolve runtime phase tables across supported fold CSV layouts.

Generated protocols write ``foldN/{train,val,test}.csv`` tables containing
complete manifest rows. Several upstream repositories instead write one wide
``splits_N.csv`` or ``foldN.csv`` with phase-specific ID columns. Runtime
loaders must interpret the same layouts that the preflight doctor accepts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from common.configuration import expand_path


_PHASES = {"train", "val", "test"}


def _read_csv(path: Path, *, id_columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={column: str for column in id_columns})
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"Cannot read split CSV {path}: {error}") from error


def _validate_phase_frame(frame: pd.DataFrame, path: Path,
                          phase: str) -> pd.DataFrame:
    missing = {"slide_id", "label"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} {phase} table is missing columns: {sorted(missing)}")
    frame = frame.copy()
    identifiers = frame["slide_id"]
    blank_ids = identifiers.isna() | identifiers.astype(str).str.strip().eq("")
    blank_labels = frame["label"].isna() | frame["label"].astype(
        str).str.strip().eq("")
    if blank_ids.any():
        raise ValueError(
            f"{path} {phase} table has {int(blank_ids.sum())} blank slide IDs")
    if blank_labels.any():
        raise ValueError(
            f"{path} {phase} table has {int(blank_labels.sum())} blank labels")
    frame["slide_id"] = identifiers.astype(str).str.strip()
    duplicates = frame.loc[
        frame["slide_id"].duplicated(), "slide_id"].unique().tolist()
    if duplicates:
        raise ValueError(
            f"{path} {phase} table repeats slide IDs: "
            + ", ".join(map(str, duplicates[:3])))
    return frame.reset_index(drop=True)


def _manifest_rows(cfg: Mapping[str, Any], identifiers: list[str],
                   wide_path: Path, phase: str) -> pd.DataFrame | None:
    manifest_value = cfg.get("dataset_csv")
    if not manifest_value:
        return None
    manifest_path = Path(expand_path(manifest_value))
    if not manifest_path.is_file():
        return None
    manifest = _read_csv(manifest_path, id_columns=("slide_id",))
    if {"slide_id", "label"} - set(manifest.columns):
        return None
    manifest["slide_id"] = manifest["slide_id"].astype(str).str.strip()
    duplicate = manifest.loc[
        manifest["slide_id"].duplicated(), "slide_id"].unique().tolist()
    if duplicate:
        raise ValueError(
            f"{manifest_path} repeats slide IDs: "
            + ", ".join(map(str, duplicate[:3])))
    indexed = manifest.set_index("slide_id", drop=False)
    missing = [identifier for identifier in identifiers
               if identifier not in indexed.index]
    if missing:
        raise ValueError(
            f"{wide_path} {phase} split contains slide IDs absent from "
            f"{manifest_path}: {', '.join(missing[:3])}")
    return indexed.loc[identifiers].reset_index(drop=True)


def _wide_phase_frame(cfg: Mapping[str, Any], path: Path,
                      phase: str) -> pd.DataFrame:
    wide = _read_csv(path, id_columns=(phase,))
    if phase not in wide.columns:
        raise ValueError(f"{path} has no {phase!r} split column")
    mask = wide[phase].notna() & wide[phase].astype(str).str.strip().ne("")
    identifiers = wide.loc[mask, phase].astype(str).str.strip().tolist()
    if not identifiers:
        raise ValueError(f"{path} has no members for {phase}")

    frame = _manifest_rows(cfg, identifiers, path, phase)
    label_column = f"{phase}_label"
    if frame is None:
        if label_column not in wide.columns:
            raise ValueError(
                f"{path} needs {label_column!r}, or dataset_csv must provide "
                "slide_id and label columns")
        frame = pd.DataFrame({
            "slide_id": identifiers,
            "label": wide.loc[mask, label_column].tolist(),
        })
    elif label_column in wide.columns:
        declared = wide.loc[mask, label_column]
        missing_labels = declared.isna() | declared.astype(str).str.strip().eq("")
        if missing_labels.any():
            raise ValueError(
                f"{path} has blank {label_column!r} values for {phase}")
        label_dict = cfg.get("label_dict", {})

        def identity(value: Any) -> Any:
            text = str(value).strip()
            if isinstance(label_dict, Mapping) and text in label_dict:
                return label_dict[text]
            try:
                number = float(text)
            except ValueError:
                return text
            return int(number) if number.is_integer() else number

        manifest_labels = [identity(value) for value in frame["label"]]
        declared_labels = [identity(value) for value in declared]
        if manifest_labels != declared_labels:
            raise ValueError(
                f"{path} {label_column!r} values do not match dataset_csv")
    return _validate_phase_frame(frame, path, phase)


def load_phase_table(cfg: Mapping[str, Any], phase: str,
                     fold: int) -> pd.DataFrame:
    """Return one complete phase table without silently changing a split."""
    if phase not in _PHASES:
        raise ValueError(f"split phase must be one of {sorted(_PHASES)}")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise ValueError("fold must be a non-negative integer")
    root = Path(expand_path(cfg["split_dir"]))
    nested = root / f"fold{fold}" / f"{phase}.csv"
    already_nested = root.name == f"fold{fold}"
    direct = root / f"{phase}.csv"

    if nested.is_file():
        return _validate_phase_frame(
            _read_csv(nested, id_columns=("slide_id",)), nested, phase)
    if already_nested and direct.is_file():
        return _validate_phase_frame(
            _read_csv(direct, id_columns=("slide_id",)), direct, phase)

    for wide in (root / f"splits_{fold}.csv", root / f"fold{fold}.csv"):
        if wide.is_file():
            return _wide_phase_frame(cfg, wide, phase)

    # A single-fold legacy run may keep phase files directly in split_dir. Do
    # not reuse that same table across a declared multi-fold experiment.
    raw_start = cfg.get("k_start", 0)
    raw_end = cfg.get("k_end", cfg.get("k", 1))
    single_fold = (
        isinstance(raw_start, int) and not isinstance(raw_start, bool)
        and isinstance(raw_end, int) and not isinstance(raw_end, bool)
        and raw_end - raw_start == 1)
    if direct.is_file() and single_fold:
        return _validate_phase_frame(
            _read_csv(direct, id_columns=("slide_id",)), direct, phase)
    if direct.is_file():
        raise ValueError(
            f"{direct} is an unscoped phase file for a multi-fold run; place "
            f"it under fold{fold}/ or provide splits_{fold}.csv")

    tried = [nested, direct, root / f"splits_{fold}.csv",
             root / f"fold{fold}.csv"]
    raise FileNotFoundError(
        f"No {phase} split table for fold {fold}. Tried:\n  "
        + "\n  ".join(str(path) for path in tried))
