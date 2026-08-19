"""Shared one-vector-per-slide feature loading.

This module is method-agnostic. Any adapter whose backbone contract declares
``FeatureLevel.SLIDE_EMBEDDING`` receives the same exact-key, exact-width,
provenance-aware HDF5/torch/pickle input behavior from the unified trainer.
Patch bags are intentionally handled elsewhere.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from common.configuration import expand_path


_FILE_SUFFIXES = {".pt", ".pth", ".h5", ".hdf5", ".svs", ".tif", ".tiff"}


def normalise_slide_id(value: Any) -> str:
    """Decode an identifier and remove only known slide/feature suffixes.

    NumPy and Python pickle producers commonly persist identifier arrays as
    fixed-width byte strings. Calling ``str`` on those values produces text
    such as ``"b'slide-a.svs'"``, which can never match a manifest slide ID.
    Decode byte-like values explicitly and reject blank identifiers at the
    source boundary instead.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        text = bytes(value).decode("utf-8")
    else:
        text = str(value)
    text = text.strip()
    if not text:
        raise ValueError("slide ID must not be blank")
    path = Path(text)
    return path.stem if path.suffix.lower() in _FILE_SUFFIXES else path.name


def infer_slide_embedding_source_type(
    path_template: str | Path | None = None,
    storage: str | None = None,
) -> str:
    """Map a registered storage declaration to the shared loader layout.

    Args:
        path_template: Representative source path or template.
        storage: Explicit storage name such as ``h5``, ``torch``, or ``pkl``.

    Returns:
        One of ``per_slide_h5``, ``per_slide_torch``, or ``pkl``.

    Raises:
        ValueError: If neither the declaration nor suffix identifies a layout.
    """
    storage_name = str(storage or "").strip().lower()
    if storage_name in {"h5", "hdf5"}:
        return "per_slide_h5"
    if storage_name in {"pt", "pth", "torch"}:
        return "per_slide_torch"
    if storage_name in {"pkl", "pickle"}:
        return "pkl"
    suffix = Path(str(path_template or "")).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return "per_slide_h5"
    if suffix in {".pt", ".pth"}:
        return "per_slide_torch"
    if suffix in {".pkl", ".pickle"}:
        return "pkl"
    raise ValueError(
        "Slide-embedding sources require storage h5/torch/pkl or a matching "
        f"path suffix, got path={path_template!r}, storage={storage!r}")


@dataclass(frozen=True)
class SlideEmbeddingSource:
    """Normalize runtime fields shared by slide-vector consumers.

    The source separates offline slide encoder provenance from any runtime
    prompt encoder selected by methods such as SLDPC.
    """

    source_type: str
    features_path: Path
    feature_path_column: str | None
    feature_key: str
    feature_dim: int
    slide_id_key: str

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SlideEmbeddingSource":
        """Validate and construct a source from a generated run config."""
        feature_dim = int(cfg.get("feature_dim", 0))
        if feature_dim <= 0:
            raise ValueError("Slide-embedding configs require positive feature_dim")
        feature_key = str(cfg.get("feature_key", "")).strip()
        if not feature_key:
            raise ValueError("Slide-embedding configs require feature_key")
        source_type = str(cfg.get("source_type", "")).strip()
        if not source_type:
            source_type = infer_slide_embedding_source_type(
                cfg.get("slide_features"), cfg.get("storage"))
        if source_type == "per_slide_pth":
            source_type = "per_slide_torch"
        if source_type not in {"pkl", "per_slide_h5", "per_slide_torch"}:
            raise ValueError(
                "Slide-embedding source_type must be pkl, per_slide_h5, or "
                f"per_slide_torch, got {source_type!r}")
        root = cfg.get("slide_features")
        if not root:
            raise ValueError("Slide-embedding configs require slide_features")
        return cls(
            source_type=source_type,
            features_path=Path(str(root)).expanduser(),
            feature_path_column=cfg.get("feature_path_column"),
            feature_key=feature_key,
            feature_dim=feature_dim,
            slide_id_key=str(cfg.get("slide_id_key", "filenames")),
        )


class SlideEmbeddingDataset(Dataset):
    """Load one exact-width vector and label for every split slide.

    Args:
        source_type: ``pkl``, ``per_slide_h5``, or ``per_slide_torch``.
        features_path: Shared pickle file or root containing per-slide files.
        csv_path: Split CSV with slide IDs and labels.
        label_dict: Mapping from string labels to integer class indices.
        feature_path_column: Optional CSV column with exact per-slide paths.
        feature_key: Exact tensor key inside HDF5 or mapping payloads.
        feature_dim: Expected flattened vector width.
        slide_id_key: Identifier key used by a shared pickle payload.

    Samples are dictionaries with ``feat``, ``label``, ``slide_id``, and
    ``case_id``. Every declared split row must match exactly one feature;
    missing or ambiguous IDs are fatal rather than silently changing a split.
    """

    def __init__(
        self,
        source_type: str,
        features_path: str | Path,
        csv_path: str | Path | pd.DataFrame,
        label_dict: Mapping[str, int],
        feature_path_column: str | None = None,
        feature_key: str = "features",
        feature_dim: int | None = None,
        slide_id_key: str = "filenames",
    ):
        self.source_type = (
            "per_slide_torch" if source_type == "per_slide_pth" else source_type)
        self.features_path = Path(expand_path(features_path))
        self.label_dict = dict(label_dict)
        self.feature_path_column = feature_path_column
        self.feature_key = feature_key
        self.slide_id_key = slide_id_key
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.entries = self._read_entries(csv_path)
        if not self.entries:
            raise ValueError(f"slide-embedding split has no rows: {csv_path}")

        if self.source_type == "pkl":
            with self.features_path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, Mapping):
                raise TypeError("pickle slide-embedding payload must be a mapping")
            missing_keys = [
                key for key in (self.feature_key, self.slide_id_key)
                if key not in payload]
            if missing_keys:
                raise KeyError(
                    "pickle slide-embedding payload is missing keys "
                    f"{missing_keys}; available keys: {list(payload.keys())}")
            self.embeddings = torch.as_tensor(
                np.asarray(payload[self.feature_key])).float()
            if self.embeddings.ndim != 2 or self.embeddings.shape[0] == 0:
                raise ValueError(
                    "pickle slide embeddings must have shape [slides, dimension], "
                    f"got {tuple(self.embeddings.shape)}")
            ids = payload[self.slide_id_key]
            if isinstance(ids, (str, bytes)):
                raise TypeError(
                    f"pickle {self.slide_id_key!r} must be a sequence of IDs")
            try:
                ids = list(ids)
            except TypeError as error:
                raise TypeError(
                    f"pickle {self.slide_id_key!r} must be a sequence of IDs") \
                    from error
            if len(ids) != len(self.embeddings):
                raise ValueError(
                    f"pickle has {len(ids)} slide IDs for "
                    f"{len(self.embeddings)} embeddings")
            indexed = [
                (normalise_slide_id(slide_id), index)
                for index, slide_id in enumerate(ids)]
        elif self.source_type == "per_slide_torch":
            self.embeddings = None
            paths = (() if all(entry[2] for entry in self.entries) else (
                path for pattern in ("*.pt", "*.pth")
                for path in self.features_path.rglob(pattern)))
            indexed = [
                (normalise_slide_id(path.name), path) for path in paths]
        elif self.source_type == "per_slide_h5":
            self.embeddings = None
            paths = (() if all(entry[2] for entry in self.entries) else (
                path for pattern in ("*.h5", "*.hdf5")
                for path in self.features_path.rglob(pattern)))
            indexed = [
                (normalise_slide_id(path.name), path) for path in paths]
        else:
            raise ValueError(
                "source_type must be pkl, per_slide_h5, or per_slide_torch")

        self.paths = {}
        duplicate_sources: dict[str, list[str]] = {}
        for slide_id, pointer in indexed:
            if slide_id in self.paths:
                duplicate_sources.setdefault(
                    slide_id, [str(self.paths[slide_id])]).append(str(pointer))
            else:
                self.paths[slide_id] = pointer
        if duplicate_sources:
            sample = "; ".join(
                f"{slide_id}: {paths}"
                for slide_id, paths in list(duplicate_sources.items())[:3])
            raise ValueError(
                "slide-embedding store has duplicate normalized slide IDs: "
                f"{sample}")

        missing: list[str] = []
        for slide_id, _label, explicit_path, _case_id in self.entries:
            if explicit_path and self.embeddings is None:
                try:
                    available = Path(expand_path(explicit_path)).is_file()
                except ValueError:
                    available = False
            else:
                available = normalise_slide_id(slide_id) in self.paths
            if not available:
                missing.append(slide_id)
        if missing:
            sample = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"{len(missing)} split slide embeddings are missing: "
                f"{sample}{suffix}")

    def _read_entries(self, csv_path: str | Path | pd.DataFrame):
        if isinstance(csv_path, pd.DataFrame):
            header = [str(column) for column in csv_path.columns]
            rows = [header] + [
                ["" if pd.isna(value) else str(value) for value in row]
                for row in csv_path.itertuples(index=False, name=None)]
        else:
            with Path(csv_path).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        if not rows:
            return []
        header = [item.strip().lower() for item in rows[0]]
        has_header = "slide_id" in header or "label" in header
        if has_header:
            duplicate_headers = sorted({
                field for field in header if header.count(field) > 1})
            if duplicate_headers:
                raise ValueError(
                    "slide-embedding split has duplicate columns: "
                    + ", ".join(duplicate_headers))
            missing = [
                field for field in ("slide_id", "label") if field not in header]
            if missing:
                raise ValueError(
                    "slide-embedding split header is missing: "
                    + ", ".join(missing))
            slide_column = header.index("slide_id")
            label_column = header.index("label")
            if (self.feature_path_column
                    and self.feature_path_column.lower() not in header):
                raise ValueError(
                    "slide-embedding split has no configured feature column "
                    f"{self.feature_path_column!r}")
            feature_column = (
                header.index(self.feature_path_column.lower())
                if self.feature_path_column
                and self.feature_path_column.lower() in header else None)
            case_column = header.index("case_id") if "case_id" in header else None
            rows = rows[1:]
        else:
            slide_column, label_column, feature_column, case_column = 0, 1, None, None

        entries = []
        seen: set[str] = set()
        valid_labels = set(self.label_dict.values())
        for row_number, row in enumerate(rows, start=2 if has_header else 1):
            if len(row) <= max(slide_column, label_column):
                raise ValueError(
                    f"split row {row_number} has fewer than two columns")
            slide_id = row[slide_column].strip()
            if not slide_id:
                raise ValueError(f"split row {row_number} has a blank slide_id")
            normalized = normalise_slide_id(slide_id)
            if normalized in seen:
                raise ValueError(
                    f"split repeats normalized slide ID {normalized!r}")
            seen.add(normalized)
            raw_label = row[label_column].strip()
            try:
                label = (
                    self.label_dict[raw_label]
                    if raw_label in self.label_dict else int(raw_label))
            except ValueError as error:
                # Wide upstream tables are rectangular CSVs, so pandas may
                # promote integer phase-label columns with blank rows to
                # floating values (``0`` becomes ``0.0``). Preserve the class
                # index when that representation is exactly integral.
                try:
                    numeric_label = float(raw_label)
                    if not numeric_label.is_integer():
                        raise ValueError
                    label = int(numeric_label)
                except ValueError:
                    raise ValueError(
                        f"split row {row_number} has unknown label "
                        f"{raw_label!r}") from error
            if label not in valid_labels:
                raise ValueError(
                    f"split row {row_number} has out-of-range label {label}")
            if feature_column is not None:
                explicit_path = (
                    row[feature_column].strip()
                    if len(row) > feature_column else "")
                if not explicit_path:
                    raise ValueError(
                        f"split row {row_number} has a blank configured "
                        f"feature path in {self.feature_path_column!r}")
            else:
                explicit_path = ""
            case_id = (
                row[case_column].strip()
                if case_column is not None and len(row) > case_column
                else slide_id)
            if not case_id:
                raise ValueError(f"split row {row_number} has a blank case_id")
            entries.append((slide_id, label, explicit_path, case_id))
        return entries

    def _load_tensor(self, path: Path) -> torch.Tensor:
        keys = (self.feature_key,)
        if path.suffix.lower() in {".h5", ".hdf5"}:
            with h5py.File(path, "r") as handle:
                key = next((name for name in keys if name in handle), None)
                if key is None:
                    raise KeyError(
                        f"No slide embedding key {self.feature_key!r} found "
                        f"in {path}; available keys: {list(handle.keys())}")
                return torch.from_numpy(handle[key][:]).float().view(-1)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            return payload.float().view(-1)
        if isinstance(payload, Mapping):
            key = next((name for name in keys if name in payload), None)
            if key is not None:
                return torch.as_tensor(payload[key]).float().view(-1)
        raise KeyError(
            f"No slide embedding key {self.feature_key!r} found in {path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        slide_id, label, explicit_path, case_id = self.entries[index]
        pointer = (
            Path(expand_path(explicit_path))
            if explicit_path and self.embeddings is None
            else self.paths[normalise_slide_id(slide_id)])
        feature = (
            self.embeddings[pointer]
            if self.embeddings is not None else self._load_tensor(pointer))
        feature = feature.float().view(-1)
        if self.feature_dim is not None and feature.numel() != self.feature_dim:
            raise ValueError(
                f"Slide {slide_id!r} has embedding width {feature.numel()}, "
                f"expected {self.feature_dim}")
        if not torch.isfinite(feature).all():
            raise ValueError(
                f"Slide {slide_id!r} contains NaN or infinite embeddings")
        return {
            "feat": feature, "label": label,
            "slide_id": slide_id, "case_id": case_id,
        }


def split_csv(cfg: Mapping[str, Any], split: str, fold: int) -> Path:
    """Resolve a fold-specific or shared split CSV.

    Raises:
        FileNotFoundError: If neither supported split layout exists.
    """
    root = Path(str(cfg["split_dir"]))
    candidates = (
        root / f"fold{fold}" / f"{split}.csv", root / f"{split}.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No slide-embedding {split} split found. Tried:\n  {rendered}")


def build_slide_embedding_loader(
    cfg: Mapping[str, Any],
    split: str,
    fold: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build the shared slide-embedding loader for a run and fold.

    Args:
        cfg: Generated run configuration.
        split: Split name such as ``train``, ``val``, or ``test``.
        fold: Outer fold index used to resolve the split path.
        shuffle: Request shuffling; only training data is shuffled.

    Returns:
        A data loader yielding slide-vector dictionaries.
    """
    from common.datasets.split_tables import load_phase_table

    source = SlideEmbeddingSource.from_config(cfg)
    dataset = SlideEmbeddingDataset(
        source.source_type,
        source.features_path,
        load_phase_table(cfg, split, fold),
        cfg["label_dict"],
        feature_path_column=source.feature_path_column,
        feature_key=source.feature_key,
        feature_dim=source.feature_dim,
        slide_id_key=source.slide_id_key,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=shuffle and split == "train",
        num_workers=int(cfg.get("num_workers", 0)),
    )
