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
import torch
from torch.utils.data import DataLoader, Dataset


_FILE_SUFFIXES = {".pt", ".pth", ".h5", ".hdf5", ".svs", ".tif", ".tiff"}


def normalise_slide_id(value: str) -> str:
    """Remove only known file suffixes while preserving periods in slide IDs."""
    path = Path(str(value))
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

    Samples are dictionaries with ``feat``, ``label``, and ``slide_id``.
    Entries without a matching feature are excluded; an entirely unmatched
    split raises instead of producing an empty loader.
    """

    def __init__(
        self,
        source_type: str,
        features_path: str | Path,
        csv_path: str | Path,
        label_dict: Mapping[str, int],
        feature_path_column: str | None = None,
        feature_key: str = "features",
        feature_dim: int | None = None,
        slide_id_key: str = "filenames",
    ):
        self.source_type = (
            "per_slide_torch" if source_type == "per_slide_pth" else source_type)
        self.features_path = Path(features_path)
        self.label_dict = dict(label_dict)
        self.feature_path_column = feature_path_column
        self.feature_key = feature_key
        self.slide_id_key = slide_id_key
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.entries = self._read_entries(csv_path)

        if self.source_type == "pkl":
            with self.features_path.open("rb") as handle:
                payload = pickle.load(handle)
            self.embeddings = torch.as_tensor(
                np.asarray(payload[self.feature_key])).float()
            ids = payload[self.slide_id_key]
            self.paths = {
                normalise_slide_id(slide_id): index
                for index, slide_id in enumerate(ids)
            }
        elif self.source_type == "per_slide_torch":
            self.embeddings = None
            paths = (
                path for pattern in ("*.pt", "*.pth")
                for path in self.features_path.rglob(pattern)
            )
            self.paths = {
                normalise_slide_id(path.name): path for path in paths}
        elif self.source_type == "per_slide_h5":
            self.embeddings = None
            paths = (
                path for pattern in ("*.h5", "*.hdf5")
                for path in self.features_path.rglob(pattern)
            )
            self.paths = {
                normalise_slide_id(path.name): path for path in paths}
        else:
            raise ValueError(
                "source_type must be pkl, per_slide_h5, or per_slide_torch")

        self.entries = [
            entry for entry in self.entries
            if ((entry[2] and Path(entry[2]).is_file())
                or normalise_slide_id(entry[0]) in self.paths)
        ]
        if not self.entries:
            raise RuntimeError(
                "No split IDs match the registered slide-embedding store")

    def _read_entries(self, csv_path: str | Path):
        with Path(csv_path).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return []
        header = [item.strip().lower() for item in rows[0]]
        has_header = "slide_id" in header or "label" in header
        if has_header:
            slide_column = header.index("slide_id") if "slide_id" in header else 0
            label_column = header.index("label") if "label" in header else 1
            feature_column = (
                header.index(self.feature_path_column.lower())
                if self.feature_path_column
                and self.feature_path_column.lower() in header else None)
            rows = rows[1:]
        else:
            slide_column, label_column, feature_column = 0, 1, None

        entries = []
        for row in rows:
            if len(row) <= max(slide_column, label_column):
                continue
            slide_id = row[slide_column].strip()
            raw_label = row[label_column].strip()
            try:
                label = (
                    self.label_dict[raw_label]
                    if raw_label in self.label_dict else int(raw_label))
            except ValueError:
                continue
            explicit_path = (
                row[feature_column].strip()
                if feature_column is not None and len(row) > feature_column
                else "")
            entries.append((slide_id, label, explicit_path))
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
        payload = torch.load(path, map_location="cpu", weights_only=False)
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
        slide_id, label, explicit_path = self.entries[index]
        pointer = (
            Path(explicit_path) if explicit_path
            else self.paths[normalise_slide_id(slide_id)])
        feature = (
            self.embeddings[pointer]
            if self.embeddings is not None else self._load_tensor(pointer))
        feature = feature.float().view(-1)
        if self.feature_dim is not None and feature.numel() != self.feature_dim:
            raise ValueError(
                f"Slide {slide_id!r} has embedding width {feature.numel()}, "
                f"expected {self.feature_dim}")
        return {"feat": feature, "label": label, "slide_id": slide_id}


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
    source = SlideEmbeddingSource.from_config(cfg)
    dataset = SlideEmbeddingDataset(
        source.source_type,
        source.features_path,
        split_csv(cfg, split, fold),
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
