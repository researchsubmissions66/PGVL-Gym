"""Data loader for the original CoD-MIL feature and map contract.

Unlike the generic dual-scale CLAM loader, CoD-MIL also needs the precomputed
low-to-high correspondence map for the exact slide in a batch.  Keeping this
loader local prevents that additional artifact from leaking into every other
method's batch format.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common.datasets.dataset_generic import _load_feature_tensor


def _artifact_candidates(root: Path, slide_id: str) -> list[Path]:
    name = Path(str(slide_id)).name
    stem = Path(name).stem
    return [
        root / f"{name}.pt",
        root / f"{stem}.pt",
        root / "pt_files" / f"{name}.pt",
        root / "pt_files" / f"{stem}.pt",
    ]


def _find_artifact(root: str | Path, slide_id: str, purpose: str) -> Path:
    candidates = _artifact_candidates(Path(root), slide_id)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {purpose} for slide '{slide_id}'. Tried:\n  {rendered}")


class CoDMILFeaturesDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, low_feature_root: str,
                 high_feature_root: str, map_dir: str, label_dict: dict,
                 low_feature_column: str | None = None,
                 high_feature_column: str | None = None,
                 feature_key: str = "features",
                 feature_dim: int | None = None,
                 include_metadata: bool = False):
        self.annotations = annotations.reset_index(drop=True)
        self.low_feature_root = low_feature_root
        self.high_feature_root = high_feature_root
        self.map_dir = map_dir
        self.label_dict = label_dict
        self.low_feature_column = low_feature_column
        self.high_feature_column = high_feature_column
        self.feature_key = feature_key
        self.feature_dim = (
            int(feature_dim) if feature_dim is not None else None)
        self.include_metadata = bool(include_metadata)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        slide_id = str(row["slide_id"])
        label_value = row["label"]
        label = self.label_dict[label_value] if isinstance(label_value, str) else int(label_value)
        low_path = (Path(str(row[self.low_feature_column]))
                    if self.low_feature_column else
                    _find_artifact(self.low_feature_root, slide_id,
                                   "low-resolution features"))
        high_path = (Path(str(row[self.high_feature_column]))
                     if self.high_feature_column else
                     _find_artifact(self.high_feature_root, slide_id,
                                    "high-resolution features"))
        low = _load_feature_tensor(low_path, self.feature_key)
        high = _load_feature_tensor(high_path, self.feature_key)
        if self.feature_dim is not None:
            for scale, features in (("low", low), ("high", high)):
                if features.shape[1] != self.feature_dim:
                    raise ValueError(
                        f"CoD-MIL {scale}-resolution features for {slide_id!r} "
                        f"have width {features.shape[1]}, expected "
                        f"{self.feature_dim}")
        map_path = _find_artifact(
            self.map_dir, slide_id, "cross-magnification map")
        mapping = torch.load(
            map_path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(mapping):
            raise TypeError(
                f"CoD-MIL correspondence map must be a tensor: {map_path}")
        if (mapping.dtype == torch.bool or torch.is_floating_point(mapping)
                or torch.is_complex(mapping)):
            raise TypeError(
                f"CoD-MIL correspondence map must contain integer indices: "
                f"{map_path}")
        if (mapping.ndim != 2 or mapping.shape[0] != low.shape[0]
                or mapping.shape[1] == 0):
            raise ValueError(
                f"CoD-MIL correspondence map for {slide_id!r} must have "
                f"shape [{low.shape[0]}, children], got "
                f"{tuple(mapping.shape)}")
        mapping = mapping.long()
        if (mapping < -1).any() or (mapping >= high.shape[0]).any():
            raise ValueError(
                f"CoD-MIL correspondence map for {slide_id!r} contains "
                f"indices outside [-1, {high.shape[0]})")
        if not (mapping >= 0).any(dim=1).all():
            raise ValueError(
                f"CoD-MIL correspondence map for {slide_id!r} has a "
                "low-resolution patch with no high-resolution match")
        if self.include_metadata:
            metadata = {
                "slide_id": slide_id,
                "case_id": str(row.get("case_id", slide_id)),
            }
            return low, high, mapping, metadata, label
        return low, high, mapping, label


def _read_split(cfg: dict, split: str, fold: int) -> pd.DataFrame:
    split_dir = Path(cfg["split_dir"])
    candidates = [split_dir / f"fold{fold}" / f"{split}.csv",
                  split_dir / f"{split}.csv",
                  split_dir / f"splits_{fold}.csv"]
    split_path = next((path for path in candidates if path.is_file()), None)
    if split_path is None:
        rendered = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No {split} split found for fold {fold}. Tried:\n  {rendered}")

    annotation = pd.read_csv(cfg["dataset_csv"], dtype={"slide_id": str})
    if {"slide_id", "label"} - set(annotation.columns):
        raise ValueError(f"{cfg['dataset_csv']} must contain slide_id and label columns")
    split_data = pd.read_csv(split_path, dtype=str)
    if "slide_id" in split_data.columns:
        selected = split_data["slide_id"].dropna().astype(str)
    elif split in split_data.columns:
        selected = split_data[split].dropna().astype(str)
    else:
        raise ValueError(f"{split_path} must contain a '{split}' or 'slide_id' column")
    annotation_ids = annotation["slide_id"].astype(str)
    duplicate_annotations = annotation_ids[annotation_ids.duplicated()].unique()
    if len(duplicate_annotations):
        raise ValueError(
            f"{cfg['dataset_csv']} repeats slide IDs: "
            + ", ".join(map(str, duplicate_annotations[:3])))
    duplicate_selected = selected[selected.duplicated()].unique()
    if len(duplicate_selected):
        raise ValueError(
            f"{split_path} repeats slide IDs: "
            + ", ".join(map(str, duplicate_selected[:3])))
    selected_ids = set(selected)
    missing = sorted(selected_ids - set(annotation_ids))
    if missing:
        raise ValueError(
            f"{split_path} contains {len(missing)} slide IDs absent from "
            f"{cfg['dataset_csv']}: {', '.join(missing[:3])}")
    result = annotation[annotation_ids.isin(selected_ids)]
    if result.empty:
        raise ValueError(f"No annotations match the {split} IDs in {split_path}")
    return result


def _collate(batch):
    if len(batch) != 1:
        raise ValueError("CoD-MIL requires batch_size=1 because bags have variable patch counts")
    item = batch[0]
    metadata = None
    if len(item) == 5:
        low, high, mapping, metadata, label = item
    else:
        low, high, mapping, label = item
    if metadata is not None:
        return (
            low, high, mapping,
            {"slide_id": [metadata["slide_id"]],
             "case_id": [metadata["case_id"]]},
            torch.tensor([label], dtype=torch.long),
        )
    return low, high, mapping, torch.tensor([label], dtype=torch.long)


def build_cod_mil_loader(cfg: dict, split: str, fold: int, shuffle: bool = True):
    map_dir = cfg.get("cross_mag_map_dir")
    if not map_dir:
        raise KeyError(
            "CoD-MIL requires 'cross_mag_map_dir'. Generate maps with "
            "scripts/generate_cross_magnification_maps.py first.")
    dataset = CoDMILFeaturesDataset(
        _read_split(cfg, split, fold), cfg.get("data_folder_s", ""),
        cfg.get("data_folder_l", ""),
        map_dir, cfg["label_dict"],
        low_feature_column=cfg.get("feature_path_column_s"),
        high_feature_column=cfg.get("feature_path_column_l"),
        feature_key=cfg.get("feature_key", "features"),
        feature_dim=cfg.get("feature_dim"),
        include_metadata=cfg.get("include_metadata", False))
    return DataLoader(dataset, batch_size=1, shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 0), collate_fn=_collate)
