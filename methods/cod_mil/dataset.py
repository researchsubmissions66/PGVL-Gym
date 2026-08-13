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
                 feature_key: str = "features"):
        self.annotations = annotations.reset_index(drop=True)
        self.low_feature_root = low_feature_root
        self.high_feature_root = high_feature_root
        self.map_dir = map_dir
        self.label_dict = label_dict
        self.low_feature_column = low_feature_column
        self.high_feature_column = high_feature_column
        self.feature_key = feature_key

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
        mapping = torch.load(_find_artifact(self.map_dir, slide_id, "cross-magnification map"),
                             map_location="cpu", weights_only=True).long()
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
    result = annotation[annotation["slide_id"].astype(str).isin(set(selected))]
    if result.empty:
        raise ValueError(f"No annotations match the {split} IDs in {split_path}")
    return result


def _collate(batch):
    if len(batch) != 1:
        raise ValueError("CoD-MIL requires batch_size=1 because bags have variable patch counts")
    low, high, mapping, label = batch[0]
    return low, high, mapping, torch.tensor([label], dtype=torch.long)


def build_cod_mil_loader(cfg: dict, split: str, fold: int, shuffle: bool = True):
    map_dir = cfg.get("cross_mag_map_dir")
    if not map_dir:
        raise KeyError(
            "CoD-MIL requires 'cross_mag_map_dir'. Generate maps with "
            "scripts/generate_cross_magnification_maps.py first.")
    dataset = CoDMILFeaturesDataset(
        _read_split(cfg, split, fold), cfg["data_folder_s"], cfg["data_folder_l"],
        map_dir, cfg["label_dict"],
        low_feature_column=cfg.get("feature_path_column_s"),
        high_feature_column=cfg.get("feature_path_column_l"),
        feature_key=cfg.get("feature_key", "features"))
    return DataLoader(dataset, batch_size=1, shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 0), collate_fn=_collate)
