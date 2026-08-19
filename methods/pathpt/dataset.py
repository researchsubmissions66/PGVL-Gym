"""PathPT-style WSI dataset.

PathPT consumes pre-extracted features stored as `.h5` files with
`features` and `coords` keys. Each slide is one bag.

CSV format (each row = one slide):
    slide_id,label
    TCGA-XX-XXXX-01Z,2
"""
from __future__ import annotations
import os
import h5py
import torch
import pandas as pd
from common.configuration import expand_path
from torch.utils.data import Dataset


class WSI_H5_Dataset(Dataset):
    def __init__(self, csv_path: str | pd.DataFrame, feature_root: str,
                 label_dict: dict, patch_num: int | None = None,
                 feature_path_column: str | None = None,
                 feature_key: str = "features",
                 include_metadata: bool = False,
                 random_subsampling: bool = True,
                 feature_dim: int | None = None):
        self.df = (csv_path.copy() if isinstance(csv_path, pd.DataFrame)
                   else pd.read_csv(csv_path))
        self.feature_root = feature_root
        self.label_dict = label_dict
        self.patch_num = patch_num
        self.feature_path_column = feature_path_column
        self.feature_key = feature_key
        self.include_metadata = include_metadata
        self.random_subsampling = bool(random_subsampling)
        self.feature_dim = int(feature_dim) if feature_dim is not None else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row["slide_id"]
        label = self.label_dict[row["label"]] \
                if isinstance(row["label"], str) else int(row["label"])

        path = (row[self.feature_path_column] if self.feature_path_column
                else os.path.join(self.feature_root, f"{slide_id}.h5"))
        path = expand_path(path)
        with h5py.File(path, "r") as f:
            feats = torch.from_numpy(f[self.feature_key][:]).float()
            coords = (torch.from_numpy(f["coords"][:]).float()
                      if "coords" in f else
                      torch.zeros((feats.shape[0], 2), dtype=torch.float32))

        if feats.ndim != 2 or feats.shape[0] == 0:
            raise ValueError(
                f"PathPT slide {slide_id!r} must contain a non-empty "
                f"[patches, dimension] tensor, got {tuple(feats.shape)}")
        if self.feature_dim is not None and feats.shape[1] != self.feature_dim:
            raise ValueError(
                f"PathPT slide {slide_id!r} has width {feats.shape[1]}, "
                f"expected {self.feature_dim}")
        if coords.ndim != 2 or coords.shape[0] != feats.shape[0]:
            raise ValueError(
                f"PathPT slide {slide_id!r} coords do not align with features")
        if not torch.isfinite(feats).all() or not torch.isfinite(coords).all():
            raise ValueError(
                f"PathPT slide {slide_id!r} contains NaN or infinity")

        if self.patch_num is not None and feats.shape[0] > self.patch_num:
            if self.random_subsampling:
                idx = torch.randperm(feats.shape[0])[:self.patch_num]
            else:
                idx = torch.linspace(
                    0, feats.shape[0] - 1, self.patch_num).round().long()
            feats = feats[idx]
            coords = coords[idx]

        if self.include_metadata:
            metadata = {
                "slide_id": str(slide_id),
                "case_id": str(row.get("case_id", slide_id)),
            }
            return feats, coords, metadata, label
        return feats, coords, label


def build_pathpt_loader(cfg, split: str = "train", shuffle: bool = True,
                        fold: int | None = None):
    from torch.utils.data import DataLoader
    from common.datasets.split_tables import load_phase_table

    fold = cfg.get("_fold_index", 0) if fold is None else fold
    phase_table = load_phase_table(cfg, split, fold)
    ds = WSI_H5_Dataset(
        csv_path=phase_table,
        feature_root=cfg.get("feature_root", ""),
        label_dict=cfg["label_dict"],
        patch_num=cfg.get("patch_num"),
        feature_path_column=cfg.get("feature_path_column"),
        feature_key=cfg.get("feature_key", "features"),
        include_metadata=cfg.get("include_metadata", False),
        random_subsampling=split == "train",
        feature_dim=cfg.get("feature_dim"))
    return DataLoader(ds, batch_size=cfg.get("batch_size", 1),
                      shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 4))
