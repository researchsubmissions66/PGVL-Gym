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
from torch.utils.data import Dataset


class WSI_H5_Dataset(Dataset):
    def __init__(self, csv_path: str, feature_root: str,
                 label_dict: dict, patch_num: int | None = None,
                 feature_path_column: str | None = None,
                 feature_key: str = "features",
                 include_metadata: bool = False):
        self.df = pd.read_csv(csv_path)
        self.feature_root = feature_root
        self.label_dict = label_dict
        self.patch_num = patch_num
        self.feature_path_column = feature_path_column
        self.feature_key = feature_key
        self.include_metadata = include_metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row["slide_id"]
        label = self.label_dict[row["label"]] \
                if isinstance(row["label"], str) else int(row["label"])

        path = (row[self.feature_path_column] if self.feature_path_column
                else os.path.join(self.feature_root, f"{slide_id}.h5"))
        with h5py.File(path, "r") as f:
            feats = torch.from_numpy(f[self.feature_key][:]).float()
            coords = (torch.from_numpy(f["coords"][:]).float()
                      if "coords" in f else
                      torch.zeros((feats.shape[0], 2), dtype=torch.float32))

        if self.patch_num is not None and feats.shape[0] > self.patch_num:
            idx = torch.randperm(feats.shape[0])[:self.patch_num]
            feats = feats[idx]
            coords = coords[idx]

        if self.include_metadata:
            metadata = {
                "slide_id": str(slide_id),
                "case_id": str(row.get("case_id", slide_id)),
            }
            return feats, coords, metadata, label
        return feats, coords, label


def build_pathpt_loader(cfg, split: str = "train", shuffle: bool = True):
    from torch.utils.data import DataLoader
    csv_path = os.path.join(cfg["split_dir"], f"{split}.csv")
    ds = WSI_H5_Dataset(
        csv_path=csv_path,
        feature_root=cfg["feature_root"],
        label_dict=cfg["label_dict"],
        patch_num=cfg.get("patch_num"),
        feature_path_column=cfg.get("feature_path_column"),
        feature_key=cfg.get("feature_key", "features"),
        include_metadata=cfg.get("include_metadata", False))
    return DataLoader(ds, batch_size=cfg.get("batch_size", 1),
                      shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 4))
