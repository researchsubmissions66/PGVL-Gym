"""MSCPT-style dataset.

MSCPT loads:
  - low-res (5x or 10x) features for the whole slide
  - high-res (20x) features for the top-K selected patches

CSV format:
    slide_id, OncoTreeCode, case_id, project_id, Diagnosis, level0_mag
"""
from __future__ import annotations
import os
import pandas as pd
from torch.utils.data import Dataset

from common.datasets.dataset_generic import _load_feature_tensor


class MSCPT_Dataset(Dataset):
    def __init__(self, csv_path: str | pd.DataFrame, feat_data_dir: str,
                 selected_5x_dir: str | None,
                 label_dict: dict, num_k: int = 100,
                 feature_path_column_s: str | None = None,
                 feature_path_column_l: str | None = None,
                 feature_key: str = "features",
                 include_metadata: bool = False,
                 feature_dim: int | None = None):
        self.df = (csv_path.copy() if isinstance(csv_path, pd.DataFrame)
                   else pd.read_csv(csv_path))
        self.feat_dir = feat_data_dir
        self.sel_dir = selected_5x_dir
        self.label_dict = label_dict
        self.num_k = num_k
        self.feature_path_column_s = feature_path_column_s
        self.feature_path_column_l = feature_path_column_l
        self.feature_key = feature_key
        self.include_metadata = include_metadata
        self.feature_dim = (
            int(feature_dim) if feature_dim is not None else None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row["slide_id"])
        label_key = row.get("label", row.get("Diagnosis"))
        label = self.label_dict[label_key] \
                if isinstance(label_key, str) else int(label_key)

        path_s = (row[self.feature_path_column_s]
                  if self.feature_path_column_s else
                  os.path.join(self.feat_dir, f"{slide_id}.pt"))
        feats = _load_feature_tensor(path_s, self.feature_key)

        if self.feature_path_column_l:
            sel = _load_feature_tensor(
                row[self.feature_path_column_l], self.feature_key)
        elif self.sel_dir is not None:
            sel = _load_feature_tensor(
                os.path.join(self.sel_dir, f"{slide_id}.pt"), self.feature_key)
        else:
            sel = feats[: self.num_k]

        if self.feature_dim is not None:
            for scale, features in (("low", feats), ("high", sel)):
                if features.shape[1] != self.feature_dim:
                    raise ValueError(
                        f"MSCPT {scale}-resolution features for {slide_id!r} "
                        f"have width {features.shape[1]}, expected "
                        f"{self.feature_dim}")

        if self.include_metadata:
            metadata = {
                "slide_id": slide_id,
                "case_id": str(row.get("case_id", slide_id)),
            }
            return feats, sel, metadata, label
        return feats, sel, label


def build_mscpt_loader(cfg, split: str = "train", shuffle: bool = True,
                       fold: int | None = None):
    from torch.utils.data import DataLoader
    from common.datasets.split_tables import load_phase_table

    fold = cfg.get("_fold_index", 0) if fold is None else fold
    phase_table = load_phase_table(cfg, split, fold)
    ds = MSCPT_Dataset(
        csv_path=phase_table,
        feat_data_dir=cfg.get("feat_data_dir", ""),
        selected_5x_dir=cfg.get("selected_5x_dir"),
        label_dict=cfg["label_dict"],
        num_k=cfg.get("num_k", 100),
        feature_path_column_s=cfg.get("feature_path_column_s"),
        feature_path_column_l=cfg.get("feature_path_column_l"),
        feature_key=cfg.get("feature_key", "features"),
        include_metadata=cfg.get("include_metadata", False),
        feature_dim=cfg.get("feature_dim"))
    return DataLoader(ds, batch_size=cfg.get("batch_size", 1),
                      shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 4))
