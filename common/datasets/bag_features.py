"""Single-scale bag-features dataset.

Used by SLIP, TOP, and any method that takes one tensor of patch
features per slide.

CSV format:
    slide_id, label
"""
from __future__ import annotations
import os
from typing import Any, Mapping

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from common.datasets.dataset_generic import _load_feature_tensor


class BagFeaturesDataset(Dataset):
    """Load one variable-length patch-feature bag per slide.

    Args:
        csv_path: Manifest containing at least ``slide_id`` and ``label``.
        feature_root: Root used when no explicit feature-path column is set.
        label_dict: Mapping from string labels to integer class indices.
        max_patches: Optional random cap applied after loading a bag.
        ext: Per-slide file suffix used with ``feature_root``.
        feature_path_column: Optional manifest column containing exact paths.
        feature_key: Tensor key used for mapping or HDF5 payloads.
        feature_dim: Expected patch width. A mismatch raises immediately.
        include_metadata: Include slide/case identifiers in returned samples.

    Each sample returns ``(features, label)`` or
    ``(features, metadata, label)``. Features have shape ``[patches, dim]``;
    batching is normally restricted to one slide because bag lengths vary.
    """
    def __init__(self, csv_path: str, feature_root: str,
                 label_dict: dict, max_patches: int | None = None,
                 ext: str = ".pt", feature_path_column: str | None = None,
                 feature_key: str = "features",
                 feature_dim: int | None = None,
                 include_metadata: bool = False):
        self.df = pd.read_csv(csv_path)
        self.feature_root = feature_root
        self.label_dict = label_dict
        self.max_patches = max_patches
        self.ext = ext
        self.feature_path_column = feature_path_column
        self.feature_key = feature_key
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.include_metadata = include_metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row["slide_id"])
        label_key = row["label"]
        label = self.label_dict[label_key] \
                if isinstance(label_key, str) else int(label_key)

        if self.feature_path_column:
            path = row[self.feature_path_column]
        else:
            path = os.path.join(self.feature_root, slide_id + self.ext)
        feats = _load_feature_tensor(path, self.feature_key)
        if feats.ndim != 2 or feats.shape[0] == 0:
            raise ValueError(
                f"Slide {slide_id!r} must contain [patches, dimension], "
                f"got {list(feats.shape)}")
        if self.feature_dim is not None and feats.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Slide {slide_id!r} has patch width {feats.shape[-1]}, "
                f"expected {self.feature_dim}")
        if self.max_patches is not None and feats.shape[0] > self.max_patches:
            idx_perm = torch.randperm(feats.shape[0])[: self.max_patches]
            feats = feats[idx_perm]

        if self.include_metadata:
            metadata = {
                "slide_id": slide_id,
                "case_id": str(row.get("case_id", slide_id)),
            }
            return feats, metadata, label
        return feats, label


def build_bag_loader(
    cfg: Mapping[str, Any], split: str = "train", shuffle: bool = True,
) -> DataLoader:
    """Construct a single-scale patch-bag loader from a run config.

    Args:
        cfg: Run configuration containing split and feature-location fields.
        split: Split filename stem, such as ``train``, ``val``, or ``test``.
        shuffle: Request shuffling; only the training split is shuffled.

    Returns:
        A PyTorch data loader yielding the dataset's bag tuples.
    """
    csv_path = os.path.join(cfg["split_dir"], f"{split}.csv")
    ds = BagFeaturesDataset(
        csv_path=csv_path,
        feature_root=cfg["data_folder_s"],
        label_dict=cfg["label_dict"],
        max_patches=cfg.get("max_patches"),
        ext=cfg.get("feature_ext", ".pt"),
        feature_path_column=cfg.get("feature_path_column"),
        feature_key=cfg.get("feature_key", "features"),
        feature_dim=cfg.get("feature_dim"),
        include_metadata=cfg.get("include_metadata", False))
    return DataLoader(ds, batch_size=cfg.get("batch_size", 1),
                      shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 4))
