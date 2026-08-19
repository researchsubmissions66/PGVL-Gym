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
        max_patches: Optional cap applied after loading a bag.
        ext: Per-slide file suffix used with ``feature_root``.
        feature_path_column: Optional manifest column containing exact paths.
        feature_key: Tensor key used for mapping or HDF5 payloads.
        feature_dim: Expected patch width. A mismatch raises immediately.
        include_metadata: Include slide/case identifiers in returned samples.

    Each sample returns ``(features, label)`` or
    ``(features, metadata, label)``. Features have shape ``[patches, dim]``;
    batching is normally restricted to one slide because bag lengths vary.
    """
    def __init__(self, csv_path: str | pd.DataFrame, feature_root: str,
                 label_dict: dict, max_patches: int | None = None,
                 ext: str = ".pt", feature_path_column: str | None = None,
                 feature_key: str = "features",
                 feature_dim: int | None = None,
                 include_metadata: bool = False,
                 random_subsampling: bool = True):
        self.df = (csv_path.copy() if isinstance(csv_path, pd.DataFrame)
                   else pd.read_csv(csv_path))
        self.feature_root = feature_root
        self.label_dict = label_dict
        self.max_patches = max_patches
        self.ext = ext
        self.feature_path_column = feature_path_column
        self.feature_key = feature_key
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.include_metadata = include_metadata
        self.random_subsampling = bool(random_subsampling)

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
            if self.random_subsampling:
                idx_perm = torch.randperm(feats.shape[0])[: self.max_patches]
            else:
                # Validation/test must be invariant to worker scheduling and
                # training duration. Evenly cover the complete slide bag.
                idx_perm = torch.linspace(
                    0, feats.shape[0] - 1, self.max_patches).round().long()
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
    fold: int | None = None,
) -> DataLoader:
    """Construct a single-scale patch-bag loader from a run config.

    Args:
        cfg: Run configuration containing split and feature-location fields.
        split: Split filename stem, such as ``train``, ``val``, or ``test``.
        shuffle: Request shuffling; only the training split is shuffled.
        fold: Fold index used for nested or wide split resolution. When omitted,
            the private ``_fold_index`` dispatch field defaults to zero.

    Returns:
        A PyTorch data loader yielding the dataset's bag tuples.
    """
    from common.datasets.split_tables import load_phase_table

    fold = cfg.get("_fold_index", 0) if fold is None else fold
    phase_table = load_phase_table(cfg, split, fold)
    feature_path_column = cfg.get("feature_path_column")
    feature_root = cfg.get("data_folder_s")
    if not feature_path_column and not feature_root:
        raise KeyError(
            "A patch-bag loader requires 'feature_path_column' or "
            "'data_folder_s'.")
    ds = BagFeaturesDataset(
        csv_path=phase_table,
        feature_root=str(feature_root or ""),
        label_dict=cfg["label_dict"],
        max_patches=cfg.get("max_patches"),
        ext=cfg.get("feature_ext", ".pt"),
        feature_path_column=feature_path_column,
        feature_key=cfg.get("feature_key", "features"),
        feature_dim=cfg.get("feature_dim"),
        include_metadata=cfg.get("include_metadata", False),
        random_subsampling=split == "train")
    return DataLoader(ds, batch_size=cfg.get("batch_size", 1),
                      shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 4))
