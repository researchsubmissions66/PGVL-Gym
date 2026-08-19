"""Legacy raw-image loader retained for ConVLM data exploration.

The benchmark path does not use this loader. ConVLM's training input is the
offline patch-embedding bag produced by its visual feature-extraction stage, so
``train.py`` dispatches registered runs through
``common.datasets.bag_features``. This module remains available for inspecting
raw upstream layouts, but using it would define a different experiment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ConVLMPatchDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_root: str | Path | None,
                 label_dict: dict[str, int], image_column: str = "image_path",
                 label_column: str = "label", training: bool = False,
                 image_size: int = 448,
                 slide_tile_path_column: str | None = None,
                 max_tiles_per_slide: int | None = None,
                 include_metadata: bool = False):
        self.data = pd.read_csv(csv_path)
        self.root = Path(image_root) if image_root else None
        self.label_dict = label_dict
        self.slide_tile_path_column = slide_tile_path_column
        self.max_tiles_per_slide = max_tiles_per_slide
        self.training = training
        self.include_metadata = include_metadata
        if (image_column not in self.data
                and (not slide_tile_path_column
                     or slide_tile_path_column not in self.data)):
            raise ValueError(
                f"{csv_path} must contain '{image_column}' or "
                f"'{slide_tile_path_column}'")
        if label_column not in self.data:
            raise ValueError(f"{csv_path} must contain '{label_column}'")
        self.image_column = image_column
        self.label_column = label_column
        ops: list[Any] = [transforms.Resize((image_size, image_size))]
        if training:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform = transforms.Compose(ops)

    @staticmethod
    def _image_files(directory: Path) -> list[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
        if not directory.is_dir():
            raise FileNotFoundError(f"ConVLM tile directory is missing: {directory}")
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions)
        if not files:
            raise FileNotFoundError(f"ConVLM tile directory is empty: {directory}")
        return files

    def _select_tiles(self, files: list[Path]) -> list[Path]:
        maximum = self.max_tiles_per_slide
        if maximum is None or len(files) <= maximum:
            return files
        if self.training:
            indices = torch.randperm(len(files))[:maximum].sort().values.tolist()
        else:
            indices = torch.linspace(
                0, len(files) - 1, steps=maximum).round().long().tolist()
        return [files[index] for index in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        raw_label = row[self.label_column]
        label = self.label_dict[raw_label] if isinstance(raw_label, str) else int(raw_label)
        if self.slide_tile_path_column:
            directory = Path(str(row[self.slide_tile_path_column]))
            if not directory.is_absolute() and self.root is not None:
                directory = self.root / directory
            files = self._select_tiles(self._image_files(directory))
            images = torch.stack([
                self.transform(Image.open(path).convert("RGB")) for path in files
            ])
            if self.include_metadata:
                metadata = {
                    "slide_id": str(row.get("slide_id", directory.name)),
                    "case_id": str(row.get("case_id", directory.name)),
                }
                return images, metadata, label
            return images, label

        image_path = Path(str(row[self.image_column]))
        if not image_path.is_absolute() and self.root is not None:
            image_path = self.root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"ConVLM patch is missing: {image_path}")
        image = self.transform(Image.open(image_path).convert("RGB"))
        return image, label


def _split_csv(cfg: dict, split: str, fold: int) -> Path:
    root = Path(cfg["split_dir"])
    candidates = (root / f"fold{fold}" / f"{split}.csv",
                  root / f"{split}.csv",
                  root / f"splits_{fold}.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No ConVLM {split} split found. Tried:\n  {rendered}")


def build_convlm_loader(cfg: dict, split: str, fold: int, shuffle: bool = True):
    dataset = ConVLMPatchDataset(
        _split_csv(cfg, split, fold),
        cfg.get("image_root"), cfg["label_dict"],
        image_column=cfg.get("image_column", "image_path"),
        label_column=cfg.get("label_column", "label"),
        training=split == "train",
        image_size=cfg.get("image_size", 448),
        slide_tile_path_column=cfg.get("slide_tile_path_column"),
        max_tiles_per_slide=cfg.get("max_tiles_per_slide"),
        include_metadata=cfg.get("include_metadata", False),
    )
    return DataLoader(dataset, batch_size=cfg.get("batch_size", 16),
                      shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 4), pin_memory=True)
