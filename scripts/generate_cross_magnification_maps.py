"""Generate CoD-MIL low-to-high patch correspondence maps from CLAM H5 files.

Each output is an ``(N_low, max_matches)`` long tensor.  Row ``i`` contains
the indices of high-magnification patches whose centres lie inside low patch
``i``; unused cells are ``-1``.  This is the format consumed by CoD-MIL.

Coordinates must be in the same level-0 reference frame.  Pass patch and
stride sizes in that coordinate system; do not use displayed magnification
labels as a substitute for physical patch dimensions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch


def load_coords(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"{path} has no 'coords' dataset")
        coords = np.asarray(handle["coords"][:], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) coords in {path}, got {coords.shape}")
    return coords


def level0_patch_size(path: Path) -> float | None:
    """Return the patch edge length in the level-0 frame, if the file records it.

    A patch cropped at a target magnification covers a different number of
    level-0 pixels depending on the scanner's base magnification: a 224 px tile
    at 10x spans 896 level-0 px on a 40x slide but only 448 on a 20x one. Both
    kinds occur in the same cohort, so a single command-line size is wrong for
    one of them -- silently, by binding roughly four times too many high-power
    patches to each low-power row. Prefer the per-slide value when present.
    """
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            return None
        value = handle["coords"].attrs.get("patch_size_level0")
    return float(value) if value is not None else None


def resolve_patch_size(path: Path, override: int | None, role: str) -> float:
    """Choose the per-slide level-0 patch size, falling back to the override."""
    recorded = level0_patch_size(path)
    if recorded is not None and recorded > 0:
        return recorded
    if override is None:
        raise ValueError(
            f"{path} records no 'patch_size_level0'; pass --{role}-patch-size "
            "with the level-0 edge length for these slides.")
    return float(override)


def build_correspondence(low: np.ndarray, high: np.ndarray,
                         low_patch_size: float, high_patch_size: float,
                         low_step_size: float) -> torch.Tensor:
    if len(low) == 0 or len(high) == 0:
        raise ValueError("Both low- and high-magnification coordinates must be non-empty")
    if min(low_patch_size, high_patch_size, low_step_size) <= 0:
        raise ValueError("Patch and stride sizes must be positive")

    # CLAM coordinates normally occupy a regular grid.  Index low-patch
    # origins by grid cell, then inspect only cells which can contain a high
    # patch centre.  This avoids an O(N_low * N_high) dense comparison.
    origin = low.min(axis=0)
    low_grid = np.floor((low - origin) / low_step_size + 1e-6).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (gx, gy) in enumerate(low_grid):
        buckets[(int(gx), int(gy))].append(index)

    centre = high + high_patch_size / 2.0
    coverage = int(np.ceil(low_patch_size / low_step_size))
    matches: list[list[int]] = [[] for _ in range(len(low))]
    for high_index, point in enumerate(centre):
        cell = np.floor((point - origin) / low_step_size + 1e-6).astype(np.int64)
        # A rectangle starting up to ``coverage - 1`` cells before the centre
        # may contain it.  The exact rectangle test handles overlaps safely.
        for gx in range(int(cell[0]) - coverage + 1, int(cell[0]) + 1):
            for gy in range(int(cell[1]) - coverage + 1, int(cell[1]) + 1):
                for low_index in buckets.get((gx, gy), []):
                    x, y = low[low_index]
                    if x <= point[0] < x + low_patch_size and y <= point[1] < y + low_patch_size:
                        matches[low_index].append(high_index)

    max_matches = max(map(len, matches), default=0)
    if max_matches == 0:
        raise ValueError(
            "No low/high correspondence was found. Verify that both coordinate "
            "sets use the same reference frame and patch/stride sizes.")
    unmatched = sum(not selected for selected in matches)
    if unmatched:
        raise ValueError(
            f"{unmatched} low-resolution patches have no high-resolution "
            "correspondence; CoD-MIL may select any low-resolution patch")
    output = torch.full((len(low), max_matches), -1, dtype=torch.long)
    for low_index, selected in enumerate(matches):
        if selected:
            output[low_index, :len(selected)] = torch.tensor(selected, dtype=torch.long)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-h5-dir", type=Path, required=True)
    parser.add_argument("--high-h5-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--low-patch-size", type=int,
        help="Level-0 edge length of a low-magnification patch. Only used for "
             "slides whose H5 does not record 'patch_size_level0'.")
    parser.add_argument(
        "--high-patch-size", type=int,
        help="Level-0 edge length of a high-magnification patch. Only used as "
             "a fallback, as above.")
    parser.add_argument("--low-step-size", type=int,
                        help="Defaults to the low patch size (non-overlapping)")
    parser.add_argument(
        "--manifest", type=Path,
        help="Cohort manifest.csv. When given, one map is written per manifest "
             "slide id and named after it, which is how CoD-MIL's loader looks "
             "maps up. Patch files are matched by slide-id prefix, so stores "
             "that suffix filenames (…_patches.h5) or carry a UUID resolve.")
    parser.add_argument("--slide-id-column", default="slide_id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _index_by_slide_id(directory: Path) -> dict[str, Path]:
    """Map a slide-id prefix to its patch H5 in a trident/CLAM store."""
    index: dict[str, Path] = {}
    for path in sorted(directory.glob("*.h5")):
        key = path.name.split(".")[0]
        index.setdefault(key, path)
    return index


def _manifest_pairs(args) -> list[tuple[str, Path, Path]]:
    """Resolve (output name, low H5, high H5) for every slide in the manifest."""
    import csv as _csv

    low_index = _index_by_slide_id(args.low_h5_dir)
    high_index = _index_by_slide_id(args.high_h5_dir)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{args.manifest} has no rows")
    if args.slide_id_column not in rows[0]:
        raise ValueError(
            f"{args.manifest} has no column {args.slide_id_column!r}")

    pairs: list[tuple[str, Path, Path]] = []
    for row in rows:
        slide_id = str(row[args.slide_id_column]).strip()
        key = slide_id.split(".")[0]
        low_path, high_path = low_index.get(key), high_index.get(key)
        if low_path is None or high_path is None:
            missing = "low" if low_path is None else "high"
            print(f"skip  {slide_id} (no {missing}-mag patch H5)")
            continue
        pairs.append((slide_id, low_path, high_path))
    return pairs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_files = sorted(args.low_h5_dir.glob("*.h5"))
    if not low_files:
        raise FileNotFoundError(f"No .h5 files in {args.low_h5_dir}")

    if args.manifest:
        pairs = _manifest_pairs(args)
    else:
        pairs = [(low_path.stem, low_path, args.high_h5_dir / low_path.name)
                 for low_path in low_files]

    written = skipped = 0
    for name, low_path, high_path in pairs:
        if not high_path.is_file():
            print(f"skip  {name} (missing high-mag H5)")
            skipped += 1
            continue
        output_path = args.output_dir / f"{name}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"skip  {output_path.name} (already exists)")
            skipped += 1
            continue
        low_size = resolve_patch_size(low_path, args.low_patch_size, "low")
        high_size = resolve_patch_size(high_path, args.high_patch_size, "high")
        # Overlap is expressed in the same level-0 frame as the patch size.
        low_step = float(args.low_step_size) if args.low_step_size else low_size
        mapping = build_correspondence(
            load_coords(low_path), load_coords(high_path), low_size,
            high_size, low_step)
        torch.save(mapping, output_path)
        print(f"write {output_path.name}: {tuple(mapping.shape)} "
              f"(low={low_size:g} high={high_size:g} level-0 px)")
        written += 1
    print(f"Completed: {written} written, {skipped} skipped")


if __name__ == "__main__":
    main()
