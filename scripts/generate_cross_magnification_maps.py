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


def build_correspondence(low: np.ndarray, high: np.ndarray,
                         low_patch_size: int, high_patch_size: int,
                         low_step_size: int) -> torch.Tensor:
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
    parser.add_argument("--low-patch-size", type=int, required=True)
    parser.add_argument("--high-patch-size", type=int, required=True)
    parser.add_argument("--low-step-size", type=int,
                        help="Defaults to --low-patch-size for non-overlapping low patches")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low_step_size = args.low_step_size or args.low_patch_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_files = sorted(args.low_h5_dir.glob("*.h5"))
    if not low_files:
        raise FileNotFoundError(f"No .h5 files in {args.low_h5_dir}")

    written = skipped = 0
    for low_path in low_files:
        high_path = args.high_h5_dir / low_path.name
        if not high_path.is_file():
            print(f"skip  {low_path.name} (missing high-mag H5)")
            skipped += 1
            continue
        output_path = args.output_dir / f"{low_path.stem}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"skip  {output_path.name} (already exists)")
            skipped += 1
            continue
        mapping = build_correspondence(
            load_coords(low_path), load_coords(high_path), args.low_patch_size,
            args.high_patch_size, low_step_size)
        torch.save(mapping, output_path)
        print(f"write {output_path.name}: {tuple(mapping.shape)}")
        written += 1
    print(f"Completed: {written} written, {skipped} skipped")


if __name__ == "__main__":
    main()
