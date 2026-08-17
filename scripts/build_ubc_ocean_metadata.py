#!/usr/bin/env python3
"""Build the UBC-OCEAN annotation table consumed by the benchmark protocol.

The table is written in the same canonical schema as ``metadata/tcga_*.csv`` so
one protocol reads every cohort the same way. The source of truth is the cleaned
release annotation table; this script only reshapes it, and always writes a
fresh file from that source rather than editing the destination in place.

``is_tma`` is derived from the recorded magnification. In the UBC-OCEAN release
the tissue microarray cores are digitised at 40x (mpp 0.25) and the whole-slide
images at 20x (mpp 0.5); the split is exact across all 538 images and matches the
25 TMAs the release documents. The protocol filters on this column, and pandas
must infer it as boolean, so it is written as ``True``/``False``.

Usage
-----
    python scripts/build_ubc_ocean_metadata.py
    python scripts/build_ubc_ocean_metadata.py --check   # verify, write nothing
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SOURCE = Path(
    "/work/hdd/bhwm/metadata/RAW_DATA_Cleaned/UBC-OCEAN/metadata.csv")
DESTINATION = REPO / "metadata" / "ubc_ocean.csv"
FEATURE_ROOT = Path("/work/hdd/bhwm/UBC-OCEAN")

# Magnification recorded for tissue microarray cores; see the module docstring.
TMA_MAGNIFICATION = "40x"

EXPECTED_LABELS = {"CC", "EC", "HGSC", "LGSC", "MC"}

COLUMNS = [
    "filename", "filepath", "label", "subclass", "patient_id", "magnification",
    "train_test", "fold", "dataset", "root_dir", "slide_id", "organ",
    "OncoTreeCode", "case_id", "image_id", "is_tma",
]


def build_rows(source: Path) -> list[dict[str, str]]:
    """Reshape the release annotation table into the canonical schema."""
    with open(source, newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"source annotation table is empty: {source}")

    rows: list[dict[str, str]] = []
    for record in records:
        filename = str(record["filename"]).strip()
        # The feature stores key slides by the bare stem, e.g. train_4.h5.
        slide_id = Path(filename).stem
        label = str(record["label"]).strip()
        magnification = str(record.get("magnification", "")).strip()
        rows.append({
            "filename": filename,
            "filepath": str(record.get("filepath", "")).strip(),
            "label": label,
            "subclass": str(record.get("subclass", "")).strip(),
            "patient_id": str(record.get("patient_id", "")).strip(),
            "magnification": magnification,
            "train_test": str(record.get("train_test", "")).strip(),
            "fold": str(record.get("fold", "")).strip(),
            "dataset": "UBC-OCEAN",
            "root_dir": str(FEATURE_ROOT),
            "slide_id": slide_id,
            "organ": "OVARY",
            "OncoTreeCode": label,
            # Each UBC-OCEAN image is its own case: patient_id is unique across
            # all 538 rows, so slide-level and case-level splits coincide.
            "case_id": slide_id,
            "image_id": slide_id.removeprefix("train_"),
            "is_tma": str(magnification == TMA_MAGNIFICATION),
        })
    return rows


def check(rows: list[dict[str, str]]) -> list[str]:
    """Return a list of problems; empty means the table is usable."""
    problems: list[str] = []

    slide_ids = [row["slide_id"] for row in rows]
    duplicates = [key for key, count in Counter(slide_ids).items() if count > 1]
    if duplicates:
        problems.append(f"duplicate slide_id values: {duplicates[:5]}")

    labels = set(row["label"] for row in rows)
    unexpected = labels - EXPECTED_LABELS
    if unexpected:
        problems.append(f"unexpected labels: {sorted(unexpected)}")
    absent = EXPECTED_LABELS - labels
    if absent:
        problems.append(f"labels with no slides: {sorted(absent)}")

    if any(row["is_tma"] not in {"True", "False"} for row in rows):
        problems.append("is_tma must be written as True/False for the filter")

    if not any(row["is_tma"] == "True" for row in rows):
        problems.append("no TMA rows found; check the magnification column")

    return problems


def report(rows: list[dict[str, str]]) -> None:
    wsi = [row for row in rows if row["is_tma"] == "False"]
    print(f"  rows            {len(rows)}")
    print(f"  whole slides    {len(wsi)}  (kept by the protocol's is_tma filter)")
    print(f"  TMA cores       {len(rows) - len(wsi)}  (filtered out)")
    counts = Counter(row["label"] for row in wsi)
    print("  label counts    "
          + ", ".join(f"{label}={counts[label]}" for label in sorted(counts)))

    for resolution, encoder in (("5x_256px_0px_overlap", "features_conch_v1"),
                                ("10x_256px_0px_overlap", "features_conch_v1"),
                                ("20x_256px_0px_overlap", "features_conch_v1"),
                                ("20x_256px_0px_overlap", "features_keep")):
        directory = FEATURE_ROOT / resolution / encoder
        available = (
            {path.stem for path in directory.glob("*.h5")}
            if directory.is_dir() else set()
        )
        covered = sum(1 for row in wsi if row["slide_id"] in available)
        print(f"  {resolution.split('_')[0]:>3} {encoder:<20} "
              f"{covered:>4}/{len(wsi)} slides have features")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--check", action="store_true",
                        help="validate and report without writing")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"FATAL: source annotation table not found: {args.source}",
              file=sys.stderr)
        return 1

    rows = build_rows(args.source)
    problems = check(rows)

    print(f"UBC-OCEAN metadata from {args.source}")
    report(rows)
    for problem in problems:
        print(f"  ! {problem}", file=sys.stderr)
    if problems:
        return 1

    if args.check:
        print("\n  --check: nothing written")
        return 0

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with open(args.destination, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
