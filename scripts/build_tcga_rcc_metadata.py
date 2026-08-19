#!/usr/bin/env python3
"""Build the TCGA-RCC annotation table from the GDC file manifest.

Subtype labels come from the GDC project a slide belongs to -- TCGA-KIRC,
TCGA-KIRP or TCGA-KICH -- which is the authoritative source rather than a
label file copied between projects. The table is written in the same canonical
schema as the other ``metadata/*.csv`` cohorts.

Only diagnostic slides (``DX``) are included; TCGA also publishes frozen tissue
slides, which have different staining and are not part of this benchmark.

Requires network access, so run it on a login node. Results are cached to
``--cache`` so the table can be rebuilt offline.

Usage
-----
    python scripts/build_tcga_rcc_metadata.py
    python scripts/build_tcga_rcc_metadata.py --check       # report, write nothing
    python scripts/build_tcga_rcc_metadata.py --cache q.json --offline
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common.configuration import expand_path  # noqa: E402
DESTINATION = REPO / "metadata" / "tcga_rcc.csv"
DEFAULT_CACHE = REPO / "metadata" / ".gdc_rcc_manifest.json"

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

# The GDC project identifies the subtype, but this benchmark labels every cohort
# by OncoTree code -- BRCA uses IDC/ILC, NSCLC uses LUAD/LUSC -- and the upstream
# prompt assets for RCC are keyed the same way. Mapping the project to its
# OncoTree code here keeps RCC consistent with the rest of the framework and
# lets the published FOCUS, MSCPT and SLDPC prompts be used unmodified.
PROJECTS = {
    "TCGA-KIRC": "CCRCC",   # renal clear cell carcinoma
    "TCGA-KIRP": "PRCC",    # papillary renal cell carcinoma
    "TCGA-KICH": "CHRCC",   # chromophobe renal cell carcinoma
}

# Where the RCC slides are currently stored. Recorded as provenance only; the
# benchmark protocol resolves feature paths itself.
FEATURE_ROOT = expand_path(
    "${PGVL_STORAGE_ROOT}/dchanda/TCGA-RCC-Unified/20x_256px_0px_overlap")

COLUMNS = [
    "filename", "filepath", "label", "subclass", "patient_id", "magnification",
    "train_test", "fold", "dataset", "root_dir", "slide_id", "organ",
    "OncoTreeCode", "case_id",
]


def fetch_manifest(cache: Path, offline: bool) -> list[dict]:
    """Return the GDC hits for RCC diagnostic slides, using the cache if asked."""
    if offline or cache.exists():
        if not cache.exists():
            raise FileNotFoundError(f"--offline given but no cache at {cache}")
        return json.loads(cache.read_text())

    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id",
                                 "value": sorted(PROJECTS)}},
        {"op": "in", "content": {"field": "files.data_type",
                                 "value": ["Slide Image"]}},
        {"op": "in", "content": {"field": "files.experimental_strategy",
                                 "value": ["Diagnostic Slide"]}},
    ]}
    params = {
        "filters": json.dumps(filters),
        "fields": "file_name,cases.project.project_id,cases.submitter_id",
        "format": "JSON",
        "size": "10000",
    }
    url = f"{GDC_FILES_ENDPOINT}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = json.load(response)

    hits = payload["data"]["hits"]
    total = payload["data"]["pagination"]["total"]
    if len(hits) != total:
        raise RuntimeError(f"GDC returned {len(hits)} of {total} records")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(hits))
    return hits


def build_rows(hits: list[dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for hit in hits:
        project = hit["cases"][0]["project"]["project_id"]
        label = PROJECTS[project]
        # TCGA-A3-3336-01Z-00-DX1.F2848BBF-....svs -> barcode, then bare slide id
        barcode = hit["file_name"].rsplit(".svs", 1)[0]
        slide_id = barcode.split(".")[0]
        case_id = hit["cases"][0]["submitter_id"]
        rows.append({
            "filename": f"{barcode}.h5",
            "filepath": "",
            "label": label,
            "subclass": "",
            "patient_id": case_id,
            "magnification": "",
            "train_test": "",
            "fold": "",
            "dataset": "TCGA",
            "root_dir": FEATURE_ROOT,
            "slide_id": slide_id,
            "organ": "KIDNEY",
            "OncoTreeCode": label,
            "case_id": case_id,
        })
    rows.sort(key=lambda row: row["slide_id"])
    return rows


def check(rows: list[dict[str, str]]) -> list[str]:
    problems: list[str] = []
    duplicates = [k for k, n in Counter(r["slide_id"] for r in rows).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate slide_id values: {duplicates[:5]}")
    labels = {r["label"] for r in rows}
    if labels != set(PROJECTS.values()):
        problems.append(f"expected {sorted(PROJECTS.values())}, got {sorted(labels)}")
    if any(not r["slide_id"].startswith("TCGA-") for r in rows):
        problems.append("some slide_id values are not TCGA barcodes")
    return problems


def report(rows: list[dict[str, str]]) -> None:
    counts = Counter(r["label"] for r in rows)
    print(f"  slides          {len(rows)}")
    print(f"  cases           {len({r['case_id'] for r in rows})}")
    print("  label counts    "
          + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)))

    store = Path(FEATURE_ROOT) / "uni_v1" / "h5_files"
    if store.is_dir():
        local = {p.name.split(".")[0] for p in store.glob("*.h5")}
        listed = {r["slide_id"] for r in rows}
        print(f"  local WSI store {len(local)} slides; "
              f"{len(listed & local)} of {len(listed)} listed slides present")
        only_local = sorted(local - listed)
        if only_local:
            print(f"  ! {len(only_local)} local slides are not in the GDC "
                  f"manifest and are excluded: {only_local[:3]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--offline", action="store_true",
                        help="build from the cached manifest without querying GDC")
    parser.add_argument("--check", action="store_true",
                        help="validate and report without writing")
    args = parser.parse_args(argv)

    try:
        hits = fetch_manifest(args.cache, args.offline)
    except Exception as error:                                   # noqa: BLE001
        print(f"FATAL: cannot obtain the GDC manifest: {error}", file=sys.stderr)
        return 1

    rows = build_rows(hits)
    problems = check(rows)
    print(f"TCGA-RCC metadata from the GDC file manifest ({len(hits)} records)")
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
