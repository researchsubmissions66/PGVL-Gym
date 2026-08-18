"""Report where every prompt asset came from, and fail on unrecorded ones.

Fidelity questions ("is this the authors' text?") were previously answerable
only by visiting four upstream repositories. That audit kept finding problems:
CoD-MIL chains authored for two cohorts but reported as upstream, SLIP tissue
vocabularies replaced by per-class descriptions to satisfy a validator that
encoded the wrong contract.

This records the answer once, in one place, and fails when an asset appears that
nobody has classified.

    python scripts/audit_prompt_provenance.py           # report
    python scripts/audit_prompt_provenance.py --check   # non-zero if unrecorded

Categories
----------
``upstream``  the authors' text, unmodified
``derived``   the authors' text, restructured without rewording
``generated`` written for this benchmark; a result using it is not
              prompt-faithful and its ``prompt_provenance`` must say so
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = REPO_ROOT / "text_prompts"
MANIFEST = PROMPT_ROOT / "PROVENANCE.json"

CATEGORIES = {"upstream", "derived", "generated"}


def inline_provenance(path: Path) -> str | None:
    """Return a marker the asset carries itself, if any."""
    try:
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload.get("_provenance")
        elif path.suffix == ".csv":
            first = path.open(encoding="utf-8").readline().strip()
            if first.startswith("#"):
                return json.loads(first[1:]).get("_provenance")
    except (OSError, ValueError):
        return None
    return None


def assets() -> list[Path]:
    return sorted(p for p in PROMPT_ROOT.rglob("*")
                  if p.is_file() and p.suffix in {".json", ".csv", ".yaml", ".yml"}
                  and p != MANIFEST)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when an asset has no recorded origin")
    args = parser.parse_args()

    manifest = (json.loads(MANIFEST.read_text(encoding="utf-8"))
                if MANIFEST.is_file() else {})
    records = manifest.get("assets", {})

    unrecorded, rows = [], []
    for path in assets():
        key = str(path.relative_to(PROMPT_ROOT))
        inline = inline_provenance(path)
        recorded = records.get(key, {}).get("provenance")
        origin = inline or recorded
        if origin not in CATEGORIES:
            unrecorded.append(key)
        rows.append((key, origin or "UNRECORDED", "inline" if inline else
                     ("manifest" if recorded else "-")))

    width = max(len(r[0]) for r in rows) if rows else 10
    for key, origin, where in rows:
        print(f"  {key:{width}s}  {origin:10s} {where}")

    counts: dict[str, int] = {}
    for _, origin, _ in rows:
        counts[origin] = counts.get(origin, 0) + 1
    print(f"\n{len(rows)} assets: " +
          ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))

    if unrecorded:
        print(f"\n{len(unrecorded)} asset(s) with no recorded origin:")
        for key in unrecorded:
            print(f"  {key}")
        print(f"\nAdd them to {MANIFEST.relative_to(REPO_ROOT)} or mark them "
              "inline with a '_provenance' key.")
        if args.check:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
