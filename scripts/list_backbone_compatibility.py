#!/usr/bin/env python3
"""Inspect method/backbone swap boundaries without loading any checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import get_backbone_contracts  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List declared encoder compatibility for every WSI/VLM method")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON")
    parser.add_argument("--method", help="show one method only")
    return parser


def main() -> None:
    args = _parser().parse_args()
    contracts = get_backbone_contracts()
    if args.method:
        key = args.method.lower().replace("-", "_")
        if key not in contracts:
            raise SystemExit(
                f"Unknown method '{args.method}'. Available: {', '.join(contracts)}")
        contracts = {key: contracts[key]}
    payload = {name: contract.as_dict() for name, contract in contracts.items()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    headings = ("method", "input", "swap policy", "backbones / requirement")
    rows = []
    for name, contract in contracts.items():
        supported = ", ".join(contract.supported_backbones)
        if not supported:
            supported = "capabilities: " + ", ".join(sorted(
                item.value for item in contract.required_capabilities))
        rows.append((name, contract.feature_level.value,
                     contract.swap_policy.value, supported))
    widths = [max(len(headings[index]), *(len(row[index]) for row in rows))
              for index in range(len(headings))]
    print("  ".join(value.ljust(widths[index])
                    for index, value in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index])
                        for index, value in enumerate(row)))


if __name__ == "__main__":
    main()
