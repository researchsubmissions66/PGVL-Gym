#!/usr/bin/env python3
"""Rebuild WSI-FiVE's structured NSCLC answer CSV from pinned workbooks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "methods" / "wsi_five" / "gpt_preprocess"
DEFAULT_OUTPUT = (
    REPO_ROOT / "text_prompts" / "wsi_five" / "nsclc_report_answers.csv")
UPSTREAM_COMMIT = "07344c9ac6eef919fcd1440877ea796feef7445a"
SOURCES = (
    (
        "LUAD_report_answer_v7_471.xlsx",
        "280269b1fc5c9dc26dd0050b1a3e7f294593b10dd58de0346dfe8a4c87bc195a",
    ),
    (
        "LUSC_report_answer_v1_468.xlsx",
        "1b47d972663eecfb8e09c2549db264fdd093502df6a0afb0cbd44d0d1fb9ae7d",
    ),
)
# The release leaves 27 answer cells blank.  Keep those cases in the bank with
# conservative, role-specific local completions that make no positive finding
# absent from the report.  This exact fill policy is part of the generated
# asset's provenance and is intentionally deterministic.
GENERATED_ANSWERS = (
    "Differentiation cannot be determined from the available pathology text.",
    "Spread through air spaces is not documented in the available pathology text.",
    "Vascular invasion is not documented in the available pathology text.",
    "Pleural invasion is not documented in the available pathology text.",
    "Extension into adjacent non-lung organs is not documented in the available pathology text.",
    "Margin status cannot be determined from the available pathology text.",
)
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_CELL = re.compile(r"([A-Z]+)")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _column_index(reference: str) -> int:
    match = _CELL.match(reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference {reference!r}")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _xlsx_rows(path: Path) -> list[list[str]]:
    """Read the simple first-sheet tables without an optional Excel package."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter(_NS + "t"))
                for item in root.findall(_NS + "si")
            ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    result: list[list[str]] = []
    for row in sheet.findall(".//" + _NS + "row"):
        cells: dict[int, str] = {}
        for cell in row.findall(_NS + "c"):
            index = _column_index(cell.attrib.get("r", ""))
            kind = cell.attrib.get("t")
            raw = cell.find(_NS + "v")
            if kind == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.iter(_NS + "t"))
            elif raw is None:
                value = ""
            elif kind == "s":
                value = shared[int(raw.text or "0")]
            else:
                value = raw.text or ""
            cells[index] = value
        width = max(cells, default=-1) + 1
        result.append([cells.get(index, "") for index in range(width)])
    return result


def build_csv() -> bytes:
    output_rows: list[list[str]] = []
    input_rows = blank_rows = 0
    for filename, expected_sha256 in SOURCES:
        path = SOURCE_ROOT / filename
        raw = path.read_bytes()
        actual_sha256 = _sha256(raw)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{path}: expected pinned source sha256 {expected_sha256}, "
                f"got {actual_sha256}")
        rows = _xlsx_rows(path)
        header = rows[0]
        indices = {name: header.index(name) for name in (
            "patient_id", "cancer_type", "answer")}
        for row_number, row in enumerate(rows[1:], start=2):
            input_rows += 1
            answer = row[indices["answer"]].strip()
            if not answer:
                blank_rows += 1
                answers = list(GENERATED_ANSWERS)
                answer = "; ".join(answers)
            else:
                answers = [value.strip() for value in answer.split(";")]
            if len(answers) != 6 or any(not value for value in answers):
                raise ValueError(
                    f"{path}: row {row_number} has {len(answers)} answer fields")
            output_rows.append([
                row[indices["patient_id"]].strip(),
                row[indices["cancer_type"]].strip(),
                answer,
                *answers,
            ])
    output_rows.sort(key=lambda row: row[0])
    if (input_rows, blank_rows, len(output_rows)) != (939, 27, 939):
        raise ValueError(
            "unexpected WSI-FiVE derivation counts: "
            f"input={input_rows}, blank={blank_rows}, output={len(output_rows)}")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow([
        "case_id", "cancer_type", "answer",
        *(f"q{index}" for index in range(1, 7)),
    ])
    writer.writerows(output_rows)
    return stream.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true",
        help="verify that output is exactly reproducible without writing it")
    args = parser.parse_args()
    generated = build_csv()
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != generated:
            print(f"drifted WSI-FiVE derived answer bank: {args.output}",
                  file=sys.stderr)
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(generated)
    print(
        f"{args.output}: 939 rows (912 upstream answers + 27 generated "
        f"completions), sha256={_sha256(generated)}, upstream={UPSTREAM_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
