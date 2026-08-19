import hashlib
import json
from pathlib import Path

import pytest

from common.preflight import preflight
from common.prompts.muse import load_muse_prompt_bank, load_muse_prompt_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
MUSE_ROOT = REPO_ROOT / "text_prompts" / "muse"
PROVENANCE = json.loads(
    (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())


UPSTREAM_BANKS = {
    "camelyon_all": (
        ["normal lymph node", "metastatic lymph node"],
        [
            "671647d7aefb14c2d043c4ecba2845a771779451bca1476a850a72d3e7266e26",
            "2a5d1b6b7e57912d68cdac161b181def80e06681905197d192717b93011e36c2",
        ],
    ),
    "tcga_brca": (
        ["invasive ductal carcinoma", "invasive lobular carcinoma"],
        [
            "3b611e4e14a1e37cd56a78f052bf1d0d922b238ae8cc62f9df4044f67d4497c4",
            "523749fddaa4267dc889b130bff71f831c1c6944513d8c5c96c7dd829b4094cb",
        ],
    ),
    "tcga_nsclc": (
        ["lung adenocarcinoma", "lung squamous cell carcinoma"],
        [
            "d014961af20a572d44016c076c01c43600a7ae37c657eec8e641288e80375ba4",
            "e59cc3b4816ff286c2b49095f4e7a060a6e2b8ddc917a06660588439bf3aeada",
        ],
    ),
}


def test_released_muse_banks_are_pinned_exact_300_row_copies():
    records = PROVENANCE["assets"]
    for task, (classnames, digests) in UPSTREAM_BANKS.items():
        paths = [MUSE_ROOT / task / f"generated_new_{index}.csv"
                 for index in range(2)]
        bank = load_muse_prompt_bank(
            dict(zip(classnames, paths)),
            classnames=classnames,
            records={
                str(path): records[str(path.relative_to(REPO_ROOT / "text_prompts"))]
                for path in paths
            },
        )
        assert [branch.row_count for branch in bank.branches] == [300, 300]
        assert [branch.file_sha256 for branch in bank.branches] == digests
        assert [hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths] == digests


def test_muse_loader_preserves_released_duplicate_descriptions():
    bank = load_muse_prompt_csv(
        MUSE_ROOT / "camelyon_all" / "generated_new_0.csv")

    assert bank.row_count == 300
    assert len(set(bank.descriptions)) == 299


def test_muse_loader_rejects_non_native_header_and_index(tmp_path: Path):
    bad_header = tmp_path / "header.csv"
    bad_header.write_text("index,text\n0,description\n")
    with pytest.raises(ValueError, match="header must be exactly"):
        load_muse_prompt_csv(bad_header)

    bad_index = tmp_path / "index.csv"
    bad_index.write_text(",0\n1,a diagnostic description\n")
    with pytest.raises(ValueError, match="expected index 0"):
        load_muse_prompt_csv(bad_index)


def test_doctor_rejects_swapped_muse_class_files():
    classnames, _digests = UPSTREAM_BANKS["tcga_nsclc"]
    paths = [MUSE_ROOT / "tcga_nsclc" / f"generated_new_{index}.csv"
             for index in range(2)]
    report = preflight({
        "method": "muse",
        "classnames": classnames,
        "prompt_csvs": {
            classnames[0]: str(paths[1]),
            classnames[1]: str(paths[0]),
        },
        "prompt_provenance": "upstream",
        "prompt_source": "muse_upstream_description_csvs",
    }, checks={"prompts"})

    assert any("provenance binds this CSV" in problem
               for problem in report.problems)


def test_doctor_rejects_generated_muse_bank_reported_as_upstream():
    root = REPO_ROOT / "benchmarks" / "ubc_ocean" / "data" / \
        "ubc_ocean" / "prompts" / "muse"
    classnames = [
        "ovarian clear cell carcinoma",
        "ovarian endometrioid carcinoma",
        "ovarian high-grade serous carcinoma",
        "ovarian low-grade serous carcinoma",
        "ovarian mucinous carcinoma",
    ]
    report = preflight({
        "method": "muse",
        "classnames": classnames,
        "prompt_csvs": {
            classname: str(root / f"generated_new_{index}.csv")
            for index, classname in enumerate(classnames)
        },
        "prompt_provenance": "upstream",
        "prompt_source": "muse_upstream_description_csvs",
    }, checks={"prompts"})

    assert any("prompt_provenance contradicts" in problem
               for problem in report.problems)
    assert any("prompt_source contradicts" in problem
               for problem in report.problems)
