import hashlib
import json
from pathlib import Path

import pytest

from common.configuration import load_yaml_config
from common.preflight import preflight
from common.prompts import FOCUS_PROMPT_FORMAT, load_focus_prompt_bank
from methods.focus.adapter import _build_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = REPO_ROOT / "text_prompts" / "focus"


def test_released_focus_banks_are_pinned_exact_assets():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    expected = {
        "CAMELYON16_two_scale_text_prompt.csv": (
            ["normal", "tumor"],
            "e73ad22cccd86b6325513566de98405f2361b8023f596d84d34a7a41cc744d01",
        ),
        "TCGA_NSCLC_two_scale_text_prompt.csv": (
            ["LUAD", "LUSC"],
            "a8df24b0adbfa15369b51fb31f17adef4a9a6b955d25cd9ab0e09f3284c76c50",
        ),
        "UBC_OCEAN_two_scale_text_prompt.csv": (
            ["CC", "HGSC", "LGSC", "EC", "MC"],
            "d8e93638bca7a12b0282710de8f540ab33187e15b594a2d783752f51ec31f1ca",
        ),
    }
    for filename, (labels, digest) in expected.items():
        path = PROMPT_ROOT / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        record = manifest["assets"][f"focus/{filename}"]
        assert record["source"].startswith(
            "https://github.com/dddavid4real/FOCUS/blob/"
            "66c4015d5ba09657f4c8183bc06947faecd5b01f/")
        bank = load_focus_prompt_bank(path, class_names=labels)
        assert bank.provenance == "upstream"
        assert bank.file_sha256 == digest


def test_focus_ubc_preserves_upstream_text_and_reorders_explicitly():
    bank = load_focus_prompt_bank(
        PROMPT_ROOT / "UBC_OCEAN_two_scale_text_prompt.csv",
        class_names=["CC", "EC", "HGSC", "LGSC", "MC"],
        file_class_names=["CC", "HGSC", "LGSC", "EC", "MC"],
        expected_provenance="upstream",
        expected_ordered_prompt_bank_sha256=(
            "c8e018033f5f2af5bdfae535048f9a717fbd6f9fef46fadd18babc0409459a37"),
    )

    assert bank.low_resolution[1].startswith(
        "A whole slide image of endometrioid carcinoma")
    assert 'Characteristic hobnail" cells' in bank.high_resolution[0]
    assert bank.high_resolution[0].endswith('visible."')


def test_focus_generated_extensions_are_not_reported_as_upstream():
    for filename, labels in {
        "TCGA_BRCA_two_scale_text_prompt.csv": ["IDC", "ILC"],
        "TCGA_RCC_two_scale_text_prompt.csv": ["CCRCC", "PRCC", "CHRCC"],
    }.items():
        bank = load_focus_prompt_bank(
            PROMPT_ROOT / filename, class_names=labels)
        assert bank.provenance == "generated"


def test_focus_loader_rejects_local_three_column_conversion(tmp_path: Path):
    path = tmp_path / "named.csv"
    path.write_text(
        "class_name,low_res_prompt,high_res_prompt\nA,low A,high A\n",
        encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one prompt"):
        load_focus_prompt_bank(
            path, class_names=["A"], file_class_names=["A"],
            record={"provenance": "generated"})


def test_focus_runtime_and_doctor_share_native_contract():
    config = {
        "method": "focus",
        "feature_dim": 512,
        "n_classes": 2,
        "label_dict": {"LUAD": 0, "LUSC": 1},
        "text_prompt_path": str(
            PROMPT_ROOT / "TCGA_NSCLC_two_scale_text_prompt.csv"),
        "focus_prompt_format": FOCUS_PROMPT_FORMAT,
        "focus_prompt_file_classnames": ["LUAD", "LUSC"],
        "focus_prompt_file_sha256": (
            "a8df24b0adbfa15369b51fb31f17adef4a9a6b955d25cd9ab0e09f3284c76c50"),
        "focus_prompt_bank_sha256": (
            "a966d04324b31fe04b81874d5bbace791fd44347eac19bef0fa73d7a4c156b3c"),
        "prompt_provenance": "upstream",
        "prompt_source": "focus_upstream_native_two_scale_csv",
    }

    native = _build_config(config)
    assert len(native.text_prompt) == 4
    assert preflight(config, checks={"prompts"}).ok

    drifted = dict(config, focus_prompt_bank_sha256="0" * 64)
    report = preflight(drifted, checks={"prompts"})
    assert not report.ok
    assert any("ordered prompt-bank sha256" in item
               for item in report.problems)


def test_checked_focus_configs_carry_strict_prompt_metadata():
    for benchmark in (
        "camelyon16", "tcga_brca", "tcga_nsclc", "tcga_rcc", "ubc_ocean",
    ):
        for path in (REPO_ROOT / "benchmarks" / benchmark / "configs").glob(
                "focus*/*.yaml"):
            config = load_yaml_config(path)
            assert config["focus_prompt_format"] == FOCUS_PROMPT_FORMAT
            assert len(config["focus_prompt_file_sha256"]) == 64
            assert len(config["focus_prompt_bank_sha256"]) == 64
