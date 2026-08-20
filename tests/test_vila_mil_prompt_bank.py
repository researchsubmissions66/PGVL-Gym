import hashlib
import json
from pathlib import Path

import pytest

from common.preflight import preflight
from common.prompts import (
    VILA_PROMPT_FORMAT,
    load_vila_prompt_bank,
)
from methods.vila_mil.adapter import _build_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = REPO_ROOT / "text_prompts" / "vila_mil"


def test_released_vila_banks_are_pinned_exact_assets():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    expected = {
        "TCGA_Lung_two_scale_text_prompt.csv": (
            ["LUAD", "LUSC"],
            "4fcea3a2016b2b0c600421da19a9b40441faef5a6b491d40ed80aa80e58319f1",
        ),
        "TCGA_RCC_two_scale_text_prompt.csv": (
            ["CCRCC", "PRCC", "CHRCC"],
            "af95c33eaaa2ab47eeba139a98fcdf7e5137fe797748c8a7caff189c4340886a",
        ),
    }
    for filename, (labels, digest) in expected.items():
        path = PROMPT_ROOT / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        record = manifest["assets"][f"vila_mil/{filename}"]
        assert record["source"].endswith(
            f"68a11cf0d5cf092dd980f0da1cb38ccac8747a82/text_prompt/{filename}")
        bank = load_vila_prompt_bank(path, class_names=labels)
        assert bank.provenance == "upstream"
        assert bank.file_sha256 == digest


def test_rcc_preserves_upstream_crcc_text_but_binds_chrcc_class():
    bank = load_vila_prompt_bank(
        PROMPT_ROOT / "TCGA_RCC_two_scale_text_prompt.csv",
        class_names=["CCRCC", "PRCC", "CHRCC"],
    )

    assert bank.class_names[2] == "CHRCC"
    assert "CRCC" in bank.low_resolution[2]
    assert "CRCC" in bank.high_resolution[2]
    assert "CHRCC" not in bank.low_resolution[2]


def test_vila_loader_reorders_only_with_explicit_file_binding():
    bank = load_vila_prompt_bank(
        PROMPT_ROOT / "UBC_OCEAN_two_scale_text_prompt.csv",
        class_names=["HGSC", "LGSC", "EC", "CC", "MC"],
        file_class_names=["CC", "EC", "HGSC", "LGSC", "MC"],
        expected_provenance="generated",
        expected_ordered_prompt_bank_sha256=(
            "a693ac313d96ed3dd588507cca47de413cee4dc570a22ac35ff2639d70e7b418"),
    )

    assert bank.low_resolution[0].startswith("High-grade serous")
    assert bank.low_resolution[3].startswith("Ovarian clear cell")


def test_vila_loader_rejects_focus_three_column_schema(tmp_path: Path):
    path = tmp_path / "focus.csv"
    path.write_text(
        "class_name,low_res_prompt,high_res_prompt\nA,low A,high A\n",
        encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one prompt"):
        load_vila_prompt_bank(
            path, class_names=["A"], file_class_names=["A"],
            record={"provenance": "generated"})


def test_vila_runtime_and_doctor_share_the_native_contract():
    config = {
        "method": "vila_mil",
        "feature_dim": 1024,
        "n_classes": 2,
        "label_dict": {"LUAD": 0, "LUSC": 1},
        "text_prompt_path": str(
            PROMPT_ROOT / "TCGA_Lung_two_scale_text_prompt.csv"),
        "vila_prompt_format": VILA_PROMPT_FORMAT,
        "vila_prompt_file_classnames": ["LUAD", "LUSC"],
        "vila_prompt_file_sha256": (
            "4fcea3a2016b2b0c600421da19a9b40441faef5a6b491d40ed80aa80e58319f1"),
        "vila_prompt_bank_sha256": (
            "61c001097a14a153dd8723cf5d85b41b8b2b0f61bb09b84b218280f1818eedfa"),
        "prompt_provenance": "upstream",
        "prompt_source": "vila_mil_upstream_native_two_scale_csv",
    }

    native = _build_config(config)
    assert len(native.text_prompt) == 4
    assert preflight(config, checks={"prompts"}).ok

    drifted = dict(config, vila_prompt_bank_sha256="0" * 64)
    report = preflight(drifted, checks={"prompts"})
    assert not report.ok
    assert any("ordered prompt-bank sha256" in item
               for item in report.problems)
