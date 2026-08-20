import json
from pathlib import Path

import pytest

from common.preflight import preflight
from common.prompts.sldpc import (
    SLDPC_ZERO_SHOT_TEMPLATES,
    load_sldpc_zero_shot_prompt_bank,
    sldpc_prompt_classname_sha256,
    sldpc_zero_shot_templates_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "text_prompts" / "sldpc"
MANIFEST = json.loads(
    (ROOT / "text_prompts" / "PROVENANCE.json").read_text())


@pytest.mark.parametrize(
    ("filename", "class_names", "sha256"),
    [
        (
            "tcga_nsclc.yaml", ["LUAD", "LUSC"],
            "a139ac9e0f43ed18365f606dca584b74039b18b192b3519522bb72c438a8aa00",
        ),
        (
            "tcga_rcc.yaml", ["CCRCC", "CHRCC", "PRCC"],
            "c720439d4833a39c806f0e5ff3f186238561ce46f86aee4068899fdf2df99ec6",
        ),
        (
            "ubc_ocean.yaml", ["CC", "EC", "HGSC", "LGSC", "MC"],
            "47b7bdd76724cb2a2454d859eaabdbca9b45efa0389b5d2cfca1e19381c58956",
        ),
        (
            "tcga_ot.yaml", MANIFEST["assets"]["sldpc/tcga_ot.yaml"]["class_names"],
            "c7defebfe4f7c591fe3ee712b9519712a66758b6fc6a5ece3192b55dbafb73b9",
        ),
    ],
)
def test_released_sldpc_zero_shot_banks_are_pinned(
    filename: str, class_names: list[str], sha256: str,
):
    bank = load_sldpc_zero_shot_prompt_bank(
        PROMPTS / filename, class_names=class_names)

    assert bank.provenance == "upstream"
    assert bank.file_sha256 == sha256
    assert MANIFEST["assets"][f"sldpc/{filename}"][
        "copied_from_upstream"] is True


@pytest.mark.parametrize("filename", ["camelyon16.yaml", "tcga_brca.yaml"])
def test_unreleased_sldpc_task_extensions_are_not_upstream(filename: str):
    record = MANIFEST["assets"][f"sldpc/{filename}"]
    bank = load_sldpc_zero_shot_prompt_bank(
        PROMPTS / filename, class_names=record["class_names"])

    assert bank.provenance == "generated"
    assert record["copied_from_upstream"] is False


def test_sldpc_zero_shot_templates_match_released_ensemble():
    assert len(SLDPC_ZERO_SHOT_TEMPLATES) == 23
    assert sldpc_zero_shot_templates_sha256() == (
        "6688a22c36820dd7b8e66a8bb1b3ea67487939f13f2e65a981afc516ae74596d"
    )


def test_sldpc_zero_shot_loader_respects_benchmark_label_order():
    bank = load_sldpc_zero_shot_prompt_bank(
        PROMPTS / "tcga_rcc.yaml",
        class_names=["CCRCC", "PRCC", "CHRCC"],
    )

    assert bank.class_names == ("CCRCC", "PRCC", "CHRCC")
    assert bank.ordered_prompt_sha256 == (
        "6261094861be32d6fe9da5eeb5563f749c7a79d9a0e5891d62f566485bcbb5fe"
    )


def test_sldpc_zero_shot_loader_rejects_incomplete_bank(tmp_path: Path):
    prompt = tmp_path / "incomplete.yaml"
    prompt.write_text("prompts:\n  A: [valid]\n")

    with pytest.raises(ValueError, match="keys must exactly match"):
        load_sldpc_zero_shot_prompt_bank(
            prompt,
            class_names=["A", "B"],
            record={"provenance": "generated"},
        )


def test_sldpc_doctor_accepts_separated_active_and_reference_contract():
    class_names = ["CCRCC", "PRCC", "CHRCC"]
    report = preflight({
        "method": "sldpc",
        "n_classes": 3,
        "label_dict": {"CCRCC": 0, "PRCC": 1, "CHRCC": 2},
        "prompt_classnames": class_names,
        "prompt_classname_sha256": sldpc_prompt_classname_sha256(class_names),
        "prompt_provenance": "derived",
        "prompt_source": "sldpc_derived_class_tokens",
        "zero_shot_prompt_path": str(PROMPTS / "tcga_rcc.yaml"),
        "zero_shot_prompt_provenance": "upstream",
        "zero_shot_prompt_sha256": (
            "c720439d4833a39c806f0e5ff3f186238561ce46f86aee4068899fdf2df99ec6"
        ),
        "zero_shot_prompt_bank_sha256": (
            "6261094861be32d6fe9da5eeb5563f749c7a79d9a0e5891d62f566485bcbb5fe"
        ),
        "zero_shot_templates_sha256": sldpc_zero_shot_templates_sha256(),
        "zero_shot_prompt_usage": "reference_only_unwired",
    }, checks={"prompts"})

    assert report.ok, report.problems
