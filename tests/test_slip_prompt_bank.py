import json
from pathlib import Path

import pytest

from common.preflight import preflight
from common.prompts.slip import load_slip_prompt_bank


REPO_ROOT = Path(__file__).resolve().parents[1]
SLIP_ROOT = REPO_ROOT / "text_prompts" / "slip"


def test_tcga_bank_preserves_complete_upstream_structure():
    bank = load_slip_prompt_bank(
        SLIP_ROOT / "TCGA_prompt_bank.json",
        fallback_slide_classnames=[
            "lung adenocarcinoma", "lung squamous cell carcinoma"
        ],
        labels=["LUAD", "LUSC"],
    )

    assert bank.provenance == "upstream"
    assert bank.templates == ("{}",)
    assert bank.slide_classnames == (
        ("lung adenocarcinoma",),
        ("lung squamous cell carcinoma",),
    )
    assert len(bank.tissue_classnames) == 17
    assert bank.tissue_classnames[0] == (
        "Alveolar tissue",
        "The normal lung tissue, consisting of air spaces and alveolar sacs. "
        "Typically seen in adenocarcinoma cases.",
    )
    assert all(len(group) == 2 for group in bank.tissue_classnames)
    assert bank.digest == (
        "acea8dc45d6cd37e124d06a155c9d84ed80d3b58dba0eda05fcf25db7edd9d62"
    )


@pytest.mark.parametrize(
    ("name", "classes", "template", "tissue_count", "digest"),
    [
        (
            "DHMC_prompt_bank.json",
            ["lepidic", "acinar", "solid"],
            "{}",
            15,
            "16a9bfb8e36c8bd158912e0a56544724a77b13d8010b3a653a33e7221923d58a",
        ),
        (
            "PatchGastricADC22_prompt_bank.json",
            ["well", "moderately", "poorly"],
            "whole slide image showing {}",
            18,
            "afb84abcf1d4c141c462f3b7043c998167c96637c80a5c148269cc3dfc3bec32",
        ),
    ],
)
def test_other_released_banks_retain_templates_and_pair_ensembles(
    name: str,
    classes: list[str],
    template: str,
    tissue_count: int,
    digest: str,
):
    bank = load_slip_prompt_bank(
        SLIP_ROOT / name,
        fallback_slide_classnames=classes,
    )

    assert bank.templates == (template,)
    assert len(bank.tissue_classnames) == tissue_count
    assert all(len(group) == 2 for group in bank.tissue_classnames)
    assert bank.digest == digest


def test_generated_extension_expands_legacy_pair_notation():
    source = SLIP_ROOT / "tcga_rcc_tissues.json"
    raw = json.loads(source.read_text())
    bank = load_slip_prompt_bank(
        source,
        fallback_slide_classnames=["ccRCC", "pRCC", "chRCC"],
    )

    assert bank.provenance == "generated"
    assert raw["_pair_separator"] == ": "
    assert bank.tissue_classnames[0] == (
        "Renal corpuscle",
        "Spherical glomerular capillary tuft enclosed by Bowman's capsule.",
    )
    assert all(len(group) == 2 for group in bank.tissue_classnames)


def test_doctor_accepts_path_only_complete_upstream_bank():
    report = preflight({
        "method": "slip",
        "tissue_classnames_path": str(SLIP_ROOT / "TCGA_prompt_bank.json"),
        "classnames": [
            "lung adenocarcinoma", "lung squamous cell carcinoma"
        ],
        "label_dict": {"LUAD": 0, "LUSC": 1},
        "n_classes": 2,
    }, checks={"prompts"})

    assert report.problems == []


def test_doctor_rejects_prompt_provenance_drift():
    report = preflight({
        "method": "slip",
        "tissue_classnames_path": str(SLIP_ROOT / "TCGA_prompt_bank.json"),
        "classnames": [
            "lung adenocarcinoma", "lung squamous cell carcinoma"
        ],
        "label_dict": {"LUAD": 0, "LUSC": 1},
        "n_classes": 2,
        "prompt_provenance": "generated",
    }, checks={"prompts"})

    assert any("prompt_provenance contradicts" in item
               for item in report.problems)


def test_loader_rejects_label_order_drift():
    with pytest.raises(ValueError, match="label_order"):
        load_slip_prompt_bank(
            SLIP_ROOT / "TCGA_prompt_bank.json",
            fallback_slide_classnames=[
                "lung squamous cell carcinoma", "lung adenocarcinoma"
            ],
            labels=["LUSC", "LUAD"],
        )


def test_loader_rejects_legacy_flattened_upstream_asset():
    with pytest.raises(ValueError, match="flat string"):
        load_slip_prompt_bank(
            SLIP_ROOT / "TCGA_tissues.json",
            fallback_slide_classnames=[
                "lung adenocarcinoma", "lung squamous cell carcinoma"
            ],
        )
