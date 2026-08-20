import json
from pathlib import Path

import pytest

from common.preflight import preflight
from common.prompts import (
    TOP_PROMPT_FORMAT,
    load_top_bag_prompt_bank,
    load_top_prompt_condition,
)
from methods.top.adapter import TOPMethod
from scripts.tcga_benchmark import _prompt_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]
TOP_ROOT = REPO_ROOT / "text_prompts" / "top"


def _method(**overrides):
    cfg = {
        "clip_arch": "RN50",
        "instance_prompt_path": str(TOP_ROOT / "instance_prototypes.json"),
        "label_dict": {"LUSC": 1, "LUAD": 0},
        "classnames": [
            "lung adenocarcinoma", "lung squamous cell carcinoma"],
    }
    cfg.update(overrides)
    condition = load_top_prompt_condition(
        cfg["instance_prompt_path"], label_dict=cfg["label_dict"],
        classnames=cfg["classnames"], bag_path=cfg.get("bag_prompt_path"))
    cfg.update(condition.config_values())
    return TOPMethod(cfg, device="cpu")


def test_top_instance_bank_matches_released_code_without_double_period():
    method = _method()
    prompts = method._instance_ctx_init()

    assert len(prompts) == 26
    assert prompts[0] == (
        "an H&E stained image of Squamous epithelium, which is Flat, "
        "plate-like cells with a centrally located nucleus."
        "* * * * * * * * * *")
    assert ".." not in prompts[0]
    assert all(prompt.count("*") == 10 for prompt in prompts)


def test_top_nsclc_bag_bank_is_exact_and_sorted_by_class_index():
    method = _method(
        bag_prompt_path=str(
            TOP_ROOT / "tcga_nsclc_upstream_code_bag_prompts.json"))

    assert method._bag_ctx_init() == [
        "Examine the lung tissue image, looking for gland patterns and mucin "
        "(Lung Adenocarcinoma). * * * * * * * * * *",
        "Examine the lung tissue image, looking for irregular cells and "
        "keratinization (Lung Squamous Cell Carcinoma). * * * * * * * * * *",
    ]
    assert method._bag_classnames() == [
        "Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"]


def test_top_camelyon_bank_overrides_benchmark_names_with_code_suffixes():
    method = _method(
        label_dict={"normal": 0, "tumor": 1},
        classnames=["normal lymph node", "metastatic lymph node"],
        instance_slot_separator=" ",
        bag_prompt_path=str(
            TOP_ROOT / "camelyon16_upstream_code_bag_prompts.json"))

    assert method._bag_ctx_init() == [
        "normal * * * * * * * * * *",
        "tumor * * * * * * * * * *",
    ]
    assert method._bag_classnames() == ["normal", "tumor"]
    assert method._instance_ctx_init()[0].endswith(
        "nucleus. * * * * * * * * * *")


def test_top_doctor_rejects_digest_order_and_slot_drift(tmp_path: Path):
    instance_payload = json.loads(
        (TOP_ROOT / "instance_prototypes.json").read_text())
    instance_payload["prototypes"][0]["prompt"] += " drift"
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps(instance_payload))

    bag_payload = json.loads((
        TOP_ROOT / "tcga_nsclc_upstream_code_bag_prompts.json").read_text())
    bag_payload["_metadata"]["label_order"] = ["LUSC", "LUAD"]
    bag_payload["ctx_init"]["LUAD"] = bag_payload["ctx_init"]["LUAD"].replace(
        " *", "", 1)
    bags = tmp_path / "bags.json"
    bags.write_text(json.dumps(bag_payload))

    report = preflight({
        "method": "top",
        "instance_prompt_path": str(instances),
        "bag_prompt_path": str(bags),
        "label_dict": {"LUAD": 0, "LUSC": 1},
        "classnames": [
            "lung adenocarcinoma", "lung squamous cell carcinoma"],
    }, checks={"prompts"})

    assert any("structured tissue/description" in item
               for item in report.problems)
    assert any("label_order" in item for item in report.problems)


def test_top_doctor_does_not_accept_missing_banks_as_upstream():
    missing_instance = preflight({
        "method": "top", "label_dict": {"A": 0, "B": 1},
    }, checks={"prompts"})
    missing_bag = preflight({
        "method": "top", "label_dict": {"A": 0, "B": 1},
        "classnames": ["A", "B"],
        "instance_prompt_path": str(TOP_ROOT / "instance_prototypes.json"),
        "prompt_provenance": "upstream",
    }, checks={"prompts"})

    assert any("requires instance_prompt_path" in item
               for item in missing_instance.problems)
    assert any("prompt_provenance contradicts active prompt roles" in item
               for item in missing_bag.problems)


def test_top_provenance_distinguishes_released_and_extension_conditions():
    nsclc = {
        "labels": ["LUAD", "LUSC"],
        "classnames": [
            "lung adenocarcinoma", "lung squamous cell carcinoma"],
        "prompts": {
            "top_instance": "text_prompts/top/instance_prototypes.json",
            "top_bag": (
                "text_prompts/top/"
                "tcga_nsclc_upstream_code_bag_prompts.json"),
        },
    }
    brca = {
        "labels": ["IDC", "ILC"],
        "classnames": [
            "invasive ductal carcinoma", "invasive lobular carcinoma"],
        "prompts": {
            "top_instance": "text_prompts/top/instance_prototypes.json",
        },
    }
    supplementary = {
        "labels": ["LUAD", "LUSC"],
        "classnames": [
            "lung adenocarcinoma", "lung squamous cell carcinoma"],
        "prompts": {
            "top_instance": "text_prompts/top/instance_prototypes.json",
            "top_bag": "text_prompts/top/tcga_nsclc_bag_prompts.json",
        },
    }

    assert _prompt_provenance(nsclc, "top") == "upstream"
    assert _prompt_provenance(
        brca, "top") == "upstream_instance_with_random_classname_bag"
    assert _prompt_provenance(
        supplementary, "top") == "upstream_supplementary_condition"


def test_top_manifest_names_code_banks_and_unwired_supplement():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["top"]

    assert set(summary["selected_code_banks"]["TCGA-NSCLC"]) == {
        "text_prompts/top/instance_prototypes.json",
        "text_prompts/top/tcga_nsclc_upstream_code_bag_prompts.json",
    }
    assert summary["preserved_unwired_alternatives"] == [
        "text_prompts/top/tcga_nsclc_bag_prompts.json"]


def test_top_manifest_hashes_bind_every_registered_bank():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    records = manifest["assets"]
    instance = records["top/instance_prototypes.json"]
    assert instance["sha256"] == (
        "33a95e5e41c33459c8cf79abe68e1275986a952758c67705da02b1f7d102d0db")
    assert instance["prompt_bank_sha256"] == instance["ordered_prompt_sha256"]

    cases = {
        "top/tcga_nsclc_upstream_code_bag_prompts.json": (
            {"LUAD": 0, "LUSC": 1},
            ["lung adenocarcinoma", "lung squamous cell carcinoma"]),
        "top/tcga_nsclc_bag_prompts.json": (
            {"LUAD": 0, "LUSC": 1},
            ["lung adenocarcinoma", "lung squamous cell carcinoma"]),
        "top/camelyon16_upstream_code_bag_prompts.json": (
            {"normal": 0, "tumor": 1},
            ["normal lymph node", "metastatic lymph node"]),
    }
    for key, (labels, classnames) in cases.items():
        bank = load_top_bag_prompt_bank(
            REPO_ROOT / "text_prompts" / key,
            label_dict=labels, classnames=classnames)
        record = records[key]
        assert bank.file_sha256 == record["sha256"]
        assert bank.prompt_bank_sha256 == record["prompt_bank_sha256"]


def test_top_runtime_rejects_supplementary_bank_under_standard_identity():
    method = _method(bag_prompt_path=str(
        TOP_ROOT / "tcga_nsclc_upstream_code_bag_prompts.json"))
    method.cfg["bag_prompt_path"] = str(
        TOP_ROOT / "tcga_nsclc_bag_prompts.json")

    with pytest.raises(ValueError, match="(sha256 mismatch|usage)"):
        method._bag_ctx_init()


def test_top_doctor_derives_provenance_and_source_from_active_roles():
    cfg = _method(bag_prompt_path=str(
        TOP_ROOT / "tcga_nsclc_upstream_code_bag_prompts.json")).cfg
    assert cfg["top_prompt_format"] == TOP_PROMPT_FORMAT
    cfg["prompt_provenance"] = "upstream_supplementary_condition"
    cfg["prompt_source"] = "top_upstream_supplementary_bag_condition"

    report = preflight(cfg, checks={"prompts"})

    assert any("prompt_provenance contradicts active prompt roles" in item
               for item in report.problems)
    assert any("prompt_source contradicts active prompt roles" in item
               for item in report.problems)
