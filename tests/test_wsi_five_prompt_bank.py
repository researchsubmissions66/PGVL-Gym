from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from common.preflight import preflight
from common.prompts import (
    load_wsi_five_answer_bank,
    load_wsi_five_evaluation_bank,
    load_wsi_five_question_bank,
)
from scripts.build_wsi_five_prompt_assets import GENERATED_ANSWERS, build_csv
from scripts.generate_configs import DATASETS, cfg_wsi_five


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = REPO_ROOT / "text_prompts" / "wsi_five"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_nsclc_native_text_roles_match_pinned_manifest():
    question_bank = load_wsi_five_question_bank(
        PROMPT_ROOT / "clinical_questions" / "nsclc.json")
    answer_bank = load_wsi_five_answer_bank(
        PROMPT_ROOT / "nsclc_report_answers.csv")
    evaluation_bank = load_wsi_five_evaluation_bank(
        PROMPT_ROOT / "nsclc_evaluation_prompts.json",
        {"LUAD": 0, "LUSC": 1},
    )

    assert question_bank.provenance == "derived"
    assert answer_bank.provenance == "generated"
    assert evaluation_bank.provenance == "derived"
    assert len(question_bank.questions) == 6
    assert len(answer_bank.records) == 939
    assert evaluation_bank.class_names == ("LUAD", "LUSC")
    assert question_bank.prompt_bank_sha256 == (
        "66689686c7d2305341dc42e71b27438412fc327275a80860e7f98499935f026f")
    assert answer_bank.answer_bank_sha256 == (
        "a853601f0cb18324627956e1a9304aa554e499e8164e798306cd80f7f611ac83")
    assert evaluation_bank.prompt_bank_sha256 == (
        "cc1647ebeb30b20f588da39992b5990a360506734d7633f50ee246c8f753b3da")
    generated = [
        record for record in answer_bank.records
        if record.answers == GENERATED_ANSWERS]
    assert len(generated) == 27


def test_nsclc_answer_csv_is_reproducible_from_pinned_workbooks():
    generated = build_csv()

    assert generated == (PROMPT_ROOT / "nsclc_report_answers.csv").read_bytes()
    assert _sha256(generated) == (
        "dbe2ebb91eb57e391f2f9fff807246815943a4a85036575375c2ec62d8b9d360")


def test_wsi_five_doctor_rejects_question_drift(tmp_path: Path):
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "wsi_five" / "lung.yaml").read_text())
    payload = json.loads(
        (PROMPT_ROOT / "clinical_questions" / "nsclc.json").read_text())
    payload["questions"][0] += " drift"
    drifted = tmp_path / "questions.json"
    drifted.write_text(json.dumps(payload))
    cfg["clinical_questions"] = str(drifted)

    report = preflight(cfg, checks={"prompts"})

    assert not report.ok
    assert any("sha256 mismatch" in problem for problem in report.problems)


def test_wsi_five_config_generator_emits_the_checked_prompt_contract():
    keys = {
        "prompt_provenance", "prompt_source", "wsi_prompt_format",
        "wsi_question_file_sha256", "wsi_question_bank_sha256",
        "wsi_question_provenance", "wsi_answer_file_sha256",
        "wsi_answer_bank_sha256", "wsi_answer_provenance",
        "wsi_evaluation_file_sha256", "wsi_evaluation_bank_sha256",
        "wsi_evaluation_provenance",
    }
    for dataset in ("lung", "rcc", "ubc"):
        generated = yaml.safe_load(cfg_wsi_five(DATASETS[dataset], dataset))
        checked_in = yaml.safe_load((
            REPO_ROOT / "configs" / "wsi_five" / f"{dataset}.yaml"
        ).read_text())
        assert {key: generated.get(key) for key in keys} == {
            key: checked_in.get(key) for key in keys}
