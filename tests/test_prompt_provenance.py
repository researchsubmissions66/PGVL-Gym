import hashlib
import json
from pathlib import Path

from scripts.tcga_benchmark import (
    _prompt_provenance,
    _wsi_five_prompt_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_focus_manifest_explicitly_classifies_every_local_bank():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["focus"]

    copied = {
        "focus/CAMELYON16_two_scale_text_prompt.csv",
        "focus/TCGA_NSCLC_two_scale_text_prompt.csv",
        "focus/UBC_OCEAN_two_scale_text_prompt.csv",
    }
    assert set(summary["copied_from_upstream"]) == copied
    for key in copied:
        record = manifest["assets"][key]
        assert record["provenance"] == "upstream"
        assert record["copied_from_upstream"] is True
    generated = {
        "focus/TCGA_BRCA_two_scale_text_prompt.csv",
        "focus/TCGA_RCC_two_scale_text_prompt.csv",
    }
    assert set(summary["generated_or_rewritten"]) == generated | {
        "prompt_spec-compiled FOCUS banks"}
    for key in generated:
        record = manifest["assets"][key]
        assert record["provenance"] == "generated"
        assert record["copied_from_upstream"] is False

    assert not any(
        path.is_file() for path in (REPO_ROOT / "text_prompts").glob("*.csv"))


def test_focus_provenance_uses_content_origin_not_path_selection():
    legacy = {
        "focus_prompt_csv": (
            "text_prompts/focus/TCGA_NSCLC_two_scale_text_prompt.csv"),
    }
    explicit = {
        "prompts": {
            "focus": "text_prompts/focus/TCGA_RCC_two_scale_text_prompt.csv",
            "vila_mil": (
                "text_prompts/vila_mil/TCGA_BRCA_two_scale_text_prompt.csv"),
        },
    }

    assert _prompt_provenance(legacy, "focus") == "upstream"
    assert _prompt_provenance(explicit, "focus") == "generated"
    assert _prompt_provenance(explicit, "vila_mil") == "generated"


def test_vila_manifest_separates_exact_copies_from_task_extensions():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["vila_mil"]

    assert set(summary["copied_from_upstream"]) == {
        "vila_mil/TCGA_Lung_two_scale_text_prompt.csv",
        "vila_mil/TCGA_RCC_two_scale_text_prompt.csv",
    }
    for key in summary["copied_from_upstream"]:
        assert manifest["assets"][key]["provenance"] == "upstream"
        assert manifest["assets"][key]["copied_from_upstream"] is True
    extensions = {
        "vila_mil/TCGA_BRCA_two_scale_text_prompt.csv",
        "vila_mil/UBC_OCEAN_two_scale_text_prompt.csv",
        "vila_mil/CAMELYON16_two_scale_text_prompt.csv",
    }
    assert extensions.issubset(summary["generated_or_rewritten"])
    for key in extensions:
        assert manifest["assets"][key]["provenance"] == "generated"
        assert manifest["assets"][key]["copied_from_upstream"] is False


def test_mscpt_manifest_classifies_copied_and_generated_banks():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["mscpt"]
    repository_assets = manifest["repository_assets"]

    assert set(summary["copied_from_upstream"]) == {
        "train_data/gpt/description/Lung.json",
        "train_data/gpt/description/RCC.json",
        "train_data/gpt/description/UBC-OCEAN.json",
    }
    assert set(summary["generated_or_rewritten"]) == {
        "text_prompts/mscpt/description/TCGA_BRCA_IDC_ILC.json",
        "benchmarks/camelyon16/data/camelyon16/prompts/gpt/description/"
        "camelyon16.json",
    }
    for key in summary["copied_from_upstream"]:
        assert repository_assets[key]["provenance"] == "upstream"
        assert repository_assets[key]["copied_from_upstream"] is True

    brca = manifest["assets"][
        "mscpt/description/TCGA_BRCA_IDC_ILC.json"]
    camelyon = repository_assets[
        "benchmarks/camelyon16/data/camelyon16/prompts/gpt/description/"
        "camelyon16.json"]
    assert brca["provenance"] == "generated"
    assert brca["copied_from_upstream"] is False
    assert camelyon["provenance"] == "generated"
    assert camelyon["copied_from_upstream"] is False


def test_checked_in_mscpt_configs_report_audited_content_origin():
    expected = {
        "camelyon16": "generated",
        "tcga_brca": "generated",
        "tcga_nsclc": "upstream",
        "tcga_rcc": "upstream",
        "ubc_ocean": "upstream",
    }
    for cohort, origin in expected.items():
        paths = sorted((REPO_ROOT / "benchmarks" / cohort / "configs").glob(
            "mscpt*/*.yaml"))
        assert len(paths) == 6
        for path in paths:
            assert f"prompt_provenance: {origin}\n" in path.read_text()


def test_mscpt_lung_upstream_issue_is_machine_readable():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    issue = manifest["repository_assets"][
        "train_data/gpt/description/Lung.json"]["known_content_issue"]

    assert issue["status"] == "upstream_preserved_and_disclosed"
    assert issue["affected_label"] == "LUSC"
    assert issue["affected_prompt_count"] == 9
    assert issue["zero_based_indices"] == {
        "small_mag": [0, 2],
        "big_mag": [0, 1, 3, 8, 9, 28, 29],
    }


def test_maple_manifest_separates_exact_banks_and_task_extensions():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["maple"]
    assets = manifest["assets"]

    assert set(summary["copied_from_upstream"]) == {
        "text_prompts/maple/LUNG_attributes.json",
        "text_prompts/maple/RCC_attributes.json",
        "text_prompts/maple/BRCA_attributes.json",
    }
    for key in summary["copied_from_upstream"]:
        record = assets[key.removeprefix("text_prompts/")]
        assert record["provenance"] == "upstream"
        assert record["copied_from_upstream"] is True
        path = REPO_ROOT / key
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    ubc = assets["maple/UBC_attributes.json"]
    assert ubc["provenance"] == "generated"
    assert ubc["copied_from_upstream"] is False
    assert summary["upstream_runtime_issue"]["status"] == \
        "corrected_and_disclosed"


def test_maple_provenance_uses_content_origin():
    copied = {
        "maple_prompt_json": "text_prompts/maple/LUNG_attributes.json",
    }
    generated = {
        "maple_prompt_json": "text_prompts/maple/UBC_attributes.json",
    }

    assert _prompt_provenance(copied, "maple") == "upstream"
    assert _prompt_provenance(generated, "maple") == "generated"


def test_muse_manifest_separates_exact_banks_and_task_extensions():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["muse"]
    assets = manifest["assets"]
    repository_assets = manifest["repository_assets"]

    assert len(summary["copied_from_upstream"]) == 6
    assert len(summary["generated_or_rewritten"]) == 8
    for key in summary["copied_from_upstream"]:
        record = assets[key.removeprefix("text_prompts/")]
        assert record["provenance"] == "upstream"
        assert record["copied_from_upstream"] is True
        assert record["rows"] == 300
        assert hashlib.sha256((REPO_ROOT / key).read_bytes()).hexdigest() == \
            record["sha256"]
    for key in summary["generated_or_rewritten"]:
        record = repository_assets[key]
        assert record["provenance"] == "generated"
        assert record["copied_from_upstream"] is False
        assert record["upstream_muse_counterpart"] is None
        assert record["rows"] == 40


def test_muse_provenance_uses_content_origin_for_csv_lists():
    upstream = {
        "muse_prompt_csvs": [
            "text_prompts/muse/tcga_nsclc/generated_new_0.csv",
            "text_prompts/muse/tcga_nsclc/generated_new_1.csv",
        ],
    }
    generated = {
        "muse_prompt_csvs": [
            "benchmarks/ubc_ocean/data/ubc_ocean/prompts/muse/"
            "generated_new_0.csv",
            "benchmarks/ubc_ocean/data/ubc_ocean/prompts/muse/"
            "generated_new_1.csv",
        ],
    }

    assert _prompt_provenance(upstream, "muse") == "upstream"
    assert _prompt_provenance(generated, "muse") == "generated"


def test_checked_in_muse_configs_report_method_specific_origin():
    expected = {
        "camelyon16": "upstream",
        "tcga_brca": "upstream",
        "tcga_nsclc": "upstream",
        "tcga_rcc": "generated",
        "ubc_ocean": "generated",
    }
    for cohort, origin in expected.items():
        paths = sorted((REPO_ROOT / "benchmarks" / cohort / "configs").glob(
            "muse*/*.yaml"))
        assert paths
        source = (
            "muse_upstream_description_csvs" if origin == "upstream"
            else "muse_generated_task_extension_csvs")
        for path in paths:
            contents = path.read_text()
            assert f"prompt_provenance: {origin}\n" in contents
            assert f"prompt_source: {source}\n" in contents


def test_slip_manifest_separates_complete_banks_from_task_extensions():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["slip"]
    assets = manifest["assets"]

    assert set(summary["copied_from_upstream"]) == {
        "text_prompts/slip/TCGA_prompt_bank.json",
        "text_prompts/slip/DHMC_prompt_bank.json",
        "text_prompts/slip/PatchGastricADC22_prompt_bank.json",
    }
    assert set(summary["generated_or_rewritten"]) == {
        "text_prompts/slip/camelyon16_tissues.json",
        "text_prompts/slip/tcga_brca_tissues.json",
        "text_prompts/slip/tcga_rcc_tissues.json",
        "text_prompts/slip/ubc_ocean_tissues.json",
    }
    assert assets["slip/TCGA_prompt_bank.json"]["provenance"] == "upstream"
    assert assets["slip/TCGA_prompt_bank.json"]["copied_from_upstream"] is True
    assert assets["slip/TCGA_tissues.json"]["provenance"] == "derived"
    assert assets["slip/TCGA_tissues.json"]["usage"] == "legacy_unwired"
    for key in summary["generated_or_rewritten"]:
        record = assets[key.removeprefix("text_prompts/")]
        assert record["provenance"] == "generated"
        assert record["copied_from_upstream"] is False


def test_slip_provenance_uses_bank_content_origin():
    upstream = {
        "slip_tissue_json": "text_prompts/slip/TCGA_prompt_bank.json",
    }
    generated = {
        "slip_tissue_json": "text_prompts/slip/tcga_rcc_tissues.json",
    }

    assert _prompt_provenance(upstream, "slip") == "upstream"
    assert _prompt_provenance(generated, "slip") == "generated"


def test_cod_mil_provenance_separates_upstream_defect_from_verified_derivative():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["cod_mil"]
    assets = manifest["assets"]

    assert "text_prompts/cod_mil/rcc_chain_of_diagnosis.csv" in \
        summary["copied_from_upstream"]
    assert summary["derived"] == [
        "text_prompts/cod_mil/rcc_text_prompt_features_clip_rn50_verified.pt"]
    released = assets["cod_mil/rcc_text_prompt_features_clip_rn50.pt"]
    verified = assets[
        "cod_mil/rcc_text_prompt_features_clip_rn50_verified.pt"]
    assert released["provenance"] == "upstream"
    assert released["usage"] == "audit_only"
    assert released["known_content_issue"]["tensor_rows"] == 30
    assert verified["provenance"] == "derived"
    assert verified["rows"] == 27
    assert verified["row_roles"] == {
        "low": [0, 3], "high": [3, 6], "background": [6, 27],
    }
    verified_path = (
        REPO_ROOT / "text_prompts/cod_mil/"
        "rcc_text_prompt_features_clip_rn50_verified.pt")
    assert hashlib.sha256(verified_path.read_bytes()).hexdigest() == \
        verified["sha256"]


def test_mscpt_provenance_uses_content_origin_not_path_selection():
    copied_legacy = {
        "mscpt_prompt_json": "train_data/gpt/description/Lung.json",
    }
    copied_explicit = {
        "prompts": {
            "mscpt": "train_data/gpt/description/RCC.json",
        },
    }
    generated_legacy = {
        "mscpt_prompt_json": (
            "text_prompts/mscpt/description/TCGA_BRCA_IDC_ILC.json"),
    }
    generated_explicit = {
        "prompts": {
            "mscpt": (
                "benchmarks/camelyon16/data/camelyon16/prompts/gpt/"
                "description/camelyon16.json"),
        },
    }

    assert _prompt_provenance(copied_legacy, "mscpt") == "upstream"
    assert _prompt_provenance(copied_explicit, "mscpt") == "upstream"
    assert _prompt_provenance(generated_legacy, "mscpt") == "generated"
    assert _prompt_provenance(generated_explicit, "mscpt") == "generated"


def test_wsi_five_provenance_separates_questions_from_comparison_bank():
    native = {
        "wsi_five_questions_json":
            "text_prompts/wsi_five/clinical_questions/nsclc.json",
    }
    extension = {
        "wsi_five_questions_json":
            "text_prompts/wsi_five/clinical_questions/rcc.json",
    }

    assert _wsi_five_prompt_provenance(
        native, "upstream_answer_bank",
    ) == "derived_questions_with_generated_answer_and_derived_evaluation_banks"
    assert _wsi_five_prompt_provenance(
        extension, "simplified_classnames",
    ) == "generated_questions_with_classname_comparison"


def test_wsi_five_manifest_names_every_derived_and_generated_role():
    manifest = json.loads(
        (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())
    summary = manifest["method_summaries"]["wsi_five"]

    derived = {
        "text_prompts/wsi_five/clinical_questions/nsclc.json",
        "text_prompts/wsi_five/nsclc_evaluation_prompts.json",
    }
    assert summary["copied_from_upstream"] == []
    assert set(summary["derived_from_upstream"]) == derived
    for path in derived:
        record = manifest["assets"][path.removeprefix("text_prompts/")]
        assert record["provenance"] == "derived"
        assert record["content_provenance"] == "upstream"
        assert record["copied_from_upstream"] is False
        assert len(record["sha256"]) == 64
    assert summary["mixed_from_upstream_and_generated"] == [
        "text_prompts/wsi_five/nsclc_report_answers.csv"]
    answer_record = manifest["assets"][
        "wsi_five/nsclc_report_answers.csv"]
    assert answer_record["provenance"] == "generated"
    assert answer_record["content_provenance"] == (
        "mixed_upstream_and_generated")
    assert answer_record["upstream_answer_rows"] == 912
    assert answer_record["generated_completion_rows"] == 27
    assert len(answer_record["generated_case_ids"]) == 27
    assert set(summary["generated_or_rewritten"]) == {
        "text_prompts/wsi_five/clinical_questions/brca.json",
        "text_prompts/wsi_five/clinical_questions/rcc.json",
        "text_prompts/wsi_five/clinical_questions/ubc_ocean.json",
    }
