import json
from pathlib import Path
import pickle
import subprocess
import sys
from unittest.mock import patch

import h5py
import numpy as np
import torch

from common.preflight import preflight
from common.prompts import load_prompt_bank_csv, prompt_feature_metadata
from scripts import preflight as preflight_cli


def _write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text(
        "slide_id,feature_a,feature_b\n" +
        "".join(f"slide-{index},{left},{right}\n"
                for index, (left, right) in enumerate(rows)))


def _base_config(tmp_path: Path, manifest: Path) -> dict:
    return {
        "dataset_csv": str(manifest),
        "feature_path_column_a": "feature_a",
        "feature_path_column_b": "feature_b",
    }


def test_preflight_accepts_upstream_fold_filename(tmp_path: Path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "fold0.csv").write_text(
        "train,val,test,train_label,val_label,test_label\n"
        "train-a,val-a,test-a,0,1,0\n")

    report = preflight(
        {"split_dir": str(split_dir), "k": 1}, checks={"splits"})

    assert not any("no fold splits exist" in problem
                   for problem in report.problems)
    assert report.ok


def test_preflight_rejects_out_of_range_upstream_phase_label(tmp_path: Path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "fold0.csv").write_text(
        "train,val,test,train_label,val_label,test_label\n"
        "train-a,val-a,test-a,0,7,1\n")

    report = preflight({
        "split_dir": str(split_dir), "k": 1, "n_classes": 2,
        "label_dict": {"A": 0, "B": 1},
    }, checks={"splits"})

    assert any("out-of-range label '7'" in problem
               for problem in report.problems)


def test_preflight_requires_complete_feature_coverage_by_default(tmp_path: Path):
    present = tmp_path / "present.pt"
    present.write_bytes(b"features")
    missing = tmp_path / "missing.pt"
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(str(present), str(present)),
                               (str(missing), str(present))])

    report = preflight(_base_config(tmp_path, manifest))

    assert not report.ok
    assert report.slides_available == 1
    assert report.slides_expected == 2
    assert report.coverage["all_required_features"] == 0.5
    assert any("below the required 100.0%" in problem
               for problem in report.problems)


def test_preflight_gates_on_joint_feature_coverage(tmp_path: Path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    missing = tmp_path / "missing.pt"
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(str(left), str(missing)),
                               (str(missing), str(right))])
    cfg = {**_base_config(tmp_path, manifest), "min_feature_coverage": 0.5}

    report = preflight(cfg, checks={"features"})

    # Each column is 50% complete, but no slide has both required inputs.
    assert not report.ok
    assert report.slides_available == 0
    assert report.coverage["all_required_features"] == 0.0


def test_preflight_accepts_explicitly_allowed_partial_single_input(
    tmp_path: Path,
):
    present = tmp_path / "present.pt"
    present.write_bytes(b"features")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(str(present), "unused"), ("", "unused")])
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_b")
    cfg["min_feature_coverage"] = 0.5

    report = preflight(cfg, checks={"features"})

    assert report.ok
    assert report.slides_available == 1
    assert any("explicitly allowed" in warning for warning in report.warnings)


def test_preflight_deduplicates_shared_feature_columns(tmp_path: Path):
    present = tmp_path / "present.pt"
    present.write_bytes(b"features")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(str(present), "unused"), ("", "unused")])
    cfg = _base_config(tmp_path, manifest)
    cfg["feature_path_column_b"] = "feature_a"
    cfg["min_feature_coverage"] = 0.5

    report = preflight(cfg, checks={"features"})

    assert report.ok
    assert list(report.coverage) == ["feature_a", "all_required_features"]
    assert len(report.warnings) == 1


def test_preflight_rejects_invalid_coverage_threshold(tmp_path: Path):
    feature = tmp_path / "feature.pt"
    feature.write_bytes(b"features")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(str(feature), str(feature))])
    cfg = {**_base_config(tmp_path, manifest), "min_feature_coverage": 1.5}

    report = preflight(cfg)

    assert not report.ok
    assert any("must be in [0, 1]" in problem for problem in report.problems)


def test_preflight_rejects_boolean_coverage_threshold(tmp_path: Path):
    feature = tmp_path / "feature.pt"
    feature.write_bytes(b"features")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(str(feature), str(feature))])
    cfg = {**_base_config(tmp_path, manifest), "min_feature_coverage": True}

    report = preflight(cfg)

    assert not report.ok
    assert any("not a boolean" in problem for problem in report.problems)


def test_preflight_rejects_duplicate_manifest_headers(tmp_path: Path):
    feature = tmp_path / "feature.pt"
    feature.write_bytes(b"features")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "slide_id,feature_a,feature_a\n"
        f"slide-a,{feature},{feature}\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_b")

    report = preflight(cfg)

    assert not report.ok
    assert any("manifest has duplicate columns" in problem
               for problem in report.problems)
    assert report.coverage == {}


def test_preflight_rejects_blank_and_duplicate_manifest_slide_ids(
    tmp_path: Path,
):
    feature = tmp_path / "feature.pt"
    feature.write_bytes(b"features")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "slide_id,feature_a\n"
        f"slide-a,{feature}\n"
        f"slide-a,{feature}\n"
        f",{feature}\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_b")

    report = preflight(cfg)

    assert not report.ok
    assert any("blank slide IDs" in problem for problem in report.problems)
    assert any("manifest repeats" in problem for problem in report.problems)


def test_preflight_explains_unresolved_feature_environment_path(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "slide_id,feature_a\n"
        "slide-a,${DEFINITELY_MISSING_PGVL_FEATURE_ROOT}/slide-a.h5\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_b")

    report = preflight(cfg)

    assert not report.ok
    assert any(
        "cannot resolve or inspect feature paths" in problem
        and "DEFINITELY_MISSING_PGVL_FEATURE_ROOT" in problem
        for problem in report.problems)


def test_preflight_rejects_incomplete_nested_fold_even_with_flat_split(
    tmp_path: Path,
):
    feature = tmp_path / "feature.pt"
    feature.write_bytes(b"features")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(str(feature), str(feature))])
    cfg = _base_config(tmp_path, manifest)
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "splits_0.csv").write_text(
        "train,val,test\nslide-train,slide-val,slide-test\n")
    cfg.update({"split_dir": str(splits), "k": 1})
    nested = Path(cfg["split_dir"]) / "fold0"
    nested.mkdir()
    (nested / "train.csv").write_text("slide_id,label\n")

    report = preflight(cfg)

    assert not report.ok
    assert any("nested fold split is incomplete" in problem
               for problem in report.problems)


def test_preflight_does_not_rewrite_zero_fold_end_to_default(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    splits = tmp_path / "splits"
    splits.mkdir()

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits),
         "k_start": 0, "k_end": 0},
        checks={"splits"})

    assert not report.ok
    assert any("invalid fold range [0, 0)" in problem
               for problem in report.problems)


def test_preflight_rejects_boolean_fold_indices(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    splits = tmp_path / "splits"
    splits.mkdir()

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits), "k": True},
        checks={"splits"})

    assert not report.ok
    assert any("must be integer fold indices" in problem
               for problem in report.problems)


def test_preflight_rejects_coercible_noninteger_fold_indices(tmp_path: Path):
    splits = tmp_path / "splits"
    splits.mkdir()

    for value in ("1", 1.0):
        report = preflight(
            {"split_dir": str(splits), "k": value}, checks={"splits"})
        assert any("must be integer fold indices" in problem
                   for problem in report.problems)


def test_preflight_rejects_patient_leakage_in_nested_splits(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    nested = splits / "fold0"
    nested.mkdir()
    for phase, slide_id, case_id in (
        ("train", "slide-a", "case-shared"),
        ("val", "slide-b", "case-shared"),
        ("test", "slide-c", "case-test"),
    ):
        (nested / f"{phase}.csv").write_text(
            "slide_id,case_id,partition\n"
            f"{slide_id},{case_id},{phase}\n")

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits), "k": 1},
        checks={"splits"})

    assert not report.ok
    assert any("patient leakage" in problem for problem in report.problems)


def test_preflight_rejects_nested_rows_that_disagree_with_manifest(
    tmp_path: Path,
):
    feature = tmp_path / "slide.pt"
    feature.write_bytes(b"feature")
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "slide_id,case_id,label,feature\n"
        f"slide-a,case-a,A,{feature}\n"
        f"slide-b,case-b,B,{feature}\n"
        f"slide-c,case-c,A,{feature}\n")
    splits = tmp_path / "splits"
    nested = splits / "fold0"
    nested.mkdir(parents=True)
    for phase, slide_id, case_id, label, path in (
        ("train", "slide-a", "case-wrong", "B", other),
        ("val", "slide-b", "case-b", "B", feature),
        ("test", "slide-c", "case-c", "A", feature),
    ):
        (nested / f"{phase}.csv").write_text(
            "slide_id,case_id,label,partition,feature\n"
            f"{slide_id},{case_id},{label},{phase},{path}\n")

    report = preflight({
        "dataset_csv": str(manifest),
        "split_dir": str(splits),
        "k": 1,
        "feature_path_column": "feature",
        "label_dict": {"A": 0, "B": 1},
        "n_classes": 2,
    }, checks={"splits"})

    assert any("case_id does not match manifest" in item
               for item in report.problems)
    assert any("label does not match manifest" in item
               for item in report.problems)
    assert any("feature column 'feature' does not match manifest" in item
               for item in report.problems)


def test_preflight_rejects_slide_leakage_in_flat_splits(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "splits_0.csv").write_text(
        "train,val,test\nshared,shared,test-only\n")

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits), "k": 1},
        checks={"splits"})

    assert not report.ok
    assert any("partition leakage" in problem for problem in report.problems)


def test_preflight_rejects_unknown_manifest_ids_in_flat_splits(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "slide_id,case_id\n"
        "slide-train,case-train\n"
        "slide-val,case-val\n"
        "slide-test,case-test\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "splits_0.csv").write_text(
        "train,val,test\nunknown,slide-val,slide-test\n")

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits), "k": 1},
        checks={"splits"})

    assert any("unknown" in problem and "absent from the manifest" in problem
               for problem in report.problems)


def test_preflight_rejects_patient_leakage_in_flat_splits(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "slide_id,case_id\n"
        "slide-a,case-shared\n"
        "slide-b,case-shared\n"
        "slide-c,case-test\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "splits_0.csv").write_text(
        "train,val,test\nslide-a,slide-b,slide-c\n")

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits), "k": 1},
        checks={"splits"})

    assert any("patient leakage" in problem for problem in report.problems)


def test_preflight_rejects_malformed_split_rows(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "splits_0.csv").write_text("train,val,test\na,b\n")

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits), "k": 1},
        checks={"splits"})

    assert not report.ok
    assert any("wrong number of fields" in problem
               for problem in report.problems)


def test_preflight_rejects_blank_case_ids_in_nested_splits(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    nested = splits / "fold0"
    nested.mkdir()
    for phase, slide_id, case_id in (
        ("train", "slide-a", "case-a"),
        ("val", "slide-b", ""),
        ("test", "slide-c", "case-c"),
    ):
        (nested / f"{phase}.csv").write_text(
            "slide_id,case_id,partition\n"
            f"{slide_id},{case_id},{phase}\n")

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits), "k": 1},
        checks={"splits"})

    assert not report.ok
    assert any("patient leakage cannot be ruled out" in problem
               for problem in report.problems)


def test_preflight_requires_case_ids_and_nonblank_partitions_in_nested_splits(
    tmp_path: Path,
):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    nested = splits / "fold0"
    nested.mkdir()
    for phase, partition in (("train", "train"), ("val", ""),
                             ("test", "test")):
        (nested / f"{phase}.csv").write_text(
            "slide_id,partition\n"
            f"slide-{phase},{partition}\n")

    report = preflight(
        {"dataset_csv": str(manifest), "split_dir": str(splits), "k": 1},
        checks={"splits"})

    assert not report.ok
    assert sum("no 'case_id' column" in problem
               for problem in report.problems) == 3
    assert any("different partition value" in problem
               for problem in report.problems)


def test_preflight_rejects_results_path_that_is_a_file(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    results_file = tmp_path / "results"
    results_file.write_text("not a directory\n")

    report = preflight(
        {"dataset_csv": str(manifest), "results_dir": str(results_file)},
        checks={"assets"})

    assert not report.ok
    assert any("results_dir" in problem and "not a directory" in problem
               for problem in report.problems)


def test_preflight_rejects_wrong_input_path_types(tmp_path: Path):
    manifest_directory = tmp_path / "manifest"
    manifest_directory.mkdir()
    (manifest_directory / "row.csv").write_text("data\n")
    feature_file = tmp_path / "features"
    feature_file.write_text("not a directory\n")

    report = preflight(
        {
            "dataset_csv": str(manifest_directory),
            "data_folder_s": str(feature_file),
        },
        checks={"assets", "features"},
        check_features=False)

    assert not report.ok
    assert any("dataset_csv" in problem and "expected a file" in problem
               for problem in report.problems)
    assert any("data_folder_s" in problem and "expected a directory" in problem
               for problem in report.problems)


def test_preflight_accepts_creatable_results_directory(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    results_dir = tmp_path / "new" / "nested" / "results"

    report = preflight(
        {"dataset_csv": str(manifest), "results_dir": str(results_dir)},
        checks={"assets"})

    assert report.ok
    assert report.checked_paths["results_dir"]["available"]
    assert report.checked_paths["results_dir"]["kind"] == (
        "creatable_output_directory")
    assert not results_dir.exists()


def test_preflight_rejects_empty_or_broad_results_directory(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")

    empty = preflight(
        {"dataset_csv": str(manifest), "results_dir": ""},
        checks={"assets"})
    broad = preflight(
        {"dataset_csv": str(manifest), "results_dir": str(Path.cwd())},
        checks={"assets"})

    assert not empty.ok
    assert any("empty or not path-like" in problem
               for problem in empty.problems)
    assert not broad.ok
    assert any("unsafe broad directory" in problem
               for problem in broad.problems)


def test_preflight_check_selection_is_independent(tmp_path: Path):
    prompt = tmp_path / "prompt.csv"
    prompt.write_text("class_name,prompt\nA,a diagnosis\n")

    report = preflight(
        {"text_prompt_path": str(prompt)}, checks={"prompts"})

    assert report.ok
    assert report.checks == ["prompts"]
    assert report.checked_paths["text_prompt_path"]["available"]
    assert not any("split_dir" in problem for problem in report.problems)


def test_preflight_checks_method_specific_configured_assets(tmp_path: Path):
    missing = tmp_path / "missing"
    cfg = {
        "cross_mag_map_dir": str(missing / "maps"),
        "text_prompt_features": str(missing / "prompts.pt"),
        "normal_structures_json": str(missing / "normal.json"),
        "report_csv": str(missing / "reports.csv"),
        "attribute_embeddings": str(missing / "attributes.pt"),
        "clinicalbert_weights": str(missing / "clinicalbert"),
        "initial_checkpoint": str(missing / "initial.pt"),
    }

    report = preflight(
        cfg, checks={"features", "prompts", "encoders"},
        check_features=False)

    assert not report.ok
    assert set(cfg) <= set(report.checked_paths)


def test_preflight_rejects_absent_required_method_assets():
    convlm = preflight(
        {"method": "convlm", "attribute_embeddings": "attributes.pt"},
        checks={"features"}, check_features=False)
    cod_mil = preflight(
        {
            "method": "cod_mil",
            "feature_path_column_s": "low",
            "feature_path_column_l": "high",
        },
        checks={"features"}, check_features=False)
    wsi_five = preflight(
        {"method": "wsi_five"}, checks={"encoders"},
        check_features=False)

    assert any("patch bags requires one of" in problem
               for problem in convlm.problems)
    assert any("requires cross_mag_map_dir" in problem
               for problem in cod_mil.problems)
    assert any("requires clinicalbert_weights" in problem
               for problem in wsi_five.problems)


def test_preflight_rejects_unsafe_runtime_settings():
    batched_bags = preflight(
        {
            "method": "wsi_five",
            "batch_size": 4,
            "feature_path_column": "feature_path",
        },
        checks={"features"}, check_features=False)
    failure_rate = preflight(
        {"max_batch_failure_rate": float("nan")}, checks={"assets"})

    assert any("requires batch_size=1" in problem
               for problem in batched_bags.problems)
    assert any("max_batch_failure_rate must be in" in problem
               for problem in failure_rate.problems)


def test_preflight_rejects_invalid_optimizer_and_staged_settings():
    report = preflight({
        "method": "sldpc",
        "epochs": 5,
        "stage1_epochs": 2,
        "stage2_epochs": 2,
        "early_stopping": True,
        "lr": float("nan"),
        "weight_decay": -1,
    }, checks={"assets"})

    assert any("lr must be a finite positive" in problem
               for problem in report.problems)
    assert any("weight_decay must be a finite non-negative" in problem
               for problem in report.problems)
    assert any("epochs must equal" in problem for problem in report.problems)
    assert any("early_stopping=false" in problem
               for problem in report.problems)

    missing_epochs = preflight({
        "method": "sldpc",
        "stage1_epochs": 2,
        "stage2_epochs": 2,
        "early_stopping": False,
    }, checks={"assets"})
    assert any("requires explicit epochs" in problem
               for problem in missing_epochs.problems)


def test_preflight_rejects_invalid_model_hyperparameters():
    report = preflight({
        "prototype_number": 0,
        "pos_ratio": 0,
        "p_drop_out": 1.0,
        "embed_dim": 10,
        "num_heads": 3,
        "num_experts": 2,
        "num_selected": 3,
    }, checks={"assets"})

    assert any("prototype_number must be a positive integer" in item
               for item in report.problems)
    assert any("pos_ratio must be a finite number in (0, 1]" in item
               for item in report.problems)
    assert any("p_drop_out must be a finite number in [0, 1)" in item
               for item in report.problems)
    assert any("must be divisible" in item for item in report.problems)
    assert any("cannot exceed" in item for item in report.problems)


def test_preflight_rejects_invalid_method_modes_and_nested_projection():
    pathpt = preflight({
        "method": "pathpt",
        "prompt_init": "typo",
        "learnable": "mystery",
        "lr_milestones": [2, 1, 1],
    }, checks={"assets"})
    sldpc = preflight({
        "method": "sldpc",
        "epochs": 2,
        "stage1_epochs": 1,
        "stage2_epochs": 1,
        "early_stopping": False,
        "class_token_position": "sideways",
        "slide_projection": {"mode": "unknown", "dropout": 1.0},
    }, checks={"assets"})

    assert any("prompt_init must be one of" in item
               for item in pathpt.problems)
    assert any("learnable must be one of" in item
               for item in pathpt.problems)
    assert any("lr_milestones must be" in item
               for item in pathpt.problems)
    assert any("class_token_position must be one of" in item
               for item in sldpc.problems)
    assert any("slide_projection.mode" in item for item in sldpc.problems)
    assert any("slide_projection.dropout" in item for item in sldpc.problems)


def test_preflight_rejects_top_binary_only_pooling_for_multiclass_tasks():
    report = preflight({
        "method": "top",
        "n_classes": 3,
        "classnames": ["A", "B", "C"],
        "label_dict": {"A": 0, "B": 1, "C": 2},
        "pooling_strategy": "learnablePrompt_multi_noCoOp",
    }, checks={"assets"})

    assert any("binary-only" in item for item in report.problems)


def test_preflight_rejects_invalid_class_and_epoch_schema():
    report = preflight({
        "epochs": 0,
        "n_classes": 2,
        "classnames": ["only one"],
        "label_dict": {"A": 0, "B": 0},
    }, checks={"assets"})

    assert not report.ok
    assert any("epochs must be" in problem for problem in report.problems)
    assert any("classnames has 1" in problem for problem in report.problems)
    assert any("label_dict values" in problem for problem in report.problems)


def test_preflight_rejects_malformed_class_and_slide_source_schema():
    report = preflight({
        "feature_dim": "eight",
        "classnames": ["A", "A", ""],
        "label_dict": ["A"],
        "source_type": "directory",
    }, checks={"assets"})

    assert not report.ok
    assert any("feature_dim must be" in problem for problem in report.problems)
    assert any("classnames entries must be non-empty" in problem
               for problem in report.problems)
    assert any("classnames entries must be unique" in problem
               for problem in report.problems)
    assert any("label_dict must be a mapping" in problem
               for problem in report.problems)
    assert any("source_type must be" in problem for problem in report.problems)


def test_preflight_checks_slide_feature_source_kind(tmp_path: Path):
    feature_file = tmp_path / "features.pkl"
    feature_file.write_bytes(b"payload")
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    (feature_root / "slide.h5").write_bytes(b"payload")

    pkl_as_directory = preflight({
        "slide_features": str(feature_root), "source_type": "pkl",
    }, checks={"features"}, check_features=False)
    directory_as_file = preflight({
        "slide_features": str(feature_file), "source_type": "per_slide_h5",
    }, checks={"features"}, check_features=False)

    assert any("expected a file" in problem
               for problem in pkl_as_directory.problems)
    assert any("expected a directory" in problem
               for problem in directory_as_file.problems)


def test_preflight_records_partial_fidelity_and_can_gate_it():
    warning = preflight({"method": "muse"}, checks={"assets"})
    strict = preflight({
        "method": "muse", "require_upstream_fidelity": True,
    }, checks={"assets"})

    assert warning.ok
    assert any("is partial" in message for message in warning.warnings)
    assert not strict.ok
    assert any("is partial" in message for message in strict.problems)


def test_deep_preflight_opens_feature_payloads_and_rejects_wrong_width(
    tmp_path: Path,
):
    feature = tmp_path / "slide.h5"
    with h5py.File(feature, "w") as handle:
        handle.create_dataset(
            "features", data=np.ones((3, 7), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        f"slide_id,feature\nslide-a,{feature}\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_a")
    cfg.pop("feature_path_column_b")
    cfg["feature_path_column"] = "feature"
    cfg["feature_dim"] = 8

    report = preflight(cfg, deep_features=True)

    assert not report.ok
    assert report.deep_features_checked == 1
    assert any("feature width 7, expected 8" in problem
               for problem in report.problems)


def test_deep_preflight_scans_middle_of_large_hdf5_payload(tmp_path: Path):
    feature = tmp_path / "slide.h5"
    values = np.ones((5000, 1), dtype=np.float32)
    values[2500, 0] = np.nan
    with h5py.File(feature, "w") as handle:
        handle.create_dataset("features", data=values)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"slide_id,feature\nslide-a,{feature}\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_a")
    cfg.pop("feature_path_column_b")
    cfg["feature_path_column"] = "feature"
    cfg["feature_dim"] = 1

    report = preflight(cfg, deep_features=True)

    assert not report.ok
    assert any("contains NaN or infinity" in problem
               for problem in report.problems)


def test_deep_preflight_validates_shared_pickle_once(tmp_path: Path):
    feature = tmp_path / "slides.pkl"
    with feature.open("wb") as handle:
        pickle.dump({
            "features": np.ones((2, 4), dtype=np.float32),
            "filenames": ["slide-a.svs", "slide-b.svs"],
        }, handle)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "slide_id,feature\n"
        f"slide-a,{feature}\n"
        f"slide-b,{feature}\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_a")
    cfg.pop("feature_path_column_b")
    cfg.update({
        "feature_path_column": "feature",
        "slide_features": str(feature),
        "source_type": "pkl",
        "feature_dim": 4,
    })

    report = preflight(cfg, deep_features=True)

    assert report.deep_features_checked == 1
    assert not any("identical feature paths" in problem
                   for problem in report.problems)
    assert not any("deep slide-embedding" in problem
                   for problem in report.problems)


def test_deep_preflight_rejects_duplicate_pickle_slide_ids(tmp_path: Path):
    feature = tmp_path / "slides.pkl"
    with feature.open("wb") as handle:
        pickle.dump({
            "features": np.ones((2, 4), dtype=np.float32),
            "filenames": ["slide-a.svs", "slide-a.pt"],
        }, handle)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_a")
    cfg.pop("feature_path_column_b")
    cfg.update({
        "slide_features": str(feature),
        "source_type": "pkl",
        "feature_dim": 4,
    })

    report = preflight(cfg, deep_features=True)

    assert any("duplicate normalized slide IDs" in problem
               for problem in report.problems)


def test_deep_preflight_rejects_pickle_missing_manifest_slide(tmp_path: Path):
    feature = tmp_path / "slides.pkl"
    with feature.open("wb") as handle:
        pickle.dump({
            "features": np.ones((1, 4), dtype=np.float32),
            "filenames": ["slide-a.svs"],
        }, handle)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\nslide-b\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_a")
    cfg.pop("feature_path_column_b")
    cfg.update({
        "slide_features": str(feature),
        "source_type": "pkl",
        "feature_dim": 4,
    })

    report = preflight(cfg, deep_features=True)

    assert any("pickle is missing 1 manifest slide IDs (slide-b)" in problem
               for problem in report.problems)


def test_deep_preflight_decodes_pickle_byte_slide_ids(tmp_path: Path):
    feature = tmp_path / "slides.pkl"
    with feature.open("wb") as handle:
        pickle.dump({
            "features": np.ones((1, 4), dtype=np.float32),
            "filenames": np.asarray([b"slide-a.svs"]),
        }, handle)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("slide_id\nslide-a\n")
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_a")
    cfg.pop("feature_path_column_b")
    cfg.update({
        "slide_features": str(feature),
        "source_type": "pkl",
        "feature_dim": 4,
    })

    report = preflight(cfg, deep_features=True)

    assert not any("deep slide-embedding" in problem
                   for problem in report.problems)


def test_deep_preflight_rejects_disabled_feature_scan():
    try:
        preflight({}, deep_features=True, check_features=False)
    except ValueError as error:
        assert "requires an enabled features scan" in str(error)
    else:
        raise AssertionError("deep preflight silently skipped its feature scan")


def test_preflight_rejects_incomplete_method_prompt_contracts(tmp_path: Path):
    focus = preflight({"method": "focus"}, checks={"prompts"})
    convlm = preflight(
        {
            "method": "convlm",
            "attribute_prompt_path": str(tmp_path / "attributes.json"),
        },
        checks={"prompts"})
    cod_mil = preflight(
        {"method": "cod_mil", "text_prompt_path": "chains.json"},
        checks={"prompts"})
    mscpt = preflight(
        {
            "method": "mscpt",
            "description_prompt_path": str(tmp_path / "wrong" / "task.json"),
        },
        checks={"prompts"})

    assert any("requires text_prompt_path" in problem
               for problem in focus.problems)
    assert any("attribute_encoder mapping" in problem
               for problem in convlm.problems)
    assert any("prompt_encoding=runtime_cached" in problem
               for problem in cod_mil.problems)
    assert any("inside a description directory" in problem
               for problem in mscpt.problems)


def test_preflight_rejects_malformed_maple_prompt_graph(tmp_path: Path):
    prompt = tmp_path / "maple.json"
    prompt.write_text(json.dumps({
        "low": {
            "tumor": "tumor",
            "global_info": {"class a": "description"},
            "entities": ["not a mapping"],
        },
    }))
    report = preflight({
        "method": "maple",
        "text_prompt_path": str(prompt),
        "classnames": ["class a"],
    }, checks={"prompts"})

    assert any("requires a 'high' prompt block" in item
               for item in report.problems)
    assert any("entity 0 must be a mapping" in item
               for item in report.problems)


def test_preflight_rejects_mscpt_prompt_cardinality_mismatch(tmp_path: Path):
    description_dir = tmp_path / "description"
    description_dir.mkdir()
    prompt = description_dir / "task.json"
    prompt.write_text(json.dumps({
        "A": {"small_mag": ["one"], "big_mag": ["one"]},
        "B": {"small_mag": ["one", "two"], "big_mag": ["one"]},
    }))
    report = preflight({
        "method": "mscpt",
        "description_prompt_path": str(prompt),
        "label_dict": {"A": 0, "B": 1},
        "n_high": 2,
    }, checks={"prompts"})

    assert any("prompt counts must match across classes" in item
               for item in report.problems)


def test_preflight_rejects_malformed_top_prompt_banks(tmp_path: Path):
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps({"prototypes": [{"prompt": ""}]}))
    bags = tmp_path / "bags.json"
    bags.write_text(json.dumps({"prompts": {"A": "description"}}))

    report = preflight({
        "method": "top",
        "instance_prompt_path": str(instances),
        "bag_prompt_path": str(bags),
        "label_dict": {"A": 0, "B": 1},
    }, checks={"prompts"})

    assert any("at least two instance prototypes" in item
               for item in report.problems)
    assert any("missing task labels ['B']" in item
               for item in report.problems)


def test_preflight_rejects_slip_prompt_drift(tmp_path: Path):
    tissues = tmp_path / "tissues.json"
    tissues.write_text(json.dumps({"tissues": ["stroma", "necrosis"]}))

    report = preflight({
        "method": "slip",
        "tissue_classnames_path": str(tissues),
        "tissue_classnames": ["stroma", "tumor"],
        "n_classes": 3,
        "text_templates": ["missing placeholder"],
    }, checks={"prompts"})

    assert any("exactly match" in item for item in report.problems)
    assert any("smaller than n_classes" in item for item in report.problems)
    assert any("contain '{}'" in item for item in report.problems)


def test_preflight_rejects_malformed_cod_prompt_chain(tmp_path: Path):
    prompt = tmp_path / "cod.json"
    prompt.write_text(json.dumps({
        "A": {"broad": [], "specific": ["high A"]},
        "unexpected": "not a prompt block",
    }))

    report = preflight({
        "method": "cod_mil",
        "prompt_encoding": "runtime_cached",
        "text_prompt_path": str(prompt),
        "classnames": ["A", "B"],
    }, checks={"prompts"})

    assert any("do not match configured classnames" in item
               for item in report.problems)
    assert any("A.broad must be a non-empty string list" in item
               for item in report.problems)
    assert any("'unexpected' must be a mapping" in item
               for item in report.problems)


def test_preflight_rejects_unbound_cod_prompt_tensor_and_accepts_verified_bank(
        tmp_path: Path):
    classnames = ["class A", "class B"]
    chain_path = tmp_path / "chain.json"
    chain_path.write_text(json.dumps({
        "class A": {"broad": ["A low"], "specific": ["A high"]},
        "class B": {"broad": ["B low"], "specific": ["B high"]},
    }))
    bank_path = tmp_path / "bank.csv"
    prompts = (
        ["A low", "B low", "A high", "B high"]
        + [f"normal tissue {index}" for index in range(15)]
    )
    bank_path.write_text("\n".join(prompts) + "\n")
    tensor_path = tmp_path / "prompts.pt"
    embeddings = torch.nn.functional.normalize(
        torch.ones((len(prompts), 8)), dim=1)
    base = {
        "method": "cod_mil",
        "text_prompt_path": str(chain_path),
        "text_prompt_bank_csv": str(bank_path),
        "text_prompt_features": str(tensor_path),
        "classnames": classnames,
        "n_classes": 2,
        "feature_dim": 8,
        "text_feature_space_id": "test/paired",
    }

    torch.save(embeddings, tensor_path)
    legacy = preflight(base, checks={"prompts"})
    assert any("unverified legacy" in item for item in legacy.problems)

    exact_prompts = load_prompt_bank_csv(bank_path)
    torch.save({
        "embeddings": embeddings,
        **prompt_feature_metadata(
            exact_prompts,
            n_classes=2,
            source_path=bank_path,
            feature_space_id="test/paired",
            encoder="test",
            checkpoint_sha256="0" * 64,
        ),
    }, tensor_path)
    verified = preflight(base, checks={"prompts"})
    assert verified.ok, verified.problems


def test_preflight_rejects_empty_muse_description_csv(tmp_path: Path):
    prompt = tmp_path / "empty.csv"
    prompt.write_text(",0\n0,\n")

    report = preflight({
        "method": "muse",
        "prompt_csvs": {"A": str(prompt)},
        "classnames": ["A"],
    }, checks={"prompts"})

    assert any("empty description" in item
               for item in report.problems)


def test_preflight_rejects_convlm_attribute_order_drift(tmp_path: Path):
    prompt = tmp_path / "attributes.json"
    prompt.write_text(json.dumps({"B": ["b attr"], "A": ["a attr"]}))

    report = preflight({
        "method": "convlm",
        "attribute_prompt_path": str(prompt),
        "attribute_encoder": {"model_name": "test", "weights": "test"},
        "classnames": ["A", "B"],
    }, checks={"prompts"})

    assert any("order must match classnames" in item
               for item in report.problems)


def test_preflight_rejects_malformed_sldpc_prompt_reference(tmp_path: Path):
    prompt = tmp_path / "sldpc.yaml"
    prompt.write_text("prompts:\n  A: [valid]\n")

    report = preflight({
        "method": "sldpc",
        "prompt_reference_yaml": str(prompt),
        "label_dict": {"A": 0, "B": 1},
    }, checks={"prompts"})

    assert any("prompt keys must match label_dict" in item
               for item in report.problems)


def test_preflight_validates_focus_prompt_table_schema(tmp_path: Path):
    prompt = tmp_path / "focus.csv"
    prompt.write_text(
        "class_name,low_res_prompt,high_res_prompt\n"
        "A,low A,high A\n")

    report = preflight({
        "method": "focus",
        "text_prompt_path": str(prompt),
        "n_classes": 2,
    }, checks={"prompts"})

    assert not report.ok
    assert any("1 rows for n_classes=2" in problem
               for problem in report.problems)


def test_preflight_rejects_misordered_vila_prompt_classes(tmp_path: Path):
    prompt = tmp_path / "vila.csv"
    prompt.write_text(
        "class_name,low_res_prompt,high_res_prompt\n"
        "B,low B,high B\n"
        "A,low A,high A\n")

    report = preflight({
        "method": "vila_mil",
        "text_prompt_path": str(prompt),
        "n_classes": 2,
        "classnames": ["class A", "class B"],
        "label_dict": {"A": 0, "B": 1},
    }, checks={"prompts"})

    assert not report.ok
    assert any("class_name order" in problem for problem in report.problems)


def test_preflight_rejects_conflicting_wsi_five_reports(tmp_path: Path):
    reports = tmp_path / "reports.csv"
    reports.write_text(
        "patient_filename,text\n"
        "TCGA-AA-0001.first,first report\n"
        "TCGA-AA-0001.second,second report\n")

    report = preflight({
        "method": "wsi_five",
        "report_csv": str(reports),
    }, checks={"prompts"})

    assert not report.ok
    assert any("conflicting reports" in problem for problem in report.problems)


def test_preflight_rejects_scalar_wsi_five_question_payload(tmp_path: Path):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": "not a list"}))

    report = preflight({
        "method": "wsi_five",
        "clinical_questions": str(questions),
    }, checks={"prompts"})

    assert not report.ok
    assert any("questions' string list" in problem
               for problem in report.problems)


def _write_wsi_five_native_assets(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    answers.write_text(
        "case_id,answer,q1,q2,q3,q4,q5,q6\n"
        "case-a,all answers,a1,a2,a3,a4,a5,a6\n")
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({
        "_provenance": "upstream",
        "questions": [f"question {index}" for index in range(1, 7)],
    }))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "_provenance": "upstream",
        "prompts": {"LUAD": "adenocarcinoma", "LUSC": "squamous"},
    }))
    return answers, questions, evaluation


def test_preflight_accepts_complete_wsi_five_native_prompt_contract(
        tmp_path: Path):
    answers, questions, evaluation = _write_wsi_five_native_assets(tmp_path)

    report = preflight({
        "method": "wsi_five",
        "training_mode": "upstream_answer_bank",
        "report_csv": str(answers),
        "clinical_questions": str(questions),
        "evaluation_prompt_path": str(evaluation),
        "require_report": True,
        "n_classes": 2,
        "label_dict": {"LUAD": 0, "LUSC": 1},
    }, checks={"prompts"})

    assert report.ok, report.problems


def test_preflight_rejects_incomplete_wsi_five_native_answer_schema(
        tmp_path: Path):
    answers, questions, evaluation = _write_wsi_five_native_assets(tmp_path)
    answers.write_text(
        "case_id,answer,q1,q2,q3,q4,q5\n"
        "case-a,all answers,a1,a2,a3,a4,a5\n")

    report = preflight({
        "method": "wsi_five",
        "training_mode": "upstream_answer_bank",
        "report_csv": str(answers),
        "clinical_questions": str(questions),
        "evaluation_prompt_path": str(evaluation),
        "require_report": True,
        "n_classes": 2,
        "label_dict": {"LUAD": 0, "LUSC": 1},
    }, checks={"prompts"})

    assert not report.ok
    assert any("missing columns" in problem and "q6" in problem
               for problem in report.problems)


def test_preflight_rejects_misaligned_wsi_five_evaluation_bank(
        tmp_path: Path):
    answers, questions, evaluation = _write_wsi_five_native_assets(tmp_path)
    evaluation.write_text(json.dumps({
        "prompts": {"LUAD": "adenocarcinoma", "OTHER": "wrong task"},
    }))

    report = preflight({
        "method": "wsi_five",
        "training_mode": "upstream_answer_bank",
        "report_csv": str(answers),
        "clinical_questions": str(questions),
        "evaluation_prompt_path": str(evaluation),
        "require_report": True,
        "n_classes": 2,
        "label_dict": {"LUAD": 0, "LUSC": 1},
    }, checks={"prompts"})

    assert not report.ok
    assert any("exactly match label_dict" in problem
               for problem in report.problems)


def test_preflight_cli_emits_json_and_failure_exit_code(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(f"text_prompt_path: {tmp_path / 'missing.csv'}\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(config), "--prompts", "--json"],
        capture_output=True, text=True, check=False)

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert not payload["healthy"]
    assert payload["schema_version"] == 1
    assert payload["results"][0]["checks"] == ["prompts"]


def test_preflight_cli_renders_doctor_diagnosis(tmp_path: Path):
    prompt = tmp_path / "prompt.csv"
    prompt.write_text("class_name,prompt\nA,a diagnosis\n")
    config = tmp_path / "config.yaml"
    config.write_text(f"text_prompt_path: {prompt}\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(config), "--prompts", "--no-color"],
        capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "PGVL-Gym doctor" in completed.stdout
    assert "PASS Diagnosis: 1/1 configs ready" in completed.stdout


def test_preflight_cli_reports_malformed_yaml_without_traceback(tmp_path: Path):
    config = tmp_path / "malformed.yaml"
    config.write_text("broken: [yaml\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(config), "--json"],
        capture_output=True, text=True, check=False)

    assert completed.returncode == 1
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["results"][0]["load_error"]


def test_preflight_cli_strict_mode_fails_on_partial_coverage(tmp_path: Path):
    present = tmp_path / "present.pt"
    present.write_bytes(b"features")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(str(present), "unused"), ("", "unused")])
    cfg = _base_config(tmp_path, manifest)
    cfg.pop("feature_path_column_b")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(f"{key}: {value}" for key, value in cfg.items())
        + "\nmin_feature_coverage: 0.5\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"

    normal = subprocess.run(
        [sys.executable, str(script), str(config), "--features", "--json"],
        capture_output=True, text=True, check=False)
    strict = subprocess.run(
        [sys.executable, str(script), str(config), "--features", "--strict",
         "--json"], capture_output=True, text=True, check=False)

    assert normal.returncode == 0
    assert strict.returncode == 1
    assert json.loads(normal.stdout)["healthy"]
    assert not json.loads(strict.stdout)["healthy"]


def test_system_doctor_reports_malformed_dotenv_instead_of_crashing():
    with patch.object(
        preflight_cli, "load_dotenv",
        side_effect=ValueError("invalid .env entry at .env:2"),
    ):
        report = preflight_cli._system_diagnostics()

    dotenv = next(
        item for item in report["checks"]
        if item["name"] == "environment.dotenv")
    assert dotenv["status"] == "fail"
    assert "invalid .env entry" in dotenv["message"]
    assert not report["ok"]


def test_system_doctor_reports_process_environment_source(tmp_path: Path):
    roots = {
        "PGVL_REPO_ROOT": str(Path.cwd()),
        "PGVL_USER_ROOT": str(tmp_path),
        "PGVL_STORAGE_ROOT": str(tmp_path),
    }
    with patch.dict(preflight_cli.os.environ, roots, clear=False), patch.object(
        preflight_cli, "load_dotenv", return_value=Path.cwd() / ".env",
    ):
        report = preflight_cli._system_diagnostics()

    repository_root = next(
        item for item in report["checks"]
        if item["name"] == "environment.PGVL_REPO_ROOT")
    assert "via process environment" in repository_root["message"]


def test_system_doctor_parses_dependency_version_prefixes():
    assert preflight_cli._major_minor("2.5.1+cu124") == (2, 5)
    assert preflight_cli._major_minor("4.40.0.dev0") == (4, 40)
    assert preflight_cli._major_minor("unknown") is None


def test_system_doctor_checks_torchvision_release_pairing():
    assert preflight_cli._torchvision_compatible("2.5.1+cu124", "0.20.1")
    assert not preflight_cli._torchvision_compatible("2.5.1", "0.21.0")


def test_doctor_advice_prioritizes_split_and_prompt_context():
    assert preflight_cli._advice_for(
        "split feature path differs from manifest").startswith("Regenerate")
    assert preflight_cli._advice_for(
        "text prompt feature file is missing").startswith("Generate/import")
