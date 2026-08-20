import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from common.datasets.dataset_generic import _load_feature_tensor
from common.datasets.bag_features import BagFeaturesDataset
from common.utils.core_utils import Accuracy_Logger
from eval import (
    _checkpoint_kind,
    _validate_checkpoint_identity,
    _validate_output_destination,
)
from train import (
    EXIT_CONFIG_ERROR,
    EXIT_SKIPPED,
    ResultsDirectoryBusyError,
    _acquire_results_lock,
    _load_resume_state,
    _archive_resume_state,
    _pending_folds,
    _resolve_run_method,
    _release_results_lock,
    _batch_metadata,
    _run_identity,
    _validate_step_output,
    _validate_device,
    _write_metrics,
    classification_metrics,
    main,
)


def test_batch_metadata_ignores_non_identity_dictionaries():
    patch_info = {"patch_inds": torch.tensor([[0.0]]), "sample_range": [1]}
    identity = {"slide_id": ["slide-a"], "case_id": ["case-a"]}
    batch = (torch.ones(1, 1, 2), ["report"], patch_info, identity,
             torch.tensor([0]))

    assert _batch_metadata(batch) == identity


def test_batch_metadata_fills_case_ids_for_tuple_batches():
    batch = (
        torch.ones(2, 1, 2),
        {"slide_id": ["patient00001-slide-a", "patient00002-slide-b"]},
        torch.tensor([0, 1]),
    )

    assert _batch_metadata(batch) == {
        "slide_id": ["patient00001-slide-a", "patient00002-slide-b"],
        "case_id": ["patient00001", "patient00002"],
    }


def test_batch_metadata_decodes_byte_identifiers_and_rejects_blanks():
    assert _batch_metadata({
        "slide_id": [b"slide-a"], "case_id": [b"case-a"],
    }) == {"slide_id": ["slide-a"], "case_id": ["case-a"]}

    with pytest.raises(ValueError, match="blank case_id"):
        _batch_metadata({"slide_id": ["slide-a"], "case_id": [" "]})


def test_evaluation_checkpoint_auto_follows_early_stopping():
    assert _checkpoint_kind("auto", {"early_stopping": True}) == "best"
    assert _checkpoint_kind("auto", {"early_stopping": False}) == "final"
    assert _checkpoint_kind("final", {"early_stopping": True}) == "final"


def test_evaluation_output_cannot_overwrite_inputs(tmp_path: Path):
    config = tmp_path / "run.yaml"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint = checkpoint_dir / "fold0_final.pt"

    assert "input config" in _validate_output_destination(
        config, config, checkpoint_dir, [checkpoint])
    assert "checkpoint metrics" in _validate_output_destination(
        checkpoint_dir / "metrics.json", config, checkpoint_dir, [checkpoint])
    assert "checkpoint fold0_final.pt" in _validate_output_destination(
        checkpoint, config, checkpoint_dir, [checkpoint])
    assert _validate_output_destination(
        tmp_path / "evaluation.json", config, checkpoint_dir, [checkpoint]) is None


def test_dual_scale_loader_reads_native_hdf5_feature_bag(tmp_path: Path):
    path = tmp_path / "slide.h5"
    expected = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=expected.numpy())

    actual = _load_feature_tensor(path)

    assert torch.equal(actual, expected)


def test_evaluation_patch_cap_is_deterministic(tmp_path: Path):
    path = tmp_path / "slide.h5"
    features = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features.numpy())
    manifest = tmp_path / "test.csv"
    manifest.write_text(
        f"slide_id,label,feature\nslide-a,A,{path}\n")
    dataset = BagFeaturesDataset(
        str(manifest), "", {"A": 0}, max_patches=5,
        feature_path_column="feature", random_subsampling=False)

    first, _ = dataset[0]
    torch.manual_seed(999)
    second, _ = dataset[0]

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, features[[0, 5, 10, 14, 19]])


def test_accuracy_logger_reports_micro_accuracy():
    logger = Accuracy_Logger(n_classes=3)
    logger.log_batch([0, 0, 2, 1], [0, 1, 2, 1])

    assert logger.get_overall_summary() == (0.75, 3, 4)


def test_runtime_device_validation_is_actionable(monkeypatch):
    assert _validate_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="supports cpu and cuda"):
        _validate_device("meta")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        _validate_device("cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="only 1 CUDA device"):
        _validate_device("cuda:2")


def test_results_directory_lock_rejects_concurrent_trainers(tmp_path: Path):
    first = _acquire_results_lock(tmp_path)
    try:
        with pytest.raises(ResultsDirectoryBusyError, match="owner PID"):
            _acquire_results_lock(tmp_path)
    finally:
        _release_results_lock(first)

    second = _acquire_results_lock(tmp_path)
    _release_results_lock(second)


def test_unified_classification_metrics_include_calibration_and_macro_scores():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    labels = np.array([0, 1, 0, 1])

    metrics = classification_metrics(probabilities, labels, n_classes=2)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["auroc_ovr"] == 1.0
    assert metrics["nll"] > 0.0
    assert metrics["ece"] > 0.0


def test_unified_classification_metrics_reject_malformed_probabilities():
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="sum to one"):
        classification_metrics(
            np.array([[0.9, 0.2], [0.1, 0.9]]), labels, n_classes=2)
    with pytest.raises(ValueError, match="NaN or infinity"):
        classification_metrics(
            np.array([[np.nan, 0.0], [0.1, 0.9]]), labels, n_classes=2)


def test_step_output_contract_rejects_wrong_class_width_and_nonfinite_values():
    with pytest.raises(ValueError, match="expected logits"):
        _validate_step_output({
            "loss": 0.1,
            "logits": torch.ones(1, 3),
            "label": torch.tensor([0]),
        }, 2, "test")

    with pytest.raises(ValueError, match="NaN or infinity"):
        _validate_step_output({
            "loss": 0.1,
            "logits": torch.tensor([[float("nan"), 0.0]]),
            "label": torch.tensor([0]),
        }, 2, "test")

    with pytest.raises(TypeError, match="integer class indices"):
        _validate_step_output({
            "loss": 0.1,
            "logits": torch.ones(1, 2),
            "label": torch.tensor([0.0]),
        }, 2, "test")


def test_resume_identity_rejects_a_changed_configuration(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "config.json"
    cfg = {"seed": 1, "results_dir": str(tmp_path)}
    config_path.write_text(json.dumps(cfg))
    _write_metrics(metrics_path, "focus", cfg, [{"fold": 0}])

    with pytest.raises(RuntimeError, match="different method or configuration"):
        _load_resume_state(
            metrics_path, config_path, "focus", {**cfg, "seed": 2})


def test_run_identity_rejects_nonfinite_configuration_values():
    with pytest.raises(ValueError, match="cannot be fingerprinted"):
        _run_identity("focus", {"lr": float("nan")})


def test_legacy_resume_requires_matching_saved_config(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "config.json"
    cfg = {"seed": 1}
    completed = {"fold": 0, "best_val_loss": 1.0, "test_acc": 0.5}
    metrics_path.write_text(json.dumps({"method": "focus", "folds": [completed]}))
    config_path.write_text(json.dumps(cfg))

    assert _load_resume_state(
        metrics_path, config_path, "focus", cfg) == [completed]


def test_resume_accepts_new_non_execution_provenance_fields(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "config.json"
    original = {"seed": 1, "method": "focus"}
    config_path.write_text(json.dumps(original))
    completed = {"fold": 0, "best_val_loss": 1.0, "test_acc": 0.5}
    _write_metrics(metrics_path, "focus", original, [completed])
    upgraded = {
        **original,
        "implementation_provenance": "vendored",
        "upstream_fidelity": "upstream",
        "fidelity_note": "documentation only",
    }

    assert _load_resume_state(
        metrics_path, config_path, "focus", upgraded) == [completed]


def test_resume_rejects_fold_without_completed_metrics(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "config.json"
    cfg = {"seed": 1}
    config_path.write_text(json.dumps(cfg))
    metrics_path.write_text(json.dumps({
        "method": "focus", "folds": [{"fold": 0}],
    }))

    with pytest.raises(RuntimeError, match="finite best_val_loss"):
        _load_resume_state(metrics_path, config_path, "focus", cfg)


def test_resume_runs_holes_in_out_of_order_fold_state():
    assert _pending_folds([{"fold": 2}, {"fold": 0}], 0, 4) == [1, 3]


def test_resume_rejects_fold_outside_configured_range():
    with pytest.raises(RuntimeError, match="outside"):
        _pending_folds([{"fold": 4}], 0, 4)


def test_resume_rejects_boolean_fold_indices(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "config.json"
    cfg = {"seed": 1}
    config_path.write_text(json.dumps(cfg))
    metrics_path.write_text(json.dumps({
        "method": "focus",
        "folds": [{"fold": True}],
    }))

    with pytest.raises(RuntimeError, match="integer index"):
        _load_resume_state(metrics_path, config_path, "focus", cfg)


def test_resume_validation_rejects_fold_outside_configured_range(
    tmp_path: Path,
):
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "config.json"
    cfg = {"k_start": 1, "k_end": 3}
    config_path.write_text(json.dumps(cfg))
    metrics_path.write_text(json.dumps({
        "method": "focus",
        "folds": [{"fold": 0, "best_val_loss": 1.0, "test_acc": 0.5}],
    }))

    with pytest.raises(RuntimeError, match=r"outside \[1, 3\)"):
        _load_resume_state(metrics_path, config_path, "focus", cfg)


def test_resume_validation_rejects_nested_nonfinite_metrics(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "config.json"
    cfg = {"k": 1}
    config_path.write_text(json.dumps(cfg))
    metrics_path.write_text(json.dumps({
        "method": "focus",
        "folds": [{
            "fold": 0,
            "best_val_loss": 1.0,
            "test_acc": 0.5,
            "slide_metrics": {"nll": float("nan")},
        }],
    }))

    with pytest.raises(RuntimeError, match=r"non-finite metric.*slide_metrics.nll"):
        _load_resume_state(metrics_path, config_path, "focus", cfg)


def test_metrics_write_records_identity_and_replaces_atomically(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    cfg = {"seed": 7}
    _write_metrics(metrics_path, "focus", cfg, [{"fold": 0}])

    state = json.loads(metrics_path.read_text())
    assert state["run_identity"] == _run_identity("focus", cfg)
    assert state["folds"] == [{"fold": 0}]
    assert not (tmp_path / "metrics.json.tmp").exists()


def test_metrics_write_preserves_canonical_file_after_serialization_error(
    tmp_path: Path,
):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"previous": true}')

    with pytest.raises(TypeError):
        _write_metrics(
            metrics_path, "focus", {"seed": 7},
            [{"fold": 0, "bad": object()}])

    assert json.loads(metrics_path.read_text()) == {"previous": True}
    assert not (tmp_path / "metrics.json.tmp").exists()


def test_rerun_archives_resume_state_instead_of_reusing_it(tmp_path: Path):
    (tmp_path / "metrics.json").write_text('{"folds": [{"fold": 0}]}')
    (tmp_path / "config.json").write_text('{"seed": 1}')
    (tmp_path / "fold0_best.pt").write_bytes(b"checkpoint")
    (tmp_path / "fold0_predictions.csv").write_text("prediction\n0\n")

    archived = _archive_resume_state(tmp_path)

    assert len(archived) == 4
    assert all(path.is_file() for path in archived)
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "fold0_best.pt").exists()
    assert not (tmp_path / "fold0_predictions.csv").exists()


def test_evaluator_requires_verifiable_checkpoint_identity(tmp_path: Path):
    with pytest.raises(RuntimeError, match="no verifiable"):
        _validate_checkpoint_identity(
            tmp_path, "focus", {"method": "focus", "seed": 1})

    cfg = {"method": "focus", "seed": 1}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    _validate_checkpoint_identity(tmp_path, "focus", cfg)


def test_run_method_accepts_aliases_but_rejects_config_mismatch():
    cfg = {"method": "vila-mil"}
    assert _resolve_run_method("vila", cfg) == "vila_mil"
    assert cfg["method"] == "vila_mil"

    with pytest.raises(ValueError, match="config declares method"):
        _resolve_run_method("focus", {"method": "pathpt"})


def test_train_rejects_method_config_mismatch_before_creating_outputs(
    tmp_path: Path, monkeypatch, capsys,
):
    results_dir = tmp_path / "should-not-exist"
    config = tmp_path / "run.yaml"
    config.write_text(
        f"method: pathpt\nresults_dir: {results_dir}\n")
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--method", "focus", "--config", str(config)],
    )

    assert main() == EXIT_CONFIG_ERROR
    assert "CONFIG ERROR" in capsys.readouterr().err
    assert not results_dir.exists()


def test_train_reports_unusable_results_directory_as_a_skip(
    tmp_path: Path, monkeypatch, capsys,
):
    results_file = tmp_path / "results"
    results_file.write_text("not a directory\n")
    config = tmp_path / "run.yaml"
    config.write_text(f"results_dir: {results_file}\n")
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--method", "focus", "--config", str(config)],
    )

    assert main() == EXIT_SKIPPED
    output = capsys.readouterr().out
    assert "results_dir" in output
    assert "reason could not be recorded" in output
