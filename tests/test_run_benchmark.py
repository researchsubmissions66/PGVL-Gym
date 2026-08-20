import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from common.readiness import refresh_benchmark_readiness
from common.run_state import run_identity
from scripts.run_benchmark import (
    ERROR,
    Plan,
    Run,
    _count,
    _fold_progress,
    _mark_plan_collisions,
    _validate_matrix_header,
    _nonnegative_int,
    _positive_int,
    _queued_job_names,
    _report_destination_error,
    _submission_environment_error,
    _validate_ready_row,
    submit,
)


def _write_state(results: Path, cfg: dict, folds: list[dict]) -> None:
    folds = [
        {"best_val_loss": 1.0, "test_acc": 0.5, **entry}
        for entry in folds]
    (results / "config.json").write_text(json.dumps(cfg))
    (results / "metrics.json").write_text(json.dumps({
        "method": cfg["method"],
        "run_identity": run_identity(cfg["method"], cfg),
        "folds": folds,
    }))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_readiness_refresh_tracks_asynchronous_features_without_regeneration(
    tmp_path: Path, monkeypatch,
):
    storage = tmp_path / "storage"
    storage.mkdir()
    first = storage / "first.h5"
    second = storage / "second.h5"
    first.write_bytes(b"ready")
    monkeypatch.setenv("PGVL_STORAGE_ROOT", str(storage))
    benchmark = tmp_path / "benchmark"
    _write_csv(benchmark / "data" / "toy" / "manifest.csv", [{
        "slide_id": "slide-a",
        "feature__first": "${PGVL_STORAGE_ROOT}/first.h5",
        "feature__second": "${PGVL_STORAGE_ROOT}/second.h5",
    }])
    coverage_rows = [
        {
            "cohort": "toy", "feature_source": source,
            "input_kind": "patch_bag", "feature_column": f"feature__{source}",
            "available_slides": 0, "annotated_slides": 1, "coverage": 0.0,
            "metadata_ready": True,
        }
        for source in ("first", "second")
    ]
    _write_csv(benchmark / "feature_coverage.csv", coverage_rows)
    base = {
        "cohort": "toy", "missing_auxiliary_files": 0,
        "metadata_ready": True, "split_ready": True,
        "encoder_ready": True, "config_valid": True,
        "auxiliary_ready": True, "preserved": "unchanged",
    }
    _write_csv(benchmark / "run_matrix.csv", [
        {
            **base, "feature_signature": "bag=first", "ready": False,
            "missing_feature_files": 99,
        },
        {
            **base, "feature_signature": "bag=second", "ready": True,
            "missing_feature_files": 0,
        },
    ])
    unrelated = benchmark / "splits" / "do-not-rebuild.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("sentinel")

    result = refresh_benchmark_readiness(benchmark)

    assert result.changed_rows == 2
    with (benchmark / "run_matrix.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["ready"], row["missing_feature_files"]) for row in rows] == [
        ("True", "0"), ("False", "1")]
    assert all(row["preserved"] == "unchanged" for row in rows)
    assert unrelated.read_text() == "sentinel"

    second.write_bytes(b"arrived later")
    result = refresh_benchmark_readiness(benchmark)

    assert result.changed_rows == 1
    with (benchmark / "run_matrix.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["ready"] == "True"
    assert rows[1]["missing_feature_files"] == "0"


def test_fold_progress_counts_completed_folds_with_holes(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    cfg = {
        "method": "focus", "results_dir": str(results),
        "k_start": 0, "k_end": 3,
    }
    _write_state(results, cfg, [{"fold": 0}, {"fold": 2}])

    completed, total, resolved = _fold_progress(cfg)

    assert (completed, total, resolved) == (2, 3, results)


def test_fold_progress_rejects_explicit_zero_end(tmp_path: Path):
    try:
        _fold_progress({
            "results_dir": str(tmp_path / "results"),
            "k_start": 0,
            "k_end": 0,
        })
    except ValueError as error:
        assert "invalid fold range [0, 0)" in str(error)
    else:
        raise AssertionError("empty fold range was reported as complete")


def test_fold_progress_rejects_coercible_noninteger_fold_indices(
    tmp_path: Path,
):
    for value in ("2", 2.0):
        try:
            _fold_progress({
                "results_dir": str(tmp_path / "results"), "k": value,
            })
        except ValueError as error:
            assert "must be integer fold indices" in str(error)
        else:
            raise AssertionError(f"coercible fold count {value!r} was accepted")


def test_fold_progress_rejects_corrupt_or_duplicate_resume_state(
    tmp_path: Path,
):
    results = tmp_path / "results"
    results.mkdir()
    metrics = results / "metrics.json"
    cfg = {"method": "focus", "results_dir": str(results), "k": 2}

    metrics.write_text("not json")
    try:
        _fold_progress(cfg)
    except ValueError as error:
        assert "cannot read resume state" in str(error)
    else:
        raise AssertionError("corrupt resume state was accepted")

    _write_state(results, cfg, [{"fold": 0}, {"fold": 0}])
    try:
        _fold_progress(cfg)
    except ValueError as error:
        assert "duplicate fold indices" in str(error)
    else:
        raise AssertionError("duplicate resume folds were accepted")


def test_fold_progress_rejects_state_from_another_configuration(
    tmp_path: Path,
):
    results = tmp_path / "results"
    results.mkdir()
    original = {
        "method": "focus", "results_dir": str(results), "k": 1, "seed": 1,
    }
    _write_state(results, original, [{"fold": 0}])

    changed = {**original, "seed": 2}
    try:
        _fold_progress(changed)
    except ValueError as error:
        assert "different method or configuration" in str(error)
    else:
        raise AssertionError("resume state from another config was accepted")


def test_rerun_is_forwarded_to_the_training_job(tmp_path: Path):
    config = tmp_path / "run.yaml"
    config.write_text("method: focus\n")
    run = Run(
        benchmark="toy", experiment="focus", method="focus", cohort="toy",
        shots="4", config=config, ready=True,
        missing_features=0, missing_auxiliary=0)
    args = SimpleNamespace(
        partition="gpu", gpus="1", cpus="2", mem="8G", time="01:00:00",
        account=None, device="cuda:0", dry_run=False, rerun=True)
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Submitted batch job 123\n", stderr="")

    with patch("scripts.run_benchmark.subprocess.run", return_value=completed) as call:
        ok, job_id = submit(run, args, tmp_path)

    assert ok and job_id == "123"
    assert call.call_args.args[0][-1] == "--rerun"


def test_queue_failure_is_not_treated_as_an_empty_queue():
    with patch(
        "scripts.run_benchmark.subprocess.run",
        side_effect=OSError("squeue unavailable"),
    ):
        assert _queued_job_names() is None


def test_real_submission_requires_an_explicit_project_environment():
    with patch.dict(os.environ, {}, clear=True):
        error = _submission_environment_error()
    assert error is not None
    assert "PGVL_CONDA_ENV is unset" in error

    with patch.dict(os.environ, {"PGVL_CONDA_ENV": " /envs/pgvl-gym "},
                    clear=True):
        assert _submission_environment_error() is None


def test_pathpt_planning_does_not_import_torch_or_h5py():
    repository = Path(__file__).resolve().parents[1]
    script = r'''
import builtins

real_import = builtins.__import__

def reject_heavy_imports(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"torch", "h5py"}:
        raise AssertionError(f"unexpected heavyweight import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_heavy_imports
from common.preflight import preflight

report = preflight({
    "method": "pathpt",
    "training_mode": "upstream_patch_ssl",
    "task": "brca",
    "n_classes": 2,
    "classnames": ["ductal", "lobular"],
    "label_dict": {"IDC": 0, "ILC": 1},
}, check_features=False, checks=("assets",))
assert report.ok, report.problems
'''
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=repository,
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_submission_rejects_unparseable_success_output(tmp_path: Path):
    config = tmp_path / "run.yaml"
    config.write_text("method: focus\n")
    run = Run(
        benchmark="toy", experiment="focus", method="focus", cohort="toy",
        shots="4", config=config, ready=True,
        missing_features=0, missing_auxiliary=0)
    args = SimpleNamespace(
        partition="gpu", gpus="1", cpus="2", mem="8G", time="01:00:00",
        account=None, device="cuda:0", dry_run=False, rerun=False)
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="submission accepted\n", stderr="")

    with patch("scripts.run_benchmark.subprocess.run", return_value=completed):
        ok, reason = submit(run, args, tmp_path)

    assert not ok
    assert "could not be parsed" in reason


def test_nonnegative_cli_counts_reject_negative_values():
    assert _nonnegative_int("0") == 0
    try:
        _nonnegative_int("-1")
    except Exception as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative campaign limit was accepted")


def test_positive_scheduler_counts_reject_zero():
    assert _positive_int("1") == 1
    for value in ("0", "-1"):
        try:
            _positive_int(value)
        except Exception as error:
            assert "positive" in str(error) or "non-negative" in str(error)
        else:
            raise AssertionError(f"invalid positive count {value!r} was accepted")


def test_matrix_counts_reject_corrupt_or_negative_cells():
    for value in ("many", "-1"):
        try:
            _count(value, "missing_feature_files")
        except ValueError as error:
            assert "non-negative integer" in str(error)
        else:
            raise AssertionError(f"corrupt matrix count {value!r} was accepted")


def test_ready_matrix_rows_cannot_contradict_blocking_fields():
    for readiness, missing_features, missing_auxiliary in (
        ({"config_valid": False}, 0, 0),
        ({"config_valid": True}, 1, 0),
        ({"config_valid": True}, 0, 1),
    ):
        try:
            _validate_ready_row(
                True, readiness, missing_features, missing_auxiliary)
        except ValueError as error:
            assert "ready=true contradicts" in str(error)
        else:
            raise AssertionError("contradictory ready matrix row was accepted")

    _validate_ready_row(False, {"config_valid": False}, 1, 1)


def test_run_matrix_header_requires_readiness_evidence():
    try:
        _validate_matrix_header(["method", "config", "ready"])
    except ValueError as error:
        assert "missing required columns" in str(error)
        assert "missing_feature_files" in str(error)
    else:
        raise AssertionError("incomplete run matrix header was accepted")

    try:
        _validate_matrix_header(["method", "method"])
    except ValueError as error:
        assert "duplicate columns" in str(error)
    else:
        raise AssertionError("duplicate run matrix header was accepted")


def test_plan_rejects_duplicate_jobs_and_results_directories(tmp_path: Path):
    shared = tmp_path / "results"
    first = Run(
        benchmark="toy", experiment="focus", method="focus", cohort="toy",
        shots="4", config=tmp_path / "a.yaml", ready=True,
        missing_features=0, missing_auxiliary=0, results_dir=shared)
    duplicate = Run(
        benchmark="toy", experiment="focus", method="focus", cohort="toy",
        shots="4", config=tmp_path / "b.yaml", ready=True,
        missing_features=0, missing_auxiliary=0, results_dir=shared)
    plan = Plan(runs=[first, duplicate])

    _mark_plan_collisions(plan)

    assert all(run.state == ERROR for run in plan.runs)
    assert all("duplicate SLURM job name" in run.reason for run in plan.runs)
    assert all("duplicate results directory" in run.reason for run in plan.runs)


def test_campaign_report_cannot_overwrite_inputs_or_run_state(tmp_path: Path):
    config = tmp_path / "run.yaml"
    results = tmp_path / "results"
    run = Run(
        benchmark="toy", experiment="focus", method="focus", cohort="toy",
        shots="4", config=config, ready=True,
        missing_features=0, missing_auxiliary=0, results_dir=results)
    plan = Plan(runs=[run])

    assert "run config" in _report_destination_error(plan, config)
    assert "run state" in _report_destination_error(
        plan, results / "metrics.json")
    assert _report_destination_error(plan, tmp_path / "report.csv") is None
