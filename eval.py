"""Evaluate unified PGVL-Gym checkpoints on their declared holdout splits.

The evaluator intentionally reuses the trainer's method-specific loader
dispatch and metric implementation. A configuration that trains with nested
fold splits, patch bags, paired bags, reports, or slide embeddings is evaluated
with that same input contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from common.configuration import load_yaml_config
from common.preflight import preflight
from common.run_state import run_identity, validate_resume_state
from methods import get_method
from train import (
    EXIT_CONFIG_ERROR,
    EXIT_SKIPPED,
    _batch_metadata,
    _resolve_run_method,
    _validate_device,
    _validate_step_output,
    _write_json_atomic,
    build_loaders,
    classification_metrics,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every configured fold with the unified loaders")
    parser.add_argument("--method", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint", choices=("auto", "best", "final"), default="auto",
        help=("checkpoint suffix to evaluate; auto selects best when early "
              "stopping is enabled and final otherwise (default: auto)"))
    parser.add_argument(
        "--output", type=Path,
        help="optional JSON output path; stdout is always populated")
    return parser.parse_args(argv)


def _load_checkpoint(path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location=device)


def _checkpoint_kind(requested: str, cfg: dict[str, Any]) -> str:
    """Resolve the checkpoint suffix without guessing nonexistent files."""
    if requested == "auto":
        return "best" if cfg.get("early_stopping", True) else "final"
    return requested


def _validate_output_destination(
    output: Path | None, config_path: str | Path, ckpt_dir: Path,
    checkpoints: Sequence[Path],
) -> str | None:
    """Reject destinations which would overwrite evaluation inputs."""
    if output is None:
        return None
    destination = output.expanduser().resolve(strict=False)
    protected = {
        Path(config_path).expanduser().resolve(strict=False): "input config",
        (ckpt_dir / "config.json").resolve(strict=False):
            "checkpoint config snapshot",
        (ckpt_dir / "metrics.json").resolve(strict=False):
            "checkpoint metrics state",
    }
    protected.update({
        path.resolve(strict=False): f"checkpoint {path.name}"
        for path in checkpoints
    })
    if destination in protected:
        return (
            f"refusing to overwrite {protected[destination]} at "
            f"{destination}")
    if output.exists() and not output.is_file():
        return f"output exists but is not a regular file: {output}"
    return None


def _metadata_values(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata[key]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _patient_metrics(
    metadata: list[dict[str, str]], probabilities: np.ndarray,
    labels: np.ndarray, n_classes: int,
) -> dict[str, Any] | None:
    if len(metadata) != len(labels):
        if metadata:
            raise ValueError(
                "prediction metadata is misaligned: "
                f"{len(metadata)} rows for {len(labels)} predictions")
        return None
    frame = pd.DataFrame(metadata)
    frame["label"] = labels
    probability_columns = []
    for index in range(n_classes):
        column = f"probability_{index}"
        probability_columns.append(column)
        frame[column] = probabilities[:, index]
    inconsistent = frame.groupby("case_id")["label"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if not inconsistent.empty:
        raise ValueError(
            "patient-level aggregation found conflicting slide labels for "
            f"case IDs: {', '.join(map(str, inconsistent.index[:5]))}")
    patients = frame.groupby("case_id", sort=True).agg({
        **{column: "mean" for column in probability_columns},
        "label": "first",
    })
    return classification_metrics(
        patients[probability_columns].to_numpy(),
        patients["label"].to_numpy(dtype=int), n_classes)


def _validate_checkpoint_identity(
    ckpt_dir: Path, method_name: str, cfg: dict[str, Any],
) -> None:
    metrics_path = ckpt_dir / "metrics.json"
    if not metrics_path.is_file():
        try:
            validate_resume_state(
                {"method": method_name, "folds": []},
                ckpt_dir / "config.json", method_name, cfg)
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "checkpoint directory has no verifiable metrics/config "
                f"identity: {error}") from error
        return
    try:
        with metrics_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        validate_resume_state(
            state, ckpt_dir / "config.json", method_name, cfg)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"checkpoint directory does not belong to this run: {error}") \
            from error


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_yaml_config(args.config)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    args.ckpt_dir = args.ckpt_dir.expanduser()
    try:
        method_name = _resolve_run_method(args.method, cfg)
    except (KeyError, ValueError) as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # Evaluation reads checkpoints from ``--ckpt_dir`` and does not write to
    # the training config's results_dir. A copied/read-only results directory
    # must therefore not make an otherwise valid evaluation skip.
    preflight_cfg = dict(cfg)
    preflight_cfg.pop("results_dir", None)
    report = preflight(preflight_cfg)
    for warning in report.warnings:
        print(f"  ! {warning}", file=sys.stderr)
    if not report.ok:
        print("Evaluation preflight failed:", file=sys.stderr)
        for problem in report.problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_SKIPPED

    try:
        args.device = _validate_device(args.device)
    except ValueError as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        _validate_checkpoint_identity(args.ckpt_dir, method_name, cfg)
    except RuntimeError as error:
        print(f"CHECKPOINT ERROR: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    k_start = int(cfg.get("k_start", 0))
    k_end = int(cfg.get("k_end", cfg.get("k", 5)))
    checkpoint_kind = _checkpoint_kind(args.checkpoint, cfg)
    checkpoints = {
        fold: args.ckpt_dir / f"fold{fold}_{checkpoint_kind}.pt"
        for fold in range(k_start, k_end)
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        print("Missing configured checkpoints:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if args.output is not None:
        args.output = args.output.expanduser()
    output_error = _validate_output_destination(
        args.output, args.config, args.ckpt_dir, list(checkpoints.values()))
    if output_error:
        print(f"OUTPUT ERROR: {output_error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    fold_results: list[dict[str, Any]] = []
    method_type = get_method(method_name)
    for fold, checkpoint_path in checkpoints.items():
        method = method_type(cfg, device=args.device)
        _, _, test_loader = build_loaders(method_name, cfg, fold)
        model = method.build_model()
        model.load_state_dict(
            _load_checkpoint(checkpoint_path, args.device), strict=True)
        method.on_checkpoint_loaded(model, checkpoint_kind, fold)
        model.eval()

        fold_logits: list[torch.Tensor] = []
        fold_labels: list[torch.Tensor] = []
        metadata_rows: list[dict[str, str]] = []
        with torch.no_grad():
            for batch in test_loader:
                output = method.eval_step(batch, model)
                logits, labels, _loss = _validate_step_output(
                    output, cfg["n_classes"], f"fold{fold} evaluation")
                logits = logits.detach().float().cpu()
                labels = labels.detach().long().cpu()
                fold_logits.append(logits)
                fold_labels.append(labels)

                metadata = _batch_metadata(batch)
                if metadata is not None:
                    slide_ids = _metadata_values(metadata, "slide_id")
                    case_ids = _metadata_values(metadata, "case_id")
                    if len(slide_ids) != len(labels) or len(case_ids) != len(labels):
                        raise ValueError(
                            f"fold{fold} metadata does not align with its labels")
                    metadata_rows.extend(
                        {"slide_id": slide_id, "case_id": case_id}
                        for slide_id, case_id in zip(slide_ids, case_ids))

        if not fold_logits:
            raise RuntimeError(f"fold{fold} test loader produced no predictions")
        logits = torch.cat(fold_logits)
        labels = torch.cat(fold_labels).numpy()
        probabilities = torch.softmax(logits, dim=1).numpy()
        slide_metrics = classification_metrics(
            probabilities, labels, cfg["n_classes"])
        if ((metadata_rows or cfg.get("include_metadata", False))
                and len(metadata_rows) != len(labels)):
            raise ValueError(
                f"fold{fold} metadata is missing or misaligned: "
                f"{len(metadata_rows)} rows for {len(labels)} predictions")
        patient_metrics = _patient_metrics(
            metadata_rows, probabilities, labels, cfg["n_classes"])
        result = {
            "fold": fold,
            "slides_evaluated": int(len(labels)),
            "slide_metrics": slide_metrics,
            "patient_metrics": patient_metrics,
        }
        fold_results.append(result)
        print(
            f"fold {fold}: accuracy={slide_metrics['accuracy']:.4f} "
            f"balanced_accuracy={slide_metrics['balanced_accuracy']:.4f} "
            f"n={len(labels)}")

    accuracies = [item["slide_metrics"]["accuracy"] for item in fold_results]
    payload = {
        "method": method_name,
        "run_identity": run_identity(method_name, cfg),
        "checkpoint": checkpoint_kind,
        "folds": fold_results,
        "mean_slide_accuracy": float(np.mean(accuracies)),
        "std_slide_accuracy": float(np.std(accuracies)),
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(args.output, payload)
        except OSError as error:
            print(
                f"OUTPUT ERROR: cannot write {args.output}: {error}",
                file=sys.stderr)
            return EXIT_CONFIG_ERROR
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
