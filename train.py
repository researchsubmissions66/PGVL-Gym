"""Unified training entry point.

Usage
-----
    python train.py --method focus     --config configs/focus_ubc.yaml
    python train.py --method pathpt    --config configs/pathpt_ubc_keep.yaml
    python train.py --method vila_mil  --config configs/vila_mil_lung.yaml

This script:
  1. Loads the YAML config.
  2. Looks up the method adapter and the right dataloader factory.
  3. Iterates over `k` cross-validation folds (CLAM convention).
  4. Builds dataset / model / optimizer / scheduler via the adapter.
  5. Runs the standard CLAM-style train + validate + test loop.
  6. Logs scalars to TensorBoard and writes per-fold metrics.
"""
from __future__ import annotations
import argparse
from collections.abc import Mapping
import fcntl
import os
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:                                     # noqa: D401
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def flush(self): pass
        def close(self): pass

from methods import canonical_method_name, get_method
from common.preflight import preflight
from common.configuration import load_yaml_config
from common.method_provenance import method_provenance
from common.run_state import run_identity as _run_identity, validate_resume_state
from common.utils.core_utils import (
    Accuracy_Logger,
    EarlyStopping,
    save_torch_state_atomic,
)

# Exit code meaning "this configuration cannot produce a result and was skipped".
# Distinct from 0 (completed) and from any real failure, so a campaign launcher
# can tell a skipped configuration apart from a broken one.
EXIT_SKIPPED = 3
EXIT_CONFIG_ERROR = 2

# Benchmark results must describe the complete declared split. Exploratory runs
# may explicitly opt into a non-zero tolerance, which is then recorded together
# with the affected slide IDs.
MAX_BATCH_FAILURE_RATE = 0.0


class ResultsDirectoryBusyError(RuntimeError):
    """Raised when another trainer owns a results directory."""


def _acquire_results_lock(results_dir: Path):
    """Take a non-blocking process lock for one experiment directory."""
    lock_path = results_dir / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        detail = f" (owner PID {owner})" if owner.isdigit() else ""
        raise ResultsDirectoryBusyError(
            f"another training process is using {results_dir}{detail}") from error
    except Exception:
        handle.close()
        raise

    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    except Exception:
        _release_results_lock(handle)
        raise
    return handle


def _release_results_lock(handle) -> None:
    """Release a lock returned by :func:`_acquire_results_lock`."""
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _load_resume_state(metrics_path: Path, config_path: Path,
                       method_name: str, cfg: Dict[str, Any]) -> list[dict]:
    """Load completed folds only when they belong to this exact experiment.

    Older metrics files have no fingerprint. They can be migrated safely when
    their saved config snapshot and method match the requested run.
    """
    with open(metrics_path, "r") as handle:
        state = json.load(handle)
    return validate_resume_state(state, config_path, method_name, cfg)


def _pending_folds(fold_metrics: list[dict], k_start: int,
                   k_end: int) -> list[int]:
    """Return every configured fold not already present in resume state."""
    completed = {entry["fold"] for entry in fold_metrics}
    unexpected = sorted(
        fold for fold in completed if fold < k_start or fold >= k_end)
    if unexpected:
        raise RuntimeError(
            f"resume state contains folds outside [{k_start}, {k_end}): "
            f"{unexpected}")
    return [fold for fold in range(k_start, k_end) if fold not in completed]


def _write_metrics(metrics_path: Path, method_name: str,
                   cfg: Dict[str, Any], folds: list[dict]) -> None:
    """Atomically persist fold metrics and their experiment identity."""
    payload = {
        "method": method_name,
        "run_identity": _run_identity(method_name, cfg),
        "folds": folds,
    }
    temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    try:
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metrics_path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Persist JSON without exposing a truncated file to a resumed job."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_dataframe_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Persist a CSV as one replacement after every row is serialized."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _archive_resume_state(results_dir: Path) -> list[Path]:
    """Move every prior fold artifact aside before a from-scratch rerun."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived: list[Path] = []
    sources = [results_dir / name for name in ("metrics.json", "config.json")]
    for pattern in (
            "fold*_best.pt", "fold*_final.pt", "fold*_predictions.csv",
            "fold*_pathpt_prompt_selection.json",
            "fold*_wsi_five_answer_bank.json"):
        sources.extend(sorted(results_dir.glob(pattern)))
    sources.append(results_dir / "tensorboard")
    for source in dict.fromkeys(sources):
        if not source.exists():
            continue
        destination = results_dir / f"{source.name}.pre-rerun-{timestamp}"
        suffix = 1
        while destination.exists():
            destination = results_dir / (
                f"{source.name}.pre-rerun-{timestamp}-{suffix}")
            suffix += 1
        source.replace(destination)
        archived.append(destination)
    return archived


def _resolve_run_method(requested: str, cfg: Dict[str, Any]) -> str:
    """Validate CLI/config agreement and return one canonical method name."""
    requested_name = canonical_method_name(requested)
    configured = cfg.get("method")
    if configured is not None:
        configured_name = canonical_method_name(configured)
        if configured_name != requested_name:
            raise ValueError(
                f"CLI --method {requested!r} resolves to {requested_name!r}, "
                f"but the config declares method {configured!r} "
                f"({configured_name!r})")
    cfg["method"] = requested_name
    provenance = method_provenance(requested_name, cfg)
    cfg.setdefault("implementation_provenance", provenance.implementation)
    cfg.setdefault("upstream_fidelity", provenance.upstream_fidelity)
    cfg.setdefault("fidelity_note", provenance.note)
    return requested_name


def _check_sample_failures(phase: str, fold: int, succeeded: int, failed: int,
                          last_error: BaseException | None,
                          threshold: float) -> None:
    """Fail loudly when too many declared samples never reached the model.

    Raises:
        RuntimeError: If no batch succeeded, or the failed fraction exceeds
            ``threshold``.
    """
    total = succeeded + failed
    if total == 0:
        raise RuntimeError(f"fold{fold} {phase}: the loader produced no batches")
    if succeeded == 0:
        raise RuntimeError(
            f"fold{fold} {phase}: every batch failed. Check the method/backbone "
            f"feature contract. Last error: {last_error}")
    rate = failed / total
    if rate > threshold:
        raise RuntimeError(
            f"fold{fold} {phase}: {failed} of {total} samples failed "
            f"({rate:.0%} > {threshold:.0%} allowed), so the result would not be "
            f"computed on the declared split. Last error: {last_error}")
    if failed:
        print(f"  ! {phase}: {failed} of {total} samples failed ({rate:.1%})")


# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified WSI/VLM training")
    p.add_argument("--method", required=True,
                   help="One of: composite, focus, vila_mil, cod_mil, maple, mscpt, "
                        "pathpt, top, slip, wsi_five, muse, convlm, sldpc")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--rerun", action="store_true",
        help="start from fold 0 and archive existing metrics/config state")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_device(device: str) -> str:
    """Validate the CPU/CUDA target before constructing loaders or models."""
    try:
        parsed = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"invalid --device {device!r}: {error}") from error
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"unsupported --device {device!r}; PGVL-Gym supports cpu and cuda")
    if parsed.type == "cpu":
        if parsed.index not in {None, 0}:
            raise ValueError(f"invalid CPU device index in {device!r}")
        return str(parsed)
    if not torch.cuda.is_available():
        raise ValueError(
            f"--device {device!r} requests CUDA, but CUDA is not available")
    count = torch.cuda.device_count()
    if parsed.index is not None and parsed.index >= count:
        raise ValueError(
            f"--device {device!r} selects GPU {parsed.index}, but only "
            f"{count} CUDA device(s) are visible")
    return str(parsed)


def _expected_calibration_error(
        probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            accuracy = (predictions[mask] == labels[mask]).mean()
            error += mask.mean() * abs(accuracy - confidence[mask].mean())
    return float(error)


def classification_metrics(
        probabilities: np.ndarray, labels: np.ndarray,
        n_classes: int) -> Dict[str, Any]:
    """Return the same slide/patient metrics for every benchmark method."""
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) \
            or n_classes < 2:
        raise ValueError("n_classes must be an integer >= 2")
    if probabilities.ndim != 2 or probabilities.shape[1] != n_classes:
        raise ValueError(
            "probabilities must have shape [samples, n_classes], got "
            f"{probabilities.shape}")
    if labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError(
            "labels must be a one-dimensional array aligned with probabilities")
    if len(labels) == 0:
        raise ValueError("classification metrics require at least one sample")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must contain integer class indices")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError(f"labels must be in [0, {n_classes})")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("each probability row must sum to one")
    predictions = probabilities.argmax(axis=1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels, predictions, labels=list(range(n_classes)),
            average="macro", zero_division=0)),
        "nll": float(-np.log(clipped[np.arange(len(labels)), labels]).mean()),
        "ece": _expected_calibration_error(probabilities, labels),
    }
    recalls = recall_score(
        labels, predictions, labels=list(range(n_classes)),
        average=None, zero_division=0)
    metrics["per_class_recall"] = {
        str(index): float(value) for index, value in enumerate(recalls)
    }
    try:
        if n_classes == 2:
            metrics["auroc_ovr"] = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            metrics["auroc_ovr"] = float(roc_auc_score(
                labels, probabilities, labels=list(range(n_classes)),
                multi_class="ovr", average="macro"))
    except ValueError:
        metrics["auroc_ovr"] = None
    return metrics


def _metadata_strings(value: Any, field: str) -> list[str]:
    """Return non-empty UTF-8 identity values from a collated batch field."""
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    result = []
    for item in values:
        if isinstance(item, (bytes, bytearray, memoryview)):
            text = bytes(item).decode("utf-8")
        else:
            text = str(item)
        text = text.strip()
        if not text:
            raise ValueError(f"prediction metadata contains a blank {field}")
        result.append(text)
    return result


def _batch_metadata(batch: Any) -> Dict[str, Any] | None:
    metadata = batch if isinstance(batch, dict) else None
    if metadata is None and isinstance(batch, (list, tuple)):
        metadata = next((
            item for item in batch[1:-1]
            if isinstance(item, dict) and "slide_id" in item
        ), None)
    if metadata is None or "slide_id" not in metadata:
        return None
    slide_ids = _metadata_strings(metadata["slide_id"], "slide_id")
    case_ids = metadata.get("case_id")
    if case_ids is None:
        case_ids = [slide_id[:12] for slide_id in slide_ids]
    else:
        case_ids = _metadata_strings(case_ids, "case_id")
    return {"slide_id": slide_ids, "case_id": case_ids}


def _batch_slide_ids(batch: Any) -> list[str]:
    """Return stable sample identifiers for failure diagnostics."""
    try:
        metadata = _batch_metadata(batch)
    except Exception as error:
        return [f"<invalid metadata: {error}>"]
    if metadata is None:
        return ["<metadata unavailable>"]
    values = metadata.get("slide_id", [])
    if isinstance(values, str):
        return [values]
    try:
        return [str(value) for value in values]
    except TypeError:
        return [str(values)]


def _batch_size(batch: Any) -> int:
    """Return the number of labelled samples represented by a batch."""
    try:
        label = batch.get("label") if isinstance(batch, dict) else batch[-1]
    except (AttributeError, IndexError, KeyError, TypeError):
        return 1
    if torch.is_tensor(label):
        return int(label.numel())
    try:
        return len(label)
    except TypeError:
        return 1


def _validate_step_output(
    output: Any, n_classes: int, context: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Enforce the adapter output contract before logging or reporting it."""
    if not isinstance(output, Mapping):
        raise TypeError(f"{context}: adapter output must be a mapping")
    missing = [key for key in ("loss", "logits", "label") if key not in output]
    if missing:
        raise KeyError(
            f"{context}: adapter output is missing {', '.join(missing)}")
    logits = output["logits"]
    labels = output["label"]
    if not torch.is_tensor(logits) or not torch.is_tensor(labels):
        raise TypeError(f"{context}: logits and label must be tensors")
    labels = labels.reshape(-1)
    if logits.ndim == 1 and labels.numel() == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2 or logits.shape != (labels.numel(), n_classes):
        raise ValueError(
            f"{context}: expected logits [{labels.numel()}, {n_classes}], "
            f"got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError(f"{context}: logits contain NaN or infinity")
    if labels.numel() == 0:
        raise ValueError(f"{context}: output contains no labels")
    if (labels.dtype == torch.bool or torch.is_floating_point(labels)
            or torch.is_complex(labels)):
        raise TypeError(f"{context}: labels must be integer class indices")
    if ((labels < 0) | (labels >= n_classes)).any():
        raise ValueError(
            f"{context}: labels must be in [0, {n_classes})")
    loss = output["loss"]
    if torch.is_tensor(loss):
        if loss.numel() != 1:
            raise ValueError(f"{context}: loss must be scalar")
        loss_value = float(loss.detach())
    else:
        try:
            loss_value = float(loss)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{context}: loss must be numeric") from error
    if not np.isfinite(loss_value):
        raise ValueError(f"{context}: loss is NaN or infinity")
    return logits, labels.long(), loss_value


# -----------------------------------------------------------------------------
def build_loaders(method_name: str, cfg: Dict[str, Any], fold: int) -> Tuple:
    """Pick the right dataloader factory for the given method."""
    method_name = method_name.lower()

    # One-vector-per-slide consumers share an exact-key/exact-width loader.
    # This dispatch is capability-based so new slide-embedding methods do not
    # need another method-name branch.
    from common.backbones import FeatureLevel
    from methods import get_method
    feature_level = get_method(method_name).get_backbone_contract().feature_level
    if feature_level is FeatureLevel.SLIDE_EMBEDDING:
        from common.datasets.slide_embeddings import (
            build_slide_embedding_loader,
        )
        return (
            build_slide_embedding_loader(cfg, "train", fold),
            build_slide_embedding_loader(
                cfg, "val", fold, shuffle=False),
            build_slide_embedding_loader(
                cfg, "test", fold, shuffle=False),
        )

    if method_name == "cod_mil":
        from methods.cod_mil.dataset import build_cod_mil_loader
        return (build_cod_mil_loader(cfg, "train", fold),
                build_cod_mil_loader(cfg, "val", fold, shuffle=False),
                build_cod_mil_loader(cfg, "test", fold, shuffle=False))

    if method_name == "focus":
        nested = os.path.join(cfg["split_dir"], f"fold{fold}")
        if os.path.isdir(nested):
            from common.datasets.bag_features import build_bag_loader
            high_column = (
                cfg.get("feature_path_column")
                or cfg.get("feature_path_column_l"))
            fold_cfg = {
                **cfg,
                "split_dir": nested,
                "_fold_index": fold,
                "feature_path_column": high_column,
                "data_folder_s": (
                    cfg.get("data_folder_l") or cfg.get("data_folder_s")),
            }
            return (
                build_bag_loader(fold_cfg, "train"),
                build_bag_loader(fold_cfg, "val", shuffle=False),
                build_bag_loader(fold_cfg, "test", shuffle=False),
            )

    if method_name in {"focus", "vila_mil", "maple"}:
        # Generated protocols use foldN/{train,val,test}.csv files containing
        # exact feature paths. Reuse the method-agnostic paired-bag loader
        # instead of the legacy CLAM ``splits_N.csv`` representation.
        nested = os.path.join(cfg["split_dir"], f"fold{fold}")
        if os.path.isdir(nested):
            from methods.mscpt.dataset import build_mscpt_loader
            fold_cfg = {
                **cfg,
                "split_dir": nested,
                "_fold_index": fold,
                "feat_data_dir": cfg.get("data_folder_s", ""),
                "selected_5x_dir": cfg.get("data_folder_l"),
            }
            return (
                build_mscpt_loader(fold_cfg, "train"),
                build_mscpt_loader(fold_cfg, "val", shuffle=False),
                build_mscpt_loader(fold_cfg, "test", shuffle=False),
            )

        from common.datasets.dataset_generic import Generic_MIL_Dataset
        from common.utils.utils import get_split_loader
        mode = cfg.get("loader_mode", "transformer")
        ds = Generic_MIL_Dataset(
            csv_path=cfg["dataset_csv"],
            data_dir_s=cfg.get("data_folder_s"),
            data_dir_l=cfg.get("data_folder_l"),
            feature_path_column_s=cfg.get("feature_path_column_s"),
            feature_path_column_l=cfg.get("feature_path_column_l"),
            feature_key=cfg.get("feature_key", "features"),
            include_metadata=cfg.get("include_metadata", False),
            mode=mode, shuffle=False, seed=cfg.get("seed", 1),
            print_info=True, label_dict=cfg["label_dict"],
            patient_strat=False, ignore=[],
            label_col=cfg.get("label_column"))
        train, val, test = ds.return_splits(
            from_id=False,
            csv_path=os.path.join(cfg["split_dir"], f"splits_{fold}.csv"))
        return (
            get_split_loader(
                train, training=True, weighted=False, mode=mode),
            get_split_loader(val, mode=mode),
            get_split_loader(test, mode=mode),
        )

    if method_name == "pathpt":
        from methods.pathpt.dataset import build_pathpt_loader
        fold_cfg = {**cfg, "_fold_index": fold}
        return (build_pathpt_loader(fold_cfg, "train"),
                build_pathpt_loader(fold_cfg, "val", shuffle=False),
                build_pathpt_loader(fold_cfg, "test", shuffle=False))

    if method_name == "mscpt":
        from methods.mscpt.dataset import build_mscpt_loader
        fold_cfg = {**cfg, "_fold_index": fold}
        return (build_mscpt_loader(fold_cfg, "train"),
                build_mscpt_loader(fold_cfg, "val", shuffle=False),
                build_mscpt_loader(fold_cfg, "test", shuffle=False))

    if method_name == "wsi_five":
        from methods.wsi_five.dataset import build_wsi_five_loader
        fold_cfg = {**cfg, "_fold_index": fold}
        return (build_wsi_five_loader(fold_cfg, "train"),
                build_wsi_five_loader(fold_cfg, "val", shuffle=False),
                build_wsi_five_loader(fold_cfg, "test", shuffle=False))

    # ConVLM joined this group once it was corrected to consume precomputed
    # patch features: upstream extracts them with UNI rather than training a
    # vision tower, so it reads a bag like the others.
    if method_name in {"top", "slip", "composite", "convlm"}:
        from common.datasets.bag_features import build_bag_loader
        fold_cfg = {**cfg, "_fold_index": fold}
        return (build_bag_loader(fold_cfg, "train"),
                build_bag_loader(fold_cfg, "val", shuffle=False),
                build_bag_loader(fold_cfg, "test", shuffle=False))

    if method_name == "muse":
        # MUSE aggregates one CONCH feature bag per slide.
        from common.datasets.bag_features import build_bag_loader
        fold_cfg = {**cfg, "_fold_index": fold, "batch_size": 1}
        return (build_bag_loader(fold_cfg, "train"),
                build_bag_loader(fold_cfg, "val", shuffle=False),
                build_bag_loader(fold_cfg, "test", shuffle=False))

    raise KeyError(f"No dataloader recipe registered for method '{method_name}'")


# -----------------------------------------------------------------------------
def train_one_fold(fold: int, cfg, method, writer):
    print(f"\n--- Fold {fold} ----------------------------------------------")
    set_seed(cfg.get("seed", 1) + fold)

    failure_threshold = float(
        cfg.get("max_batch_failure_rate", MAX_BATCH_FAILURE_RATE))
    # Worst epoch per phase, so a shortfall is recorded in metrics.json rather
    # than only printed into a log nobody reads.
    failure_counts = {"train": 0, "val": 0, "test": 0}
    failure_samples: dict[str, set[str]] = {
        "train": set(), "val": set(), "test": set()}

    train_loader, val_loader, test_loader = build_loaders(method.name, cfg, fold)

    model = method.build_model()
    # Some paper-native objectives derive frozen supervision state from the
    # training fold (for example PathPT's WSI prompt selector). This hook runs
    # after both model and loaders exist but before optimizer construction or
    # any validation/test access, preventing prompt-selection leakage.
    method.prepare_fold(fold, model, train_loader)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Model has no trainable parameters")
    print(
        "  trainable parameters: "
        f"{sum(count for _, count in trainable):,} "
        f"across {[name for name, _ in trainable]}")
    optimizer = method.build_optimizer(model)
    scheduler = method.build_scheduler(optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()

    early = EarlyStopping(patience=cfg.get("es_patience", 20),
                          stop_epoch=cfg.get("es_stop_epoch", 50),
                          verbose=True) \
            if cfg.get("early_stopping", True) else None

    best_val_loss = float("inf")
    if cfg.get("evaluate_initial", False):
        model.eval()
        initial_val_loss = 0.0
        initial_samples, initial_failed_samples = 0, 0
        last_initial_error = None
        for batch in val_loader:
            try:
                out = method.eval_step(batch, model, loss_fn)
                _logits, labels, loss_value = _validate_step_output(
                    out, cfg["n_classes"],
                    f"fold{fold} initial validation")
            except Exception as error:
                if failure_threshold == 0:
                    raise RuntimeError(
                        "initial validation failed for "
                        f"{', '.join(_batch_slide_ids(batch))}: {error}") from error
                last_initial_error = error
                initial_failed_samples += _batch_size(batch)
                failed_ids = _batch_slide_ids(batch)
                failure_samples["val"].update(failed_ids)
                print(
                    "  ! initial eval_step failed for "
                    f"{', '.join(failed_ids)}: {error}")
                continue
            sample_count = int(labels.numel())
            initial_val_loss += loss_value * sample_count
            initial_samples += sample_count
        _check_sample_failures(
            "initial validation", fold, initial_samples,
            initial_failed_samples, last_initial_error, failure_threshold)
        failure_counts["val"] = max(
            failure_counts["val"], initial_failed_samples)
        initial_val_loss /= initial_samples
        best_val_loss = initial_val_loss
        writer.add_scalar(f"fold{fold}/val_loss", initial_val_loss, -1)
        print(f"  initial validation loss: {initial_val_loss:.4f}")
        method.on_validation_end(
            -1, {"val_loss": initial_val_loss})
        if early is not None:
            ckpt = Path(cfg["results_dir"]) / f"fold{fold}_best.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            early(-1, initial_val_loss, model, ckpt_name=str(ckpt))

    for epoch in range(cfg.get("epochs", 200)):
        method.on_train_epoch_start(epoch, model)
        model.train()
        run_loss = 0.0
        n_successful_samples, n_failed_samples = 0, 0
        last_train_error = None
        train_log = Accuracy_Logger(n_classes=cfg["n_classes"])
        for batch in train_loader:
            try:
                out = method.train_step(batch, model, optimizer, loss_fn)
                logits, labels, loss_value = _validate_step_output(
                    out, cfg["n_classes"], f"fold{fold} train")
            except Exception as e:
                # Adapters own optimizer.step(), so after an exception the
                # parameter/optimizer state cannot be proven unchanged. Never
                # continue training from that ambiguous state, even when an
                # exploratory evaluation failure tolerance is configured.
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"fold{fold} train failed for "
                    f"{', '.join(_batch_slide_ids(batch))}: {e}") from e
            sample_count = int(labels.numel())
            run_loss += loss_value * sample_count
            n_successful_samples += sample_count
            preds = logits.argmax(dim=1)
            train_log.log_batch(preds.cpu().numpy(),
                                labels.cpu().numpy())
        _check_sample_failures("train", fold, n_successful_samples, n_failed_samples,
                              last_train_error, failure_threshold)
        failure_counts["train"] = max(
            failure_counts["train"], n_failed_samples)
        train_loss = run_loss / n_successful_samples
        writer.add_scalar(f"fold{fold}/train_loss", train_loss, epoch)

        model.eval()
        val_loss = 0.0
        n_v_samples, n_v_failed_samples = 0, 0
        last_val_error = None
        val_log = Accuracy_Logger(n_classes=cfg["n_classes"])
        for batch in val_loader:
            try:
                out = method.eval_step(batch, model, loss_fn)
                logits, labels, loss_value = _validate_step_output(
                    out, cfg["n_classes"], f"fold{fold} validation")
            except Exception as e:
                if failure_threshold == 0:
                    raise RuntimeError(
                        f"fold{fold} validation failed for "
                        f"{', '.join(_batch_slide_ids(batch))}: {e}") from e
                last_val_error = e
                n_v_failed_samples += _batch_size(batch)
                failed_ids = _batch_slide_ids(batch)
                failure_samples["val"].update(failed_ids)
                print(
                    "  ! eval_step failed for "
                    f"{', '.join(failed_ids)}: {e}")
                continue
            sample_count = int(labels.numel())
            val_loss += loss_value * sample_count
            n_v_samples += sample_count
            preds = logits.argmax(dim=1)
            val_log.log_batch(preds.cpu().numpy(),
                              labels.cpu().numpy())
        _check_sample_failures("val", fold, n_v_samples, n_v_failed_samples,
                              last_val_error, failure_threshold)
        failure_counts["val"] = max(
            failure_counts["val"], n_v_failed_samples)
        val_loss /= n_v_samples
        writer.add_scalar(f"fold{fold}/val_loss", val_loss, epoch)

        if scheduler is not None:
            if isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        method.on_validation_end(
            epoch, {"train_loss": train_loss, "val_loss": val_loss})
        print(f"  epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")
        best_val_loss = min(best_val_loss, val_loss)

        if early is not None:
            ckpt = Path(cfg["results_dir"]) / f"fold{fold}_best.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            early(epoch, val_loss, model, ckpt_name=str(ckpt))
            if early.early_stop:
                print(f"  >>> early stopping at epoch {epoch}")
                break

    # Unconditionally save the final epoch model
    final_ckpt = Path(cfg["results_dir"]) / f"fold{fold}_final.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_torch_state_atomic(model.state_dict(), final_ckpt)

    # ---- test ------------------------------------------------------
    if not cfg.get("evaluate_test", True):
        print(
            f"  >>> fold{fold} holdout test evaluation skipped "
            "(evaluate_test=false)")
        return {"test_acc": None, "best_val_loss": best_val_loss,
                "sample_failures": dict(failure_counts),
                "failed_slide_ids": {
                    phase: sorted(values)
                    for phase, values in failure_samples.items()}}

    if early is not None:
        ckpt = Path(cfg["results_dir"]) / f"fold{fold}_best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"best checkpoint was not created for fold {fold}: {ckpt}")
        try:
            checkpoint = torch.load(
                ckpt, map_location=method.device, weights_only=True)
        except TypeError:  # torch < 2.0
            checkpoint = torch.load(ckpt, map_location=method.device)
        model.load_state_dict(checkpoint, strict=True)
        method.on_checkpoint_loaded(model, "best", fold)
    model.eval()
    test_log = Accuracy_Logger(n_classes=cfg["n_classes"])
    n_test_samples = 0
    n_test_failed_samples = 0
    last_test_error = None
    test_logits = []
    test_labels = []
    prediction_metadata: list[dict[str, str]] = []
    for batch in test_loader:
        try:
            out = method.eval_step(batch, model)
            logits, labels, _loss_value = _validate_step_output(
                out, cfg["n_classes"], f"fold{fold} test")
            logits = logits.detach().float().cpu()
            labels = labels.detach().long().cpu()
            metadata = _batch_metadata(batch)
            batch_metadata: list[dict[str, str]] = []
            if metadata is not None:
                if (len(metadata["slide_id"]) != len(labels)
                        or len(metadata["case_id"]) != len(labels)):
                    raise ValueError(
                        "prediction metadata does not align with labels")
                for index in range(len(labels)):
                    batch_metadata.append({
                        "slide_id": str(metadata["slide_id"][index]),
                        "case_id": str(metadata["case_id"][index]),
                    })
            preds = logits.argmax(dim=1)
            test_log.log_batch(preds.numpy(), labels.numpy())
            test_logits.append(logits)
            test_labels.append(labels)
            prediction_metadata.extend(batch_metadata)
            n_test_samples += len(labels)
        except Exception as e:
            if failure_threshold == 0:
                raise RuntimeError(
                    f"fold{fold} test failed for "
                    f"{', '.join(_batch_slide_ids(batch))}: {e}") from e
            last_test_error = e
            n_test_failed_samples += _batch_size(batch)
            failed_ids = _batch_slide_ids(batch)
            failure_samples["test"].update(failed_ids)
            print(
                "  ! test_step failed for "
                f"{', '.join(failed_ids)}: {e}")
    # The held-out split is the number that gets reported, so it is held to the
    # same standard as training rather than merely being non-empty.
    _check_sample_failures("test", fold, n_test_samples, n_test_failed_samples,
                          last_test_error, failure_threshold)
    failure_counts["test"] = n_test_failed_samples
    logits = torch.cat(test_logits).numpy()
    labels = torch.cat(test_labels).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    slide_metrics = classification_metrics(
        probabilities, labels, cfg["n_classes"])
    test_acc = slide_metrics["accuracy"]
    assert test_acc is not None
    print(f"  >>> fold{fold} test acc: {test_acc:.4f}")

    patient_metrics = None
    if ((prediction_metadata or cfg.get("include_metadata", False))
            and len(prediction_metadata) != len(labels)):
        raise ValueError(
            "prediction metadata is missing or misaligned: "
            f"{len(prediction_metadata)} rows for {len(labels)} predictions")
    if len(prediction_metadata) == len(labels):
        predictions = pd.DataFrame(prediction_metadata)
        predictions["label"] = labels
        predictions["prediction"] = probabilities.argmax(axis=1)
        for class_index in range(cfg["n_classes"]):
            predictions[f"probability_{class_index}"] = probabilities[:, class_index]
        probability_columns = [
            f"probability_{index}" for index in range(cfg["n_classes"])
        ]
        inconsistent = predictions.groupby("case_id")["label"].nunique()
        inconsistent = inconsistent[inconsistent > 1]
        if not inconsistent.empty:
            raise ValueError(
                "patient-level aggregation found conflicting slide labels for "
                f"case IDs: {', '.join(map(str, inconsistent.index[:5]))}")
        patient_frame = predictions.groupby("case_id", sort=True).agg(
            {**{column: "mean" for column in probability_columns}, "label": "first"}
        )
        patient_metrics = classification_metrics(
            patient_frame[probability_columns].to_numpy(),
            patient_frame["label"].to_numpy(dtype=int), cfg["n_classes"])
        prediction_path = (
            Path(cfg["results_dir"]) / f"fold{fold}_predictions.csv")
        _write_dataframe_atomic(prediction_path, predictions)

    return {
        "test_acc": test_acc,
        "best_val_loss": best_val_loss,
        "slide_metrics": slide_metrics,
        "patient_metrics": patient_metrics,
        "sample_failures": dict(failure_counts),
        "failed_slide_ids": {
            phase: sorted(values)
            for phase, values in failure_samples.items()},
    }


def _run_ready_experiment(args: argparse.Namespace, cfg: Dict[str, Any],
                          method_name: str, results_dir: Path) -> int:
    """Run or resume one preflight-approved experiment while its lock is held."""
    config_path = results_dir / "config.json"
    skip_marker = results_dir / "skipped.json"
    # Preconditions are met now, so clear any marker from an earlier attempt.
    skip_marker.unlink(missing_ok=True)

    if args.rerun:
        archived = _archive_resume_state(results_dir)
        if archived:
            print("Archived previous run state:")
            for path in archived:
                print(f"  - {path}")

    method_type = get_method(method_name)

    metrics_path = Path(cfg["results_dir"]) / "metrics.json"
    fold_metrics = []
    k_start = cfg.get("k_start", 0)
    k_end = cfg.get("k_end", cfg.get("k", 5))
    resume_state_loaded = metrics_path.exists() and not args.rerun

    # Validate resume state before replacing the saved configuration snapshot.
    # This also permits safe migration from the legacy, unfingerprinted format.
    if resume_state_loaded:
        try:
            fold_metrics = _load_resume_state(
                metrics_path, config_path, method_name, cfg)
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"Refusing to resume from {metrics_path}: {error}. "
                "Use a new results_dir for the new experiment.") from error
        print(f"Resuming from {len(fold_metrics)} completed folds...")

    try:
        pending_folds = _pending_folds(fold_metrics, k_start, k_end)
    except RuntimeError as error:
        raise RuntimeError(
            f"Refusing to resume from {metrics_path}: {error}. "
            "Use a new results_dir for the new experiment.") from error

    # Persist successful legacy/documentation-only migrations even when every
    # fold is already complete. Write the self-identifying metrics first: if
    # the subsequent config snapshot write fails, resume validation can still
    # establish ownership directly from the metrics fingerprint.
    if resume_state_loaded:
        _write_metrics(metrics_path, method_name, cfg, fold_metrics)
    _write_json_atomic(config_path, cfg)

    writer = SummaryWriter(log_dir=str(results_dir / "tensorboard"))
    try:
        for fold in pending_folds:
            # Adapters can own caches, pseudo-label banks, staged-training
            # state, and optimizer references.  Cross-validation folds are
            # independent experiments, so none of that state may survive a
            # fold boundary.  Seed before construction for adapters which
            # initialize any private stochastic state in __init__.
            set_seed(cfg.get("seed", 1) + fold)
            method = method_type(cfg, device=args.device)
            m = {"fold": fold, **train_one_fold(fold, cfg, method, writer)}
            fold_metrics.append(m)
            method.on_fold_end(fold, m)

            # Incrementally save metrics so a preempted job resumes safely.
            _write_metrics(metrics_path, method_name, cfg, fold_metrics)

        test_accs = [m["test_acc"] for m in fold_metrics
                     if m["test_acc"] is not None]
        print("\n" + "=" * 60)
        print(f"  Method: {method_name}")
        if test_accs:
            print(f"  Mean test acc: {np.mean(test_accs):.4f} "
                  f"+- {np.std(test_accs):.4f}")
        else:
            print("  Holdout test evaluation: skipped")
        print("=" * 60)
        return 0
    finally:
        writer.flush()
        writer.close()


# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    try:
        cfg = load_yaml_config(args.config)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    try:
        method_name = _resolve_run_method(args.method, cfg)
    except (KeyError, ValueError) as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("results_dir", f"./results/{method_name}")
    results_dir = Path(cfg["results_dir"])

    # A benchmark matrix always contains configurations whose assets have not
    # been produced yet. Those must skip visibly rather than fail on a GPU or
    # train on an empty set, so the campaign as a whole keeps moving.
    skip_marker = results_dir / "skipped.json"
    report = preflight(cfg)
    for warning in report.warnings:
        print(f"  ! {warning}")
    if not report.ok:
        print(f"\nSKIPPED {method_name} / {args.config}")
        for problem in report.problems:
            print(f"  - {problem}")
        payload = {"method": method_name, "config": args.config,
                   **report.as_dict()}
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(skip_marker, payload)
            print(f"\n  reason recorded in {skip_marker}")
        except OSError as error:
            # An unusable results directory is itself a preflight diagnosis.
            # Preserve the clean skipped exit even when no marker can be
            # persisted there.
            print(f"\n  reason could not be recorded: {error}")
        return EXIT_SKIPPED

    try:
        args.device = _validate_device(args.device)
    except ValueError as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # Only mutate output directories after the read-only doctor passes. This
    # lets an invalid results_dir produce a useful skip instead of an early
    # mkdir traceback.
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = _acquire_results_lock(results_dir)
    except (OSError, ResultsDirectoryBusyError) as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        return _run_ready_experiment(args, cfg, method_name, results_dir)
    finally:
        _release_results_lock(lock_handle)


if __name__ == "__main__":
    raise SystemExit(main())
