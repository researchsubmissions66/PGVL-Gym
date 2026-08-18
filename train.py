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
import os
import json
import random
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

from methods import get_method
from common.preflight import preflight
from common.utils.core_utils import EarlyStopping, Accuracy_Logger

# Exit code meaning "this configuration cannot produce a result and was skipped".
# Distinct from 0 (completed) and from any real failure, so a campaign launcher
# can tell a skipped configuration apart from a broken one.
EXIT_SKIPPED = 3

# Individual batches are allowed to fail -- one unreadable slide should not end a
# five-fold run -- but a number computed from a fraction of the declared split is
# worse than no number at all, because it still looks comparable to a complete
# one. Past this fraction the run is no longer measuring the split the protocol
# declared, so it fails instead of reporting. Override per run with
# `max_batch_failure_rate`.
MAX_BATCH_FAILURE_RATE = 0.1


def _check_batch_failures(phase: str, fold: int, succeeded: int, failed: int,
                          last_error: BaseException | None,
                          threshold: float) -> None:
    """Fail loudly when too much of a split never reached the model.

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
            f"fold{fold} {phase}: {failed} of {total} batches failed "
            f"({rate:.0%} > {threshold:.0%} allowed), so the result would not be "
            f"computed on the declared split. Last error: {last_error}")
    if failed:
        print(f"  ! {phase}: {failed} of {total} batches failed ({rate:.1%})")


# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified WSI/VLM training")
    p.add_argument("--method", required=True,
                   help="One of: focus, vila_mil, cod_mil, maple, mscpt, "
                        "pathpt, top, slip, wsi_five, muse, convlm, sldpc")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _batch_metadata(batch: Any) -> Dict[str, Any] | None:
    if isinstance(batch, dict) and "slide_id" in batch:
        slide_ids = batch["slide_id"]
        case_ids = batch.get("case_id")
        if case_ids is None:
            case_ids = [str(slide_id)[:12] for slide_id in slide_ids]
        return {"slide_id": slide_ids, "case_id": case_ids}
    if not isinstance(batch, (list, tuple)):
        return None
    return next((item for item in batch[1:-1] if isinstance(item, dict)), None)


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

    if method_name in {"focus", "vila_mil", "maple"}:
        # CLAM-style dual-scale loader
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
            mode=mode,
            shuffle=False, seed=cfg.get("seed", 1), print_info=True,
            label_dict=cfg["label_dict"], patient_strat=False, ignore=[],
            label_col=cfg.get("label_column"))
        train, val, test = ds.return_splits(
            from_id=False,
            csv_path=os.path.join(cfg["split_dir"], f"splits_{fold}.csv"))
        return (get_split_loader(train, training=True, weighted=False, mode=mode),
                get_split_loader(val,   mode=mode),
                get_split_loader(test,  mode=mode))

    if method_name == "pathpt":
        from methods.pathpt.dataset import build_pathpt_loader
        return (build_pathpt_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "train"),
                build_pathpt_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "val", shuffle=False),
                build_pathpt_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "test", shuffle=False))

    if method_name == "mscpt":
        from methods.mscpt.dataset import build_mscpt_loader
        return (build_mscpt_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "train"),
                build_mscpt_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "val", shuffle=False),
                build_mscpt_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "test", shuffle=False))

    if method_name == "wsi_five":
        from methods.wsi_five.dataset import build_wsi_five_loader
        return (build_wsi_five_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "train"),
                build_wsi_five_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "val", shuffle=False),
                build_wsi_five_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "test", shuffle=False))

    if method_name in {"top", "slip", "composite"}:
        from common.datasets.bag_features import build_bag_loader
        return (build_bag_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "train"),
                build_bag_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "val", shuffle=False),
                build_bag_loader({**cfg, "split_dir": _fold_dir(cfg, fold)}, "test", shuffle=False))

    if method_name == "muse":
        # MUSE aggregates one CONCH feature bag per slide.
        from common.datasets.bag_features import build_bag_loader
        fold_cfg = {**cfg, "split_dir": _fold_dir(cfg, fold), "batch_size": 1}
        return (build_bag_loader(fold_cfg, "train"),
                build_bag_loader(fold_cfg, "val", shuffle=False),
                build_bag_loader(fold_cfg, "test", shuffle=False))

    if method_name == "convlm":
        # ConVLM is the only raw pathology-patch method in this collection.
        from methods.convlm.dataset import build_convlm_loader
        return (build_convlm_loader(cfg, "train", fold),
                build_convlm_loader(cfg, "val", fold, shuffle=False),
                build_convlm_loader(cfg, "test", fold, shuffle=False))

    raise KeyError(f"No dataloader recipe registered for method '{method_name}'")


def _fold_dir(cfg: Dict[str, Any], fold: int) -> str:
    """Methods may have per-fold subfolders (e.g. fold0/train.csv)
    or a flat folder (splits_{fold}.csv).  We try the per-fold layout
    first, fall back to flat."""
    cand = os.path.join(cfg["split_dir"], f"fold{fold}")
    return cand if os.path.isdir(cand) else cfg["split_dir"]


# -----------------------------------------------------------------------------
def train_one_fold(fold: int, cfg, method, writer):
    print(f"\n--- Fold {fold} ----------------------------------------------")
    set_seed(cfg.get("seed", 1) + fold)

    failure_threshold = float(
        cfg.get("max_batch_failure_rate", MAX_BATCH_FAILURE_RATE))
    # Worst epoch per phase, so a shortfall is recorded in metrics.json rather
    # than only printed into a log nobody reads.
    failure_counts = {"train": 0, "val": 0, "test": 0}

    train_loader, val_loader, test_loader = build_loaders(method.name, cfg, fold)

    model = method.build_model()
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
        initial_val_loss, initial_n = 0.0, 0
        last_initial_error = None
        for batch in val_loader:
            try:
                out = method.eval_step(batch, model, loss_fn)
            except Exception as error:
                last_initial_error = error
                print(f"  ! initial eval_step failed: {error}")
                continue
            initial_val_loss += out["loss"]
            initial_n += 1
        if initial_n == 0:
            raise RuntimeError(
                "Every initial validation batch failed. Last error: "
                f"{last_initial_error}")
        initial_val_loss /= initial_n
        best_val_loss = initial_val_loss
        writer.add_scalar(f"fold{fold}/val_loss", initial_val_loss, -1)
        print(f"  initial validation loss: {initial_val_loss:.4f}")
        if early is not None:
            ckpt = Path(cfg["results_dir"]) / f"fold{fold}_best.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            early(-1, initial_val_loss, model, ckpt_name=str(ckpt))

    for epoch in range(cfg.get("epochs", 200)):
        model.train()
        run_loss, n_batches, n_failed = 0.0, 0, 0
        last_train_error = None
        train_log = Accuracy_Logger(n_classes=cfg["n_classes"])
        for batch in train_loader:
            try:
                out = method.train_step(batch, model, optimizer, loss_fn)
            except Exception as e:
                last_train_error = e
                n_failed += 1
                print(f"  ! train_step failed for one batch: {e}")
                continue
            run_loss += out["loss"]
            n_batches += 1
            preds = out["logits"].argmax(dim=1)
            train_log.log_batch(preds.cpu().numpy(),
                                out["label"].cpu().numpy())
        _check_batch_failures("train", fold, n_batches, n_failed,
                              last_train_error, failure_threshold)
        failure_counts["train"] = max(failure_counts["train"], n_failed)
        train_loss = run_loss / n_batches
        writer.add_scalar(f"fold{fold}/train_loss", train_loss, epoch)

        model.eval()
        val_loss, n_v, n_v_failed = 0.0, 0, 0
        last_val_error = None
        val_log = Accuracy_Logger(n_classes=cfg["n_classes"])
        for batch in val_loader:
            try:
                out = method.eval_step(batch, model, loss_fn)
            except Exception as e:
                last_val_error = e
                n_v_failed += 1
                print(f"  ! eval_step failed: {e}")
                continue
            val_loss += out["loss"]
            n_v += 1
            preds = out["logits"].argmax(dim=1)
            val_log.log_batch(preds.cpu().numpy(),
                              out["label"].cpu().numpy())
        _check_batch_failures("val", fold, n_v, n_v_failed,
                              last_val_error, failure_threshold)
        failure_counts["val"] = max(failure_counts["val"], n_v_failed)
        val_loss /= n_v
        writer.add_scalar(f"fold{fold}/val_loss", val_loss, epoch)

        if scheduler is not None:
            try:
                scheduler.step(val_loss)
            except TypeError:
                scheduler.step()

        method.on_epoch_end(epoch, {"train_loss": train_loss, "val_loss": val_loss})
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
    torch.save(model.state_dict(), str(final_ckpt))

    # ---- test ------------------------------------------------------
    if not cfg.get("evaluate_test", True):
        print(
            f"  >>> fold{fold} holdout test evaluation skipped "
            "(evaluate_test=false)")
        return {"test_acc": None, "best_val_loss": best_val_loss,
                "batch_failures": dict(failure_counts)}

    if early is not None:
        ckpt = Path(cfg["results_dir"]) / f"fold{fold}_best.pt"
        if ckpt.exists():
            try:
                model.load_state_dict(
                    torch.load(
                        ckpt, map_location=method.device, weights_only=True))
            except Exception as error:
                print(f"  ! cannot load best ckpt: {error}")
    model.eval()
    test_log = Accuracy_Logger(n_classes=cfg["n_classes"])
    n_test = 0
    n_test_failed = 0
    last_test_error = None
    test_logits = []
    test_labels = []
    prediction_metadata: list[dict[str, str]] = []
    for batch in test_loader:
        try:
            out = method.eval_step(batch, model)
            preds = out["logits"].argmax(dim=1)
            test_log.log_batch(preds.cpu().numpy(),
                               out["label"].cpu().numpy())
            logits = out["logits"].detach().float().cpu()
            labels = out["label"].detach().long().cpu().reshape(-1)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            test_logits.append(logits)
            test_labels.append(labels)
            metadata = _batch_metadata(batch)
            if metadata is not None:
                for index in range(len(labels)):
                    prediction_metadata.append({
                        "slide_id": str(metadata["slide_id"][index]),
                        "case_id": str(metadata["case_id"][index]),
                    })
            n_test += 1
        except Exception as e:
            last_test_error = e
            n_test_failed += 1
            print(f"  ! test_step failed: {e}")
    # The held-out split is the number that gets reported, so it is held to the
    # same standard as training rather than merely being non-empty.
    _check_batch_failures("test", fold, n_test, n_test_failed,
                          last_test_error, failure_threshold)
    failure_counts["test"] = n_test_failed
    logits = torch.cat(test_logits).numpy()
    labels = torch.cat(test_labels).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    slide_metrics = classification_metrics(
        probabilities, labels, cfg["n_classes"])
    test_acc = slide_metrics["accuracy"]
    assert test_acc is not None
    print(f"  >>> fold{fold} test acc: {test_acc:.4f}")

    patient_metrics = None
    if len(prediction_metadata) == len(labels):
        predictions = pd.DataFrame(prediction_metadata)
        predictions["label"] = labels
        predictions["prediction"] = probabilities.argmax(axis=1)
        for class_index in range(cfg["n_classes"]):
            predictions[f"probability_{class_index}"] = probabilities[:, class_index]
        prediction_path = Path(cfg["results_dir"]) / f"fold{fold}_predictions.csv"
        predictions.to_csv(prediction_path, index=False)

        probability_columns = [
            f"probability_{index}" for index in range(cfg["n_classes"])
        ]
        patient_frame = predictions.groupby("case_id", sort=True).agg(
            {**{column: "mean" for column in probability_columns}, "label": "first"}
        )
        patient_metrics = classification_metrics(
            patient_frame[probability_columns].to_numpy(),
            patient_frame["label"].to_numpy(dtype=int), cfg["n_classes"])

    return {
        "test_acc": test_acc,
        "best_val_loss": best_val_loss,
        "slide_metrics": slide_metrics,
        "patient_metrics": patient_metrics,
        "batch_failures": dict(failure_counts),
    }


# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("results_dir", f"./results/{args.method}")
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    with open(results_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # A benchmark matrix always contains configurations whose assets have not
    # been produced yet. Those must skip visibly rather than fail on a GPU or
    # train on an empty set, so the campaign as a whole keeps moving.
    skip_marker = results_dir / "skipped.json"
    report = preflight(cfg)
    for warning in report.warnings:
        print(f"  ! {warning}")
    if not report.ok:
        print(f"\nSKIPPED {args.method} / {args.config}")
        for problem in report.problems:
            print(f"  - {problem}")
        payload = {"method": args.method, "config": args.config,
                   **report.as_dict()}
        with open(skip_marker, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n  reason recorded in {skip_marker}")
        return EXIT_SKIPPED

    # Preconditions are met now, so clear any marker from an earlier attempt.
    skip_marker.unlink(missing_ok=True)

    method = get_method(args.method)(cfg, device=args.device)
    writer = SummaryWriter(log_dir=f"logs/{args.method}")

    metrics_path = Path(cfg["results_dir"]) / "metrics.json"
    fold_metrics = []

    # Load existing metrics to resume
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as handle:
                old_metrics = json.load(handle)
                if "folds" in old_metrics:
                    fold_metrics = old_metrics["folds"]
                    print(f"Resuming from {len(fold_metrics)} completed folds...")
        except Exception as e:
            print(f"Failed to load existing metrics for resume: {e}")

    k_start = cfg.get("k_start", 0)
    k_end = cfg.get("k_end", cfg.get("k", 5))

    # Resume from the highest recorded fold index rather than from the number of
    # entries, so a partially written or out-of-order metrics.json cannot cause a
    # fold to be silently skipped or repeated. Files written before the fold
    # index was recorded fall back to the old length-based behaviour.
    completed = [entry["fold"] for entry in fold_metrics
                 if isinstance(entry, dict) and isinstance(entry.get("fold"), int)]
    start_fold = max(completed) + 1 if completed else k_start + len(fold_metrics)

    for fold in range(start_fold, k_end):
        m = {"fold": fold, **train_one_fold(fold, cfg, method, writer)}
        fold_metrics.append(m)
        method.on_fold_end(fold, m)

        # Incrementally save metrics so we don't lose them if the job is preempted
        with open(metrics_path, "w") as handle:
            json.dump({"method": args.method, "folds": fold_metrics}, handle, indent=2)

    test_accs = [m["test_acc"] for m in fold_metrics
                 if m["test_acc"] is not None]
    print("\n" + "=" * 60)
    print(f"  Method: {args.method}")
    if test_accs:
        print(f"  Mean test acc: {np.mean(test_accs):.4f} "
              f"+- {np.std(test_accs):.4f}")
    else:
        print("  Holdout test evaluation: skipped")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
