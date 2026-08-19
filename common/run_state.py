"""Shared validation for resumable benchmark metrics."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


RESUME_STATE_VERSION = 1

# These fields document the implementation but do not alter model construction,
# data, optimization, or evaluation. They were added after early runs existed;
# allowing an absent/present difference keeps those exact runs resumable while
# every executable configuration field remains fingerprinted.
NON_EXECUTION_CONFIG_KEYS = frozenset({
    "fidelity_note", "implementation_provenance", "upstream_fidelity",
})


def _execution_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in cfg.items()
        if key not in NON_EXECUTION_CONFIG_KEYS}


def _load_saved_config(config_path: Path) -> Any:
    try:
        with config_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"cannot read config snapshot {config_path}: {error}") from error


def run_identity(method_name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable identity of one method and resolved configuration."""
    try:
        canonical = json.dumps(
            {"method": method_name, "config": cfg},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"run configuration cannot be fingerprinted as JSON: {error}") \
            from error
    return {
        "version": RESUME_STATE_VERSION,
        "method": method_name,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _validate_finite_metrics(value: Any, location: str) -> None:
    """Reject non-finite values anywhere in a persisted fold record."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_metrics(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_metrics(item, f"{location}[{index}]")
    elif (isinstance(value, Real) and not isinstance(value, bool)
          and not math.isfinite(float(value))):
        raise RuntimeError(
            f"resume state contains a non-finite metric at {location}")


def validate_resume_state(
    state: Any, config_path: Path, method_name: str,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate metrics ownership and fold indices before a run is resumed.

    Fingerprinted state must exactly match the current method and resolved
    configuration. Legacy state is accepted only when its method and saved
    ``config.json`` snapshot provide the same evidence.
    """
    if not isinstance(state, dict) or not isinstance(state.get("folds"), list):
        raise RuntimeError("invalid resume state")

    if state.get("method") != method_name:
        raise RuntimeError(
            "resume state method does not match the requested method")

    expected = run_identity(method_name, cfg)
    actual = state.get("run_identity")
    if actual is None:
        if state.get("method") != method_name or not config_path.exists():
            raise RuntimeError(
                "legacy resume state has no verifiable method/config snapshot")
        saved_cfg = _load_saved_config(config_path)
        if (not isinstance(saved_cfg, dict)
                or _execution_config(saved_cfg) != _execution_config(cfg)):
            raise RuntimeError("saved configuration differs from this run")
    elif not isinstance(actual, dict):
        raise RuntimeError("resume state has an invalid run identity")
    elif actual != expected:
        saved_cfg = (
            _load_saved_config(config_path) if config_path.exists() else None)
        documented_upgrade = (
            isinstance(saved_cfg, dict)
            and actual == run_identity(method_name, saved_cfg)
            and _execution_config(saved_cfg) == _execution_config(cfg))
        if not documented_upgrade:
            raise RuntimeError(
                "resume state belongs to a different method or configuration "
                f"(expected {expected['config_sha256']}, "
                f"found {actual.get('config_sha256', 'missing')})")

    folds = state["folds"]
    indices = [
        entry.get("fold") for entry in folds if isinstance(entry, dict)
    ]
    if len(indices) != len(folds) or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in indices):
        raise RuntimeError(
            "resume state contains a fold without an integer index")
    if len(indices) != len(set(indices)):
        raise RuntimeError("resume state contains duplicate fold indices")
    raw_start = cfg.get("k_start", 0)
    raw_end = cfg.get("k_end", cfg.get("k", 5))
    if (isinstance(raw_start, bool) or not isinstance(raw_start, int)
            or isinstance(raw_end, bool) or not isinstance(raw_end, int)
            or raw_start < 0 or raw_end <= raw_start):
        raise RuntimeError(
            "cannot validate resume folds against invalid configured range "
            f"[{raw_start!r}, {raw_end!r})")
    outside = sorted(
        index for index in indices
        if index < raw_start or index >= raw_end)
    if outside:
        raise RuntimeError(
            f"resume state contains folds outside [{raw_start}, {raw_end}): "
            f"{outside}")
    evaluate_test = cfg.get("evaluate_test", True)
    for entry in folds:
        best_loss = entry.get("best_val_loss")
        if (isinstance(best_loss, bool) or not isinstance(best_loss, Real)
                or not math.isfinite(float(best_loss))):
            raise RuntimeError(
                f"resume fold {entry['fold']} has no finite best_val_loss")
        test_accuracy = entry.get("test_acc")
        if evaluate_test is False:
            if test_accuracy is not None:
                raise RuntimeError(
                    f"resume fold {entry['fold']} should have test_acc=null "
                    "when evaluate_test=false")
        elif (isinstance(test_accuracy, bool)
              or not isinstance(test_accuracy, Real)
              or not math.isfinite(float(test_accuracy))
              or not 0.0 <= float(test_accuracy) <= 1.0):
            raise RuntimeError(
                f"resume fold {entry['fold']} has no valid test_acc in [0, 1]")
        _validate_finite_metrics(entry, f"folds[{entry['fold']}]")
    return folds
