#!/usr/bin/env python3
"""Prepare and validate feature-agnostic pathology benchmark protocols."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Each cohort owns its protocol, so there is no meaningful default: generating
# against the wrong cohort silently produces a benchmark for the wrong data.
# --protocol is required instead.
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def _load_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if protocol.get("version") != 3:
        raise ValueError("Only resolution-registry protocol version 3 is supported")
    _validate_protocol_registry(protocol)
    return protocol


def _absolute_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _feature_column(feature_source: str) -> str:
    return f"feature__{feature_source}"


def _input_kind(feature_cfg: dict[str, Any]) -> str:
    return str(feature_cfg.get("input_kind", "patch_bag"))


def _feature_path(
    feature_cfg: dict[str, Any], slide_id: str, task: str | None = None
) -> Path:
    template = str(feature_cfg["path_template"])
    if "{slide_id}" not in template:
        raise ValueError(f"Feature path template must include {{slide_id}}: {template}")
    try:
        resolved = template.format(
            slide_id=slide_id, task=task or "validation-task")
    except KeyError as error:
        raise ValueError(
            f"Feature path template uses undefined keys: {template}") from error
            
    path = Path(resolved)
    if not path.exists() and path.parent.exists():
        # Handle TCGA UUID suffixes (e.g. slide_id.UUID.h5)
        matches = list(path.parent.glob(f"{slide_id}*.*"))
        if matches:
            return matches[0]
    return path.expanduser()


def _source_present(feature_cfg: dict[str, Any], path: Path) -> bool:
    if _input_kind(feature_cfg) == "raw_tile_directory":
        if not path.is_dir():
            return False
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
        return any(item.is_file() and item.suffix.lower() in extensions
                   for item in path.iterdir())
    return path.is_file()


def _feature_root(feature_cfg: dict[str, Any], task: str | None = None) -> str:
    return str(_feature_path(feature_cfg, "__slide_id__", task).parent)


def _slide_source_type(
    feature_cfg: dict[str, Any], task: str | None = None
) -> str:
    from common.datasets.slide_embeddings import (
        infer_slide_embedding_source_type,
    )
    return infer_slide_embedding_source_type(
        str(_feature_path(feature_cfg, "__slide_id__", task)),
        feature_cfg.get("storage"),
    )


def _experiment_supports_task(
    experiment_cfg: dict[str, Any], task: str
) -> bool:
    """Return whether an experiment is registered for a benchmark task."""
    tasks = experiment_cfg.get("tasks")
    return tasks is None or task in tasks


def _validate_protocol_registry(protocol: dict[str, Any]) -> None:
    """Reject ambiguous encoder provenance before generating any configs."""
    from common.backbones import get_spec

    required = {"resolution", "backbone", "feature_space_id", "path_template"}
    valid_kinds = {
        "patch_bag", "slide_embedding", "patch_sequence",
        "raw_tile_directory",
    }
    sources = protocol.get("feature_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Protocol requires a non-empty feature_sources registry")
    for source, config in sources.items():
        missing = required.difference(config)
        if missing:
            raise ValueError(f"Feature source {source} missing {sorted(missing)}")
        kind = _input_kind(config)
        if kind not in valid_kinds:
            raise ValueError(
                f"Feature source {source} has unknown input_kind {kind!r}")
        if kind != "raw_tile_directory":
            for field in ("feature_key", "feature_dim"):
                if config.get(field) is None:
                    raise ValueError(f"Feature source {source} requires {field}")

        model_owned = bool(config.get("model_owned", False))
        runtime_encoder = bool(config.get("runtime_encoder", True))
        if not runtime_encoder and kind != "slide_embedding":
            raise ValueError(
                f"Feature source {source} may disable runtime_encoder only "
                "for offline slide embeddings")
        if not model_owned and not config.get("encoder_weights"):
            raise ValueError(
                f"Feature source {source} requires encoder_weights for "
                "offline provenance")
        if not model_owned and runtime_encoder:
            spec = get_spec(config["backbone"])
            declared_space = str(config["feature_space_id"])
            exact_space = declared_space == spec.feature_space_id
            revision_space = (
                spec.name == "titan"
                and declared_space.startswith(spec.feature_space_id + "@")
            )
            if not (exact_space or revision_space):
                raise ValueError(
                    f"Feature source {source} declares feature space "
                    f"{declared_space!r}, but encoder {spec.name!r} uses "
                    f"{spec.feature_space_id!r}")
            native_dim = (spec.slide_input_dim
                          if kind == "slide_embedding" else spec.tile_dim)
            if (native_dim is not None
                    and int(config["feature_dim"]) != native_dim):
                raise ValueError(
                    f"Feature source {source} width {config['feature_dim']} does "
                    f"not match encoder {spec.name} width {native_dim}")
        weights_value = config.get("encoder_weights")
        if weights_value:
            weights = Path(str(weights_value)).expanduser()
            if (not weights.exists()
                    and config.get("availability") != "future"):
                raise FileNotFoundError(
                    f"Feature source {source} encoder weights are missing: {weights}")
        _feature_path(config, "validation-slide-id", "validation-task")

    experiments = protocol.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError("Protocol requires at least one experiment")
    cohorts = protocol.get("cohorts")
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("Protocol requires at least one cohort")
    for cohort, config in cohorts.items():
        labels = config.get("labels")
        classnames = config.get("classnames")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Cohort {cohort} requires ordered labels")
        if not isinstance(classnames, list) or len(classnames) != len(labels):
            raise ValueError(
                f"Cohort {cohort} classnames must align one-to-one with labels")
        if len(set(map(str, labels))) != len(labels):
            raise ValueError(f"Cohort {cohort} labels must be unique")
        if config.get("prompt_spec"):
            from common.prompts import load_prompt_profile

            load_prompt_profile(
                config["prompt_spec"], labels=labels, classnames=classnames,
                repo_root=REPO_ROOT)
    for experiment, config in experiments.items():
        tasks = config.get("tasks")
        if tasks is not None:
            if not isinstance(tasks, list) or not tasks:
                raise ValueError(
                    f"Experiment {experiment} tasks must be a non-empty list")
            unknown = set(tasks).difference(protocol.get("cohorts", {}))
            if unknown:
                raise ValueError(
                    f"Experiment {experiment} references unknown tasks "
                    f"{sorted(unknown)}")
        if not isinstance(config.get("features"), dict):
            raise ValueError(f"Experiment {experiment} requires feature role bindings")
        bindings = _resolve_feature_bindings(protocol, experiment, config)
        _validate_feature_roles(config["method"], experiment, bindings)
        if config["method"] == "sldpc":
            _validate_sldpc_protocol(experiment, config, bindings)
        elif config["method"] == "muse":
            _validate_muse_protocol(experiment, config)


def _feature_tensor_shape(path: Path, feature_key: str) -> tuple[int, ...]:
    """Inspect an HDF5 or torch feature payload without model assumptions."""
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            key = next(
                (name for name in (feature_key, "features", "embeddings", "feats")
                 if name in handle), None)
            if key is None:
                raise ValueError(f"No feature tensor found in {path}")
            return tuple(handle[key].shape)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        key = next(
            (name for name in (feature_key, "features", "embeddings", "feats")
             if name in payload), None)
        if key is None:
            raise ValueError(f"No feature tensor found in {path}")
        payload = payload[key]
    shape = tuple(torch.as_tensor(payload).shape)
    return shape[1:] if len(shape) == 3 and shape[0] == 1 else shape


def build_generated_prompt_assets(
    protocol: dict[str, Any], output_dir: Path
) -> dict[str, list[str]]:
    """Compile canonical task prompts, retaining legacy description support."""
    generated: dict[str, list[str]] = {}
    for cohort, cohort_cfg in protocol["cohorts"].items():
        if cohort_cfg.get("prompt_spec"):
            from common.prompts import compile_task_prompt_assets

            assets = compile_task_prompt_assets(
                cohort, cohort_cfg, output_dir, repo_root=REPO_ROOT)
            # The in-memory registry is consumed by _method_config during this
            # generation pass. It is intentionally not serialized into YAML.
            cohort_cfg["_generated_prompt_assets"] = assets
            generated[cohort] = list(assets["muse"])
            continue
        source_value = cohort_cfg.get("muse_prompt_json")
        if not source_value:
            continue
        source = _absolute_repo_path(source_value)
        with source.open(encoding="utf-8") as handle:
            descriptions = json.load(handle)
        prompt_dir = output_dir / "data" / cohort / "muse_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for class_index, label in enumerate(cohort_cfg["labels"]):
            if label not in descriptions:
                raise ValueError(f"{source}: missing MUSE descriptions for {label}")
            class_descriptions = []
            for scale in ("small_mag", "big_mag"):
                class_descriptions.extend(descriptions[label].get(scale, []))
            if not class_descriptions:
                raise ValueError(f"{source}: no MUSE descriptions for {label}")
            path = prompt_dir / f"generated_new_{class_index}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["", "0"])
                writer.writerows(enumerate(class_descriptions))
            paths.append(str(path))
        generated[cohort] = paths

        if (not cohort_cfg.get("maple_prompt_json")
                and cohort_cfg.get("classnames")):
            classnames = list(cohort_cfg["classnames"])
            labels = list(cohort_cfg["labels"])
            maple_payload: dict[str, Any] = {}
            for level, source_level in (("low", "small_mag"),
                                        ("high", "big_mag")):
                global_info: dict[str, str] = {}
                attributes: dict[str, str] = {}
                for label, classname in zip(labels, classnames):
                    prompts = _nonempty_strings(
                        descriptions[label].get(source_level),
                        f"{source}:{label}.{source_level}",
                    )
                    global_info[classname] = " ".join(prompts)
                    attributes[classname] = " ".join(prompts[:3])
                maple_payload[level] = {
                    "tumor": (
                        "histologic architecture and cellular morphology "
                        "used to distinguish the diagnostic classes"
                    ),
                    "global_info": global_info,
                    "entities": [{
                        "name": "Diagnostic morphology",
                        "general_feature": (
                            "Class-specific architectural and cytologic "
                            "findings visible at this magnification"
                        ),
                        "attributes": attributes,
                    }],
                }
            maple_path = output_dir / "data" / cohort / "maple_attributes.json"
            with maple_path.open("w", encoding="utf-8") as handle:
                json.dump(maple_payload, handle, indent=2)
    return generated


def _load_task_metadata(
    task: str, task_cfg: dict[str, Any]
) -> pd.DataFrame | None:
    """Load one or more annotation tables into the canonical task schema."""
    values = task_cfg.get("metadata_csvs")
    if values is None:
        value = task_cfg.get("metadata_csv")
        values = [value] if value else []
    if not isinstance(values, list) or not values:
        raise ValueError(f"{task}: metadata_csv or metadata_csvs is required")

    frames: list[pd.DataFrame] = []
    for index, value in enumerate(values):
        if isinstance(value, dict):
            path_value = value.get("path")
            partition = value.get("partition", f"source_{index}")
        else:
            path_value = value
            partition = f"source_{index}"
        if not path_value:
            raise ValueError(f"{task}: metadata source {index} has no path")
        path = Path(str(path_value)).expanduser()
        if not path.is_file():
            if task_cfg.get("metadata_availability") == "future":
                return None
            raise FileNotFoundError(f"{task}: metadata does not exist: {path}")
        frame = pd.read_csv(path)
        frame["source_partition"] = str(partition)
        frames.append(frame)

    source = pd.concat(frames, ignore_index=True, sort=False)
    for column, allowed in task_cfg.get("filters", {}).items():
        if column not in source.columns:
            raise ValueError(f"{task}: filter column {column!r} is missing")
        values_allowed = allowed if isinstance(allowed, list) else [allowed]
        source = source[source[column].isin(values_allowed)].copy()
    slide_column = str(task_cfg.get("slide_id_column", "slide_id"))
    case_column = task_cfg.get("case_id_column", "case_id")
    label_column = str(task_cfg["label_column"])
    required = {slide_column, label_column}
    if case_column is not None:
        required.add(str(case_column))
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"{task}: metadata missing {sorted(missing)}")

    canonical = source.copy()
    canonical["slide_id"] = canonical[slide_column].astype(str)
    suffixes = task_cfg.get("slide_suffixes", [".svs"])
    for suffix in suffixes:
        canonical["slide_id"] = canonical["slide_id"].str.removesuffix(
            str(suffix))
    canonical["case_id"] = (
        canonical[str(case_column)].astype(str)
        if case_column is not None else canonical["slide_id"]
    )
    labels = canonical[label_column].astype(str).str.strip()
    transform = task_cfg.get("label_transform")
    if transform == "lower":
        labels = labels.str.lower()
    elif transform == "upper":
        labels = labels.str.upper()
    elif transform not in {None, "identity"}:
        raise ValueError(f"{task}: unknown label_transform {transform!r}")
    mapping = {
        str(key): str(value)
        for key, value in task_cfg.get("label_mapping", {}).items()
    }
    canonical["label"] = labels.map(lambda value: mapping.get(value, value))
    keep = ["slide_id", "case_id", "label", "source_partition"]
    for column in task_cfg.get("metadata_columns", []):
        if column not in canonical.columns:
            raise ValueError(f"{task}: metadata column {column!r} is missing")
        if column not in keep:
            keep.append(column)
    return canonical[keep].copy()


def build_manifests(protocol: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    """Create annotation-defined manifests plus independent feature coverage."""
    rows: list[dict[str, Any]] = []

    for cohort, cohort_cfg in protocol["cohorts"].items():
        source = _load_task_metadata(cohort, cohort_cfg)
        labels = list(cohort_cfg["labels"])
        if source is None:
            frame = pd.DataFrame(columns=[
                "slide_id", "case_id", "label", "source_partition"])
        else:
            frame = source[source["label"].isin(labels)].copy()
        frame["label_id"] = frame["label"].map(dict(zip(labels, range(len(labels)))))
        frame.insert(0, "cohort", cohort)

        for feature_source, feature_cfg in protocol["feature_sources"].items():
            column = _feature_column(feature_source)
            frame[column] = [
                str(_feature_path(feature_cfg, slide_id, cohort))
                for slide_id in frame["slide_id"]
            ]
            present = frame[column].map(
                lambda value: _source_present(feature_cfg, Path(value)))
            rows.append(
                {
                    "cohort": cohort,
                    "feature_source": feature_source,
                    "input_kind": _input_kind(feature_cfg),
                    "resolution": feature_cfg["resolution"],
                    "backbone": feature_cfg["backbone"],
                    "feature_column": column,
                    "feature_dim": (
                        int(feature_cfg["feature_dim"])
                        if feature_cfg.get("feature_dim") is not None else None),
                    "available_slides": int(present.sum()),
                    "annotated_slides": int(len(frame)),
                    "coverage": float(present.mean()) if len(frame) else 0.0,
                    "metadata_ready": source is not None and not frame.empty,
                }
            )

        eligible = frame.sort_values(["case_id", "slide_id"]).reset_index(drop=True)
        if eligible.empty and cohort_cfg.get("metadata_availability") != "future":
            raise ValueError(f"{cohort}: no annotated slides after label filtering")

        case_label_counts = eligible.groupby("case_id")["label"].nunique()
        inconsistent = case_label_counts[case_label_counts != 1]
        if not inconsistent.empty:
            raise ValueError(f"{cohort}: cases have conflicting labels: {inconsistent.index[:5]}")

        cohort_dir = output_dir / "data" / cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)
        eligible.to_csv(cohort_dir / "manifest.csv", index=False)

    coverage = pd.DataFrame(rows).sort_values(
        ["cohort", "backbone", "resolution", "feature_source"]
    )
    coverage.to_csv(output_dir / "feature_coverage.csv", index=False)
    return coverage


def _representative_slides(manifest: pd.DataFrame, cases: list[str]) -> pd.DataFrame:
    """Use one deterministic slide per few-shot patient."""
    selected = manifest[manifest["case_id"].isin(cases)].copy()
    return (
        selected.sort_values(["case_id", "slide_id"])
        .groupby("case_id", sort=False, as_index=False)
        .first()
    )


def build_splits(protocol: dict[str, Any], output_dir: Path) -> None:
    shots = sorted(int(value) for value in protocol["shots"])
    max_shot = max(shots)
    folds = int(protocol["folds"])
    seed = int(protocol["seed"])

    for cohort, cohort_cfg in protocol["cohorts"].items():
        manifest = pd.read_csv(output_dir / "data" / cohort / "manifest.csv")
        if manifest.empty:
            if cohort_cfg.get("metadata_availability") != "future":
                raise ValueError(f"{cohort}: empty manifest")
            for shot in shots:
                (output_dir / "splits" / cohort / f"{shot}shot").mkdir(
                    parents=True, exist_ok=True)
            continue
        patients = (
            manifest[["case_id", "label", "label_id"]]
            .drop_duplicates()
            .sort_values("case_id")
            .reset_index(drop=True)
        )
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_indices = list(splitter.split(patients["case_id"], patients["label_id"]))

        for fold, (development_index, test_index) in enumerate(fold_indices):
            development = patients.iloc[development_index]
            test_cases = patients.iloc[test_index]["case_id"].tolist()
            train_order: dict[str, list[str]] = {}
            val_order: dict[str, list[str]] = {}

            for label_index, label in enumerate(cohort_cfg["labels"]):
                cases = sorted(
                    development.loc[development["label"] == label, "case_id"].tolist()
                )
                if len(cases) < 2 * max_shot:
                    raise ValueError(
                        f"{cohort} fold {fold} {label}: need {2 * max_shot} "
                        f"development patients, found {len(cases)}"
                    )
                rng = np.random.default_rng(seed + fold * 1009 + label_index * 100003)
                ordered = np.asarray(cases, dtype=object)[rng.permutation(len(cases))]
                train_order[label] = ordered[:max_shot].tolist()
                val_order[label] = ordered[max_shot : 2 * max_shot].tolist()

            test_frame = manifest[manifest["case_id"].isin(test_cases)].copy()
            test_frame = test_frame.sort_values(["case_id", "slide_id"])
            for shot in shots:
                train_cases = sum(
                    (train_order[label][:shot] for label in cohort_cfg["labels"]), []
                )
                val_cases = sum(
                    (val_order[label][:shot] for label in cohort_cfg["labels"]), []
                )
                partitions = {
                    "train": _representative_slides(manifest, train_cases),
                    "val": _representative_slides(manifest, val_cases),
                    "test": test_frame.copy(),
                }
                fold_dir = output_dir / "splits" / cohort / f"{shot}shot" / f"fold{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                for name, frame in partitions.items():
                    frame.insert(0, "partition", name)
                    frame.insert(0, "shots", shot)
                    frame.insert(0, "fold", fold)
                    frame.to_csv(fold_dir / f"{name}.csv", index=False)

                wide = pd.concat(
                    {
                        name: frame["slide_id"].reset_index(drop=True)
                        for name, frame in partitions.items()
                    },
                    axis=1,
                )
                split_root = fold_dir.parent
                wide.to_csv(split_root / f"splits_{fold}.csv", index=False)

                descriptor_rows = []
                for label in cohort_cfg["labels"]:
                    item: dict[str, Any] = {"label": label}
                    for name, frame in partitions.items():
                        subset = frame[frame["label"] == label]
                        item[f"{name}_slides"] = len(subset)
                        item[f"{name}_patients"] = subset["case_id"].nunique()
                    descriptor_rows.append(item)
                pd.DataFrame(descriptor_rows).to_csv(
                    split_root / f"splits_{fold}_descriptor.csv", index=False
                )


def _common_config(
    protocol: dict[str, Any], cohort: str, cohort_cfg: dict[str, Any], shot: int,
    bindings: dict[str, dict[str, Any]], output_dir: Path,
) -> dict[str, Any]:
    labels = list(cohort_cfg["labels"])
    primary = bindings.get(
        "bag", bindings.get("high", bindings.get("tiles")))
    if primary is None:
        raise ValueError("Feature bindings require bag, high, or tiles input")
    feature_cfg = primary["config"]
    weights_value = feature_cfg.get("encoder_weights")
    encoder_weights = (str(Path(str(weights_value)).expanduser().resolve())
                       if weights_value else None)
    config = {
        "backbone": feature_cfg["backbone"],
        "backbone_weights": encoder_weights,
        "encoder": {
            "name": feature_cfg["backbone"],
            "weights": encoder_weights,
            "feature_space_id": feature_cfg["feature_space_id"],
            "feature_dim": (
                int(feature_cfg["feature_dim"])
                if feature_cfg.get("feature_dim") is not None else None),
            "initialization": feature_cfg.get("initialization", "pretrained"),
        },
        "feature_space_id": feature_cfg["feature_space_id"],
        "n_classes": len(labels),
        "classnames": list(cohort_cfg["classnames"]),
        "label_dict": dict(zip(labels, range(len(labels)))),
        "shots": shot,
        "k": int(protocol["folds"]),
        "k_start": 0,
        "k_end": int(protocol["folds"]),
        "seed": int(protocol["seed"]),
        "batch_size": 1,
        "num_workers": 4,
        "feature_key": feature_cfg.get("feature_key"),
        "feature_sources": {
            role: binding["source"] for role, binding in bindings.items()},
        "feature_resolutions": {
            role: binding["config"]["resolution"]
            for role, binding in bindings.items()},
        "feature_input_kinds": {
            role: _input_kind(binding["config"])
            for role, binding in bindings.items()},
        "include_metadata": True,
        "dataset_csv": str(output_dir / "data" / cohort / "manifest.csv"),
        "split_dir": str(output_dir / "splits" / cohort / f"{shot}shot"),
        "task": cohort,
        "benchmark": protocol.get("name", "pathology_benchmark"),
    }
    if feature_cfg.get("feature_dim") is not None:
        config["feature_dim"] = int(feature_cfg["feature_dim"])
    if _input_kind(feature_cfg) == "slide_embedding":
        source_weights_value = feature_cfg.get("encoder_weights")
        source_weights = (
            str(Path(str(source_weights_value)).expanduser().resolve())
            if source_weights_value else None)
        config.update({
            "slide_encoder": {
                "input_kind": "slide_embedding",
                "name": feature_cfg["backbone"],
                "weights": source_weights,
                "feature_space_id": feature_cfg["feature_space_id"],
                "feature_dim": int(feature_cfg["feature_dim"]),
                "feature_key": feature_cfg["feature_key"],
                "resolution": feature_cfg["resolution"],
                "path_template": str(
                    _feature_path(feature_cfg, "{slide_id}", cohort)),
                "runtime_encoder": bool(
                    feature_cfg.get("runtime_encoder", True)),
            },
            "source_type": _slide_source_type(feature_cfg, cohort),
            "slide_features": _feature_root(feature_cfg, cohort),
            "feature_path_column": _feature_column(primary["source"]),
        })
    return config


def _validate_feature_roles(
    method: str, experiment: str, bindings: dict[str, dict[str, Any]]
) -> None:
    expected = {
        "pathpt": {"bag"},
        "muse": {"bag"},
        "focus": {"low", "high"},
        "mscpt": {"low", "high"},
        "maple": {"low", "high"},
        "vila_mil": {"low", "high"},
        "cod_mil": {"low", "high"},
        "top": {"bag"},
        "slip": {"bag"},
        "wsi_five": {"bag"},
        "sldpc": {"bag"},
        "convlm": {"tiles"},
        "composite": {"bag"},
    }
    if method not in expected:
        raise KeyError(f"No feature-registry template for method {method}")
    expected_inputs = expected[method]
    actual_inputs = set(bindings)
    if actual_inputs != expected_inputs:
        raise ValueError(
            f"Experiment {experiment} requires inputs {sorted(expected_inputs)}, "
            f"got {sorted(actual_inputs)}")

    required_kinds = {
        "pathpt": "patch_bag", "muse": "patch_bag",
        "focus": "patch_bag", "mscpt": "patch_bag",
        "maple": "patch_bag", "vila_mil": "patch_bag",
        "cod_mil": "patch_bag", "top": "patch_bag",
        "slip": "patch_bag", "wsi_five": "patch_sequence",
        "sldpc": "slide_embedding", "convlm": "raw_tile_directory",
        "composite": "patch_bag",
    }
    kinds = {_input_kind(binding["config"]) for binding in bindings.values()}
    if kinds != {required_kinds[method]}:
        raise ValueError(
            f"Experiment {experiment} method {method} requires "
            f"{required_kinds[method]}, got {sorted(kinds)}")

    configs = [binding["config"] for binding in bindings.values()]
    if len(configs) > 1:
        for field in (
            "backbone", "encoder_weights", "feature_dim", "feature_space_id",
            "feature_key",
        ):
            values = {config.get(field, "features") for config in configs}
            if len(values) != 1:
                raise ValueError(
                    f"Experiment {experiment} mixes incompatible {field}: "
                    f"{sorted(str(value) for value in values)}")
    if {"low", "high"}.issubset(bindings):
        low = str(bindings["low"]["config"]["resolution"]).lower().removesuffix("x")
        high = str(bindings["high"]["config"]["resolution"]).lower().removesuffix("x")
        try:
            if float(low) > float(high):
                raise ValueError(
                    f"Experiment {experiment} low resolution exceeds high "
                    f"resolution: {low}x > {high}x")
        except ValueError as error:
            if "low resolution exceeds" in str(error):
                raise


def _validate_sldpc_protocol(
    experiment: str, experiment_cfg: dict[str, Any],
    bindings: dict[str, dict[str, Any]],
) -> None:
    """Validate independent slide-feature and prompt-backbone provenance."""
    from common.backbones import BackboneCapability as Cap, get_spec

    prompt = experiment_cfg.get("prompt_encoder")
    if not isinstance(prompt, dict):
        raise ValueError(
            f"SLDPC experiment {experiment} requires prompt_encoder metadata")
    required = {"name", "weights", "feature_space_id"}
    missing = required.difference(prompt)
    if missing:
        raise ValueError(
            f"SLDPC experiment {experiment} prompt_encoder is missing "
            f"{sorted(missing)}")
    spec = get_spec(prompt["name"])
    needed = {Cap.TEXT_ENCODE, Cap.SOFT_PROMPT}
    if not needed.issubset(spec.capabilities):
        raise ValueError(
            f"SLDPC prompt encoder {spec.name} lacks "
            f"{sorted(item.value for item in needed - spec.capabilities)}")
    prompt_space = str(prompt["feature_space_id"])
    exact_space = prompt_space == spec.feature_space_id
    revision_space = (
        spec.name == "titan"
        and prompt_space.startswith(spec.feature_space_id + "@"))
    if not (exact_space or revision_space):
        raise ValueError(
            f"SLDPC prompt encoder space {prompt_space!r} does not match "
            f"registered encoder {spec.feature_space_id!r}")
    prompt_weights = Path(str(prompt["weights"])).expanduser()
    if not prompt_weights.exists():
        raise FileNotFoundError(
            f"SLDPC prompt encoder weights are missing: {prompt_weights}")

    projection = experiment_cfg.get("slide_projection")
    if not isinstance(projection, dict):
        raise ValueError(
            f"SLDPC experiment {experiment} requires slide_projection")
    mode = str(projection.get("mode", "")).lower()
    if mode not in {"native", "linear", "mlp"}:
        raise ValueError(
            f"SLDPC experiment {experiment} slide_projection.mode must be "
            "native, linear, or mlp")
    source = bindings["bag"]["config"]
    if mode == "native":
        needed_native = {Cap.SLIDE_PROJECT, Cap.PAIRED_SLIDE_TEXT}
        if not needed_native.issubset(spec.capabilities):
            raise ValueError(
                f"SLDPC native projection encoder {spec.name} lacks "
                f"{sorted(item.value for item in needed_native - spec.capabilities)}")
        if str(source["feature_space_id"]) != prompt_space:
            raise ValueError(
                f"SLDPC native projection requires identical slide/prompt "
                f"spaces, got {source['feature_space_id']!r} and "
                f"{prompt_space!r}")
        if (spec.slide_input_dim is not None
                and int(source["feature_dim"]) != spec.slide_input_dim):
            raise ValueError(
                f"SLDPC native projection expects {spec.slide_input_dim}D "
                f"slide embeddings, got {source['feature_dim']}")
    elif spec.shared_dim is None:
        raise ValueError(
            f"SLDPC learned projection requires {spec.name} to declare "
            "shared_dim")


def _validate_muse_protocol(
    experiment: str, experiment_cfg: dict[str, Any],
) -> None:
    """Validate MUSE's prompt tower independently from its patch source."""
    from common.backbones import BackboneCapability as Cap, get_spec

    prompt = experiment_cfg.get("prompt_encoder")
    if not isinstance(prompt, dict):
        raise ValueError(
            f"MUSE experiment {experiment} requires prompt_encoder metadata")
    required = {"name", "weights", "feature_space_id"}
    missing = required.difference(prompt)
    if missing:
        raise ValueError(
            f"MUSE experiment {experiment} prompt_encoder is missing "
            f"{sorted(missing)}")
    spec = get_spec(prompt["name"])
    if Cap.TEXT_ENCODE not in spec.capabilities:
        raise ValueError(
            f"MUSE prompt encoder {spec.name} lacks text_encode")
    if spec.shared_dim is None:
        raise ValueError(
            f"MUSE prompt encoder {spec.name} does not declare shared_dim")
    if str(prompt["feature_space_id"]) != spec.feature_space_id:
        raise ValueError(
            f"MUSE prompt encoder space {prompt['feature_space_id']!r} does "
            f"not match registered encoder {spec.feature_space_id!r}")
    weights = Path(str(prompt["weights"])).expanduser()
    if not weights.exists():
        raise FileNotFoundError(
            f"MUSE prompt encoder weights are missing: {weights}")


def _resolve_feature_bindings(
    protocol: dict[str, Any], experiment: str, experiment_cfg: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    bindings = {}
    for role, source in experiment_cfg["features"].items():
        if source not in protocol["feature_sources"]:
            raise KeyError(f"Experiment {experiment}: unknown feature source {source}")
        bindings[role] = {
            "source": source,
            "config": protocol["feature_sources"][source],
        }
    return bindings


def _muse_prompt_paths(
    cohort: str, cohort_cfg: dict[str, Any], output_dir: Path
) -> list[str]:
    explicit = _explicit_prompt(cohort_cfg, "muse")
    if explicit is not None:
        return list(explicit) if isinstance(explicit, list) else [explicit]
    generated = cohort_cfg.get("_generated_prompt_assets", {})
    if str(cohort_cfg.get("prompt_precedence", "upstream")).lower() == "upstream" \
            and cohort_cfg.get("muse_prompt_csvs"):
        return [
            str(_absolute_repo_path(value))
            for value in cohort_cfg["muse_prompt_csvs"]
        ]
    if generated.get("muse"):
        return list(generated["muse"])
    if cohort_cfg.get("muse_prompt_csvs"):
        return [
            str(_absolute_repo_path(value))
            for value in cohort_cfg["muse_prompt_csvs"]
        ]
    return [
        str(output_dir / "data" / cohort / "muse_prompts"
            / f"generated_new_{index}.csv")
        for index in range(len(cohort_cfg["labels"]))
    ]


def _explicit_prompt(cohort_cfg: dict[str, Any], method: str) -> Any:
    """Return the path a cohort's ``prompts:`` block names for one method.

    The block maps a method name to the prompt file that method reads, so the
    prompt in force for a run is stated in the protocol rather than resolved::

        cohorts:
          ubc_ocean:
            prompts:
              focus: text_prompts/focus/UBC_OCEAN_two_scale_text_prompt.csv
              mscpt: train_data/gpt/description/UBC-OCEAN.json
              muse:
                - text_prompts/muse/ubc_ocean/generated_new_0.csv
                - text_prompts/muse/ubc_ocean/generated_new_1.csv

    Paths are repository-relative or absolute. A named file that does not exist
    is an error rather than a silent fallback: a prompt the author asked for and
    did not get would change what the model reads without saying so.

    Returns:
        A resolved path, a list of resolved paths, or None when the method has
        no explicit entry.
    """
    prompts = cohort_cfg.get("prompts")
    if not prompts:
        return None
    if not isinstance(prompts, dict):
        raise ValueError("cohort `prompts` must map method names to paths")
    if method not in prompts:
        return None

    value = prompts[method]
    values = value if isinstance(value, (list, tuple)) else [value]
    resolved: list[str] = []
    for item in values:
        path = _absolute_repo_path(item)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"prompts.{method} names a file that does not exist: {path}")
        resolved.append(str(path))
    return resolved if isinstance(value, (list, tuple)) else resolved[0]


def _prompt_asset(
    cohort_cfg: dict[str, Any], method: str, legacy_key: str | None = None,
) -> Any:
    """Resolve the prompt asset a method reads, honouring declared precedence.

    Most upstream repositories ship their prompts as an explicit per-task file --
    FOCUS reads a ``class_name,low_res_prompt,high_res_prompt`` CSV, MUSE reads
    per-class description CSVs, MSCPT reads a GPT description JSON. When a cohort
    declares that published asset, it is what the paper actually used, so it wins
    by default; the ``prompt_spec`` compiler is the fallback for tasks that have
    no published prompts.

    Set ``prompt_precedence: generated`` on the cohort (or on the protocol, as a
    default for every cohort) to deliberately prefer compiled prompts instead --
    for example to compare methods under one uniform prompt style. The choice is
    recorded per config in ``prompt_provenance``.
    """
    # An explicit `prompts:` entry names the file for this method outright and
    # always wins. Nothing is inferred, so what a run embeds is readable from
    # the cohort definition without tracing resolution rules.
    explicit = _explicit_prompt(cohort_cfg, method)
    if explicit is not None:
        return explicit

    generated = cohort_cfg.get("_generated_prompt_assets", {})
    precedence = str(cohort_cfg.get("prompt_precedence", "upstream")).lower()
    if precedence not in {"upstream", "generated"}:
        raise ValueError(
            f"prompt_precedence must be 'upstream' or 'generated', "
            f"got {precedence!r}")

    upstream: str | None = None
    if legacy_key and cohort_cfg.get(legacy_key) is not None:
        candidate = _absolute_repo_path(cohort_cfg[legacy_key])
        if Path(candidate).exists():
            upstream = str(candidate)

    order = ((upstream, generated.get(method)) if precedence == "upstream"
             else (generated.get(method), upstream))
    for asset in order:
        if asset is not None:
            return asset

    raise ValueError(
        f"Task has no prompt_spec-generated {method} asset"
        + (f" or {legacy_key}" if legacy_key else ""))


# The cohort key naming each method's published, paper-native prompt asset.
_UPSTREAM_PROMPT_KEYS = {
    "focus": "focus_prompt_csv",
    "vila_mil": "focus_prompt_csv",
    "mscpt": "mscpt_prompt_json",
    "maple": "maple_prompt_json",
    "cod_mil": "cod_prompt_json",
    "slip": "slip_tissue_json",
    "sldpc": "sldpc_prompt_yaml",
    "muse": "muse_prompt_csvs",
    "convlm": "convlm_attribute_prompt_json",
}

# Methods that build their prompt from `classnames` rather than reading a file.
# Methods whose entire embedded text is built from `classnames`. WSI-FiVE
# left this set once it began embedding a clinical question set and
# per-slide report text; its provenance comes from that question set.
_CLASSNAME_PROMPT_METHODS = frozenset({"pathpt"})


def _prompt_provenance(cohort_cfg: dict[str, Any], method: str) -> str:
    """Describe which prompt source *this method* actually resolved to.

    Reported per method, not per cohort: a cohort commonly has a published asset
    for one method and only a compiled one for another, and recording the
    cohort's best case for every method would misreport what a run embedded.
    """
    if _explicit_prompt_declared(cohort_cfg, method):
        return "explicit"
    if method in _CLASSNAME_PROMPT_METHODS:
        return "classname_template"

    generated = cohort_cfg.get("_generated_prompt_assets", {})
    precedence = str(cohort_cfg.get("prompt_precedence", "upstream")).lower()
    legacy_key = _UPSTREAM_PROMPT_KEYS.get(method)
    has_upstream = bool(legacy_key and cohort_cfg.get(legacy_key) is not None)

    if precedence == "upstream" and has_upstream:
        return "upstream"
    if method in generated:
        return generated.get("provenance", "generated")
    return "upstream" if has_upstream else "generated"


def _explicit_prompt_declared(cohort_cfg: dict[str, Any], method: str) -> bool:
    prompts = cohort_cfg.get("prompts")
    return isinstance(prompts, dict) and method in prompts


def _augment_provenance(provenance: str, cohort_cfg: dict[str, Any],
                        method: str) -> str:
    """Downgrade a provenance claim when part of the bank was authored here.

    CoD-MIL embeds a class chain *and* a normal-tissue bank. Upstream publishes
    the latter for kidney only, so another cohort supplies its own -- and a run
    whose contrast set was written for this benchmark is not the published
    prompt condition, even though its class prompts are. Reporting plain
    ``upstream`` there would overstate fidelity in the one field a reader would
    check.
    """
    if method == "wsi_five":
        source = cohort_cfg.get("wsi_five_questions_json")
        if not source:
            return provenance
        try:
            with _absolute_repo_path(source).open(encoding="utf-8") as handle:
                declared = json.load(handle).get("_provenance")
        except (OSError, ValueError, AttributeError):
            return provenance
        if declared == "generated":
            return "generated_questions_with_free_text_reports"
        return "upstream_questions_and_answers"
    if provenance not in {"upstream", "explicit"}:
        return provenance

    # A declared path does not make its contents upstream. CoD-MIL publishes a
    # chain for kidney only, so the BRCA and NSCLC chains are authored here and
    # say so in the file. Read the asset rather than trusting the declaration.
    if method == "cod_mil":
        chain = cohort_cfg.get("cod_prompt_json")
        if chain:
            try:
                with _absolute_repo_path(chain).open(encoding="utf-8") as handle:
                    if json.load(handle).get("_provenance") == "generated":
                        provenance = "generated"
            except (OSError, ValueError, AttributeError):
                pass

    source = {
        "cod_mil": "cod_normal_structures_json",
        # WSI-FiVE cross-attends a fixed set of clinical questions. The released
        # set is lung-specific ("spread through air spaces", "pleural invasion",
        # "excluding the current lung organ"), so any other cohort supplies an
        # authored one -- and a run whose questions were written here is not the
        # published prompt condition even when its answers are.
        "wsi_five": "wsi_five_questions_json",
    }.get(method)
    source = cohort_cfg.get(source) if source else None
    if not source:
        return provenance
    try:
        with _absolute_repo_path(source).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return provenance
    if isinstance(payload, dict) and payload.get("_provenance") == "generated":
        suffix = ("chain_with_generated_normal_tissue" if method == "cod_mil"
                  else "reports_with_generated_questions")
        return f"{provenance}_{suffix}"
    return provenance


def _method_config(
    experiment: str, experiment_cfg: dict[str, Any], method: str,
    protocol: dict[str, Any], cohort: str, cohort_cfg: dict[str, Any],
    shot: int, output_dir: Path,
) -> dict[str, Any]:
    bindings = _resolve_feature_bindings(protocol, experiment, experiment_cfg)
    _validate_feature_roles(method, experiment, bindings)
    primary = bindings.get(
        "bag", bindings.get("high", bindings.get("tiles")))
    assert primary is not None
    feature_cfg = primary["config"]
    cfg = _common_config(
        protocol, cohort, cohort_cfg, shot, bindings, output_dir)
    cfg.update({
        "method": method,
        "experiment": experiment,
        "epochs": int(experiment_cfg["epochs"]),
        "prompt_provenance": _augment_provenance(
            _prompt_provenance(cohort_cfg, method), cohort_cfg, method),
        "results_dir": str(
            REPO_ROOT / "results"
            / protocol.get("results_namespace", "tcga_benchmark") / experiment
            / cohort / f"{shot}shot"),
    })

    if method == "pathpt":
        cfg.update({
            "lr": 1e-4,
            "n_ctx": 32,
            "prompt_init": "template",
            "aux_weight": 0.5,
            "learnable": "token",
            "vision_only": False,
            "vision_grad": True,
            "use_aug": False,
            "loss_weight": [1.0, 0.5, 0.1],
            "feature_root": _feature_root(bindings["bag"]["config"], cohort),
            "feature_path_column": _feature_column(bindings["bag"]["source"]),
            "patch_num": None,
            "loader_mode": "h5",
            "prompt_source": "pathpt_classname_template",
        })
    elif method == "muse":
        from common.backbones import get_spec

        prompt_paths = _muse_prompt_paths(cohort, cohort_cfg, output_dir)
        prompt_cfg = experiment_cfg["prompt_encoder"]
        prompt_spec = get_spec(prompt_cfg["name"])
        prompt_weights = str(
            Path(str(prompt_cfg["weights"])).expanduser().resolve())
        patch_weights_value = feature_cfg.get("encoder_weights")
        patch_weights = (
            str(Path(str(patch_weights_value)).expanduser().resolve())
            if patch_weights_value else None)
        cfg.update({
            "backbone": prompt_spec.name,
            "backbone_weights": prompt_weights,
            "prompt_feature_space_id": prompt_cfg["feature_space_id"],
            "encoder": {
                "name": prompt_spec.name,
                "weights": prompt_weights,
                "feature_space_id": prompt_cfg["feature_space_id"],
                "feature_dim": int(prompt_spec.shared_dim),
                "initialization": prompt_cfg.get(
                    "initialization", "pretrained"),
            },
            "patch_encoder": {
                "input_kind": "patch_bag",
                "name": feature_cfg["backbone"],
                "weights": patch_weights,
                "feature_space_id": feature_cfg["feature_space_id"],
                "feature_dim": int(feature_cfg["feature_dim"]),
                "feature_key": feature_cfg["feature_key"],
                "resolution": feature_cfg["resolution"],
                "path_template": str(
                    _feature_path(feature_cfg, "{slide_id}", cohort)),
            },
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": True,
            "es_patience": 20,
            "es_stop_epoch": 50,
            "embed_dim": int(prompt_spec.shared_dim),
            "num_heads": 8,
            "num_experts": 8,
            "num_selected": 2,
            "retrieval_k": 8,
            "dropout": 0.25,
            "text_batch_size": 64,
            "prompt_csvs": dict(zip(cohort_cfg["classnames"], prompt_paths)),
            "prompt_source": "muse_description_csvs",
            "conch_ckpt": (
                prompt_weights if prompt_spec.name == "conch" else None),
            "data_folder_s": _feature_root(bindings["bag"]["config"], cohort),
            "feature_path_column": _feature_column(bindings["bag"]["source"]),
        })
    elif method == "focus":
        cfg.update({
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "bag_loss": "ce",
            "drop_out": True,
            "early_stopping": True,
            "es_patience": 20,
            "es_stop_epoch": 40,
            "prototype_number": 16,
            "mode": "transformer",
            "loader_mode": "transformer",
            "data_folder_s": _feature_root(bindings["low"]["config"], cohort),
            "data_folder_l": _feature_root(bindings["high"]["config"], cohort),
            "feature_path_column_s": _feature_column(bindings["low"]["source"]),
            "feature_path_column_l": _feature_column(bindings["high"]["source"]),
            "text_prompt_path": _prompt_asset(
                cohort_cfg, "focus", "focus_prompt_csv"),
            "prompt_source": "focus_two_scale_csv",
            "conch_ckpt": cfg["backbone_weights"],
        })
    elif method == "mscpt":
        prompt_path = Path(_prompt_asset(
            cohort_cfg, "mscpt", "mscpt_prompt_json")).resolve()
        if prompt_path.parent.name != "description":
            raise ValueError(
                f"MSCPT prompt must be inside a description directory: {prompt_path}")
        cfg.update({
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "dataset_name": prompt_path.stem,
            "n_tpro": 2,
            "n_vpro": 2,
            "n_set": 5,
            "n_high": 10,
            "num_k": 100,
            "input_mode": "precomputed_shared_features",
            "low_input_kind": "patch_feature_bag",
            "high_input_kind": "patch_feature_bag",
            "feat_data_dir": _feature_root(bindings["low"]["config"], cohort),
            "selected_5x_dir": _feature_root(bindings["high"]["config"], cohort),
            "feature_path_column_s": _feature_column(bindings["low"]["source"]),
            "feature_path_column_l": _feature_column(bindings["high"]["source"]),
            "gpt_dir": str(prompt_path.parent.parent),
            "description_prompt_path": str(prompt_path),
            "prompt_source": "mscpt_multiscale_description_json",
        })
    elif method == "maple":
        cfg.update({
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "early_stopping": True,
            "es_patience": 20,
            "es_stop_epoch": 50,
            "prompt_mode": "attribute",
            "attr_edge_topk": 7,
            "entity_weight": 0.3,
            "pos_ratio": 0.8,
            "neg_ratio": 0.2,
            "n_ctx": 0,
            "csc": False,
            "all_ctx_trainable": False,
            "p_drop_out": 0.0,
            "p_bag_drop_out": 0.0,
            "loader_mode": "transformer",
            "data_folder_s": _feature_root(bindings["low"]["config"], cohort),
            "data_folder_l": _feature_root(bindings["high"]["config"], cohort),
            "feature_path_column_s": _feature_column(bindings["low"]["source"]),
            "feature_path_column_l": _feature_column(bindings["high"]["source"]),
            "text_prompt_path": _prompt_asset(
                cohort_cfg, "maple", "maple_prompt_json"),
            "prompt_source": "maple_attribute_json",
        })
    elif method == "vila_mil":
        cfg.update({
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "bag_loss": "ce",
            "drop_out": True,
            "early_stopping": True,
            "es_patience": 20,
            "es_stop_epoch": 80,
            "prototype_number": 16,
            "mode": "transformer",
            "loader_mode": "transformer",
            "data_folder_s": _feature_root(bindings["low"]["config"], cohort),
            "data_folder_l": _feature_root(bindings["high"]["config"], cohort),
            "feature_path_column_s": _feature_column(bindings["low"]["source"]),
            "feature_path_column_l": _feature_column(bindings["high"]["source"]),
            "text_prompt_path": _prompt_asset(
                cohort_cfg, "vila_mil", "focus_prompt_csv"),
            "prompt_source": "vila_two_scale_csv",
        })
    elif method == "cod_mil":
        cfg.update({
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": True,
            "es_patience": 20,
            "es_stop_epoch": 50,
            "batch_size": 1,
            "data_folder_s": _feature_root(bindings["low"]["config"], cohort),
            "data_folder_l": _feature_root(bindings["high"]["config"], cohort),
            "feature_path_column_s": _feature_column(bindings["low"]["source"]),
            "feature_path_column_l": _feature_column(bindings["high"]["source"]),
            "text_prompt_path": _prompt_asset(
                cohort_cfg, "cod_mil", "cod_prompt_json"),
            "text_prompt_features": (
                str(_absolute_repo_path(cohort_cfg["cod_prompt_features"]))
                if cohort_cfg.get("cod_prompt_features") else None),
            # Organ-specific normal structures for the second diagnostic chain.
            # Absent, the runtime bank falls back to the organ-independent rows
            # alone, which is a weaker contrast set than the released one.
            "normal_structures_json": (
                str(_absolute_repo_path(cohort_cfg["cod_normal_structures_json"]))
                if cohort_cfg.get("cod_normal_structures_json") else None),
            "prompt_encoding": (
                "precomputed" if cohort_cfg.get("cod_prompt_features")
                else "runtime_cached"),
            "text_feature_space_id": feature_cfg["feature_space_id"],
            "cross_mag_map_dir": str(
                output_dir / "maps" / "cod_mil"
                / f"{bindings['low']['source']}__{bindings['high']['source']}"),
            "prompt_source": (
                "cod_chain_precomputed_clip_rn50"
                if cohort_cfg.get("cod_prompt_features")
                else "cod_chain_runtime_clip_rn50"),
        })
    elif method == "top":
        cfg.update({
            "clip_arch": "RN50",
            "lr": 0.02,
            "lr_TB": 0.02,
            "lr_IB": 0.02,
            "weight_decay": 0.0,
            "early_stopping": False,
            "n_ctx_bag": 4,
            "n_ctx_inst": 4,
            "ctx_init_bag": "",
            "ctx_init_inst": "",
            # TOP's instance branch is prototype-based: 26 task-agnostic tissue
            # phenotypes, not the bag class names. Declared under the cohort's
            # `prompts` block as `top_instance` / `top_bag`.
            "instance_prompt_path": str(_absolute_repo_path(
                cohort_cfg.get("prompts", {}).get(
                    "top_instance", "text_prompts/top/instance_prototypes.json"))),
            "csc": True,
            "p_drop_out": 0.2,
            "p_bag_drop_out": 0.2,
            "weight_lossA": 25,
            "pooling_strategy": "learnablePrompt_multi",
            "data_folder_s": _feature_root(bindings["bag"]["config"], cohort),
            "feature_path_column": _feature_column(bindings["bag"]["source"]),
            "prompt_source": "top_instance_prototypes_and_bag_context",
        })
        top_bag = cohort_cfg.get("prompts", {}).get("top_bag")
        if top_bag:
            cfg["bag_prompt_path"] = str(_absolute_repo_path(top_bag))
    elif method == "slip":
        tissue_path = Path(_prompt_asset(
            cohort_cfg, "slip", "slip_tissue_json")).resolve()
        with tissue_path.open(encoding="utf-8") as handle:
            tissue_classnames = json.load(handle)
        # A marked asset wraps the vocabulary; an upstream one is a bare list.
        if isinstance(tissue_classnames, dict):
            tissue_classnames = tissue_classnames["tissues"]
        cfg.update({
            "lr": 2e-3,
            "weight_decay": 0.0,
            "early_stopping": False,
            "context_size": 1,
            "context_gain": 0.01,
            "topk": 50,
            "temp": 0.01,
            "image_size": 224,
            "text_templates": ["a histopathology image of {}."],
            "tissue_classnames_path": str(tissue_path),
            "tissue_classnames": tissue_classnames,
            "data_folder_s": _feature_root(bindings["bag"]["config"], cohort),
            "feature_path_column": _feature_column(bindings["bag"]["source"]),
            "prompt_source": "slip_class_and_tissue_context",
        })
    elif method == "wsi_five":
        cfg.update({
            "lr": 8e-6,
            "weight_decay": 0.05,
            "early_stopping": False,
            "batch_size": 1,
            "num_frames": 2048,
            "T_mit": 8,
            "is_img_pth": True,
            "feature_root": _feature_root(bindings["bag"]["config"], cohort),
            "feature_path_column": _feature_column(bindings["bag"]["source"]),
            "report_csv": (
                str(_absolute_repo_path(cohort_cfg["wsi_report_csv"]))
                if cohort_cfg.get("wsi_report_csv") else None),
            "default_report": cohort_cfg.get(
                "_generated_prompt_assets", {}).get(
                    "wsi_five_default_report"),
            "require_report": not bool(cohort_cfg.get(
                "_generated_prompt_assets", {}).get(
                    "wsi_five_default_report")),
            "clinicalbert_weights": os.environ.get(
                "CLINICALBERT_CKPT",
                "/path/to/model-cache/huggingface/hub/"
                "models--emilyalsentzer--Bio_ClinicalBERT/snapshots/"
                "d5892b39a4adaed74b92212a44081509db72f87b"),
            # The released question set is lung-specific. A cohort outside lung
            # must declare its own, and that set is generated, not the authors'.
            "clinical_questions": (
                str(_absolute_repo_path(cohort_cfg["wsi_five_questions_json"]))
                if cohort_cfg.get("wsi_five_questions_json") else None),
            "prompt_source": (
                "wsi_five_clinicalbert_reports_and_classnames"
                if cohort_cfg.get("wsi_report_csv")
                else "wsi_five_class_agnostic_task_context"),
        })
    elif method == "sldpc":
        from common.backbones import get_spec

        prompt_path = Path(_prompt_asset(
            cohort_cfg, "sldpc", "sldpc_prompt_yaml")).resolve()
        prompt_cfg = experiment_cfg["prompt_encoder"]
        prompt_spec = get_spec(prompt_cfg["name"])
        prompt_weights = str(
            Path(str(prompt_cfg["weights"])).expanduser().resolve())
        projection_cfg = dict(experiment_cfg["slide_projection"])
        projection_mode = str(projection_cfg["mode"]).lower()
        projection_cfg.update({
            "mode": projection_mode,
            "input_dim": int(feature_cfg["feature_dim"]),
            "output_dim": int(prompt_spec.shared_dim),
            "trainable": projection_mode != "native",
        })
        cfg.update({
            "backbone": prompt_spec.name,
            "backbone_weights": prompt_weights,
            "prompt_feature_space_id": prompt_cfg["feature_space_id"],
            "encoder": {
                "name": prompt_spec.name,
                "weights": prompt_weights,
                "feature_space_id": prompt_cfg["feature_space_id"],
                "feature_dim": int(prompt_spec.shared_dim),
                "initialization": prompt_cfg.get(
                    "initialization", "pretrained"),
            },
            "slide_projection": projection_cfg,
            "prompt_encoder_model_id": prompt_cfg.get("model_id"),
            "prompt_encoder_revision": prompt_cfg.get("revision"),
            "local_files_only": bool(
                prompt_cfg.get("local_files_only", True)),
            "stage1_epochs": 50,
            "stage2_epochs": 50,
            "stage1_lr": 1e-3,
            "stage2_lr": 1e-3,
            "weight_decay": 0.0,
            "n_ctx": 8,
            "csc": False,
            "class_token_position": "end",
            "omega": 0.8,
            "tau": 0.07,
            "topk": 8,
            "stage1_apply_tau": False,
            "early_stopping": False,
            "monitor_metric": "F1",
            "prompt_reference_yaml": str(prompt_path),
            "prompt_source": "sldpc_learnable_context_from_classnames",
        })
    elif method == "convlm":
        cfg.update({
            "seen_class_indices": list(range(len(cohort_cfg["labels"]))),
            "image_root": _feature_root(bindings["tiles"]["config"], cohort),
            "slide_tile_path_column": _feature_column(bindings["tiles"]["source"]),
            "image_layout": "per_slide_directory",
            "attribute_embeddings": (
                str(_absolute_repo_path(
                    cohort_cfg["convlm_attribute_embeddings"]))
                if cohort_cfg.get("convlm_attribute_embeddings") else None),
            "attribute_prompt_path": (
                None if cohort_cfg.get("convlm_attribute_embeddings")
                else _prompt_asset(
                    cohort_cfg, "convlm", "convlm_attribute_prompt_json")),
            "attribute_encoder": dict(
                experiment_cfg.get("attribute_encoder", {})),
            "attribute_feature_space_id": "hf:wisdomik/QuiltNet-B-32",
            "image_size": 448,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "keep_rate": 0.7,
            "batch_size": 1,
            "max_tiles_per_slide": 64,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "lr_milestones": [10, 20, 30],
            "loss_global_alignment": 1.0,
            "loss_sr": 1.0,
            "early_stopping": False,
            "prompt_source": (
                "convlm_quiltnet_attribute_embeddings"
                if cohort_cfg.get("convlm_attribute_embeddings")
                else "convlm_runtime_quiltnet_attribute_prompts"),
        })
    elif method == "composite":
        cfg.update({
            "selectors": [],
            "prompts": {
                "coop_flat": {"enabled": True, "n_ctx": 16},
                "fusion": {"mode": "average"},
            },
            "aggregators": {
                "attn_pool": {"enabled": True, "hidden_dim": 192},
                "fusion": "logit_ensemble",
                "logit_mode": "mean",
            },
            "recipe": {"type": "focus", "lr": 1e-4,
                       "epochs": int(experiment_cfg["epochs"])},
            "lr": 1e-4,
            "early_stopping": True,
            "es_patience": 20,
            "es_stop_epoch": 50,
            "data_folder_s": _feature_root(bindings["bag"]["config"], cohort),
            "feature_path_column": _feature_column(bindings["bag"]["source"]),
            "prompt_source": "composite_coop_classname_context",
        })
    else:
        raise KeyError(f"No feature-registry template for method {method}")
    cfg.update(experiment_cfg.get("config_overrides", {}))
    return cfg


def _require_asset(value: Any, label: str) -> Path:
    if not value:
        raise ValueError(f"Missing required {label}")
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _nonempty_strings(values: Any, label: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list")
    output = [str(value).strip() for value in values]
    if any(not value for value in output):
        raise ValueError(f"{label} contains an empty prompt")
    return output


def _validate_focus_prompt(cfg: dict[str, Any]) -> None:
    path = _require_asset(cfg.get("text_prompt_path"), "FOCUS prompt CSV")
    frame = pd.read_csv(path)
    required = {"class_name", "low_res_prompt", "high_res_prompt"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: FOCUS prompt columns must include {sorted(required)}")
    labels = list(cfg["label_dict"])
    if frame["class_name"].astype(str).tolist() != labels:
        raise ValueError(
            f"{path}: FOCUS class order must be {labels}, got "
            f"{frame['class_name'].astype(str).tolist()}")
    for column in ("low_res_prompt", "high_res_prompt"):
        values = frame[column].astype(str).str.strip()
        if values.eq("").any() or values.str.lower().eq("nan").any():
            raise ValueError(f"{path}: FOCUS {column} contains an empty prompt")


def _validate_maple_prompt(cfg: dict[str, Any]) -> None:
    path = _require_asset(cfg.get("text_prompt_path"), "MAPLE prompt JSON")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    classnames = set(cfg["classnames"])
    for level in ("low", "high"):
        block = payload.get(level)
        if not isinstance(block, dict):
            raise ValueError(f"{path}: MAPLE requires a {level!r} prompt block")
        if not str(block.get("tumor", "")).strip():
            raise ValueError(f"{path}: MAPLE {level}.tumor is empty")
        global_info = block.get("global_info")
        if not isinstance(global_info, dict) or set(global_info) != classnames:
            raise ValueError(
                f"{path}: MAPLE {level}.global_info classes must be "
                f"{sorted(classnames)}")
        entities = block.get("entities")
        if not isinstance(entities, list) or not entities:
            raise ValueError(f"{path}: MAPLE {level}.entities must be non-empty")
        for index, entity in enumerate(entities):
            if not str(entity.get("name", "")).strip():
                raise ValueError(f"{path}: MAPLE {level} entity {index} has no name")
            if not str(entity.get("general_feature", "")).strip():
                raise ValueError(
                    f"{path}: MAPLE {level} entity {index} has no general_feature")
            attributes = entity.get("attributes")
            if not isinstance(attributes, dict) or set(attributes) != classnames:
                raise ValueError(
                    f"{path}: MAPLE {level} entity {index} attributes must "
                    f"cover {sorted(classnames)}")


def _validate_mscpt_prompt(cfg: dict[str, Any]) -> None:
    path = _require_asset(
        cfg.get("description_prompt_path"), "MSCPT description prompt JSON")
    expected = (
        Path(cfg["gpt_dir"]).expanduser().resolve()
        / "description" / f"{cfg['dataset_name']}.json"
    )
    if path != expected:
        raise ValueError(
            f"MSCPT description_prompt_path {path} does not match native "
            f"gpt_dir/dataset_name resolution {expected}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    labels = set(cfg["label_dict"])
    prompt_labels = {key for key in payload if not str(key).startswith("_")}
    if prompt_labels != labels:
        raise ValueError(
            f"{path}: MSCPT prompt labels {sorted(prompt_labels)} do not "
            f"match task labels {sorted(labels)}")
    small_counts, big_counts = set(), set()
    for label in cfg["label_dict"]:
        block = payload[label]
        small = _nonempty_strings(block.get("small_mag"), f"{path}:{label}.small_mag")
        big = _nonempty_strings(block.get("big_mag"), f"{path}:{label}.big_mag")
        small_counts.add(len(small))
        big_counts.add(len(big))
    if len(small_counts) != 1 or len(big_counts) != 1:
        raise ValueError(f"{path}: MSCPT prompt counts must match across classes")
    expected_high = int(cfg.get("n_high", 10))
    if small_counts != {expected_high}:
        raise ValueError(
            f"{path}: MSCPT requires {expected_high} small_mag prompts per "
            f"class for n_high={expected_high}, got {sorted(small_counts)}")


def _validate_muse_prompts(cfg: dict[str, Any]) -> None:
    prompt_csvs = cfg.get("prompt_csvs")
    if not isinstance(prompt_csvs, dict):
        raise ValueError("MUSE prompt_csvs must map classnames to CSV paths")
    classnames = list(cfg["classnames"])
    if list(prompt_csvs) != classnames:
        raise ValueError(
            f"MUSE prompt_csvs order must be {classnames}, got {list(prompt_csvs)}")
    for classname, value in prompt_csvs.items():
        path = _require_asset(value, f"MUSE prompt CSV for {classname}")
        from methods.muse.adapter import _csv_descriptions
        _csv_descriptions(path)


def _validate_muse_feature_config(cfg: dict[str, Any]) -> None:
    """Validate offline patch provenance independently from prompt encoding."""
    from common.backbones import get_spec

    patch = cfg.get("patch_encoder")
    if not isinstance(patch, dict):
        raise ValueError("MUSE requires explicit patch_encoder provenance")
    required = {
        "input_kind", "name", "weights", "feature_space_id", "feature_dim",
        "feature_key", "resolution", "path_template",
    }
    missing = required.difference(patch)
    if missing:
        raise ValueError(f"MUSE patch_encoder is missing {sorted(missing)}")
    if patch["input_kind"] != "patch_bag":
        raise ValueError("MUSE patch_encoder.input_kind must be patch_bag")
    for field in ("feature_space_id", "feature_key"):
        if patch[field] != cfg[field]:
            raise ValueError(
                f"MUSE patch_encoder.{field} must match input {field}")
    if int(patch["feature_dim"]) != int(cfg["feature_dim"]):
        raise ValueError(
            "MUSE patch_encoder.feature_dim must match input feature_dim")
    prompt_spec = get_spec(cfg["backbone"])
    if int(cfg["embed_dim"]) != int(prompt_spec.shared_dim or -1):
        raise ValueError(
            "MUSE embed_dim must match the prompt encoder shared dimension")


# CoD-MIL's released kidney bank carries 21 normal-tissue prompts (6 organ
# structures + 15 organ-independent phenotypes). The organ-independent set is
# the floor any cohort can meet, so it is the minimum a bank must supply.
MIN_BACKGROUND_PROMPTS = 15


def _validate_cod_prompt(cfg: dict[str, Any]) -> None:
    path = _require_asset(cfg.get("text_prompt_path"), "CoD-MIL chain JSON")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    # Keys beginning with an underscore are metadata (provenance, notes), not
    # classes. Comparing them against classnames would reject a marked asset.
    payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    if list(payload) != list(cfg["classnames"]):
        raise ValueError(
            f"{path}: CoD-MIL class order must be {cfg['classnames']}")
    for classname, block in payload.items():
        _nonempty_strings(block.get("broad"), f"{path}:{classname}.broad")
        _nonempty_strings(block.get("specific"), f"{path}:{classname}.specific")
    if not cfg.get("text_prompt_features"):
        from common.backbones import BackboneCapability as Cap, get_spec

        if cfg.get("prompt_encoding") != "runtime_cached":
            raise ValueError(
                "CoD-MIL without text_prompt_features requires "
                "prompt_encoding=runtime_cached")
        spec = get_spec(cfg["backbone"])
        if Cap.TEXT_ENCODE not in spec.capabilities:
            raise ValueError(
                f"CoD-MIL runtime prompt encoder {spec.name} lacks text_encode")
        if int(spec.shared_dim or -1) != int(cfg["feature_dim"]):
            raise ValueError(
                "CoD-MIL runtime text width must match patch feature_dim")
        if cfg.get("text_feature_space_id") != cfg.get("feature_space_id"):
            raise ValueError(
                "CoD-MIL runtime prompt and patch feature spaces must match")
        _require_asset(cfg.get("backbone_weights"), "CoD-MIL text encoder weights")
        return
    tensor_path = _require_asset(
        cfg.get("text_prompt_features"), "CoD-MIL prompt features")
    prompt_payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    embedded_space = None
    if isinstance(prompt_payload, dict):
        embedded_space = prompt_payload.get("feature_space_id")
        prompt_payload = prompt_payload.get("embeddings", prompt_payload.get("features"))
    if not torch.is_tensor(prompt_payload) or prompt_payload.ndim != 2:
        raise ValueError(f"{tensor_path}: expected a rank-2 prompt tensor")
    # The released kidney bank is 3 low + 3 high + 21 normal-tissue rows, and
    # the model slices [2C:-1] for its auxiliary contrastive branch. One spare
    # row satisfies the arithmetic while leaving that branch with a single
    # negative, which is not the published objective -- so require a bank, not
    # a row. MIN_BACKGROUND_PROMPTS is the organ-independent set alone.
    minimum = 2 * int(cfg["n_classes"]) + MIN_BACKGROUND_PROMPTS
    if prompt_payload.shape[0] < minimum:
        background = prompt_payload.shape[0] - 2 * int(cfg["n_classes"])
        raise ValueError(
            f"{tensor_path}: CoD-MIL needs {2 * int(cfg['n_classes'])} class "
            f"prompts plus at least {MIN_BACKGROUND_PROMPTS} normal-tissue "
            f"prompts for its contrastive branch; this bank has "
            f"{prompt_payload.shape[0]} rows ({background} background).")
    if prompt_payload.shape[1] != int(cfg["feature_dim"]):
        raise ValueError(
            f"{tensor_path}: prompt width {prompt_payload.shape[1]} does not "
            f"match feature_dim {cfg['feature_dim']}")
    declared_space = cfg.get("text_feature_space_id")
    if embedded_space and declared_space != embedded_space:
        raise ValueError(
            f"{tensor_path}: embedded feature space {embedded_space!r} does "
            f"not match {declared_space!r}")


def _validate_top_prompts(cfg: dict[str, Any]) -> None:
    """Check TOP's prototype bank and, when declared, its bag prompts.

    TOP scores each instance prototype against the bag class prompts, so the
    prototype bank sizes the instance branch while the class list sizes the bag
    branch. They are independent, and a bank that silently collapses to the
    class count reproduces the defect this validation exists to prevent.
    """
    path = _require_asset(cfg.get("instance_prompt_path"),
                          "TOP instance prototype bank")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    prototypes = payload["prototypes"] if isinstance(payload, dict) else payload
    if len(prototypes) < 2:
        raise ValueError(f"{path}: TOP needs at least two instance prototypes")
    missing = [item for item in prototypes if "prompt" not in item]
    if missing:
        raise ValueError(f"{path}: every prototype needs a 'prompt' field")

    bag_path = cfg.get("bag_prompt_path")
    if not bag_path:
        return
    resolved = _require_asset(bag_path, "TOP bag prompt file")
    with resolved.open(encoding="utf-8") as handle:
        prompts = json.load(handle)["prompts"]
    labels = list(cfg["label_dict"])
    absent = [label for label in labels if label not in prompts]
    if absent:
        raise ValueError(
            f"{resolved}: TOP bag prompts missing {absent}; "
            f"file provides {sorted(prompts)}")


def _validate_slip_prompt(cfg: dict[str, Any]) -> None:
    path = _require_asset(
        cfg.get("tissue_classnames_path"), "SLIP tissue prompt JSON")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    # SLIP routes patches through a vocabulary of *tissue phenotypes*, not one
    # bank per class: `slidecoop.py` builds tissue weights from these and takes
    # a top-k similarity over them. Upstream ships 17 phenotypes for its TCGA
    # lung task against 2 classes, so tying the count to `n_classes` would
    # reject the authors' own asset and force a per-class substitute that
    # collapses the routing.
    tissues = payload.get("tissues") if isinstance(payload, dict) else payload
    if not isinstance(tissues, list) or not tissues:
        raise ValueError(f"{path}: SLIP needs a list of tissue phenotypes")
    if tissues != cfg.get("tissue_classnames"):
        raise ValueError("SLIP tissue_classnames must exactly match its prompt JSON")
    if len(tissues) < int(cfg["n_classes"]):
        raise ValueError(
            f"{path}: SLIP tissue vocabulary ({len(tissues)}) is smaller than "
            f"the class count ({cfg['n_classes']}); routing would be degenerate")
    for index, prompt in enumerate(tissues):
        _nonempty_strings(prompt, f"{path}:tissue {index}")


def _validate_sldpc_prompt(cfg: dict[str, Any]) -> None:
    path = _require_asset(
        cfg.get("prompt_reference_yaml"), "SLDPC prompt reference YAML")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if (not isinstance(prompts, dict)
            or set(prompts) != set(cfg["label_dict"])):
        raise ValueError(
            f"{path}: SLDPC prompt keys must be {list(cfg['label_dict'])}")
    for label, values in prompts.items():
        _nonempty_strings(values, f"{path}:{label}")


def _validate_slide_embedding_source_config(cfg: dict[str, Any]) -> None:
    """Validate shared provenance/storage fields for any slide-vector method."""
    from common.datasets.slide_embeddings import SlideEmbeddingSource

    slide = cfg.get("slide_encoder")
    if not isinstance(slide, dict):
        raise ValueError(
            "Slide-embedding methods require explicit slide_encoder provenance")
    required = {
        "input_kind", "name", "weights", "feature_space_id", "feature_dim",
        "feature_key", "resolution", "path_template", "runtime_encoder",
    }
    missing = required.difference(slide)
    if missing:
        raise ValueError(f"slide_encoder is missing {sorted(missing)}")
    if slide["input_kind"] != "slide_embedding":
        raise ValueError("slide_encoder.input_kind must be slide_embedding")
    if slide["feature_space_id"] != cfg["feature_space_id"]:
        raise ValueError(
            "slide_encoder.feature_space_id must match input features")
    if int(slide["feature_dim"]) != int(cfg["feature_dim"]):
        raise ValueError(
            "slide_encoder.feature_dim must match input feature_dim")
    if slide["feature_key"] != cfg["feature_key"]:
        raise ValueError(
            "slide_encoder.feature_key must match input feature_key")
    SlideEmbeddingSource.from_config(cfg)


def _validate_sldpc_encoder_config(cfg: dict[str, Any]) -> None:
    """Validate SLDPC's model-specific prompt/slide alignment choice."""
    from common.backbones import BackboneCapability as Cap, get_spec

    projection = cfg.get("slide_projection")
    if not isinstance(projection, dict):
        raise ValueError("SLDPC requires an explicit slide_projection block")
    mode = str(projection.get("mode", "")).lower()
    if mode not in {"native", "linear", "mlp"}:
        raise ValueError(
            "SLDPC slide_projection.mode must be native, linear, or mlp")
    if int(projection.get("input_dim", -1)) != int(cfg["feature_dim"]):
        raise ValueError(
            "SLDPC slide_projection.input_dim must match feature_dim")
    spec = get_spec(cfg["backbone"])
    if int(projection.get("output_dim", -1)) != int(spec.shared_dim or -1):
        raise ValueError(
            "SLDPC slide_projection.output_dim must match the prompt space")
    if bool(projection.get("trainable")) != (mode != "native"):
        raise ValueError(
            "SLDPC slide_projection.trainable is inconsistent with its mode")
    if mode == "native":
        required_native = {Cap.SLIDE_PROJECT, Cap.PAIRED_SLIDE_TEXT}
        if not required_native.issubset(spec.capabilities):
            raise ValueError(
                f"SLDPC native prompt backbone {spec.name} lacks paired "
                "slide-text projection")
        if cfg["feature_space_id"] != cfg["prompt_feature_space_id"]:
            raise ValueError(
                "SLDPC native projection requires identical input and prompt "
                "feature spaces")
        if (spec.slide_input_dim is not None
                and int(cfg["feature_dim"]) != spec.slide_input_dim):
            raise ValueError(
                "SLDPC native projection input width does not match the "
                "prompt backbone")

def _validate_wsi_five_assets(cfg: dict[str, Any]) -> None:
    if cfg.get("report_csv"):
        report_path = _require_asset(cfg.get("report_csv"), "WSI-FiVE report CSV")
        columns = set(pd.read_csv(report_path, nrows=1).columns)
        if not columns.intersection(
                {"slide_id", "case_id", "patient_id", "patient_filename"}):
            raise ValueError(f"{report_path}: missing a report identifier column")
        if not columns.intersection({"report", "text"}):
            raise ValueError(f"{report_path}: missing a report text column")
    else:
        if not str(cfg.get("default_report", "")).strip():
            raise ValueError(
                "WSI-FiVE requires report_csv or a class-agnostic default_report")
        if cfg.get("require_report", True):
            raise ValueError(
                "WSI-FiVE default_report mode requires require_report=false")
    _require_asset(
        cfg.get("clinicalbert_weights"), "WSI-FiVE ClinicalBERT weights")


def _validate_convlm_assets(cfg: dict[str, Any]) -> None:
    if not cfg.get("attribute_embeddings"):
        path = _require_asset(
            cfg.get("attribute_prompt_path"), "ConVLM attribute prompt JSON")
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        # Keys beginning with an underscore are metadata (provenance, notes),
        # not classes; comparing them against classnames would reject a marked
        # asset.
        payload = {k: v for k, v in payload.items()
                   if not str(k).startswith("_")}
        if list(payload) != list(cfg["classnames"]):
            raise ValueError(
                f"{path}: ConVLM prompt classes must be {cfg['classnames']}")
        for classname, prompts in payload.items():
            _nonempty_strings(prompts, f"{path}:{classname}")
        encoder = cfg.get("attribute_encoder")
        if not isinstance(encoder, dict):
            raise ValueError(
                "ConVLM runtime prompts require attribute_encoder metadata")
        required = {"model_name", "weights", "feature_space_id"}
        missing = required.difference(encoder)
        if missing:
            raise ValueError(
                f"ConVLM attribute_encoder missing {sorted(missing)}")
        _require_asset(encoder["weights"], "ConVLM attribute encoder weights")
        if encoder["feature_space_id"] != cfg.get("attribute_feature_space_id"):
            raise ValueError(
                "ConVLM attribute encoder feature space does not match config")
        return
    path = _require_asset(
        cfg.get("attribute_embeddings"), "ConVLM attribute embeddings")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    feature_space = None
    classnames = None
    if isinstance(payload, dict):
        feature_space = payload.get("feature_space_id")
        classnames = payload.get("classnames")
        payload = payload.get("embeddings", payload.get("attributes"))
    if not torch.is_tensor(payload) or payload.ndim != 2:
        raise ValueError(f"{path}: ConVLM attributes must be rank 2")
    if payload.shape[0] != cfg["n_classes"]:
        raise ValueError(f"{path}: ConVLM attribute row count is wrong")
    if feature_space != cfg.get("attribute_feature_space_id"):
        raise ValueError(f"{path}: ConVLM attribute feature space is wrong")
    if classnames is not None and list(classnames) != list(cfg["classnames"]):
        raise ValueError(f"{path}: ConVLM attribute class order is wrong")


def validate_generated_config_assets(cfg: dict[str, Any]) -> None:
    """Validate paths and prompt schemas without allocating an encoder."""
    from common.backbones import get_spec

    _require_asset(cfg.get("dataset_csv"), "dataset manifest")
    _require_asset(cfg.get("split_dir"), "split directory")
    encoder = cfg.get("encoder")
    if not isinstance(encoder, dict):
        raise ValueError("Generated config requires an explicit encoder block")
    method = cfg["method"]
    weights_value = cfg.get("backbone_weights")
    weights = (Path(str(weights_value)).expanduser().resolve()
               if weights_value else None)
    encoder_space = (
        cfg.get("prompt_feature_space_id")
        if method in {"muse", "sldpc"} else cfg["feature_space_id"])
    encoder_dim = (
        int(get_spec(cfg["backbone"]).shared_dim)
        if method in {"muse", "sldpc"}
        else (int(cfg["feature_dim"])
              if cfg.get("feature_dim") is not None else None))
    expected_encoder = {
        "name": cfg["backbone"],
        "weights": str(weights) if weights is not None else None,
        "feature_space_id": encoder_space,
        "feature_dim": encoder_dim,
        "initialization": encoder.get("initialization", "pretrained"),
    }
    if encoder != expected_encoder:
        raise ValueError(
            f"Encoder block does not match feature provenance: "
            f"{encoder} != {expected_encoder}")
    model_owned = cfg["backbone"] in {"wsi-five-vit", "convlm-vit"}
    if not model_owned:
        if weights is None or not weights.exists():
            raise FileNotFoundError(f"encoder weights do not exist: {weights}")
        spec = get_spec(cfg["backbone"])
        declared_space = str(encoder_space)
        if not (declared_space == spec.feature_space_id or (
                spec.name == "titan"
                and declared_space.startswith(spec.feature_space_id + "@"))):
            raise ValueError(
                f"Config feature space {declared_space} does not match "
                f"encoder {spec.name}: {spec.feature_space_id}")

    if "slide_embedding" in set(cfg.get("feature_input_kinds", {}).values()):
        _validate_slide_embedding_source_config(cfg)

    if method == "focus":
        _validate_focus_prompt(cfg)
        if Path(cfg["conch_ckpt"]).expanduser().resolve() != weights:
            raise ValueError("FOCUS conch_ckpt must match encoder weights")
    elif method == "maple":
        _validate_maple_prompt(cfg)
    elif method == "mscpt":
        _validate_mscpt_prompt(cfg)
        if cfg.get("input_mode") != "precomputed_shared_features":
            raise ValueError(
                "MSCPT configs must declare precomputed_shared_features")
    elif method == "muse":
        _validate_muse_prompts(cfg)
        _validate_muse_feature_config(cfg)
        if (cfg["backbone"] == "conch"
                and Path(cfg["conch_ckpt"]).expanduser().resolve() != weights):
            raise ValueError("MUSE conch_ckpt must match prompt encoder weights")
    elif method == "pathpt":
        if cfg.get("prompt_init") not in {"template", "rand"}:
            raise ValueError("PathPT prompt_init must be 'template' or 'rand'")
        if int(cfg.get("n_ctx", 0)) <= 0:
            raise ValueError("PathPT n_ctx must be positive")
    elif method == "vila_mil":
        _validate_focus_prompt(cfg)
    elif method == "cod_mil":
        _validate_cod_prompt(cfg)
    elif method == "top":
        if cfg.get("clip_arch") != "RN50":
            raise ValueError("TOP configs require clip_arch RN50")
        _validate_top_prompts(cfg)
    elif method == "slip":
        _validate_slip_prompt(cfg)
    elif method == "wsi_five":
        _validate_wsi_five_assets(cfg)
    elif method == "sldpc":
        _validate_sldpc_prompt(cfg)
        _validate_sldpc_encoder_config(cfg)
    elif method == "convlm":
        _validate_convlm_assets(cfg)
        if cfg.get("image_layout") != "per_slide_directory":
            raise ValueError("ConVLM configs require per_slide_directory")
    elif method == "composite":
        if not cfg.get("prompts", {}).get("coop_flat", {}).get("enabled"):
            raise ValueError("TCGA composite baseline requires CoOp prompts")
    else:
        raise KeyError(f"No asset validator for benchmark method {method}")


def _encoder_provenance(method: str, cfg: dict[str, Any],
                        feature_cfg: dict[str, Any]) -> str:
    """Say whether this run measures the encoder or the encoder plus a bridge.

    ``native`` means the method's published code supports this encoder directly.
    ``adapted`` means a trainable projection stands between the offline patch
    source and the prompt space, so the result measures the method *and* that
    projection and must not share a results table with a native row.

    Derived from the declared contract rather than set by hand, so it cannot
    drift from what the run actually does.
    """
    from common.backbones import SwapPolicy
    from methods import get_method

    contract = get_method(method).get_backbone_contract()
    # A patch source that is not the prompt backbone is bridged by the method's
    # learned visual adapter (MUSE is the registered case).
    if cfg["backbone"] != feature_cfg["backbone"]:
        return "adapted"
    if contract.swap_policy is SwapPolicy.CAPABILITY:
        expected = contract.feature_dims.get(cfg["backbone"])
        width = cfg.get("feature_dim")
        if expected and width is not None and int(width) not in expected:
            return "adapted"
    return "native"


def _prune_stale_configs(output_dir: Path, written: set[Path]) -> None:
    """Delete generated configs the protocol no longer produces.

    An experiment that stops generating -- because its prompts went missing, or
    it was removed from the protocol -- leaves its previous YAML behind. That
    file still looks valid and still names a results directory, so anything that
    walks the config tree instead of ``run_matrix.csv`` will happily run an
    experiment this protocol has already rejected. The protocol is the source of
    truth, so whatever it did not just write is removed.
    """
    config_root = output_dir / "configs"
    if not config_root.is_dir():
        return
    removed = [path for path in sorted(config_root.glob("*/*.yaml"))
               if path.resolve() not in written]
    for path in removed:
        path.unlink()
    # Drop the experiment directory too once it holds nothing.
    for directory in sorted(config_root.iterdir()):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if removed:
        print(f"  pruned {len(removed)} stale config(s) no longer generated: "
              + ", ".join(sorted({path.parent.name for path in removed})))


def generate_configs(protocol: dict[str, Any], output_dir: Path) -> None:
    build_generated_prompt_assets(protocol, output_dir)
    from methods import get_method

    matrix_rows = []
    written_configs: set[Path] = set()
    skipped_configs: list[dict[str, Any]] = []
    for experiment, experiment_cfg in protocol["experiments"].items():
        method = experiment_cfg["method"]
        bindings = _resolve_feature_bindings(protocol, experiment, experiment_cfg)
        _validate_feature_roles(method, experiment, bindings)
        primary = bindings.get(
            "bag", bindings.get("high", bindings.get("tiles")))
        assert primary is not None
        feature_cfg = primary["config"]
        for cohort, cohort_cfg in protocol["cohorts"].items():
            if not _experiment_supports_task(experiment_cfg, cohort):
                continue
            manifest = pd.read_csv(output_dir / "data" / cohort / "manifest.csv")
            metadata_ready = not manifest.empty
            required_columns = [
                _feature_column(binding["source"])
                for binding in bindings.values()
            ]
            missing_files = 0
            for binding, column in zip(bindings.values(), required_columns):
                missing_files += int((~manifest[column].map(
                    lambda value, source_cfg=binding["config"]:
                    _source_present(source_cfg, Path(value)))).sum())
            for shot in sorted(int(value) for value in protocol["shots"]):
                split_root = output_dir / "splits" / cohort / f"{shot}shot"
                split_ready = all(
                    (split_root / f"fold{fold}" / f"{partition}.csv").is_file()
                    for fold in range(int(protocol["folds"]))
                    for partition in ("train", "val", "test")
                )
                # A cohort declared `metadata_availability: future` has no
                # slides and, usually, no prompt assets compiled yet. Building
                # or validating its config is expected to fail, and that must
                # not abort generation for every other cohort and experiment in
                # the protocol.
                pending = (not metadata_ready
                           and cohort_cfg.get("metadata_availability") == "future")
                try:
                    cfg = _method_config(
                        experiment, experiment_cfg, method, protocol,
                        cohort, cohort_cfg, shot, output_dir)
                    get_method(method).get_backbone_contract().validate_config(cfg)
                    config_valid = True
                    if pending:
                        # Assets cannot be checked against a cohort that has no
                        # slides, so the row is generated but never marked valid.
                        config_valid = False
                    else:
                        validate_generated_config_assets(cfg)
                except (ValueError, KeyError, FileNotFoundError) as error:
                    # One experiment lacking an asset for one cohort -- MUSE has
                    # no published RCC descriptions, for instance -- must not
                    # abort the rest of the matrix. Report it, leave the row out,
                    # and carry on; the launcher then has nothing to submit for
                    # that combination rather than a config that cannot run.
                    reason = ("awaiting metadata and prompt assets" if pending
                              else f"{type(error).__name__}: {str(error)[:110]}")
                    print(f"  ! {experiment}/{cohort}/{shot}shot skipped: {reason}")
                    skipped_configs.append({
                        "experiment": experiment, "method": method,
                        "cohort": cohort, "shots": shot, "reason": reason,
                    })
                    continue
                missing_auxiliary = 0
                if method == "cod_mil":
                    map_root = Path(cfg["cross_mag_map_dir"])
                    missing_auxiliary = sum(
                        not (map_root / f"{slide_id}.pt").is_file()
                        for slide_id in manifest["slide_id"].astype(str))
                elif method == "wsi_five" and cfg.get("report_csv"):
                    reports = pd.read_csv(cfg["report_csv"])
                    id_column = next(column for column in (
                        "slide_id", "case_id", "patient_id", "patient_filename")
                        if column in reports.columns)
                    report_ids = {
                        str(value)[:12] for value in reports[id_column].dropna()
                    }
                    missing_auxiliary = int((
                        ~manifest["case_id"].astype(str).isin(report_ids)).sum())

                weights_value = cfg.get("backbone_weights")
                encoder_ready = (
                    cfg["encoder"].get("initialization") == "scratch"
                    or bool(weights_value and Path(weights_value).exists())
                )
                # Recorded in the config as well as the matrix, so a results
                # file can be traced to its provenance without re-deriving it.
                cfg["encoder_provenance"] = _encoder_provenance(
                    method, cfg, feature_cfg)
                path = (
                    output_dir / "configs" / experiment
                    / f"{cohort}_{shot}shot.yaml")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(cfg, handle, sort_keys=False)
                matrix_rows.append({
                    "experiment": experiment,
                    "method": method,
                    "cohort": cohort,
                    "shots": shot,
                    "feature_signature": ";".join(
                        f"{role}={binding['source']}"
                        for role, binding in bindings.items()),
                    "resolution_signature": ";".join(
                        f"{role}={binding['config']['resolution']}"
                        for role, binding in bindings.items()),
                    "feature_level": get_method(
                        method).get_backbone_contract().feature_level.value,
                    "backbone": cfg["backbone"],
                    "feature_encoder": feature_cfg["backbone"],
                    "encoder_provenance": cfg["encoder_provenance"],
                    "slide_encoder": (
                        feature_cfg["backbone"]
                        if _input_kind(feature_cfg) == "slide_embedding"
                        else None),
                    "slide_projection_mode": (
                        cfg.get("slide_projection", {}).get("mode")
                        if _input_kind(feature_cfg) == "slide_embedding"
                        else None),
                    "encoder_weights": cfg["backbone_weights"],
                    "encoder_ready": encoder_ready,
                    "prompt_source": cfg["prompt_source"],
                    "prompt_asset": (
                        cfg.get("text_prompt_path")
                        or cfg.get("description_prompt_path")
                        or ";".join(cfg.get("prompt_csvs", {}).values())
                        or cfg.get("tissue_classnames_path")
                        or cfg.get("prompt_reference_yaml")
                        or cfg.get("attribute_embeddings")
                        or cfg.get("attribute_prompt_path")
                        or cfg.get("report_csv")
                        or cfg.get("default_report")
                        or cfg.get("prompt_init")
                    ),
                    "config_valid": config_valid,
                    "bag_resolution": (
                        bindings["bag"]["config"]["resolution"]
                        if "bag" in bindings else None),
                    "low_resolution": (
                        bindings["low"]["config"]["resolution"]
                        if "low" in bindings else None),
                    "high_resolution": (
                        bindings["high"]["config"]["resolution"]
                        if "high" in bindings else None),
                    "auxiliary_ready": missing_auxiliary == 0,
                    "metadata_ready": metadata_ready,
                    "split_ready": split_ready,
                    "ready": (
                        missing_files == 0 and missing_auxiliary == 0
                        and encoder_ready and metadata_ready and split_ready),
                    "missing_feature_files": missing_files,
                    "missing_auxiliary_files": missing_auxiliary,
                    "config": str(path),
                    "command": f"python train.py --method {method} --config {path}",
                })
                written_configs.add(path.resolve())
    _prune_stale_configs(output_dir, written_configs)
    pd.DataFrame(matrix_rows).to_csv(output_dir / "run_matrix.csv", index=False)
    # Record what could not be generated, so a cohort missing one method's
    # assets is visible rather than merely absent from the matrix.
    if skipped_configs:
        pd.DataFrame(skipped_configs).to_csv(
            output_dir / "skipped_configs.csv", index=False)
        print(f"  {len(skipped_configs)} configs skipped; see "
              f"{output_dir / 'skipped_configs.csv'}")


def _assert_disjoint(frames: dict[str, pd.DataFrame], cohort: str, fold: int) -> None:
    case_sets = {name: set(frame["case_id"]) for name, frame in frames.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = case_sets[left].intersection(case_sets[right])
        if overlap:
            raise AssertionError(f"{cohort} fold {fold}: {left}/{right} leak {overlap}")


def audit_generated_configs(
    protocol: dict[str, Any], output_dir: Path
) -> pd.DataFrame:
    """Recheck every serialized config against the protocol and its assets."""
    from methods import get_method

    matrix_path = output_dir / "run_matrix.csv"
    _require_asset(matrix_path, "benchmark run matrix")
    matrix = pd.read_csv(matrix_path)
    rows = []
    for run in matrix.to_dict("records"):
        path = _require_asset(run["config"], "generated benchmark config")
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        for field in ("experiment", "method", "shots"):
            expected = run[field]
            if field == "shots":
                expected = int(expected)
            if cfg[field] != expected:
                raise ValueError(
                    f"{path}: {field}={cfg[field]!r} does not match run "
                    f"matrix value {expected!r}")
        get_method(cfg["method"]).get_backbone_contract().validate_config(cfg)
        # Rows generated for a cohort still awaiting metadata carry
        # config_valid=False and reference prompt assets that do not exist yet.
        # Their structure is still audited below; only the asset check is
        # deferred until the cohort's data lands.
        if bool(run.get("config_valid", True)):
            validate_generated_config_assets(cfg)

        for role, source in cfg["feature_sources"].items():
            source_cfg = protocol["feature_sources"].get(source)
            if source_cfg is None:
                raise KeyError(f"{path}: unknown serialized feature source {source}")
            if cfg["feature_resolutions"].get(role) != source_cfg["resolution"]:
                raise ValueError(
                    f"{path}: {role} resolution does not match source {source}")
        rows.append({
            "experiment": cfg["experiment"],
            "method": cfg["method"],
            "cohort": run["cohort"],
            "shots": int(cfg["shots"]),
            "config": str(path),
            "encoder": cfg["backbone"],
            "encoder_weights": cfg["backbone_weights"],
            "feature_encoder": (
                cfg.get("patch_encoder", {}).get("name")
                or cfg.get("slide_encoder", {}).get("name")
                or cfg["backbone"]),
            "encoder_provenance": cfg.get("encoder_provenance"),
            "slide_encoder": (
                cfg.get("slide_encoder", {}).get("name")
                if "slide_embedding" in set(
                    cfg.get("feature_input_kinds", {}).values()) else None),
            "slide_feature_space_id": (
                cfg.get("slide_encoder", {}).get("feature_space_id")
                if "slide_embedding" in set(
                    cfg.get("feature_input_kinds", {}).values()) else None),
            "slide_projection_mode": (
                cfg.get("slide_projection", {}).get("mode")
                if "slide_embedding" in set(
                    cfg.get("feature_input_kinds", {}).values()) else None),
            "prompt_source": cfg["prompt_source"],
            "valid": True,
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "config_audit.csv", index=False)
    return audit


def validate(protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"valid": True, "cohorts": {}, "missing_features": []}
    shots = sorted(int(value) for value in protocol["shots"])
    base_shot = shots[0]

    for cohort, cohort_cfg in protocol["cohorts"].items():
        labels = list(cohort_cfg["labels"])
        manifest = pd.read_csv(output_dir / "data" / cohort / "manifest.csv")
        if manifest.empty:
            if cohort_cfg.get("metadata_availability") != "future":
                raise AssertionError(f"{cohort}: manifest is unexpectedly empty")
            report["cohorts"][cohort] = {
                "slides": 0,
                "patients": 0,
                "labels": {},
                "status": "awaiting_metadata",
            }
            continue
        all_test_cases: list[str] = []
        cohort_report = {
            "slides": len(manifest),
            "patients": manifest["case_id"].nunique(),
            "labels": manifest["label"].value_counts().to_dict(),
        }
        for fold in range(int(protocol["folds"])):
            test_reference: set[str] | None = None
            train_nested: set[str] = set()
            val_nested: set[str] = set()
            for shot in shots:
                fold_dir = output_dir / "splits" / cohort / f"{shot}shot" / f"fold{fold}"
                frames = {
                    name: pd.read_csv(fold_dir / f"{name}.csv")
                    for name in ("train", "val", "test")
                }
                _assert_disjoint(frames, cohort, fold)
                for name in ("train", "val"):
                    counts = frames[name]["label"].value_counts().to_dict()
                    expected = {label: shot for label in labels}
                    if counts != expected:
                        raise AssertionError(
                            f"{cohort} fold {fold} {shot}-shot {name}: {counts} != {expected}"
                        )
                test_cases = set(frames["test"]["case_id"])
                if test_reference is None:
                    test_reference = test_cases
                elif test_reference != test_cases:
                    raise AssertionError(f"{cohort} fold {fold}: test set changed by shot")
                current_train = set(frames["train"]["slide_id"])
                current_val = set(frames["val"]["slide_id"])
                if not train_nested.issubset(current_train):
                    raise AssertionError(f"{cohort} fold {fold}: train shots are not nested")
                if not val_nested.issubset(current_val):
                    raise AssertionError(f"{cohort} fold {fold}: val shots are not nested")
                train_nested, val_nested = current_train, current_val
            base_test = pd.read_csv(
                output_dir
                / "splits"
                / cohort
                / f"{base_shot}shot"
                / f"fold{fold}"
                / "test.csv"
            )
            all_test_cases.extend(base_test["case_id"].drop_duplicates().tolist())

        counts = pd.Series(all_test_cases).value_counts()
        if set(counts.index) != set(manifest["case_id"]) or not (counts == 1).all():
            raise AssertionError(f"{cohort}: outer test folds do not partition patients")
        report["cohorts"][cohort] = cohort_report

        for feature_source, feature_cfg in protocol["feature_sources"].items():
            column = _feature_column(feature_source)
            missing = manifest.loc[
                ~manifest[column].map(
                    lambda value: _source_present(feature_cfg, Path(value))),
                column,
            ]
            if not missing.empty:
                report["missing_features"].append(
                    {
                        "cohort": cohort,
                        "feature_source": feature_source,
                        "input_kind": _input_kind(feature_cfg),
                        "resolution": feature_cfg["resolution"],
                        "feature_column": column,
                        "missing": len(missing),
                    }
                )
            available_samples = manifest.loc[
                manifest[column].map(
                    lambda value: _source_present(feature_cfg, Path(value))),
                column,
            ]
            kind = _input_kind(feature_cfg)
            if not available_samples.empty and kind != "raw_tile_directory":
                sample_path = Path(available_samples.iloc[0])
                shape = _feature_tensor_shape(
                    sample_path, feature_cfg.get("feature_key", "features"))
                expected_dim = int(feature_cfg["feature_dim"])
                if kind == "slide_embedding":
                    valid_shape = shape in {(expected_dim,), (1, expected_dim)}
                    expected_text = f"[{expected_dim}]"
                else:
                    valid_shape = (
                        len(shape) == 2 and shape[0] > 0
                        and shape[1] == expected_dim)
                    expected_text = f"[patches,{expected_dim}]"
                if not valid_shape:
                    raise AssertionError(
                        f"{cohort}/{feature_source}: expected "
                        f"{expected_text}, got {shape}")

    matrix_path = output_dir / "run_matrix.csv"
    if matrix_path.is_file():
        config_audit = audit_generated_configs(protocol, output_dir)
        report["configs"] = {
            "total": int(len(config_audit)),
            "valid": int(config_audit["valid"].sum()),
        }
    else:
        report["configs"] = {"total": 0, "valid": 0}
    path = output_dir / "validation_report.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _flatten_metrics(value: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            flattened[name] = float(item)
    return flattened


def aggregate_results(output_dir: Path) -> pd.DataFrame:
    """Aggregate completed fold metrics without changing the frozen protocol."""
    matrix = pd.read_csv(output_dir / "run_matrix.csv")
    fold_rows: list[dict[str, Any]] = []
    for run in matrix.to_dict("records"):
        with Path(run["config"]).open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        metrics_path = Path(cfg["results_dir"]) / "metrics.json"
        if not metrics_path.is_file():
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for fold, metrics in enumerate(payload.get("folds", [])):
            fold_rows.append(
                {
                    "experiment": run["experiment"],
                    "method": run["method"],
                    "feature_signature": run["feature_signature"],
                    "resolution_signature": run["resolution_signature"],
                    "backbone": run["backbone"],
                    # A native result and an adapted one are different claims,
                    # and so are an upstream prompt and a generated one. They
                    # must stay separable in the table people actually read, not
                    # just in the matrix that produced it.
                    "encoder_provenance": run.get("encoder_provenance"),
                    "prompt_provenance": cfg.get("prompt_provenance"),
                    "cohort": run["cohort"],
                    "shots": int(run["shots"]),
                    "fold": fold,
                    **_flatten_metrics(metrics),
                }
            )

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output_dir / "fold_results.csv", index=False)
    if folds.empty:
        aggregate = pd.DataFrame(
            columns=[
                "experiment", "method", "feature_signature",
                "resolution_signature", "backbone", "encoder_provenance",
                "prompt_provenance", "cohort", "shots",
                "metric", "mean", "std", "folds"]
        )
    else:
        id_columns = [
            "experiment", "method", "feature_signature",
            "resolution_signature", "backbone", "encoder_provenance",
            "prompt_provenance", "cohort", "shots", "fold"]
        long = folds.melt(
            id_vars=id_columns, var_name="metric", value_name="value"
        ).dropna(subset=["value"])
        aggregate = (
            long.groupby([
                "experiment", "method", "feature_signature",
                "resolution_signature", "backbone", "encoder_provenance",
                "prompt_provenance", "cohort", "shots",
                "metric"], dropna=False)["value"]
            .agg(mean="mean", std="std", folds="count")
            .reset_index()
        )
    aggregate.to_csv(output_dir / "aggregate_results.csv", index=False)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("inventory", "prepare", "configs", "validate", "aggregate", "all"),
    )
    parser.add_argument(
        "--protocol", type=Path, required=True,
        help="path to a cohort's protocol.yaml, e.g. "
             "benchmarks/tcga_brca/protocol.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    output_dir = (args.output_dir or protocol_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command in {"inventory", "prepare", "all"}:
        coverage = build_manifests(protocol, output_dir)
        print(coverage.to_string(index=False))
    if args.command in {"prepare", "all"}:
        build_splits(protocol, output_dir)
    if args.command in {"configs", "all"}:
        generate_configs(protocol, output_dir)
    if args.command in {"validate", "all"}:
        report = validate(protocol, output_dir)
        print(json.dumps(report, indent=2))
    if args.command == "aggregate":
        aggregate = aggregate_results(output_dir)
        print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
