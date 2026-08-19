"""Preconditions a benchmark configuration must meet before it can run.

A benchmark is a large matrix of configurations, and at any moment some of them
reference assets that have not been produced yet -- a prompt file that was never
imported, an encoder whose weights are not downloaded, features that have not
been extracted for a cohort. Those configurations must *skip*, cleanly and
visibly, rather than abort the campaign or (worse) train on nothing and report a
number.

:func:`preflight` inspects one resolved run config and answers a single
question: can this configuration produce a meaningful result? Normal checks read
only filesystem metadata, so they are cheap enough to run at the top of every
job. The opt-in deep check also opens feature payloads; no model is constructed.

Fatal findings become ``problems`` and the run is skipped. Feature coverage is
complete by default; a config may explicitly lower ``min_feature_coverage`` for
an exploratory partial-cohort run, which produces a visible warning.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from common.configuration import REPO_ROOT, expand_path, load_yaml_file
from common.method_provenance import method_provenance

PREFLIGHT_CHECKS = frozenset(
    {"assets", "features", "prompts", "encoders", "splits"})

# Paths are grouped so callers can ask focused health questions. Absent keys
# are fine; a configured path that is unavailable is always a problem.
PATH_KEYS_BY_CHECK = {
    "assets": ("dataset_csv",),
    "features": (
        "dataset_csv", "feat_data_dir", "selected_5x_dir", "feature_root",
        "data_folder_s", "data_folder_l", "slide_features", "data_path",
        "image_root", "cross_mag_map_dir",
    ),
    "prompts": (
        "text_prompt_path", "description_prompt_path",
        "prompt_reference_yaml", "gpt_dir", "instance_prompt_path",
        "bag_prompt_path", "prompt_features", "text_prompt_features",
        "text_prompt_bank_csv", "normal_structures_json", "report_csv", "attribute_embeddings",
        "attribute_prompt_path", "tissue_classnames_path",
        "clinical_questions", "evaluation_prompt_path",
    ),
    "encoders": (
        "backbone_weights", "conch_ckpt", "clinicalbert_weights",
        "initial_checkpoint",
    ),
    "splits": ("dataset_csv",),
}

# Encoder blocks each name their own checkpoint, and those are separate assets
# from ``backbone_weights``: SLDPC declares an offline slide encoder and a
# runtime prompt encoder independently, and either can be absent on its own.
# Only ``weights`` is checked here -- sibling keys such as ``path_template``
# hold a ``{slide_id}`` pattern rather than a path, and are resolved per slide.
ASSET_BLOCK_KEYS = (
    "encoder",
    "patch_encoder",
    "slide_encoder",
    "prompt_encoder",
    "attribute_encoder",
)
ASSET_BLOCK_FIELDS = ("weights",)

# Keys holding a collection of asset paths. The shape varies by method: a plain
# list of paths, a list of {path, partition} records, or -- for MUSE -- a
# mapping of classname to that class's description CSV.
LIST_KEYS_BY_CHECK = {
    "assets": ("metadata_csvs",),
    "prompts": ("prompt_csvs", "muse_prompt_csvs"),
}

# Output directories are different from input assets: they may not exist yet,
# but the closest existing parent must be writable so the trainer can create
# them. Probing is read-only; preflight never creates the directory itself.
OUTPUT_KEYS_BY_CHECK = {
    "assets": ("results_dir",),
}

FILE_PATH_KEYS = frozenset({
    "dataset_csv", "text_prompt_path", "description_prompt_path",
    "prompt_reference_yaml", "instance_prompt_path", "bag_prompt_path",
    "prompt_features", "text_prompt_features", "text_prompt_bank_csv",
    "normal_structures_json",
    "report_csv", "attribute_embeddings", "attribute_prompt_path",
    "tissue_classnames_path", "initial_checkpoint", "clinical_questions",
    "evaluation_prompt_path",
})

DIRECTORY_PATH_KEYS = frozenset({
    "feat_data_dir", "selected_5x_dir", "feature_root", "data_folder_s",
    "data_folder_l", "data_path", "image_root", "cross_mag_map_dir",
    "gpt_dir",
})


def _configured(cfg: dict[str, Any], key: str) -> bool:
    """Return whether a config key contains a meaningful non-empty value."""
    value = cfg.get(key)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return value is not None


def _require_any_config_key(
    cfg: dict[str, Any], report: "PreflightReport", purpose: str,
    keys: tuple[str, ...],
) -> None:
    if not any(_configured(cfg, key) for key in keys):
        report.problems.append(
            f"{purpose} requires one of: {', '.join(keys)}")


def _check_focus_prompt_schema(
    cfg: dict[str, Any], report: "PreflightReport", *,
    consumer: str = "FOCUS",
) -> None:
    """Validate the ordered two-scale prompt table used by FOCUS/ViLa."""
    value = cfg.get("text_prompt_path")
    if not _exists(value):
        return
    path = Path(expand_path(value))
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        report.problems.append(
            f"cannot read {consumer} prompt CSV {path}: {error}")
        return
    required = {"class_name", "low_res_prompt", "high_res_prompt"}
    missing = sorted(required - fields)
    if missing:
        report.problems.append(
            f"{consumer} prompt CSV is missing columns: "
            + ", ".join(missing))
        return
    raw_n_classes = cfg.get("n_classes")
    if (isinstance(raw_n_classes, int) and not isinstance(raw_n_classes, bool)
            and len(rows) != raw_n_classes):
        report.problems.append(
            f"{consumer} prompt CSV has {len(rows)} rows for "
            f"n_classes={raw_n_classes}")
    class_names = [_cell(row, "class_name") for row in rows]
    duplicate_names = _duplicate_values(
        [name for name in class_names if name])
    if duplicate_names:
        report.problems.append(
            f"{consumer} prompt CSV repeats class names: "
            + ", ".join(duplicate_names[:3]))
    expected_orders: list[list[str]] = []
    label_dict = cfg.get("label_dict")
    if isinstance(label_dict, dict) and all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in label_dict.values()):
        expected_orders.append([
            str(label) for label, _index in sorted(
                label_dict.items(), key=lambda item: item[1])])
    classnames = cfg.get("classnames")
    if isinstance(classnames, list):
        expected_orders.append([str(value) for value in classnames])
    if (class_names and expected_orders
            and class_names not in expected_orders):
        report.problems.append(
            f"{consumer} prompt CSV class_name order {class_names} does not "
            "match label_dict/classnames class-index order")
    for column in required:
        blanks = sum(not _cell(row, column) for row in rows)
        if blanks:
            report.problems.append(
                f"{consumer} prompt CSV has {blanks} blank values in {column}")


def _check_report_csv_schema(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Catch malformed or ambiguous WSI-FiVE supervision before model build."""
    value = cfg.get("report_csv")
    if not _exists(value):
        return
    path = Path(expand_path(value))
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        report.problems.append(f"cannot read WSI-FiVE report CSV {path}: {error}")
        return
    id_column = next((column for column in (
        "slide_id", "case_id", "patient_id", "patient_filename")
        if column in fields), None)
    text_column = next((column for column in ("answer", "report", "text")
                        if column in fields), None)
    if id_column is None or text_column is None:
        report.problems.append(
            "WSI-FiVE report CSV needs an ID column and an "
            "answer/report/text column")
        return
    native = cfg.get("training_mode") == "upstream_answer_bank"
    question_columns = [f"q{index}" for index in range(1, 7)]
    if native:
        missing = [column for column in ("answer", *question_columns)
                   if column not in fields]
        if missing:
            report.problems.append(
                "WSI-FiVE native answer CSV is missing columns: "
                + ", ".join(missing))
            return
    reports: dict[str, str] = {}
    structured_answers: dict[str, tuple[str, ...]] = {}
    for row_number, row in enumerate(rows, start=2):
        identifier, text = _cell(row, id_column), _cell(row, text_column)
        if not identifier or not text or text.lower() == "nan":
            report.problems.append(
                f"WSI-FiVE report CSV row {row_number} has a blank ID or report")
            continue
        if native:
            blank_questions = [
                column for column in question_columns
                if not _cell(row, column)
                or _cell(row, column).casefold() == "nan"]
            if blank_questions:
                report.problems.append(
                    f"WSI-FiVE report CSV row {row_number} has blank "
                    "structured answers: " + ", ".join(blank_questions))
            answer_fields = tuple(_cell(row, column)
                                  for column in question_columns)
        for key in dict.fromkeys((identifier, identifier[:12])):
            if key in reports and reports[key] != text:
                report.problems.append(
                    f"WSI-FiVE report CSV has conflicting reports for ID {key!r}")
            reports[key] = text
            if native:
                existing = structured_answers.get(key)
                if existing is not None and existing != answer_fields:
                    report.problems.append(
                        "WSI-FiVE native answer CSV has conflicting structured "
                        f"answers for ID {key!r}")
                structured_answers[key] = answer_fields


def _check_wsi_five_evaluation_prompts(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Validate the released diagnostic descriptions and class order."""
    value = cfg.get("evaluation_prompt_path")
    if not _exists(value):
        return
    path = Path(expand_path(value))
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        prompts = payload.get("prompts") if isinstance(payload, dict) else None
        if not isinstance(prompts, dict):
            raise ValueError("evaluation prompts must be a mapping")
        label_dict = cfg.get("label_dict")
        if not isinstance(label_dict, dict) or not label_dict:
            raise ValueError("label_dict must be a non-empty mapping")
        if set(prompts) != set(label_dict):
            raise ValueError(
                "evaluation prompt labels must exactly match label_dict: "
                f"expected {sorted(label_dict)}, got {sorted(prompts)}")
        indices = list(label_dict.values())
        if (any(isinstance(index, bool) or not isinstance(index, int)
                for index in indices)
                or sorted(indices) != list(range(len(indices)))):
            raise ValueError(
                "label_dict values must be contiguous class indices")
        ordered = [None] * len(label_dict)
        for label, index in label_dict.items():
            value = prompts[label]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"evaluation prompt for {label!r} is blank")
            ordered[index] = value.strip()
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        report.problems.append(
            f"invalid WSI-FiVE evaluation prompt bank: {error}")
        return
    n_classes = cfg.get("n_classes")
    if (isinstance(n_classes, int) and not isinstance(n_classes, bool)
            and len(ordered) != n_classes):
        report.problems.append(
            "WSI-FiVE evaluation prompt count does not match n_classes")


def _check_clinical_questions_schema(
    value: str | Path, report: "PreflightReport", *,
    expected_count: int | None = None,
) -> None:
    """Validate the JSON question-bank shape consumed by WSI-FiVE."""
    if not _exists(value):
        return
    path = Path(expand_path(value))
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, ValueError) as error:
        report.problems.append(
            f"cannot read WSI-FiVE clinical questions {path}: {error}")
        return
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if (not isinstance(questions, list) or not questions
            or any(not isinstance(item, str) or not item.strip()
                   for item in questions)):
        report.problems.append(
            "WSI-FiVE clinical question JSON must contain a non-empty "
            "'questions' string list")
    elif expected_count is not None and len(questions) != expected_count:
        report.problems.append(
            f"WSI-FiVE native question bank must contain exactly "
            f"{expected_count} questions, got {len(questions)}")


def _read_prompt_json(
    value: str | Path, consumer: str, report: "PreflightReport",
) -> tuple[Path, Any] | None:
    """Decode a prompt JSON asset while keeping failures diagnostic."""
    if not _exists(value):
        return None
    path = Path(expand_path(value))
    try:
        with path.open(encoding="utf-8") as handle:
            return path, json.load(handle)
    except (OSError, UnicodeError, ValueError) as error:
        report.problems.append(
            f"cannot read {consumer} prompt JSON {path}: {error}")
        return None


def _prompt_manifest_record(path: Path) -> dict[str, Any]:
    """Return audited metadata for a text_prompts asset, if registered."""
    prompt_root = (REPO_ROOT / "text_prompts").resolve()
    try:
        key = str(path.resolve().relative_to(prompt_root))
    except ValueError:
        return {}
    try:
        with (prompt_root / "PROVENANCE.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, ValueError):
        return {}
    record = manifest.get("assets", {}).get(key, {}) \
        if isinstance(manifest, dict) else {}
    return record if isinstance(record, dict) else {}


def _valid_prompt_list(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(item, str) and item.strip() for item in value))


def _valid_slip_prompt_groups(value: Any) -> bool:
    """Accept SLIP's native per-concept text ensembles, not just flat text."""
    if not isinstance(value, list) or not value:
        return False
    return all(
        (isinstance(group, str) and bool(group.strip()))
        or _valid_prompt_list(group)
        for group in value
    )


def _check_maple_prompt_schema(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Validate MAPLE's order-sensitive two-scale prompt graph."""
    value = cfg.get("text_prompt_path")
    if not _exists(value):
        return
    path = Path(expand_path(value))
    classnames = cfg.get("classnames")
    if not _valid_prompt_list(classnames):
        report.problems.append(
            "MAPLE requires ordered classnames to validate prompt/logit order")
        return
    try:
        from common.prompts.maple import load_maple_prompt_bank
        bank = load_maple_prompt_bank(path, classnames=classnames)
    except (OSError, UnicodeError, ValueError) as error:
        report.problems.append(str(error))
        return

    record = _prompt_manifest_record(path)
    origin = bank.provenance
    if origin == "unknown":
        origin = str(record.get("provenance", "unknown"))
    declared = cfg.get("prompt_provenance")
    if declared and origin != "unknown" and declared != origin:
        report.problems.append(
            "MAPLE prompt_provenance contradicts the active bank: "
            f"declared {declared!r}, expected {origin!r}")
    expected_source = {
        "upstream": "maple_upstream_attribute_json",
        "generated": "maple_task_extension_attribute_json",
    }.get(origin)
    declared_source = cfg.get("prompt_source")
    if (declared_source and expected_source
            and declared_source != expected_source):
        report.problems.append(
            "MAPLE prompt_source contradicts the active bank: "
            f"declared {declared_source!r}, expected {expected_source!r}")
    declared_sha256 = record.get("sha256")
    if declared_sha256 is not None:
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != declared_sha256:
            report.problems.append(
                f"{path}: MAPLE file sha256 does not match the provenance "
                "manifest")


def _check_mscpt_prompt_schema(
    cfg: dict[str, Any], value: str | Path, report: "PreflightReport",
) -> None:
    """Validate MSCPT's native description bank before model construction."""
    loaded = _read_prompt_json(value, "MSCPT", report)
    if loaded is None:
        return
    path, payload = loaded
    if not isinstance(payload, dict):
        report.problems.append(f"{path}: MSCPT prompt payload must be a mapping")
        return
    label_dict = cfg.get("label_dict")
    labels = set(label_dict) if isinstance(label_dict, dict) else None
    prompt_labels = {
        key for key in payload if not str(key).startswith("_")}
    if labels is not None and prompt_labels != labels:
        report.problems.append(
            f"{path}: MSCPT prompt labels {sorted(map(str, prompt_labels))} "
            f"do not match task labels {sorted(map(str, labels))}")
    small_counts: set[int] = set()
    big_counts: set[int] = set()
    for label in prompt_labels:
        block = payload.get(label)
        if not isinstance(block, dict):
            report.problems.append(
                f"{path}: MSCPT prompt block {label!r} must be a mapping")
            continue
        for field, counts in (("small_mag", small_counts),
                              ("big_mag", big_counts)):
            prompts = block.get(field)
            if not _valid_prompt_list(prompts):
                report.problems.append(
                    f"{path}: MSCPT {label}.{field} must be a non-empty "
                    "string list")
            else:
                counts.add(len(prompts))
    if len(small_counts) > 1 or len(big_counts) > 1:
        report.problems.append(
            f"{path}: MSCPT prompt counts must match across classes")
    n_high = cfg.get("n_high", 10)
    if (isinstance(n_high, int) and not isinstance(n_high, bool)
            and small_counts and small_counts != {n_high}):
        report.problems.append(
            f"{path}: MSCPT requires {n_high} small_mag prompts per class "
            f"for n_high={n_high}, got {sorted(small_counts)}")


def _check_top_prompt_schema(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Validate TOP's instance prototypes and optional bag descriptions."""
    instance_value = cfg.get("instance_prompt_path")
    if not _configured(cfg, "instance_prompt_path"):
        report.problems.append(
            "Method 'top' requires instance_prompt_path; the released method "
            "always initializes its instance learner from tissue prototypes")
    else:
        loaded = _read_prompt_json(instance_value, "TOP instance", report)
        if loaded is not None:
            path, payload = loaded
            prototypes = (
                payload.get("prototypes") if isinstance(payload, dict)
                else payload)
            if not isinstance(prototypes, list) or len(prototypes) < 2:
                report.problems.append(
                    f"{path}: TOP needs at least two instance prototypes")
            else:
                invalid = [
                    index for index, item in enumerate(prototypes)
                    if (not isinstance(item, dict)
                        or not isinstance(item.get("prompt"), str)
                        or not item["prompt"].strip())]
                if invalid:
                    report.problems.append(
                        f"{path}: TOP instance prototypes at indices "
                        f"{invalid[:5]} need a non-empty 'prompt' string")
                valid_prompts = [
                    item["prompt"] for item in prototypes
                    if isinstance(item, dict)
                    and isinstance(item.get("prompt"), str)
                    and item["prompt"].strip()]
                if len(valid_prompts) != len(set(valid_prompts)):
                    report.problems.append(
                        f"{path}: TOP instance prompts must be unique")
                mismatched = []
                for index, item in enumerate(prototypes):
                    if not isinstance(item, dict):
                        continue
                    tissue = item.get("tissue")
                    description = item.get("description")
                    prompt = item.get("prompt")
                    if (isinstance(tissue, str)
                            and isinstance(description, str)
                            and isinstance(prompt, str)
                            and prompt != f"an H&E stained image of {tissue}, "
                            f"which is {description}"):
                        mismatched.append(index)
                if mismatched:
                    report.problems.append(
                        f"{path}: TOP instance prototypes at indices "
                        f"{mismatched[:5]} do not match their structured "
                        "tissue/description fields")
                slotted = [
                    index for index, item in enumerate(prototypes)
                    if isinstance(item, dict)
                    and isinstance(item.get("prompt"), str)
                    and "*" in item["prompt"]]
                if slotted:
                    report.problems.append(
                        f"{path}: TOP base instance prompts at indices "
                        f"{slotted[:5]} already contain learnable slots")
                metadata = payload.get("_metadata", {}) \
                    if isinstance(payload, dict) else {}
                if not isinstance(metadata, dict):
                    report.problems.append(
                        f"{path}: TOP instance _metadata must be a mapping")
                else:
                    declared_count = metadata.get("count")
                    if (declared_count is not None
                            and declared_count != len(prototypes)):
                        report.problems.append(
                            f"{path}: TOP instance metadata count "
                            f"{declared_count} does not match "
                            f"{len(prototypes)} prototypes")
                    declared_digest = metadata.get("ordered_prompt_sha256")
                    if declared_digest is not None and not invalid:
                        actual_digest = hashlib.sha256(
                            "\n".join(valid_prompts).encode()).hexdigest()
                        if declared_digest != actual_digest:
                            report.problems.append(
                                f"{path}: TOP ordered_prompt_sha256 does not "
                                "match the instance bank")
                    separator = metadata.get("slot_separator", "")
                    separator = cfg.get(
                        "instance_slot_separator", separator)
                    if separator not in {"", " "}:
                        report.problems.append(
                            f"{path}: TOP instance slot_separator must be "
                            "empty or one space")

    if not _configured(cfg, "bag_prompt_path"):
        if cfg.get("prompt_provenance") == "upstream":
            report.problems.append(
                "TOP prompt_provenance='upstream' requires bag_prompt_path; "
                "otherwise the bag learner uses random classname context")
        return
    loaded = _read_prompt_json(cfg["bag_prompt_path"], "TOP bag", report)
    if loaded is None:
        return
    path, payload = loaded
    exact_ctx = payload.get("ctx_init") if isinstance(payload, dict) else None
    prompts = (exact_ctx if exact_ctx is not None
               else payload.get("prompts") if isinstance(payload, dict)
               else None)
    if not isinstance(prompts, dict) or not prompts:
        report.problems.append(
            f"{path}: TOP bag prompt payload needs a non-empty 'prompts' or "
            "'ctx_init' mapping")
        return
    invalid = [
        label for label, prompt in prompts.items()
        if not isinstance(prompt, str) or not prompt.strip()]
    if invalid:
        report.problems.append(
            f"{path}: TOP bag prompts are blank for {invalid[:5]}")
    label_dict = cfg.get("label_dict")
    if isinstance(label_dict, dict):
        labels = list(label_dict)
        if all(isinstance(index, int) and not isinstance(index, bool)
               for index in label_dict.values()):
            labels = [label for label, _ in sorted(
                label_dict.items(), key=lambda item: item[1])]
        missing = [label for label in labels if label not in prompts]
        if missing:
            report.problems.append(
                f"{path}: TOP bag prompts are missing task labels {missing}")
        extra = [label for label in prompts if label not in label_dict]
        if extra:
            report.problems.append(
                f"{path}: TOP bag prompts contain unknown task labels {extra}")
        metadata = payload.get("_metadata", {})
        if not isinstance(metadata, dict):
            report.problems.append(
                f"{path}: TOP bag _metadata must be a mapping")
            metadata = {}
        declared_order = metadata.get("label_order")
        if declared_order is not None and declared_order != labels:
            report.problems.append(
                f"{path}: TOP bag label_order {declared_order} does not match "
                f"classifier order {labels}")
        upstream_classnames = metadata.get("upstream_classnames")
        if (upstream_classnames is not None
                and (not isinstance(upstream_classnames, list)
                     or len(upstream_classnames) != len(labels)
                     or any(not isinstance(name, str) or not name.strip()
                            for name in upstream_classnames))):
            report.problems.append(
                f"{path}: TOP upstream_classnames must contain one non-empty "
                "suffix per classifier label")
    if exact_ctx is not None:
        wrong_slots = [
            label for label, prompt in prompts.items()
            if isinstance(prompt, str) and prompt.count("*") != 10]
        if wrong_slots:
            report.problems.append(
                f"{path}: TOP ctx_init needs exactly ten learnable slots per "
                f"label; invalid labels {wrong_slots}")
    else:
        slotted = [
            label for label, prompt in prompts.items()
            if isinstance(prompt, str) and "*" in prompt]
        if slotted:
            report.problems.append(
                f"{path}: TOP base bag prompts already contain learnable "
                f"slots for labels {slotted}")


def _check_slip_prompt_schema(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Validate all three text roles used by SLIP's routing branch."""
    configured_tissues = cfg.get("tissue_classnames")
    path_is_configured = _configured(cfg, "tissue_classnames_path")
    if configured_tissues is not None \
            and not _valid_slip_prompt_groups(configured_tissues):
        report.problems.append(
            "SLIP tissue_classnames must be a non-empty list of text groups")
        configured_tissues = None
    if configured_tissues is None and not path_is_configured:
        report.problems.append(
            "Method 'slip' requires tissue_classnames_path or "
            "tissue_classnames")

    path_value = cfg.get("tissue_classnames_path")
    if path_is_configured:
        loaded = _read_prompt_json(path_value, "SLIP tissue", report)
        if loaded is not None:
            path, payload = loaded
            raw_tissues = payload.get(
                "tissue_classnames", payload.get("tissues"),
            ) if isinstance(payload, dict) else payload
            pair_separator = payload.get("_pair_separator") \
                if isinstance(payload, dict) else None
            if not _valid_slip_prompt_groups(raw_tissues):
                report.problems.append(
                    f"{path}: SLIP tissue prompt JSON needs a non-empty "
                    "tissue_classnames (or legacy tissues) list")
            elif (configured_tissues is not None and not pair_separator
                  and raw_tissues != configured_tissues):
                report.problems.append(
                    f"{path}: SLIP tissue_classnames must exactly match its "
                    "prompt JSON")

            classnames = cfg.get("classnames")
            n_classes = cfg.get("n_classes")
            if not _valid_prompt_list(classnames):
                count = n_classes if (
                    isinstance(n_classes, int)
                    and not isinstance(n_classes, bool)
                    and n_classes > 0) else 1
                classnames = [f"class {index}" for index in range(count)]
            label_dict = cfg.get("label_dict")
            labels = None
            if (isinstance(label_dict, dict)
                    and all(isinstance(index, int)
                            and not isinstance(index, bool)
                            for index in label_dict.values())):
                labels = [label for label, _ in sorted(
                    label_dict.items(), key=lambda item: item[1])]
            try:
                from common.prompts.slip import load_slip_prompt_bank
                bank = load_slip_prompt_bank(
                    path,
                    fallback_slide_classnames=classnames,
                    labels=labels,
                )
            except (OSError, UnicodeError, ValueError) as error:
                report.problems.append(str(error))
            else:
                expected = bank.config_values()
                for key in ("text_templates", "slip_slide_classnames",
                            "tissue_classnames"):
                    if key in cfg and cfg[key] != expected[key]:
                        report.problems.append(
                            f"{path}: SLIP {key} must exactly match its "
                            "complete prompt bank")
                if bank.provenance not in {
                        "upstream", "derived", "generated"}:
                    report.problems.append(
                        f"{path}: SLIP prompt bank must declare _provenance "
                        "as upstream, derived, or generated")
                declared = cfg.get("prompt_provenance")
                if declared and declared != bank.provenance:
                    report.problems.append(
                        "SLIP prompt_provenance contradicts the active bank: "
                        f"declared {declared!r}, expected "
                        f"{bank.provenance!r}")
                expected_source = {
                    "upstream": "slip_upstream_complete_prompt_bank",
                    "generated": "slip_generated_task_extension_prompt_bank",
                }.get(bank.provenance)
                declared_source = cfg.get("prompt_source")
                if (declared_source and expected_source
                        and declared_source != expected_source):
                    report.problems.append(
                        "SLIP prompt_source contradicts the active bank: "
                        f"declared {declared_source!r}, expected "
                        f"{expected_source!r}")
                configured_tissues = expected["tissue_classnames"]

    n_classes = cfg.get("n_classes")
    if (configured_tissues is not None
            and isinstance(n_classes, int) and not isinstance(n_classes, bool)
            and len(configured_tissues) < n_classes):
        report.problems.append(
            "SLIP tissue vocabulary is smaller than n_classes; tissue routing "
            "would be degenerate")
    templates = cfg.get("text_templates")
    if templates is not None and (
            not _valid_prompt_list(templates)
            or any(template.count("{}") != 1 for template in templates)):
        report.problems.append(
            "SLIP text_templates must be a non-empty string list whose "
            "entries contain exactly one '{}'")


def _check_cod_prompt_schema(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Validate CoD-MIL's chain and its position-bound feature bank."""
    if not _configured(cfg, "text_prompt_path"):
        return
    loaded = _read_prompt_json(cfg["text_prompt_path"], "CoD-MIL", report)
    if loaded is None:
        return
    path, payload = loaded
    if not isinstance(payload, dict):
        report.problems.append(
            f"{path}: CoD-MIL prompt payload must be a mapping")
        return
    chain = {
        key: value for key, value in payload.items()
        if not str(key).startswith("_")}
    classnames = cfg.get("classnames")
    if isinstance(classnames, list) and set(chain) != set(classnames):
        report.problems.append(
            f"{path}: CoD-MIL prompt labels {sorted(map(str, chain))} do not "
            f"match configured classnames {sorted(map(str, classnames))}")
    for label, block in chain.items():
        if not isinstance(block, dict):
            report.problems.append(
                f"{path}: CoD-MIL prompt block {label!r} must be a mapping")
            continue
        for field in ("broad", "specific"):
            if not _valid_prompt_list(block.get(field)):
                report.problems.append(
                    f"{path}: CoD-MIL {label}.{field} must be a non-empty "
                    "string list")

    if not _configured(cfg, "text_prompt_features"):
        return
    bank_value = cfg.get("text_prompt_bank_csv")
    if not _configured(cfg, "text_prompt_bank_csv"):
        report.problems.append(
            "Precomputed CoD-MIL prompts require text_prompt_bank_csv so "
            "positional rows can be verified")
        return
    if not _exists(bank_value) or not _exists(cfg.get("text_prompt_features")):
        return

    from common.prompts import (
        load_prompt_bank_csv,
        validate_prompt_feature_metadata,
    )

    bank_path = Path(expand_path(bank_value))
    tensor_path = Path(expand_path(cfg["text_prompt_features"]))
    try:
        prompts = load_prompt_bank_csv(bank_path)
        n_classes = int(cfg["n_classes"])
        expected_class_rows = [
            chain[name]["broad"][0] for name in cfg.get("classnames", [])
        ] + [
            chain[name]["specific"][0] for name in cfg.get("classnames", [])
        ]
        if prompts[:2 * n_classes] != expected_class_rows:
            raise ValueError(
                f"{bank_path}: leading rows do not match the configured "
                "low/high diagnosis chain")
        try:
            import torch
        except ModuleNotFoundError as error:
            raise ValueError(
                "checking a CoD-MIL prompt tensor requires the core PyTorch "
                "environment") from error
        try:
            raw_payload = torch.load(
                tensor_path, map_location="cpu", weights_only=True)
        except TypeError:
            raw_payload = torch.load(tensor_path, map_location="cpu")
        embeddings = validate_prompt_feature_metadata(
            raw_payload,
            prompts=prompts,
            n_classes=n_classes,
            source_path=bank_path,
            context=tensor_path,
        )
        if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
            raise ValueError(f"{tensor_path}: embeddings must be a rank-2 tensor")
        if embeddings.shape[0] != len(prompts):
            raise ValueError(
                f"{tensor_path}: {embeddings.shape[0]} tensor rows do not "
                f"match {len(prompts)} source prompts")
        expected_dim = cfg.get("feature_dim")
        if (isinstance(expected_dim, int)
                and embeddings.shape[1] != expected_dim):
            raise ValueError(
                f"{tensor_path}: width {embeddings.shape[1]} does not match "
                f"feature_dim {expected_dim}")
        embedded_space = raw_payload.get("feature_space_id")
        declared_space = cfg.get("text_feature_space_id")
        if declared_space and embedded_space != declared_space:
            raise ValueError(
                f"{tensor_path}: feature space {embedded_space!r} does not "
                f"match {declared_space!r}")
        if not torch.isfinite(embeddings).all():
            raise ValueError(f"{tensor_path}: embeddings contain non-finite values")
    except Exception as error:
        report.problems.append(str(error))


def _check_muse_prompt_schema(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Validate every class-specific MUSE description CSV."""
    prompt_csvs = cfg.get("prompt_csvs")
    if not isinstance(prompt_csvs, dict):
        return
    for classname, value in prompt_csvs.items():
        if not _exists(value):
            continue
        path = Path(expand_path(value))
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        except (OSError, UnicodeError, csv.Error) as error:
            report.problems.append(
                f"cannot read MUSE prompt CSV {path}: {error}")
            continue
        descriptions = []
        for row in rows:
            values = [cell.strip() for cell in row if cell.strip()]
            if not values or values == ["0"]:
                continue
            text = max(values, key=len)
            if len(text) > 8:
                descriptions.append(text)
        if not descriptions:
            report.problems.append(
                f"{path}: MUSE prompt CSV for {classname!r} contains no "
                "descriptions")


def _check_convlm_prompt_schema(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Validate the ordered class-to-attribute graph used at runtime."""
    if not _configured(cfg, "attribute_prompt_path"):
        return
    loaded = _read_prompt_json(
        cfg["attribute_prompt_path"], "ConVLM attribute", report)
    if loaded is None:
        return
    path, payload = loaded
    if not isinstance(payload, dict):
        report.problems.append(
            f"{path}: ConVLM attribute prompt payload must be a mapping")
        return
    prompts = {
        key: value for key, value in payload.items()
        if not str(key).startswith("_")}
    classnames = cfg.get("classnames")
    if isinstance(classnames, list) and list(prompts) != classnames:
        report.problems.append(
            f"{path}: ConVLM attribute prompt order must match classnames")
    for classname, values in prompts.items():
        if not _valid_prompt_list(values):
            report.problems.append(
                f"{path}: ConVLM attributes for {classname!r} must be a "
                "non-empty string list")


def _check_sldpc_prompt_schema(
    cfg: dict[str, Any], report: "PreflightReport",
) -> None:
    """Validate SLDPC's declared task prompt reference."""
    if not _configured(cfg, "prompt_reference_yaml"):
        return
    value = cfg["prompt_reference_yaml"]
    if not _exists(value):
        return
    path = Path(expand_path(value))
    try:
        payload = load_yaml_file(path)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        report.problems.append(
            f"cannot read SLDPC prompt reference {path}: {error}")
        return
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, dict):
        report.problems.append(
            f"{path}: SLDPC prompt reference needs a 'prompts' mapping")
        return
    label_dict = cfg.get("label_dict")
    if isinstance(label_dict, dict) and set(prompts) != set(label_dict):
        report.problems.append(
            f"{path}: SLDPC prompt keys must match label_dict")
    for label, values in prompts.items():
        if not _valid_prompt_list(values):
            report.problems.append(
                f"{path}: SLDPC prompts for {label!r} must be a non-empty "
                "string list")


def _check_method_requirements(
    cfg: dict[str, Any], report: "PreflightReport", selected: set[str],
) -> None:
    """Catch missing method inputs that path-existence checks cannot see."""
    method = str(cfg.get("method", "")).strip().lower().replace("-", "_")
    method = {
        "vila": "vila_mil", "vilamil": "vila_mil",
        "codmil": "cod_mil", "five": "wsi_five",
        "wsifive": "wsi_five",
    }.get(method, method)
    if not method:
        return

    if "features" in selected:
        if (method == "focus"
                and str(cfg.get("experiment", "")).startswith("focus_5x")):
            report.problems.append(
                "This FOCUS low-scale variant is obsolete: upstream FOCUS "
                "consumes only the high-resolution bag. Regenerate configs "
                "from the protocol and use the 'focus' experiment.")
        if method in {
            "top", "slip", "composite", "convlm", "muse", "pathpt",
            "focus", "vila_mil", "maple", "mscpt", "cod_mil", "wsi_five",
        }:
            batch_size = cfg.get("batch_size", 1)
            if (isinstance(batch_size, bool)
                    or not isinstance(batch_size, int)
                    or batch_size != 1):
                report.problems.append(
                    f"Method '{method}' requires batch_size=1 for "
                    "variable-length feature bags")
        if method in {"top", "slip", "composite", "convlm", "muse"}:
            _require_any_config_key(
                cfg, report, f"Method '{method}' patch bags",
                ("feature_path_column", "data_folder_s"))
        elif method == "pathpt":
            _require_any_config_key(
                cfg, report, "Method 'pathpt' patch bags",
                ("feature_path_column", "feature_root"))
        elif method == "focus":
            _require_any_config_key(
                cfg, report, "Method 'focus' high-resolution bags",
                ("feature_path_column", "feature_path_column_l",
                 "data_folder_l", "data_folder_s"))
        elif method in {"vila_mil", "maple"}:
            _require_any_config_key(
                cfg, report, f"Method '{method}' low-scale bags",
                ("feature_path_column_s", "data_folder_s"))
            _require_any_config_key(
                cfg, report, f"Method '{method}' high-scale bags",
                ("feature_path_column_l", "data_folder_l"))
        elif method == "mscpt":
            _require_any_config_key(
                cfg, report, "Method 'mscpt' low-scale bags",
                ("feature_path_column_s", "feat_data_dir"))
            if cfg.get("input_mode") != "precomputed_shared_features":
                report.problems.append(
                    "Method 'mscpt' unified loading requires "
                    "input_mode=precomputed_shared_features")
        elif method == "cod_mil":
            _require_any_config_key(
                cfg, report, "Method 'cod_mil' low-scale bags",
                ("feature_path_column_s", "data_folder_s"))
            _require_any_config_key(
                cfg, report, "Method 'cod_mil' high-scale bags",
                ("feature_path_column_l", "data_folder_l"))
            if not _configured(cfg, "cross_mag_map_dir"):
                report.problems.append(
                    "Method 'cod_mil' requires cross_mag_map_dir")
        elif method == "wsi_five":
            _require_any_config_key(
                cfg, report, "Method 'wsi_five' patch bags",
                ("feature_path_column", "feature_root", "data_folder_s",
                 "data_path"))
        elif method == "sldpc" and not _configured(cfg, "slide_features"):
            report.problems.append("Method 'sldpc' requires slide_features")

    if "prompts" in selected:
        if method == "top":
            _check_top_prompt_schema(cfg, report)
        elif method == "slip":
            _check_slip_prompt_schema(cfg, report)
        elif method in {"focus", "vila_mil", "maple"}:
            if not _configured(cfg, "text_prompt_path"):
                report.problems.append(
                    f"Method '{method}' requires text_prompt_path")
            elif method in {"focus", "vila_mil"}:
                _check_focus_prompt_schema(
                    cfg, report,
                    consumer="FOCUS" if method == "focus" else "ViLa-MIL")
            elif method == "maple":
                _check_maple_prompt_schema(cfg, report)
        elif method == "mscpt":
            if _configured(cfg, "description_prompt_path"):
                description = Path(str(cfg["description_prompt_path"]))
                if description.parent.name != "description":
                    report.problems.append(
                        "Method 'mscpt' description_prompt_path must be "
                        "inside a description directory")
                _check_mscpt_prompt_schema(cfg, description, report)
            elif (_configured(cfg, "gpt_dir")
                  and _configured(cfg, "dataset_name")):
                description = (
                    Path(str(cfg["gpt_dir"])) / "description"
                    / f"{cfg['dataset_name']}.json")
                _check_path(
                    report, "mscpt.description_prompt", description, "file")
                _check_mscpt_prompt_schema(cfg, description, report)
            else:
                report.problems.append(
                    "Method 'mscpt' requires description_prompt_path or both "
                    "gpt_dir and dataset_name")
        elif method == "muse":
            _require_any_config_key(
                cfg, report, "Method 'muse' prompt bank",
                ("prompt_features", "prompt_csvs"))
            prompt_csvs = cfg.get("prompt_csvs")
            if prompt_csvs is not None:
                if not isinstance(prompt_csvs, dict):
                    report.problems.append(
                        "Method 'muse' prompt_csvs must map classnames to paths")
                elif isinstance(cfg.get("classnames"), (list, tuple)):
                    missing = [
                        name for name in cfg["classnames"]
                        if not _configured(prompt_csvs, str(name))]
                    if missing:
                        report.problems.append(
                            "Method 'muse' prompt_csvs is missing classnames: "
                            + ", ".join(map(str, missing)))
                _check_muse_prompt_schema(cfg, report)
        elif method == "convlm":
            _require_any_config_key(
                cfg, report, "Method 'convlm' attribute bank",
                ("attribute_embeddings", "attribute_prompt_path"))
            if (_configured(cfg, "attribute_prompt_path")
                    and not _configured(cfg, "attribute_embeddings")):
                encoder = cfg.get("attribute_encoder")
                if not isinstance(encoder, dict):
                    report.problems.append(
                        "Method 'convlm' runtime attribute prompts require "
                        "an attribute_encoder mapping")
                else:
                    missing = [
                        key for key in ("model_name", "weights")
                        if not _configured(encoder, key)]
                    if missing:
                        report.problems.append(
                            "Method 'convlm' attribute_encoder requires: "
                            + ", ".join(missing))
            _check_convlm_prompt_schema(cfg, report)
        elif method == "sldpc":
            if not _configured(cfg, "prompt_reference_yaml"):
                report.problems.append(
                    "Method 'sldpc' requires prompt_reference_yaml")
            _check_sldpc_prompt_schema(cfg, report)
        elif method == "cod_mil":
            if not _configured(cfg, "text_prompt_path"):
                report.problems.append("Method 'cod_mil' requires text_prompt_path")
            if not _configured(cfg, "text_prompt_features"):
                if cfg.get("prompt_encoding") != "runtime_cached":
                    report.problems.append(
                        "Method 'cod_mil' requires text_prompt_features or "
                        "prompt_encoding=runtime_cached")
                if not _configured(cfg, "text_prompt_path"):
                    report.problems.append(
                        "Method 'cod_mil' runtime prompt encoding requires "
                        "text_prompt_path")
            elif not _configured(cfg, "text_prompt_bank_csv"):
                report.problems.append(
                    "Method 'cod_mil' precomputed prompts require "
                    "text_prompt_bank_csv")
            _check_cod_prompt_schema(cfg, report)
        elif method == "wsi_five":
            mode = cfg.get("training_mode", "simplified_classnames")
            if mode not in {"upstream_answer_bank", "simplified_classnames"}:
                report.problems.append(
                    "Method 'wsi_five' training_mode must be "
                    "upstream_answer_bank or simplified_classnames")
            else:
                if not _configured(cfg, "clinical_questions"):
                    report.problems.append(
                        "Method 'wsi_five' requires an explicit six-question "
                        "prompt bank (clinical_questions)")
                elif isinstance(cfg["clinical_questions"], (str, Path)):
                    _check_clinical_questions_schema(
                        cfg["clinical_questions"], report, expected_count=6)
                else:
                    questions = cfg["clinical_questions"]
                    if (not isinstance(questions, (list, tuple))
                            or len(questions) != 6
                            or any(not isinstance(item, str) or not item.strip()
                                   for item in questions)):
                        report.problems.append(
                            "Method 'wsi_five' clinical_questions must contain "
                            "exactly six non-empty strings")
            if mode == "upstream_answer_bank":
                for key, purpose in (
                        ("report_csv", "structured answer CSV"),
                        ("evaluation_prompt_path", "evaluation prompt bank")):
                    if not _configured(cfg, key):
                        report.problems.append(
                            f"Method 'wsi_five' native mode requires {purpose} "
                            f"({key})")
                if _configured(cfg, "report_csv"):
                    _check_report_csv_schema(cfg, report)
                if _configured(cfg, "evaluation_prompt_path"):
                    _check_wsi_five_evaluation_prompts(cfg, report)
                if cfg.get("require_report") is not True:
                    report.problems.append(
                        "Method 'wsi_five' native mode requires "
                        "require_report=true for training answers")
            elif mode == "simplified_classnames":
                if cfg.get("require_report", False):
                    report.problems.append(
                        "Method 'wsi_five' simplified mode must not require "
                        "privileged report text")
                if _configured(cfg, "report_csv"):
                    _check_report_csv_schema(cfg, report)
                    report.warnings.append(
                        "WSI-FiVE simplified mode ignores report_csv; per-slide "
                        "text is not supplied at evaluation")

    if ("encoders" in selected and method == "wsi_five"
            and not _configured(cfg, "clinicalbert_weights")):
        report.problems.append(
            "Method 'wsi_five' requires clinicalbert_weights")


def _check_runtime_settings(
    cfg: dict[str, Any], report: "PreflightReport", selected: set[str],
) -> None:
    """Validate generic safety knobs used by the unified training loop."""
    if "assets" not in selected:
        return
    if "max_batch_failure_rate" in cfg:
        value = cfg["max_batch_failure_rate"]
        if isinstance(value, bool):
            report.problems.append(
                "max_batch_failure_rate must be a number in [0, 1], not a boolean")
        else:
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                report.problems.append(
                    "max_batch_failure_rate must be a number in [0, 1], "
                    f"got {value!r}")
            else:
                if not 0.0 <= fraction <= 1.0:
                    report.problems.append(
                        "max_batch_failure_rate must be in [0, 1], "
                        f"got {fraction}")

    for key in ("epochs", "n_classes", "batch_size", "num_workers"):
        if key not in cfg:
            continue
        value = cfg[key]
        minimum = 0 if key == "num_workers" else 2 if key == "n_classes" else 1
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < minimum):
            report.problems.append(
                f"{key} must be an integer >= {minimum}, got {value!r}")
    for key in (
            "feature_dim", "max_patches", "patch_num", "num_frames",
            "num_k", "high_patch_topk", "prototype_number", "hidden_size",
            "window_size", "max_context_length", "embed_dim", "depth",
            "num_heads", "num_experts", "num_selected", "retrieval_k",
            "text_batch_size", "n_high", "n_set", "n_tpro", "n_vpro",
            "target_size", "learnable_prompts", "prompt_context_length",
            "context_size", "image_size", "topk"):
        value = cfg.get(key)
        if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
                or value <= 0):
            report.problems.append(
                f"{key} must be a positive integer or null, got {value!r}")

    for key in ("seed", "es_stop_epoch", "n_ctx", "bag_n_ctx",
                "n_ctx_bag", "n_ctx_inst", "attr_edge_topk"):
        value = cfg.get(key)
        if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
                or value < 0):
            report.problems.append(
                f"{key} must be a non-negative integer, got {value!r}")
    if "es_patience" in cfg:
        value = cfg["es_patience"]
        if (isinstance(value, bool) or not isinstance(value, int)
                or value <= 0):
            report.problems.append(
                f"es_patience must be a positive integer, got {value!r}")

    positive_numbers = (
        "lr", "lr_TB", "lr_IB", "stage1_lr", "stage2_lr", "tau",
        "temp", "context_gain", "logit_scale", "mlp_ratio")
    nonnegative_numbers = (
        "weight_decay", "weight_lossA", "loss_global_alignment", "loss_sr")
    for key, minimum_inclusive in (
            *((key, False) for key in positive_numbers),
            *((key, True) for key in nonnegative_numbers)):
        if key not in cfg:
            continue
        value = cfg[key]
        if isinstance(value, bool):
            valid = False
        else:
            try:
                number = float(value)
                valid = math.isfinite(number) and (
                    number >= 0.0 if minimum_inclusive else number > 0.0)
            except (TypeError, ValueError):
                valid = False
        if not valid:
            relation = "non-negative" if minimum_inclusive else "positive"
            report.problems.append(
                f"{key} must be a finite {relation} number, got {value!r}")

    for key in ("dropout", "drop", "p_drop_out", "p_bag_drop_out"):
        if key not in cfg:
            continue
        value = cfg[key]
        try:
            dropout = float(value)
            valid_dropout = (
                not isinstance(value, bool) and math.isfinite(dropout)
                and 0.0 <= dropout < 1.0)
        except (TypeError, ValueError):
            valid_dropout = False
        if not valid_dropout:
            report.problems.append(
                f"{key} must be a finite number in [0, 1), got {value!r}")

    for key in ("keep_rate", "pos_ratio"):
        if key not in cfg:
            continue
        value = cfg[key]
        try:
            fraction = float(value)
            valid_fraction = (
                not isinstance(value, bool) and math.isfinite(fraction)
                and 0.0 < fraction <= 1.0)
        except (TypeError, ValueError):
            valid_fraction = False
        if not valid_fraction:
            report.problems.append(
                f"{key} must be a finite number in (0, 1], got {value!r}")

    for key in ("entity_weight", "neg_ratio", "omega"):
        if key not in cfg:
            continue
        value = cfg[key]
        try:
            fraction = float(value)
            valid_fraction = (
                not isinstance(value, bool) and math.isfinite(fraction)
                and 0.0 <= fraction <= 1.0)
        except (TypeError, ValueError):
            valid_fraction = False
        if not valid_fraction:
            report.problems.append(
                f"{key} must be a finite number in [0, 1], got {value!r}")

    if "sim_threshold" in cfg:
        value = cfg["sim_threshold"]
        try:
            threshold = float(value)
            valid_threshold = (
                not isinstance(value, bool) and math.isfinite(threshold)
                and -1.0 <= threshold <= 1.0)
        except (TypeError, ValueError):
            valid_threshold = False
        if not valid_threshold:
            report.problems.append(
                "sim_threshold must be a finite number in [-1, 1], "
                f"got {value!r}")

    embed_dim, num_heads = cfg.get("embed_dim"), cfg.get("num_heads")
    if (isinstance(embed_dim, int) and not isinstance(embed_dim, bool)
            and isinstance(num_heads, int) and not isinstance(num_heads, bool)
            and embed_dim > 0 and num_heads > 0
            and embed_dim % num_heads != 0):
        report.problems.append(
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
    num_selected, num_experts = (
        cfg.get("num_selected"), cfg.get("num_experts"))
    if (isinstance(num_selected, int) and not isinstance(num_selected, bool)
            and isinstance(num_experts, int) and not isinstance(num_experts, bool)
            and num_selected > num_experts > 0):
        report.problems.append(
            f"num_selected={num_selected} cannot exceed "
            f"num_experts={num_experts}")

    for key in (
            "early_stopping", "evaluate_initial", "evaluate_test",
            "include_metadata", "require_upstream_fidelity", "drop_out",
            "all_ctx_trainable", "csc", "vision_only", "vision_grad",
            "use_aug", "prompt_select", "balance", "enable_pseudo",
            "pathpt_synthetic_normal", "stage1_apply_tau", "local_files_only",
            "require_report"):
        if key in cfg and not isinstance(cfg[key], bool):
            report.problems.append(
                f"{key} must be a boolean, got {cfg[key]!r}")

    method_name = str(cfg.get("method", "")).lower().replace("-", "_")
    method_name = {
        "vila": "vila_mil", "vilamil": "vila_mil",
        "codmil": "cod_mil", "five": "wsi_five",
        "wsifive": "wsi_five",
    }.get(method_name, method_name)
    choices: dict[tuple[str, str], set[str]] = {
        ("focus", "trainable_scope"): {"all", "soft_context"},
        ("pathpt", "prompt_init"): {"template", "rand"},
        ("pathpt", "learnable"): {"token", "embedding", "both"},
        ("pathpt", "training_mode"): {
            "simplified_slide_ce", "upstream_patch_ssl"},
        ("mscpt", "input_mode"): {"precomputed_shared_features"},
        ("mscpt", "precision"): {"fp16", "fp32", "amp"},
        ("sldpc", "class_token_position"): {"end", "front", "middle"},
        ("top", "pooling_strategy"): {
            "NoCoOp", "ABMIL", "mean", "max", "first-one", "CoOp",
            "learnablePrompt", "learnablePrompt_noCoOp",
            "learnablePrompt_argmax", "learnablePrompt_multi",
            "learnablePrompt_multi_noCoOp",
        },
    }
    for (consumer, key), allowed in choices.items():
        if method_name != consumer or key not in cfg:
            continue
        value = cfg[key]
        if not isinstance(value, str) or value not in allowed:
            report.problems.append(
                f"{consumer} {key} must be one of {sorted(allowed)}, "
                f"got {value!r}")

    if (method_name == "pathpt"
            and cfg.get("training_mode") == "upstream_patch_ssl"):
        for key, default in (
                ("prompt_classifier_count", 200),
                ("prompt_select_count", 100),
                ("prompt_top_patches", 100),
                ("eval_patch_batch_size", 50_000)):
            value = cfg.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                report.problems.append(
                    f"PathPT {key} must be a positive integer, got {value!r}")
        classifier_count = cfg.get("prompt_classifier_count", 200)
        select_count = cfg.get("prompt_select_count", 100)
        if (isinstance(classifier_count, int)
                and isinstance(select_count, int)
                and not isinstance(classifier_count, bool)
                and not isinstance(select_count, bool)
                and select_count > classifier_count):
            report.problems.append(
                "PathPT prompt_select_count cannot exceed "
                "prompt_classifier_count")
        if cfg.get("vision_only", False):
            report.problems.append(
                "PathPT upstream_patch_ssl requires vision_only=false")
        try:
            from methods.pathpt.prompts import resolve_prompt_bank
            bank = resolve_prompt_bank(cfg)
        except (KeyError, TypeError, ValueError) as error:
            report.problems.append(f"PathPT prompt bank is invalid: {error}")
        else:
            declared = cfg.get("prompt_provenance")
            if declared and declared != bank.provenance:
                report.problems.append(
                    "PathPT prompt_provenance contradicts the active bank: "
                    f"declared {declared!r}, expected {bank.provenance!r}")
            declared_source = cfg.get("pathpt_prompt_bank_source")
            if declared_source and declared_source != bank.source:
                report.problems.append(
                    "PathPT pathpt_prompt_bank_source contradicts the active "
                    f"bank: declared {declared_source!r}, expected "
                    f"{bank.source!r}")
            declared_normal = cfg.get("pathpt_synthetic_normal")
            if (declared_normal is not None
                    and declared_normal is not bank.synthetic_normal):
                report.problems.append(
                    "PathPT pathpt_synthetic_normal contradicts the active "
                    f"bank: declared {declared_normal!r}, expected "
                    f"{bank.synthetic_normal!r}")
            if bank.task == "camelyon16":
                report.warnings.append(
                    "PathPT CAMELYON preserves the upstream malformed "
                    "concatenated Normal synonym and uses a disclosed local "
                    "binary slide-classification adaptation")

    if method_name == "top":
        strategy = cfg.get("pooling_strategy", "learnablePrompt_multi")
        raw_classes = cfg.get("n_classes")
        class_count = (
            raw_classes if isinstance(raw_classes, int)
            and not isinstance(raw_classes, bool) else
            len(cfg["label_dict"]) if isinstance(cfg.get("label_dict"), dict)
            else len(cfg["classnames"])
            if isinstance(cfg.get("classnames"), list) else None)
        binary_only = {
            "NoCoOp", "learnablePrompt_noCoOp",
            "learnablePrompt_multi_noCoOp",
        }
        if strategy in binary_only and class_count is not None \
                and class_count != 2:
            report.problems.append(
                f"TOP pooling_strategy={strategy!r} is binary-only in the "
                f"upstream implementation, but the task has {class_count} "
                "classes; use a prompt-classifier pooling strategy")

    if "lr_milestones" in cfg:
        milestones = cfg["lr_milestones"]
        if (not isinstance(milestones, list)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 0 for value in milestones)
                or milestones != sorted(set(milestones))):
            report.problems.append(
                "lr_milestones must be a sorted list of unique non-negative "
                "integers")

    if "seen_class_indices" in cfg:
        indices = cfg["seen_class_indices"]
        if (not isinstance(indices, list) or not indices
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in indices)
                or len(indices) != len(set(indices))
                or (isinstance(cfg.get("n_classes"), int)
                    and any(value < 0 or value >= cfg["n_classes"]
                            for value in indices))):
            report.problems.append(
                "seen_class_indices must be a non-empty list of unique class "
                "indices in [0, n_classes)")

    if method_name == "sldpc":
        projection = cfg.get("slide_projection", {})
        if not isinstance(projection, dict):
            report.problems.append("SLDPC slide_projection must be a mapping")
        else:
            mode = projection.get("mode", "native")
            if not isinstance(mode, str) or mode.lower() not in {
                    "native", "linear", "mlp"}:
                report.problems.append(
                    "SLDPC slide_projection.mode must be native, linear, or "
                    f"mlp, got {mode!r}")
            for key in ("input_dim", "output_dim", "hidden_dim"):
                value = projection.get(key)
                if value is not None and (
                        isinstance(value, bool) or not isinstance(value, int)
                        or value <= 0):
                    report.problems.append(
                        f"SLDPC slide_projection.{key} must be a positive "
                        f"integer or null, got {value!r}")
            if "dropout" in projection:
                value = projection["dropout"]
                try:
                    dropout = float(value)
                    valid = (not isinstance(value, bool)
                             and math.isfinite(dropout)
                             and 0.0 <= dropout < 1.0)
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    report.problems.append(
                        "SLDPC slide_projection.dropout must be a finite "
                        f"number in [0, 1), got {value!r}")
        stage1 = cfg.get("stage1_epochs", 50)
        stage2 = cfg.get("stage2_epochs", 50)
        for key, value in (("stage1_epochs", stage1),
                           ("stage2_epochs", stage2)):
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                report.problems.append(
                    f"{key} must be a positive integer, got {value!r}")
        if "epochs" not in cfg:
            report.problems.append(
                "SLDPC requires explicit epochs equal to "
                "stage1_epochs + stage2_epochs")
        if (isinstance(stage1, int) and not isinstance(stage1, bool)
                and isinstance(stage2, int) and not isinstance(stage2, bool)
                and isinstance(cfg.get("epochs"), int)
                and not isinstance(cfg.get("epochs"), bool)
                and cfg["epochs"] != stage1 + stage2):
            report.problems.append(
                "SLDPC epochs must equal stage1_epochs + stage2_epochs")
        if cfg.get("early_stopping", True) is not False:
            report.problems.append(
                "SLDPC requires early_stopping=false so both stages run")
        monitor = str(cfg.get("monitor_metric", "F1")).upper()
        if monitor not in {"F1", "ACC", "AUC", "LOSS", "VAL_LOSS"}:
            report.problems.append(
                "SLDPC monitor_metric must be F1, ACC, AUC, or val_loss")

    label_dict = cfg.get("label_dict")
    n_classes = cfg.get("n_classes")
    if label_dict is not None and not isinstance(label_dict, dict):
        report.problems.append("label_dict must be a mapping of labels to indices")
    elif isinstance(label_dict, dict):
        values = list(label_dict.values())
        if (any(isinstance(value, bool) or not isinstance(value, int)
                for value in values)
                or sorted(values) != list(range(len(values)))):
            report.problems.append(
                "label_dict values must be unique contiguous indices starting at 0")
        if isinstance(n_classes, int) and len(values) != n_classes:
            report.problems.append(
                f"label_dict has {len(values)} classes but n_classes={n_classes}")
    classnames = cfg.get("classnames")
    if classnames is not None and not isinstance(classnames, list):
        report.problems.append("classnames must be a list")
    elif isinstance(classnames, list):
        invalid_names = [
            value for value in classnames
            if not isinstance(value, str) or not value.strip()]
        if invalid_names:
            report.problems.append(
                "classnames entries must be non-empty strings")
        valid_names = [
            value for value in classnames
            if isinstance(value, str) and value.strip()]
        if len(set(valid_names)) != len(valid_names):
            report.problems.append("classnames entries must be unique")
        if isinstance(n_classes, int) and not isinstance(n_classes, bool) \
                and len(classnames) != n_classes:
            report.problems.append(
                f"classnames has {len(classnames)} entries but "
                f"n_classes={n_classes}")

    source_type = cfg.get("source_type")
    if source_type is not None and source_type not in {
            "pkl", "per_slide_h5", "per_slide_torch", "per_slide_pth"}:
        report.problems.append(
            "source_type must be pkl, per_slide_h5, or per_slide_torch, "
            f"got {source_type!r}")

    method = cfg.get("method")
    if method:
        try:
            provenance = method_provenance(str(method), cfg)
        except KeyError as error:
            report.problems.append(str(error))
        else:
            declared = cfg.get("implementation_provenance")
            if declared and declared != provenance.implementation:
                report.problems.append(
                    "implementation_provenance contradicts the adapter registry: "
                    f"declared {declared!r}, expected {provenance.implementation!r}")
            declared_fidelity = cfg.get("upstream_fidelity")
            if declared_fidelity and declared_fidelity != provenance.upstream_fidelity:
                report.problems.append(
                    "upstream_fidelity contradicts the adapter registry: "
                    f"declared {declared_fidelity!r}, expected "
                    f"{provenance.upstream_fidelity!r}")
            if provenance.upstream_fidelity == "partial":
                message = f"Method '{method}' is partial: {provenance.note}"
                if cfg.get("require_upstream_fidelity", False):
                    report.problems.append(message)
                else:
                    report.warnings.append(message)


@dataclass
class PreflightReport:
    """Outcome of inspecting one run configuration."""

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    checked_paths: dict[str, dict[str, Any]] = field(default_factory=dict)
    checks: list[str] = field(default_factory=list)
    slides_expected: int = 0
    slides_available: int = 0
    deep_features_checked: int = 0

    @property
    def ok(self) -> bool:
        """Return True when the configuration can produce a real result."""
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "problems": self.problems,
            "warnings": self.warnings,
            "coverage": self.coverage,
            "checked_paths": self.checked_paths,
            "checks": self.checks,
            "slides_expected": self.slides_expected,
            "slides_available": self.slides_available,
            "deep_features_checked": self.deep_features_checked,
        }

    def summary(self) -> str:
        return "; ".join(self.problems) if self.problems else "ready"


def _exists(value: Any) -> bool:
    """Return True when the path exists and is not an empty placeholder.

    An interrupted download leaves a zero-byte file behind, which satisfies a
    plain existence check and then fails at model build with an opaque
    ``EOFError: Ran out of input``. Treating an empty file as absent turns that
    into a skip naming the actual asset.
    """
    if not value:
        return False
    try:
        path = expand_path(value)
        if not os.path.exists(path):
            return False
        if os.path.isfile(path) and os.path.getsize(path) == 0:
            return False
        if os.path.isdir(path) and not os.listdir(path):
            return False
    except (OSError, ValueError):
        return False
    return True


def _path_detail(value: Any) -> dict[str, Any]:
    """Return a diagnostic record for a configured filesystem path.

    The boolean result remains compatible with the original preflight report,
    while the additional fields let the CLI explain *why* a path failed.  In
    particular, an empty checkpoint and a missing checkpoint need different
    remedies even though neither can be used for a run.
    """
    detail: dict[str, Any] = {
        "path": str(value),
        "resolved_path": None,
        "available": False,
        "kind": None,
        "size_bytes": None,
        "reason": None,
    }
    if not value:
        detail["reason"] = "path is empty"
        return detail
    try:
        resolved = expand_path(value)
        detail["resolved_path"] = resolved
        path = Path(resolved)
        if not path.exists():
            detail["reason"] = "path does not exist"
            return detail
        if path.is_file():
            detail["kind"] = "file"
            detail["size_bytes"] = path.stat().st_size
            if detail["size_bytes"] == 0:
                detail["reason"] = "file is empty"
                return detail
            if not os.access(path, os.R_OK):
                detail["reason"] = "file is not readable"
                return detail
        elif path.is_dir():
            detail["kind"] = "directory"
            if not os.access(path, os.R_OK | os.X_OK):
                detail["reason"] = "directory is not readable/searchable"
                return detail
            try:
                next(path.iterdir())
            except StopIteration:
                detail["reason"] = "directory is empty"
                return detail
        else:
            detail["kind"] = "other"
            detail["reason"] = "path is not a regular file or directory"
            return detail
        detail["available"] = True
        return detail
    except ValueError as error:
        detail["reason"] = str(error)
    except OSError as error:
        detail["reason"] = f"cannot inspect path: {error}"
    return detail


def _feature_file_exists(value: Any) -> bool:
    """Return True only for a non-empty feature file."""
    return _feature_file_status(value)[0]


def _feature_file_status(value: Any) -> tuple[bool, str | None]:
    """Return feature availability plus an actionable inspection failure."""
    if not isinstance(value, str) or not value.strip():
        return False, None
    try:
        value = expand_path(value)
        if not os.path.isfile(value) or os.path.getsize(value) == 0:
            return False, None
        if not os.access(value, os.R_OK):
            return False, "feature file is not readable"
        return True, None
    except ValueError as error:
        return False, str(error)
    except OSError as error:
        return False, f"cannot inspect feature path: {error}"


def _inspect_feature_payload(
    value: str, feature_key: str, expected_dim: int | None,
    slide_embedding: bool, *, source_type: str | None = None,
    slide_id_key: str = "filenames",
    required_slide_ids: Iterable[str] | None = None,
) -> str | None:
    """Open one feature file and return a semantic validation error."""
    try:
        path = Path(expand_path(value))
        suffix = path.suffix.lower()
        shared_pickle = source_type == "pkl" or suffix in {".pkl", ".pickle"}
        if suffix in {".h5", ".hdf5"} and not shared_pickle:
            import h5py
            import numpy as np
            with h5py.File(path, "r") as handle:
                keys = (feature_key, "features", "embeddings", "feats")
                key = next((item for item in keys if item in handle), None)
                if key is None:
                    return (
                        f"{path}: no feature key {feature_key!r}; available "
                        f"keys: {list(handle.keys())}")
                dataset = handle[key]
                shape = tuple(dataset.shape)
                if dataset.size == 0:
                    return f"{path}: feature tensor is empty"
                if dataset.ndim == 0:
                    finite = bool(np.isfinite(dataset[()]).all())
                else:
                    # Deep mode is an integrity check, not a sample. Scan in
                    # bounded chunks so a corrupt value in the middle of a
                    # large slide cannot pass without loading the whole bag.
                    finite = all(
                        bool(np.isfinite(
                            dataset[start:start + 4096]).all())
                        for start in range(0, dataset.shape[0], 4096))
        elif suffix in {".pt", ".pth"} and not shared_pickle:
            import torch
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # torch < 2.0
                payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict):
                keys = (feature_key, "features", "embeddings", "feats")
                key = next((item for item in keys if item in payload), None)
                if key is None:
                    return (
                        f"{path}: no feature key {feature_key!r}; available "
                        f"keys: {list(payload.keys())}")
                payload = payload[key]
            tensor = torch.as_tensor(payload)
            shape = tuple(tensor.shape)
            finite = bool(torch.isfinite(tensor).all())
        elif shared_pickle:
            import pickle
            import numpy as np
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                return f"{path}: pickle payload must be a mapping"
            missing_keys = [
                key for key in (feature_key, slide_id_key)
                if key not in payload]
            if missing_keys:
                return (
                    f"{path}: pickle payload is missing keys {missing_keys}; "
                    f"available keys: {list(payload.keys())}")
            values = np.asarray(payload[feature_key])
            shape = tuple(values.shape)
            if values.ndim != 2 or values.shape[0] == 0:
                return (
                    f"{path}: pickle slide embeddings must have shape "
                    f"[slides, dimension], got {shape}")
            identifiers = payload[slide_id_key]
            if isinstance(identifiers, (str, bytes)):
                return f"{path}: {slide_id_key!r} must be a sequence of IDs"
            try:
                identifiers = list(identifiers)
            except TypeError:
                return f"{path}: {slide_id_key!r} must be a sequence of IDs"
            if len(identifiers) != values.shape[0]:
                return (
                    f"{path}: pickle has {len(identifiers)} slide IDs for "
                    f"{values.shape[0]} embeddings")
            normalized_ids = []
            known_suffixes = {
                ".pt", ".pth", ".h5", ".hdf5", ".svs", ".tif", ".tiff"}
            for identifier in identifiers:
                if isinstance(identifier, (bytes, bytearray, memoryview)):
                    try:
                        text = bytes(identifier).decode("utf-8").strip()
                    except UnicodeDecodeError as error:
                        return (
                            f"{path}: pickle slide ID is not valid UTF-8 "
                            f"({error})")
                else:
                    text = str(identifier).strip()
                if not text:
                    return f"{path}: pickle contains a blank slide ID"
                identifier_path = Path(text)
                normalized_ids.append(
                    identifier_path.stem
                    if identifier_path.suffix.lower() in known_suffixes
                    else identifier_path.name)
            duplicates = _duplicate_values(normalized_ids)
            if duplicates:
                return (
                    f"{path}: pickle has duplicate normalized slide IDs "
                    f"({', '.join(duplicates[:3])})")
            if required_slide_ids is not None:
                required = []
                for identifier in required_slide_ids:
                    identifier_path = Path(str(identifier).strip())
                    required.append(
                        identifier_path.stem
                        if identifier_path.suffix.lower() in known_suffixes
                        else identifier_path.name)
                missing = sorted(set(required) - set(normalized_ids))
                if missing:
                    return (
                        f"{path}: pickle is missing {len(missing)} manifest "
                        f"slide IDs ({', '.join(missing[:3])})")
            finite = bool(np.isfinite(values).all())
        else:
            return f"{path}: unsupported deep-inspection suffix {suffix!r}"
    except Exception as error:  # doctor turns decoder failures into diagnoses
        return f"{value}: cannot decode feature payload ({error})"

    if shared_pickle:
        valid_shape = len(shape) == 2 and shape[0] > 0
        width = shape[-1] if shape else 0
        expected_shape = "[slides, dimension]"
    elif slide_embedding:
        valid_shape = (
            len(shape) == 1 or (len(shape) == 2 and shape[0] == 1))
        width = shape[-1] if shape else 0
        expected_shape = "[dimension] or [1, dimension]"
    else:
        valid_shape = len(shape) == 2 and shape[0] > 0
        width = shape[-1] if shape else 0
        expected_shape = "[patches, dimension]"
    if not valid_shape:
        return f"{value}: expected {expected_shape}, got {shape}"
    if expected_dim is not None and width != expected_dim:
        return f"{value}: feature width {width}, expected {expected_dim}"
    if not finite:
        return f"{value}: feature payload contains NaN or infinity"
    return None


def _minimum_feature_coverage(cfg: dict[str, Any],
                              report: PreflightReport) -> float:
    """Read and validate the required fraction of complete feature rows."""
    value = cfg.get("min_feature_coverage", 1.0)
    if isinstance(value, bool):
        report.problems.append(
            "min_feature_coverage must be a number in [0, 1], not a boolean")
        return 1.0
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        report.problems.append(
            f"min_feature_coverage must be a number in [0, 1], got {value!r}")
        return 1.0
    if not 0.0 <= threshold <= 1.0:
        report.problems.append(
            f"min_feature_coverage must be in [0, 1], got {threshold}")
        return 1.0
    return threshold


def _asset_paths(value: Any) -> list[str]:
    """Extract asset paths from any of the shapes listed on ASSET_LIST_KEYS."""
    if not value:
        return []
    if isinstance(value, dict):
        # classname -> path, e.g. MUSE's per-class description CSVs
        entries: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple)):
        entries = value
    else:
        entries = [value]

    paths: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            entry = entry.get("path")
        if isinstance(entry, str) and entry:
            paths.append(entry)
    return paths


def _check_path(
    report: PreflightReport, label: str, value: Any,
    expected_kind: str | None = None,
) -> None:
    """Record and validate one configured path."""
    if label in report.checked_paths:
        return
    detail = _path_detail(value)
    if (detail["available"] and expected_kind is not None
            and detail["kind"] != expected_kind):
        detail["available"] = False
        detail["reason"] = (
            f"expected a {expected_kind}, found {detail['kind']}")
    report.checked_paths[label] = detail
    if not detail["available"]:
        reason = detail["reason"] or "path is unavailable"
        report.problems.append(f"{label}: {reason}: {value}")


def _check_output_path(report: PreflightReport, label: str, value: Any) -> None:
    """Check that an output directory exists or can be created read-only."""
    if label in report.checked_paths:
        return
    detail: dict[str, Any] = {
        "path": str(value),
        "resolved_path": None,
        "available": False,
        "kind": "output_directory",
        "size_bytes": None,
        "reason": None,
    }
    if not isinstance(value, (str, Path)) or not str(value).strip():
        detail["reason"] = "output path is empty or not path-like"
        report.checked_paths[label] = detail
        report.problems.append(f"{label}: {detail['reason']}: {value}")
        return
    try:
        resolved = Path(expand_path(value))
        detail["resolved_path"] = str(resolved)
        canonical = resolved.resolve(strict=False)
        if canonical == Path(canonical.anchor) or canonical == REPO_ROOT.resolve():
            detail["reason"] = (
                "output path is an unsafe broad directory; choose a dedicated "
                "run directory")
        elif resolved.exists():
            if not resolved.is_dir():
                detail["reason"] = "output path exists but is not a directory"
            elif not os.access(resolved, os.W_OK | os.X_OK):
                detail["reason"] = "output directory is not writable/searchable"
            else:
                detail["available"] = True
        else:
            parent = resolved.parent
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not parent.is_dir():
                detail["reason"] = (
                    f"closest existing parent is not a directory: {parent}")
            elif not os.access(parent, os.W_OK | os.X_OK):
                detail["reason"] = (
                    f"closest existing parent is not writable/searchable: "
                    f"{parent}")
            else:
                detail["kind"] = "creatable_output_directory"
                detail["available"] = True
    except ValueError as error:
        detail["reason"] = str(error)
    except (OSError, RuntimeError) as error:
        detail["reason"] = f"cannot inspect output path: {error}"
    report.checked_paths[label] = detail
    if not detail["available"]:
        reason = detail["reason"] or "output path is unavailable"
        report.problems.append(f"{label}: {reason}: {value}")


def _iter_split_files(split_dir: Path, k_start: int, k_end: int) -> Iterable[Path]:
    """Yield the split files for each fold, in either supported layout."""
    for fold in range(k_start, k_end):
        flat = split_dir / f"splits_{fold}.csv"
        upstream_flat = split_dir / f"fold{fold}.csv"
        nested = split_dir / f"fold{fold}"
        # A nested fold takes precedence in the loaders that support both.
        if nested.is_dir():
            yield nested
        elif _exists(flat):
            yield flat
        elif _exists(upstream_flat):
            yield upstream_flat
        elif (k_end - k_start == 1 and all(
                _exists(split_dir / f"{phase}.csv")
                for phase in ("train", "val", "test"))):
            yield split_dir
        else:
            yield flat  # also names the conventional file when neither exists


def _feature_columns(cfg: dict[str, Any]) -> list[str]:
    # Two inputs may intentionally use the same cached feature column. Avoid
    # scanning it twice and emitting duplicate warnings while preserving the
    # configuration's insertion order.
    return list(dict.fromkeys(
        str(value) for key, value in cfg.items()
        if key.startswith("feature_path_column") and value))


def _check_features(cfg: dict[str, Any], report: PreflightReport,
                    threshold: float, *, deep: bool = False) -> None:
    """Measure how many of the manifest's referenced feature files exist."""
    manifest = cfg.get("dataset_csv")
    if not _exists(manifest):
        return                                   # already reported as a problem

    columns = _feature_columns(cfg)
    if not columns:
        return              # method resolves features by root + slide id instead

    try:
        with open(expand_path(manifest), newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        report.problems.append(f"cannot read manifest {manifest}: {error}")
        return

    if not fields:
        report.problems.append(f"manifest has no header: {manifest}")
        return
    blank_fields = [index for index, field in enumerate(fields) if not field.strip()]
    duplicate_fields = sorted({field for field in fields if fields.count(field) > 1})
    malformed = sum(
        None in row or any(row.get(field) is None for field in fields)
        for row in rows)
    if blank_fields:
        report.problems.append(
            f"manifest has blank header fields at positions {blank_fields}: "
            f"{manifest}")
    if duplicate_fields:
        report.problems.append(
            f"manifest has duplicate columns {duplicate_fields}: {manifest}")
    if malformed:
        report.problems.append(
            f"manifest has {malformed} rows with the wrong number of fields: "
            f"{manifest}")
    if blank_fields or duplicate_fields or malformed:
        return
    if not rows:
        report.problems.append(f"manifest has no rows: {manifest}")
        return
    if "slide_id" not in fields:
        report.problems.append(
            f"manifest has no 'slide_id' column: {manifest}")
        return
    slide_ids = [_cell(row, "slide_id") for row in rows]
    blank_slide_ids = sum(not value for value in slide_ids)
    duplicate_slide_ids = _duplicate_values(
        [value for value in slide_ids if value])
    if blank_slide_ids:
        report.problems.append(
            f"manifest has {blank_slide_ids} blank slide IDs: {manifest}")
    if duplicate_slide_ids:
        sample = ", ".join(duplicate_slide_ids[:3])
        report.problems.append(
            f"manifest repeats {len(duplicate_slide_ids)} slide IDs "
            f"({sample}): {manifest}")

    availability_by_column: dict[str, tuple[bool, ...]] = {}
    deep_cache: dict[tuple[str, str, int | None, bool], str | None] = {}
    for column in columns:
        if column not in fields:
            report.problems.append(
                f"manifest {manifest} has no column {column!r}")
            continue
        statuses = tuple(
            _feature_file_status(row.get(column)) for row in rows)
        availability = tuple(status[0] for status in statuses)
        inspection_failures = sorted({
            status[1] for status in statuses if status[1] is not None})
        if inspection_failures:
            report.problems.append(
                f"cannot resolve or inspect feature paths in {column!r}: "
                + "; ".join(inspection_failures[:3]))
        availability_by_column[column] = availability
        resolved_references: list[str] = []
        for row in rows:
            value = row.get(column)
            if not isinstance(value, str) or not value.strip():
                continue
            try:
                resolved_references.append(str(Path(expand_path(value))))
            except (OSError, UnicodeError, ValueError):
                # The inspection failure above already reports this path.
                continue
        duplicate_references = _duplicate_values(resolved_references)
        if duplicate_references and cfg.get("source_type") != "pkl":
            report.problems.append(
                f"manifest column {column!r} maps multiple slide IDs to "
                f"{len(duplicate_references)} identical feature paths "
                f"({', '.join(duplicate_references[:3])})")
        available = sum(availability)
        expected = len(rows)
        fraction = available / expected if expected else 0.0
        report.coverage[column] = round(fraction, 4)
        report.slides_expected = max(report.slides_expected, expected)

        if available == 0:
            report.problems.append(
                f"no feature files exist for {column} "
                f"(0 of {expected} slides)")
        elif fraction < threshold:
            report.problems.append(
                f"feature coverage for {column!r} is {available}/{expected} "
                f"({fraction:.1%}), below the required {threshold:.1%}")
        elif available < expected:
            report.warnings.append(
                f"{expected - available} of {expected} slides have no features "
                f"for {column!r}; partial coverage was explicitly allowed "
                f"by min_feature_coverage={threshold:g}")

        # A shared pickle is one dataset-level store, not one payload per
        # manifest row. It is inspected once below from ``slide_features``.
        if deep and cfg.get("source_type") != "pkl":
            feature_key = str(cfg.get("feature_key", "features"))
            raw_dim = cfg.get("feature_dim")
            expected_dim = (
                raw_dim if isinstance(raw_dim, int)
                and not isinstance(raw_dim, bool) and raw_dim > 0 else None)
            input_kinds = set(cfg.get("feature_input_kinds", {}).values())
            slide_embedding = (
                cfg.get("source_type") is not None
                or input_kinds == {"slide_embedding"})
            errors: list[str] = []
            for row, is_available in zip(rows, availability):
                value = row.get(column)
                if not is_available or not isinstance(value, str):
                    continue
                cache_key = (
                    value, feature_key, expected_dim, slide_embedding)
                if cache_key not in deep_cache:
                    deep_cache[cache_key] = _inspect_feature_payload(
                        value, feature_key, expected_dim, slide_embedding,
                        source_type=str(cfg.get("source_type", "")).strip()
                        or None,
                        slide_id_key=str(cfg.get("slide_id_key", "filenames")))
                    report.deep_features_checked += 1
                error = deep_cache[cache_key]
                if error:
                    errors.append(error)
            if errors:
                report.problems.append(
                    f"deep feature validation failed for {len(errors)} files "
                    f"in {column!r}: " + "; ".join(errors[:3]))

    # Multi-input methods can run only on the intersection, not the least
    # incomplete column. Gate on the exact joint coverage as well.
    if availability_by_column:
        complete_rows = tuple(
            all(values) for values in zip(*availability_by_column.values()))
        complete = sum(complete_rows)
        expected = len(rows)
        fraction = complete / expected if expected else 0.0
        report.coverage["all_required_features"] = round(fraction, 4)
        report.slides_expected = expected
        report.slides_available = complete
        if len(availability_by_column) > 1 and complete == 0:
            report.problems.append(
                "no manifest row has every required feature file")
        elif len(availability_by_column) > 1 and fraction < threshold:
            report.problems.append(
                f"joint feature coverage is {complete}/{expected} "
                f"({fraction:.1%}), below the required {threshold:.1%}")


def _check_splits(cfg: dict[str, Any], report: PreflightReport,
                  split_dir: Path) -> None:
    """Validate configured flat and nested fold definitions."""
    manifest_slide_ids: set[str] | None = None
    manifest_case_ids: dict[str, str] | None = None
    manifest_rows: dict[str, dict[str | None, Any]] | None = None
    manifest = cfg.get("dataset_csv")
    if _exists(manifest):
        try:
            with open(
                    expand_path(manifest), newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                rows = list(reader)
                if "slide_id" in fields:
                    manifest_values = [
                        _cell(row, "slide_id") for row in rows]
                    blank_manifest_ids = sum(
                        not value for value in manifest_values)
                    if blank_manifest_ids:
                        report.problems.append(
                            f"manifest has {blank_manifest_ids} blank slide IDs: "
                            f"{manifest}")
                    duplicate_manifest_ids = _duplicate_values(
                        [value for value in manifest_values if value])
                    if duplicate_manifest_ids:
                        report.problems.append(
                            "manifest repeats slide IDs used for split "
                            "validation: "
                            + ", ".join(duplicate_manifest_ids[:3]))
                    manifest_slide_ids = {
                        value for value in manifest_values if value}
                    manifest_rows = {
                        _cell(row, "slide_id"): row for row in rows
                        if _cell(row, "slide_id")}
                    if "case_id" in fields:
                        manifest_case_ids = {
                            _cell(row, "slide_id"): _cell(row, "case_id")
                            for row in rows
                            if (_cell(row, "slide_id")
                                and _cell(row, "case_id"))}
                else:
                    report.problems.append(
                        f"manifest has no 'slide_id' column: {manifest}")
        except (OSError, UnicodeError, csv.Error) as error:
            report.problems.append(f"cannot read manifest {manifest}: {error}")
    raw_start = cfg.get("k_start", 0)
    raw_end = cfg["k_end"] if "k_end" in cfg else cfg.get("k", 5)
    try:
        if (isinstance(raw_start, bool) or not isinstance(raw_start, int)
                or isinstance(raw_end, bool) or not isinstance(raw_end, int)):
            raise ValueError
        k_start = raw_start
        k_end = raw_end
    except (TypeError, ValueError):
        report.problems.append(
            "k_start and k_end/k must be integer fold indices")
        k_start, k_end = 0, 0
    valid_fold_range = k_start >= 0 and k_end > k_start
    if not valid_fold_range:
        report.problems.append(f"invalid fold range [{k_start}, {k_end})")

    missing = ([
        str(path) for path in _iter_split_files(split_dir, k_start, k_end)
        if not _exists(path)
    ] if valid_fold_range else [])
    if valid_fold_range and len(missing) == k_end - k_start:
        report.problems.append(f"no fold splits exist under {split_dir}")
    elif valid_fold_range and missing:
        report.problems.append(
            f"{len(missing)} of {k_end - k_start} configured fold splits "
            "are missing or empty: " + ", ".join(missing))

    # Several loaders prefer nested phase files when foldN exists, even if a
    # flat split is also available.
    incomplete_nested: list[str] = []
    for fold in range(k_start, k_end) if valid_fold_range else ():
        nested = split_dir / f"fold{fold}"
        if not nested.exists():
            continue
        for phase in ("train", "val", "test"):
            path = nested / f"{phase}.csv"
            if not _exists(path):
                incomplete_nested.append(str(path))
    if incomplete_nested:
        report.problems.append(
            "nested fold split is incomplete; missing or empty: "
            + ", ".join(incomplete_nested))

    root_phase_paths = tuple(
        split_dir / f"{phase}.csv" for phase in ("train", "val", "test"))
    root_phase_layout = valid_fold_range and k_end - k_start == 1 and any(
        path.exists() for path in root_phase_paths)
    if root_phase_layout:
        incomplete_root = [str(path) for path in root_phase_paths
                           if not _exists(path)]
        if incomplete_root:
            report.problems.append(
                "root phase split is incomplete; missing or empty: "
                + ", ".join(incomplete_root))

    for fold in range(k_start, k_end) if valid_fold_range else ():
        nested = split_dir / f"fold{fold}"
        if nested.is_dir() and not any(
                str(path) in incomplete_nested
                for path in (nested / "train.csv", nested / "val.csv",
                             nested / "test.csv")):
            _check_nested_split_contents(
                report, nested, fold, cfg, manifest_slide_ids, manifest_rows)
            continue
        if root_phase_layout and all(_exists(path) for path in root_phase_paths):
            _check_nested_split_contents(
                report, split_dir, fold, cfg, manifest_slide_ids, manifest_rows)
            continue
        conventional = split_dir / f"splits_{fold}.csv"
        upstream = split_dir / f"fold{fold}.csv"
        flat = conventional if _exists(conventional) else upstream
        if _exists(flat):
            _check_flat_split_contents(
                report, flat, fold, manifest_slide_ids, manifest_case_ids,
                manifest_rows, cfg)


def _read_split_csv(
    path: Path, report: PreflightReport,
) -> tuple[list[str], list[dict[str | None, Any]]] | None:
    """Read a split CSV and turn parser failures into doctor findings."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        report.problems.append(f"cannot read split CSV {path}: {error}")
        return None
    if not fields:
        report.problems.append(f"split CSV has no header: {path}")
        return None
    blank_fields = [index for index, field in enumerate(fields) if not field.strip()]
    duplicate_fields = sorted({field for field in fields if fields.count(field) > 1})
    if blank_fields:
        report.problems.append(
            f"split CSV has blank header fields at positions {blank_fields}: {path}")
    if duplicate_fields:
        report.problems.append(
            f"split CSV has duplicate columns {duplicate_fields}: {path}")
    malformed = sum(
        None in row or any(row.get(field) is None for field in fields)
        for row in rows)
    if malformed:
        report.problems.append(
            f"split CSV has {malformed} rows with the wrong number of fields: "
            f"{path}")
    return fields, rows


def _duplicate_values(values: list[str]) -> list[str]:
    """Return sorted non-empty values occurring more than once."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _cell(row: dict[str | None, Any], field: str) -> str:
    """Normalize one CSV cell without turning a missing value into ``'None'``."""
    value = row.get(field)
    return "" if value is None else str(value).strip()


def _check_flat_split_contents(
    report: PreflightReport, path: Path, fold: int,
    manifest_slide_ids: set[str] | None = None,
    manifest_case_ids: dict[str, str] | None = None,
    manifest_rows: dict[str, dict[str | None, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> None:
    """Validate the CLAM-style ``train,val,test`` fold representation."""
    loaded = _read_split_csv(path, report)
    if loaded is None:
        return
    fields, rows = loaded
    phases = ("train", "val", "test")
    missing = [phase for phase in phases if phase not in fields]
    if missing:
        report.problems.append(
            f"fold{fold} split {path} is missing columns: {', '.join(missing)}")
        return
    empty = [phase for phase in phases
             if not any(_cell(row, phase) for row in rows)]
    if empty:
        report.problems.append(
            f"fold{fold} split {path} has no members for: {', '.join(empty)}")
        return
    identities: dict[str, set[str]] = {}
    case_identities: dict[str, set[str]] = {}
    cfg = cfg or {}
    label_dict = cfg.get("label_dict", {})
    if not isinstance(label_dict, dict):
        label_dict = {}
    raw_n_classes = cfg.get("n_classes")
    n_classes = (
        raw_n_classes if isinstance(raw_n_classes, int)
        and not isinstance(raw_n_classes, bool) and raw_n_classes > 0
        else len(label_dict) if label_dict else None)

    def parse_label(raw: str) -> int | None:
        if raw in label_dict:
            value = label_dict[raw]
            return value if isinstance(value, int) and not isinstance(value, bool) \
                else None
        try:
            return int(raw)
        except ValueError:
            try:
                numeric = float(raw)
            except ValueError:
                return None
            return int(numeric) if numeric.is_integer() else None

    for phase in phases:
        values = [
            _cell(row, phase) for row in rows if _cell(row, phase)
        ]
        label_column = f"{phase}_label"
        if label_column in fields:
            blank_labels = sum(
                bool(_cell(row, phase)) and not _cell(row, label_column)
                for row in rows)
            orphan_labels = sum(
                not _cell(row, phase) and bool(_cell(row, label_column))
                for row in rows)
            if blank_labels:
                report.problems.append(
                    f"fold{fold} {phase} split has {blank_labels} members "
                    f"without {label_column}")
            if orphan_labels:
                report.problems.append(
                    f"fold{fold} {phase} split has {orphan_labels} labels "
                    "without slide IDs")
            for row_number, row in enumerate(rows, start=2):
                slide_id = _cell(row, phase)
                raw_label = _cell(row, label_column)
                if not slide_id or not raw_label:
                    continue
                label = parse_label(raw_label)
                if (label is None or label < 0
                        or (n_classes is not None and label >= n_classes)):
                    expected = (f"[0, {n_classes})" if n_classes is not None
                                else "a non-negative integer")
                    report.problems.append(
                        f"fold{fold} {phase} row {row_number} has unknown or "
                        f"out-of-range label {raw_label!r}; expected {expected}")
                    continue
                manifest_label = _cell(
                    (manifest_rows or {}).get(slide_id, {}), "label")
                if (manifest_label and parse_label(manifest_label) is not None
                        and label != parse_label(manifest_label)):
                    report.problems.append(
                        f"fold{fold} {phase} row {row_number} {label_column} "
                        f"does not match manifest for slide {slide_id!r}")
        else:
            missing_manifest_labels = [
                value for value in values
                if (manifest_rows is None
                    or not _cell(manifest_rows.get(value, {}), "label"))]
            if missing_manifest_labels:
                report.problems.append(
                    f"fold{fold} {phase} split has no {label_column} column "
                    "and dataset_csv does not provide labels for "
                    f"{len(missing_manifest_labels)} members "
                    f"({', '.join(missing_manifest_labels[:3])})")
            for slide_id in values:
                raw_label = _cell(
                    (manifest_rows or {}).get(slide_id, {}), "label")
                if not raw_label:
                    continue
                label = parse_label(raw_label)
                if (label is None or label < 0
                        or (n_classes is not None and label >= n_classes)):
                    expected = (f"[0, {n_classes})" if n_classes is not None
                                else "a non-negative integer")
                    report.problems.append(
                        f"fold{fold} {phase} manifest label {raw_label!r} for "
                        f"slide {slide_id!r} is invalid; expected {expected}")
        duplicates = _duplicate_values(values)
        if duplicates:
            sample = ", ".join(duplicates[:3])
            report.problems.append(
                f"fold{fold} {phase} split repeats {len(duplicates)} slide IDs "
                f"({sample})")
        if manifest_slide_ids is not None:
            unknown = sorted(set(values) - manifest_slide_ids)
            if unknown:
                report.problems.append(
                    f"fold{fold} {phase} split has {len(unknown)} slide IDs "
                    "absent from the manifest "
                    f"({', '.join(unknown[:3])})")
        if manifest_case_ids is not None:
            missing_cases = sorted(
                value for value in set(values)
                if value in (manifest_slide_ids or set())
                and value not in manifest_case_ids)
            if missing_cases:
                report.problems.append(
                    f"fold{fold} {phase} split has {len(missing_cases)} slides "
                    "without manifest case IDs; patient leakage cannot be "
                    f"ruled out ({', '.join(missing_cases[:3])})")
            case_identities[phase] = {
                manifest_case_ids[value] for value in values
                if value in manifest_case_ids}
        identities[phase] = set(values)
    _check_partition_overlap(report, identities, fold, "slide IDs")
    if len(case_identities) == 3:
        _check_partition_overlap(report, case_identities, fold, "case IDs")


def _check_nested_split_contents(
    report: PreflightReport, directory: Path, fold: int,
    cfg: dict[str, Any], manifest_slide_ids: set[str] | None = None,
    manifest_rows: dict[str, dict[str | None, Any]] | None = None,
) -> None:
    """Validate nested phase files and detect case leakage across phases."""
    slide_identities: dict[str, set[str]] = {}
    case_identities: dict[str, set[str]] = {}
    case_labels: dict[str, set[int]] = {}
    label_dict = cfg.get("label_dict", {})
    if not isinstance(label_dict, dict):
        label_dict = {}
    raw_n_classes = cfg.get("n_classes")
    n_classes = (
        raw_n_classes if isinstance(raw_n_classes, int)
        and not isinstance(raw_n_classes, bool) and raw_n_classes > 0
        else len(label_dict) if label_dict else None)
    feature_columns = _feature_columns(cfg)
    for phase in ("train", "val", "test"):
        path = directory / f"{phase}.csv"
        loaded = _read_split_csv(path, report)
        if loaded is None:
            continue
        fields, rows = loaded
        if "slide_id" not in fields:
            report.problems.append(
                f"fold{fold} {phase} split has no 'slide_id' column: {path}")
            continue
        missing_feature_columns = [
            column for column in feature_columns if column not in fields]
        if missing_feature_columns:
            report.problems.append(
                f"fold{fold} {phase} split is missing configured feature "
                "columns: " + ", ".join(missing_feature_columns))
        for column in feature_columns:
            if column in fields:
                blanks = sum(not _cell(row, column) for row in rows)
                if blanks:
                    report.problems.append(
                        f"fold{fold} {phase} split has {blanks} blank values "
                        f"in configured feature column {column!r}")
        if not rows:
            report.problems.append(
                f"fold{fold} {phase} split has no data rows: {path}")
            continue
        slide_values = [
            _cell(row, "slide_id") for row in rows
        ]
        blank_slides = sum(not value for value in slide_values)
        if blank_slides:
            report.problems.append(
                f"fold{fold} {phase} split has {blank_slides} blank slide IDs")
        slide_values = [value for value in slide_values if value]
        duplicate_slides = _duplicate_values(slide_values)
        if duplicate_slides:
            sample = ", ".join(duplicate_slides[:3])
            report.problems.append(
                f"fold{fold} {phase} split repeats "
                f"{len(duplicate_slides)} slide IDs ({sample})")
        if manifest_slide_ids is not None:
            unknown = sorted(set(slide_values) - manifest_slide_ids)
            if unknown:
                report.problems.append(
                    f"fold{fold} {phase} split has {len(unknown)} slide IDs "
                    "absent from the manifest "
                    f"({', '.join(unknown[:3])})")
        if manifest_rows is not None:
            for row_number, row in enumerate(rows, start=2):
                slide_id = _cell(row, "slide_id")
                source = manifest_rows.get(slide_id)
                if source is None:
                    continue
                for column in ("case_id", "label"):
                    expected = _cell(source, column)
                    actual = _cell(row, column)
                    if expected and actual and actual != expected:
                        report.problems.append(
                            f"fold{fold} {phase} row {row_number} {column} "
                            f"does not match manifest for slide {slide_id!r}")
                for column in feature_columns:
                    if column not in fields:
                        continue
                    expected = _cell(source, column)
                    actual = _cell(row, column)
                    if ((bool(expected) != bool(actual))
                            or (expected and actual and not _same_resolved_path(
                                expected, actual))):
                        report.problems.append(
                            f"fold{fold} {phase} row {row_number} feature "
                            f"column {column!r} does not match manifest for "
                            f"slide {slide_id!r}")
        slide_identities[phase] = set(slide_values)
        if "partition" in fields:
            mismatched = sum(
                _cell(row, "partition") != phase
                for row in rows)
            if mismatched:
                report.problems.append(
                    f"fold{fold} {phase} split has {mismatched} rows with a "
                    "different partition value")
        if "label" not in fields:
            report.problems.append(
                f"fold{fold} {phase} split has no 'label' column: {path}")
        else:
            for row_number, row in enumerate(rows, start=2):
                raw_label = _cell(row, "label")
                try:
                    label = (
                        label_dict[raw_label]
                        if raw_label in label_dict else int(raw_label))
                except (TypeError, ValueError):
                    report.problems.append(
                        f"fold{fold} {phase} row {row_number} has unknown "
                        f"label {raw_label!r}")
                    continue
                if (not isinstance(label, int) or isinstance(label, bool)
                        or label < 0
                        or (n_classes is not None
                            and not 0 <= label < n_classes)):
                    expected = (
                        f"[0, {n_classes})" if n_classes is not None
                        else "a non-negative integer")
                    report.problems.append(
                        f"fold{fold} {phase} row {row_number} has label index "
                        f"{label!r} outside {expected}")
                case_id = _cell(row, "case_id")
                if case_id:
                    case_labels.setdefault(case_id, set()).add(label)
        if "case_id" in fields:
            case_values = [
                _cell(row, "case_id") for row in rows
            ]
            blank_cases = sum(not value for value in case_values)
            if blank_cases:
                report.problems.append(
                    f"fold{fold} {phase} split has {blank_cases} blank case IDs; "
                    "patient leakage cannot be ruled out")
            case_identities[phase] = {
                value for value in case_values if value}
        else:
            report.problems.append(
                f"fold{fold} {phase} split has no 'case_id' column; "
                "patient leakage cannot be ruled out")

    _check_partition_overlap(report, slide_identities, fold, "slide IDs")
    if len(case_identities) == 3:
        _check_partition_overlap(report, case_identities, fold, "case IDs")
    inconsistent = sorted(
        case_id for case_id, labels in case_labels.items() if len(labels) > 1)
    if inconsistent:
        report.problems.append(
            f"fold{fold} has conflicting labels for {len(inconsistent)} case "
            f"IDs ({', '.join(inconsistent[:3])})")


def _check_partition_overlap(
    report: PreflightReport, identities: dict[str, set[str]], fold: int,
    identity_label: str,
) -> None:
    """Report identities assigned to more than one fold partition."""
    leakage_kind = (
        "patient leakage" if identity_label == "case IDs"
        else "partition leakage")
    for left, right in (("train", "val"), ("train", "test"),
                        ("val", "test")):
        overlap = sorted(
            identities.get(left, set()) & identities.get(right, set()))
        if overlap:
            sample = ", ".join(overlap[:3])
            suffix = " ..." if len(overlap) > 3 else ""
            report.problems.append(
                f"fold{fold} {leakage_kind}: {len(overlap)} "
                f"{identity_label} occur in "
                f"both {left} and {right} ({sample}{suffix})")


def _same_resolved_path(left: str, right: str) -> bool:
    """Compare portable and absolute spellings of the same feature path."""
    try:
        return (Path(expand_path(left)).resolve(strict=False)
                == Path(expand_path(right)).resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return left == right


def preflight(cfg: dict[str, Any], *, check_features: bool = True,
              deep_features: bool = False,
              checks: Iterable[str] | None = None) -> PreflightReport:
    """Check whether one resolved run configuration can produce a result.

    Args:
        cfg: The loaded run configuration.
        check_features: When True, stat every feature file the manifest names.
            That is the authoritative check and belongs at the start of a job.
            A campaign planner should pass False and rely on the run matrix's
            recorded coverage instead: the scan costs thousands of network
            filesystem stats per cohort, which is worth paying once per job but
            not once per row of a whole-matrix plan.
        deep_features: Open every available manifest-referenced feature and
            validate its key, rank, width, and finite values. This is intended
            for the interactive doctor, not every training launch.
        checks: Optional subset of ``assets``, ``features``, ``prompts``,
            ``encoders``, and ``splits``. The default runs every check.

    Returns:
        A :class:`PreflightReport`. ``report.ok`` is False when the run must be
        skipped, and ``report.problems`` explains why in terms of the specific
        asset that is absent.
    """
    selected = set(PREFLIGHT_CHECKS if checks is None else checks)
    unknown = selected - PREFLIGHT_CHECKS
    if unknown:
        raise ValueError(
            "unknown preflight checks: " + ", ".join(sorted(unknown)))
    if deep_features and (not check_features or "features" not in selected):
        raise ValueError(
            "deep feature validation requires an enabled features scan")
    report = PreflightReport(checks=sorted(selected))
    _check_method_requirements(cfg, report, selected)
    _check_runtime_settings(cfg, report, selected)

    for category in sorted(selected):
        for key in PATH_KEYS_BY_CHECK.get(category, ()):
            if key in cfg and cfg[key] is not None:
                if key == "slide_features":
                    source_type = str(cfg.get("source_type", "")).strip()
                    suffix = Path(str(cfg[key])).suffix.lower()
                    expected_kind = (
                        "file" if source_type == "pkl"
                        or (not source_type and suffix in {".pkl", ".pickle"})
                        else "directory")
                else:
                    expected_kind = (
                        "file" if key in FILE_PATH_KEYS else
                        "directory" if key in DIRECTORY_PATH_KEYS else None)
                _check_path(report, key, cfg[key], expected_kind)
        for key in LIST_KEYS_BY_CHECK.get(category, ()):
            for index, path in enumerate(_asset_paths(cfg.get(key))):
                _check_path(report, f"{key}[{index}]", path, "file")
        for key in OUTPUT_KEYS_BY_CHECK.get(category, ()):
            if key in cfg and cfg[key] is not None:
                _check_output_path(report, key, cfg[key])

    if "encoders" in selected:
        for block in ASSET_BLOCK_KEYS:
            section = cfg.get(block)
            if not isinstance(section, dict):
                continue
            for field in ASSET_BLOCK_FIELDS:
                value = section.get(field)
                if value and "{" not in str(value):
                    _check_path(report, f"{block}.{field}", value)

    if "prompts" in selected:
        clinical_questions = cfg.get("clinical_questions")
        if isinstance(clinical_questions, (str, Path)):
            _check_path(
                report, "clinical_questions", clinical_questions, "file")
            _check_clinical_questions_schema(clinical_questions, report)
        elif clinical_questions is not None and not (
                isinstance(clinical_questions, (list, tuple))
                and bool(clinical_questions)
                and all(isinstance(item, str) and item.strip()
                        for item in clinical_questions)):
            report.problems.append(
                "clinical_questions must be a path or a non-empty string list")

    if "splits" in selected:
        split_dir = cfg.get("split_dir")
        if not split_dir:
            report.problems.append("split_dir is not set")
        else:
            try:
                resolved_split_dir = Path(expand_path(split_dir))
            except ValueError as error:
                report.problems.append(f"cannot resolve split_dir: {error}")
            else:
                if not resolved_split_dir.is_dir():
                    report.problems.append(
                        f"split_dir does not exist: {split_dir}")
                else:
                    report.checked_paths["split_dir"] = _path_detail(split_dir)
                    _check_splits(cfg, report, resolved_split_dir)

    if "features" in selected:
        coverage_threshold = _minimum_feature_coverage(cfg, report)
        if check_features:
            _check_features(
                cfg, report, coverage_threshold, deep=deep_features)
        if (deep_features and cfg.get("source_type") == "pkl"
                and _feature_file_exists(cfg.get("slide_features"))):
            feature_key = str(cfg.get("feature_key", "features"))
            slide_id_key = str(cfg.get("slide_id_key", "filenames"))
            raw_dim = cfg.get("feature_dim")
            expected_dim = (
                raw_dim if isinstance(raw_dim, int)
                and not isinstance(raw_dim, bool) and raw_dim > 0 else None)
            required_slide_ids = None
            manifest = cfg.get("dataset_csv")
            if _exists(manifest):
                try:
                    with open(
                            expand_path(manifest), newline="",
                            encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        if "slide_id" in (reader.fieldnames or []):
                            required_slide_ids = [
                                _cell(row, "slide_id") for row in reader
                                if _cell(row, "slide_id")]
                except (OSError, UnicodeError, csv.Error):
                    # The manifest/split checks emit the actionable decoder
                    # error; do not duplicate it under the pickle diagnosis.
                    pass
            error = _inspect_feature_payload(
                str(cfg["slide_features"]), feature_key, expected_dim, True,
                source_type="pkl", slide_id_key=slide_id_key,
                required_slide_ids=required_slide_ids)
            report.deep_features_checked += 1
            if error:
                report.problems.append(
                    f"deep slide-embedding validation failed: {error}")
    return report
