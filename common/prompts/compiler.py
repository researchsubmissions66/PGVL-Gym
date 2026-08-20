"""Compile one dataset prompt profile into supported method prompt schemas.

The benchmark protocol should describe diagnostic semantics once.  This
module turns that description into the CSV/JSON/YAML layouts expected by the
vendored methods without putting dataset names in model code. WSI-FiVE is
deliberately excluded: a generic class-description profile cannot supply its
six aligned clinical questions, per-case training answers, and separate
evaluation bank without inventing supervision.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from common.configuration import load_yaml_file


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = load_yaml_file(path)
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        raise ValueError(f"Prompt profile must be YAML or JSON: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt profile must contain a mapping: {path}")
    return payload


def _strings(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty string list")
    output = [str(item).strip() for item in value]
    if any(not item for item in output):
        raise ValueError(f"{field} contains an empty prompt")
    return output


def load_prompt_profile(
    specification: str | Path | Mapping[str, Any], *, labels: Sequence[str],
    classnames: Sequence[str], repo_root: Path,
) -> dict[str, Any]:
    """Load and normalize a dataset prompt profile.

    A profile may contain descriptions directly or reference a released
    multiscale description JSON through ``description_source``.  The canonical
    fields are ``low_res``, ``high_res``, ``aliases``, ``broad``, ``specific``,
    and ``attributes``.  Only the two resolution descriptions are mandatory;
    safe method-specific defaults are derived from them.

    Args:
        specification: YAML/JSON path or inline profile mapping.
        labels: Ordered machine-readable task labels.
        classnames: Ordered human-readable class names paired with ``labels``.
        repo_root: Base directory used to resolve relative asset paths.

    Returns:
        A versioned canonical profile with normalized prompt lists and source
        provenance.

    Raises:
        TypeError: If the specification or profile blocks have invalid types.
        ValueError: If versions, labels, class names, or prompt lists violate
            the canonical schema.
    """
    if isinstance(specification, (str, Path)):
        profile_path = _resolve_path(specification, repo_root)
        raw = _load_mapping(profile_path)
        profile_source = str(profile_path.resolve())
    elif isinstance(specification, Mapping):
        raw = dict(specification)
        profile_source = "inline"
    else:
        raise TypeError("prompt_spec must be a path or mapping")
    if int(raw.get("version", 1)) != 1:
        raise ValueError("Only prompt profile version 1 is supported")

    descriptions: dict[str, Any] = {}
    description_source = raw.get("description_source")
    if description_source:
        description_path = _resolve_path(str(description_source), repo_root)
        descriptions = _load_mapping(description_path)
    classes = raw.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("Prompt profile requires a classes mapping")
    if len(labels) != len(classnames):
        raise ValueError("labels and classnames must have the same length")
    missing = set(labels).difference(classes)
    extra = set(classes).difference(labels)
    if missing or extra:
        raise ValueError(
            f"Prompt classes must exactly match labels; missing={sorted(missing)}, "
            f"extra={sorted(extra)}")

    normalized_classes: dict[str, Any] = {}
    for label, classname in zip(labels, classnames):
        block = classes[label]
        if not isinstance(block, dict):
            raise ValueError(f"Prompt class {label} must be a mapping")
        declared_name = str(block.get("classname", classname)).strip()
        if declared_name != classname:
            raise ValueError(
                f"Prompt classname for {label!r} is {declared_name!r}, "
                f"expected {classname!r}")
        released = descriptions.get(label, {})
        if not isinstance(released, dict):
            raise ValueError(f"Description source entry {label!r} must be a mapping")
        low = _strings(
            block.get("low_res", block.get("small_mag", released.get("small_mag"))),
            f"{label}.low_res",
        )
        high = _strings(
            block.get("high_res", block.get("big_mag", released.get("big_mag"))),
            f"{label}.high_res",
        )
        aliases = _strings(
            block.get("aliases", [classname, label]), f"{label}.aliases")
        tissues = _strings(
            block.get("tissues", aliases), f"{label}.tissues")
        broad = _strings(block.get("broad", low), f"{label}.broad")
        specific = _strings(block.get("specific", high), f"{label}.specific")
        attributes = _strings(
            block.get("attributes", low[:3] + high[:3]),
            f"{label}.attributes",
        )
        normalized_classes[label] = {
            "classname": classname,
            "low_res": low,
            "high_res": high,
            "aliases": aliases,
            "tissues": tissues,
            "broad": broad,
            "specific": specific,
            "attributes": attributes,
        }

    context = str(raw.get(
        "context",
        "Histopathology whole-slide classification among: "
        + ", ".join(classnames) + ".",
    )).strip()
    if not context:
        raise ValueError("Prompt profile context must be non-empty")
    return {
        "version": 1,
        "source": profile_source,
        "description_source": (
            str(_resolve_path(str(description_source), repo_root).resolve())
            if description_source else None),
        "provenance": str(raw.get("provenance", "user_defined")),
        "context": context,
        "mscpt_prompts_per_scale": int(raw.get("mscpt_prompts_per_scale", 10)),
        "classes": normalized_classes,
    }


def compile_task_prompt_assets(
    task: str, task_cfg: Mapping[str, Any], output_dir: Path,
    *, repo_root: Path,
) -> dict[str, Any]:
    """Compile a task's canonical profile into method-native prompt files.

    Args:
        task: Stable task/cohort identifier used for output directories.
        task_cfg: Task configuration containing ``prompt_spec``, ``labels``,
            and ``classnames``.
        output_dir: Benchmark output directory.
        repo_root: Base directory used to resolve relative prompt sources.

    Returns:
        Paths and provenance for every emitted method-native prompt asset.
    """
    profile = load_prompt_profile(
        task_cfg["prompt_spec"], labels=list(task_cfg["labels"]),
        classnames=list(task_cfg["classnames"]), repo_root=repo_root)
    labels = list(task_cfg["labels"])
    classes = profile["classes"]
    root = output_dir / "data" / task / "prompts"
    root.mkdir(parents=True, exist_ok=True)

    canonical_path = root / "canonical_profile.json"
    canonical_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    focus_path = root / "focus_two_scale.csv"
    with focus_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for label in labels:
            writer.writerow([classes[label]["low_res"][0]])
        for label in labels:
            writer.writerow([classes[label]["high_res"][0]])

    # Keep method-owned files distinct even though both released loaders use
    # the same positional schema; provenance and future task selection differ.
    vila_path = root / "vila_mil.csv"
    with vila_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for label in labels:
            writer.writerow([classes[label]["low_res"][0]])
        for label in labels:
            writer.writerow([classes[label]["high_res"][0]])

    muse_dir = root / "muse"
    muse_dir.mkdir(parents=True, exist_ok=True)
    muse_paths: list[str] = []
    for index, label in enumerate(labels):
        path = muse_dir / f"generated_new_{index}.csv"
        descriptions = classes[label]["low_res"] + classes[label]["high_res"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "0"])
            writer.writerows(enumerate(descriptions))
        muse_paths.append(str(path))

    count = int(profile["mscpt_prompts_per_scale"])
    if count <= 0:
        raise ValueError("mscpt_prompts_per_scale must be positive")
    mscpt_payload: dict[str, Any] = {}
    for label in labels:
        low, high = classes[label]["low_res"], classes[label]["high_res"]
        if len(low) < count or len(high) < count:
            raise ValueError(
                f"{task}/{label}: MSCPT needs at least {count} low/high prompts; "
                f"got {len(low)}/{len(high)}")
        mscpt_payload[label] = {
            "small_mag": low[:count], "big_mag": high[:count]}
    mscpt_path = root / "gpt" / "description" / f"{task}.json"
    mscpt_path.parent.mkdir(parents=True, exist_ok=True)
    mscpt_path.write_text(
        json.dumps(mscpt_payload, indent=2) + "\n", encoding="utf-8")

    maple_payload: dict[str, Any] = {}
    for level, key in (("low", "low_res"), ("high", "high_res")):
        maple_payload[level] = {
            "tumor": profile["context"],
            "global_info": {
                classes[label]["classname"]: " ".join(classes[label][key])
                for label in labels
            },
            "entities": [{
                "name": "Diagnostic morphology",
                "general_feature": (
                    "Class-specific architectural and cytologic findings at "
                    f"{level} magnification"),
                "attributes": {
                    classes[label]["classname"]: " ".join(
                        classes[label]["attributes"])
                    for label in labels
                },
            }],
        }
    maple_digest = hashlib.sha256(json.dumps(
        {level: maple_payload[level] for level in ("low", "high")},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()).hexdigest()
    maple_payload.update({
        "_provenance": "generated",
        "_metadata": {
            "role": "generated_maple_task_extension",
            "source_profile": profile["provenance"],
            "classnames": [classes[label]["classname"] for label in labels],
            "prompt_bank_sha256": maple_digest,
        },
    })
    maple_path = root / "maple_attributes.json"
    maple_path.write_text(
        json.dumps(maple_payload, indent=2) + "\n", encoding="utf-8")

    cod_payload = {
        classes[label]["classname"]: {
            "broad": classes[label]["broad"],
            "specific": classes[label]["specific"],
        }
        for label in labels
    }
    cod_path = root / "cod_chain.json"
    cod_path.write_text(json.dumps(cod_payload, indent=2) + "\n", encoding="utf-8")

    slip_path = root / "slip_tissues.json"
    slip_path.write_text(json.dumps(
        [classes[label]["tissues"] for label in labels], indent=2) + "\n",
        encoding="utf-8")

    sldpc_path = root / "sldpc.yaml"
    sldpc_path.write_text(yaml.safe_dump({
        "prompts": {label: classes[label]["aliases"] for label in labels}},
        sort_keys=False), encoding="utf-8")

    convlm_prompts = {
        classes[label]["classname"]: [
            f"a histopathology image of {classes[label]['classname']}",
            classes[label]["low_res"][0],
            classes[label]["high_res"][0],
            *classes[label]["attributes"],
        ]
        for label in labels
    }
    convlm_digest = hashlib.sha256(json.dumps(
        convlm_prompts, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()
    convlm_payload = dict(convlm_prompts)
    convlm_payload.update({
        "_provenance": "generated",
        "_metadata": {
            "role": "generated_convlm_attribute_prompts",
            "source_profile": profile["provenance"],
            "classnames": list(convlm_prompts),
            "prompt_counts_per_class": {
                name: len(values) for name, values in convlm_prompts.items()
            },
            "prompt_bank_sha256": convlm_digest,
        },
    })
    convlm_path = root / "convlm_attributes.json"
    convlm_path.write_text(
        json.dumps(convlm_payload, indent=2) + "\n", encoding="utf-8")

    return {
        "canonical": str(canonical_path),
        # Every emitted method file is a locally selected/reformatted task
        # extension. Keep the profile's text source separately so a released
        # MSCPT description bank cannot masquerade as a released MUSE bank.
        "provenance": "generated",
        "source_profile_provenance": profile["provenance"],
        "context": profile["context"],
        "focus": str(focus_path),
        "vila_mil": str(vila_path),
        "muse": muse_paths,
        "mscpt": str(mscpt_path),
        "maple": str(maple_path),
        "cod_mil": str(cod_path),
        "slip": str(slip_path),
        # This YAML is for SLDPC's separately reported TITAN zero-shot
        # baseline. Stage 1/2 learns context around prompt_classnames and does
        # not consume synonym banks.
        "sldpc_zero_shot": str(sldpc_path),
        "convlm": str(convlm_path),
    }
