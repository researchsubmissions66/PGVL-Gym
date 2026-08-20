"""Strict loading for TOP's instance- and bag-level prompt roles.

TOP consumes two independent text assets.  The instance learner is initialized
from a task-agnostic tissue-prototype bank, while the bag learner either uses a
task-specific initializer published by TOP or random context paired with the
configured class names.  Keeping those roles separate is important: a valid
JSON shape alone does not establish that a bank is the released condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TOP_PROMPT_FORMAT = "instance_prototypes_plus_optional_bag_context.v1"
TOP_LEARNABLE_SLOT_COUNT = 10
TOP_LEARNABLE_SLOTS = " ".join(["*"] * TOP_LEARNABLE_SLOT_COUNT)
_PROVENANCE = frozenset({"upstream", "derived", "generated"})
_BAG_USAGES = frozenset({"standard_upstream_recipe", "alternative_unwired"})


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest_record(path: Path) -> dict[str, Any]:
    prompt_root = Path(__file__).resolve().parents[2] / "text_prompts"
    try:
        key = str(path.resolve().relative_to(prompt_root.resolve()))
    except ValueError:
        return {}
    try:
        manifest = json.loads(
            (prompt_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    record = manifest.get("assets", {}).get(key, {}) \
        if isinstance(manifest, dict) else {}
    return record if isinstance(record, dict) else {}


def _non_empty_strings(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"TOP {label} must be a sequence, not one string")
    result = tuple(values)
    if (not result
            or any(not isinstance(value, str) or not value.strip()
                   for value in result)
            or len(set(result)) != len(result)):
        raise ValueError(f"TOP {label} must be unique non-empty strings")
    return result


def _ordered_labels(label_dict: Mapping[str, int]) -> tuple[str, ...]:
    if not isinstance(label_dict, Mapping) or not label_dict:
        raise ValueError("TOP label_dict must be a non-empty mapping")
    if (any(not isinstance(label, str) or not label.strip()
            for label in label_dict)
            or any(isinstance(index, bool) or not isinstance(index, int)
                   for index in label_dict.values())):
        raise ValueError("TOP label_dict needs non-empty labels and integer indices")
    ordered = tuple(
        label for label, _ in sorted(label_dict.items(), key=lambda item: item[1]))
    if sorted(label_dict.values()) != list(range(len(label_dict))):
        raise ValueError("TOP label_dict indices must be contiguous from zero")
    return ordered


def _metadata(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    value = payload.get("_metadata", {})
    if not isinstance(value, dict):
        raise ValueError(f"{path}: TOP _metadata must be a mapping")
    return value


def _origin(
    path: Path,
    *,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    record: Mapping[str, Any],
    role: str,
    usage: str,
    expected_provenance: str | None,
) -> tuple[str, str]:
    inline = payload.get("_provenance")
    recorded = record.get("provenance")
    declared = [
        ("inline provenance", inline),
        ("manifest provenance", recorded),
        ("configured provenance", expected_provenance),
    ]
    present = [(label, value) for label, value in declared if value is not None]
    provenance = present[0][1] if present else None
    if provenance not in _PROVENANCE:
        raise ValueError(
            f"{path}: TOP prompt asset needs audited upstream/derived/generated "
            "provenance")
    for label, value in present[1:]:
        if value != provenance:
            raise ValueError(
                f"{path}: TOP {label} {value!r} contradicts {provenance!r}")

    role_declarations = [("inline role", metadata.get("role"))]
    if record:
        role_declarations.append(("manifest role", record.get("role")))
    for label, value in role_declarations:
        if value != role:
            raise ValueError(
                f"{path}: TOP {label} must be {role!r}, got {value!r}")
    usage_declarations = []
    if metadata.get("usage") is not None:
        usage_declarations.append(("inline usage", metadata.get("usage")))
    if record:
        usage_declarations.append(("manifest usage", record.get("usage")))
    if not usage_declarations:
        raise ValueError(f"{path}: TOP prompt asset needs an audited usage")
    for label, value in usage_declarations:
        if value != usage:
            raise ValueError(
                f"{path}: TOP {label} must be {usage!r}, got {value!r}")

    copied_values = [metadata.get("copied_from_upstream")]
    if record:
        copied_values.append(record.get("copied_from_upstream"))
    if provenance == "upstream" and any(value is not True for value in copied_values):
        raise ValueError(
            f"{path}: upstream TOP asset must be recorded as an exact copy "
            "inline and in the provenance manifest")
    if provenance == "generated" and any(value is True for value in copied_values):
        raise ValueError(
            f"{path}: generated TOP asset cannot be marked copied upstream")
    source = record.get("source", metadata.get("source_url", path))
    return str(provenance), str(source)


def _verify_digests(
    path: Path,
    *,
    file_sha256: str,
    prompt_bank_sha256: str,
    record: Mapping[str, Any],
    expected_file_sha256: str | None,
    expected_prompt_bank_sha256: str | None,
) -> None:
    checks = (
        ("manifest file sha256", record.get("sha256"), file_sha256),
        ("manifest prompt-bank sha256", record.get("prompt_bank_sha256"),
         prompt_bank_sha256),
        ("configured file sha256", expected_file_sha256, file_sha256),
        ("configured prompt-bank sha256", expected_prompt_bank_sha256,
         prompt_bank_sha256),
    )
    for label, expected, actual in checks:
        if expected is not None and expected != actual:
            raise ValueError(
                f"{path}: TOP {label} mismatch: expected {expected}, got {actual}")


def top_instance_prompt_bank_sha256(prompts: Sequence[str]) -> str:
    """Hash the ordered rendered instance prompts, matching TOP's asset card."""
    return hashlib.sha256("\n".join(prompts).encode()).hexdigest()


def top_bag_prompt_bank_sha256(
    *,
    mode: str,
    labels: Sequence[str],
    prompts: Sequence[str],
    classnames: Sequence[str],
) -> str:
    """Hash every order-sensitive bag prompt component used by the learner."""
    return _json_sha256({
        "mode": mode,
        "label_order": list(labels),
        "prompts": list(prompts),
        "classnames": list(classnames),
    })


@dataclass(frozen=True)
class TOPInstancePromptBank:
    path: Path
    prompts: tuple[str, ...]
    default_slot_separator: str
    file_sha256: str
    prompt_bank_sha256: str
    provenance: str
    source: str
    role: str
    usage: str

    def initialized_prompts(self, separator: str | None = None) -> tuple[str, ...]:
        selected = self.default_slot_separator if separator is None else separator
        if selected not in {"", " "}:
            raise ValueError(
                "TOP instance_slot_separator must be empty or one space")
        return tuple(
            f"{prompt}{selected}{TOP_LEARNABLE_SLOTS}"
            for prompt in self.prompts)

    def config_values(self) -> dict[str, str]:
        return {
            "top_instance_file_sha256": self.file_sha256,
            "top_instance_prompt_bank_sha256": self.prompt_bank_sha256,
            "top_instance_provenance": self.provenance,
        }


@dataclass(frozen=True)
class TOPBagPromptBank:
    path: Path
    labels: tuple[str, ...]
    prompts: tuple[str, ...]
    classnames: tuple[str, ...]
    mode: str
    default_slot_separator: str
    file_sha256: str
    prompt_bank_sha256: str
    provenance: str
    source: str
    role: str
    usage: str

    @property
    def initialized_prompts(self) -> tuple[str, ...]:
        if self.mode == "ctx_init":
            return self.prompts
        return tuple(
            f"{prompt}{self.default_slot_separator}{TOP_LEARNABLE_SLOTS}"
            for prompt in self.prompts)

    def config_values(self) -> dict[str, str]:
        return {
            "top_bag_file_sha256": self.file_sha256,
            "top_bag_prompt_bank_sha256": self.prompt_bank_sha256,
            "top_bag_provenance": self.provenance,
            "top_bag_usage": self.usage,
        }


@dataclass(frozen=True)
class TOPPromptCondition:
    instance: TOPInstancePromptBank
    bag: TOPBagPromptBank | None
    provenance: str
    source: str

    def config_values(self) -> dict[str, str]:
        values = {
            "top_prompt_format": TOP_PROMPT_FORMAT,
            **self.instance.config_values(),
            "prompt_provenance": self.provenance,
            "prompt_source": self.source,
        }
        if self.bag is not None:
            values.update(self.bag.config_values())
        return values


def load_top_instance_prompt_bank(
    path: str | Path,
    *,
    record: Mapping[str, Any] | None = None,
    expected_file_sha256: str | None = None,
    expected_prompt_bank_sha256: str | None = None,
    expected_provenance: str | None = None,
) -> TOPInstancePromptBank:
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{source_path}: TOP instance prompt bank must be a mapping")
    metadata = _metadata(payload, source_path)
    prototypes = payload.get("prototypes")
    if not isinstance(prototypes, list) or len(prototypes) < 2:
        raise ValueError(f"{source_path}: TOP needs at least two instance prototypes")
    prompts: list[str] = []
    for index, item in enumerate(prototypes):
        if not isinstance(item, dict):
            raise ValueError(
                f"{source_path}: TOP instance prototype {index} must be a mapping")
        tissue = item.get("tissue")
        description = item.get("description")
        prompt = item.get("prompt")
        if (not isinstance(tissue, str) or not tissue.strip()
                or not isinstance(description, str) or not description.strip()
                or not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError(
                f"{source_path}: TOP instance prototype {index} needs non-empty "
                "tissue, description, and prompt fields")
        expected = f"an H&E stained image of {tissue}, which is {description}"
        if prompt != expected:
            raise ValueError(
                f"{source_path}: TOP instance prototype {index} does not match "
                "its structured tissue/description fields")
        if "*" in prompt:
            raise ValueError(
                f"{source_path}: TOP instance prototype {index} already "
                "contains learnable slots")
        prompts.append(prompt)
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"{source_path}: TOP instance prompts must be unique")
    if metadata.get("count") != len(prompts):
        raise ValueError(
            f"{source_path}: TOP instance metadata count must equal "
            f"{len(prompts)}")
    separator = metadata.get("slot_separator")
    if separator not in {"", " "}:
        raise ValueError(
            f"{source_path}: TOP instance slot_separator must be empty or one space")
    prompt_digest = top_instance_prompt_bank_sha256(prompts)
    if metadata.get("ordered_prompt_sha256") != prompt_digest:
        raise ValueError(
            f"{source_path}: TOP ordered_prompt_sha256 does not match the "
            "instance bank")
    audited = dict(record) if record is not None else _manifest_record(source_path)
    provenance, source = _origin(
        source_path, payload=payload, metadata=metadata, record=audited,
        role="instance_level_prototypes", usage="standard_upstream_recipe",
        expected_provenance=expected_provenance)
    file_digest = _file_sha256(source_path)
    _verify_digests(
        source_path, file_sha256=file_digest,
        prompt_bank_sha256=prompt_digest, record=audited,
        expected_file_sha256=expected_file_sha256,
        expected_prompt_bank_sha256=expected_prompt_bank_sha256)
    return TOPInstancePromptBank(
        source_path, tuple(prompts), separator, file_digest, prompt_digest,
        provenance, source, "instance_level_prototypes",
        "standard_upstream_recipe")


def load_top_bag_prompt_bank(
    path: str | Path,
    *,
    label_dict: Mapping[str, int],
    classnames: Sequence[str],
    record: Mapping[str, Any] | None = None,
    expected_file_sha256: str | None = None,
    expected_prompt_bank_sha256: str | None = None,
    expected_provenance: str | None = None,
    expected_usage: str | None = None,
) -> TOPBagPromptBank:
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{source_path}: TOP bag prompt bank must be a mapping")
    metadata = _metadata(payload, source_path)
    exact = payload.get("ctx_init")
    base = payload.get("prompts")
    if (exact is None) == (base is None):
        raise ValueError(
            f"{source_path}: TOP bag bank needs exactly one of ctx_init or prompts")
    mode = "ctx_init" if exact is not None else "prompts"
    prompt_mapping = exact if exact is not None else base
    if not isinstance(prompt_mapping, dict) or not prompt_mapping:
        raise ValueError(
            f"{source_path}: TOP bag {mode} must be a non-empty mapping")
    labels = _ordered_labels(label_dict)
    if tuple(prompt_mapping) != labels:
        raise ValueError(
            f"{source_path}: TOP bag prompt order must match classifier order; "
            f"expected {list(labels)}, got {list(prompt_mapping)}")
    values = _non_empty_strings(tuple(prompt_mapping.values()), label="bag prompts")
    if metadata.get("label_order") != list(labels):
        raise ValueError(
            f"{source_path}: TOP bag label_order does not match classifier order")

    configured_classnames = _non_empty_strings(classnames, label="classnames")
    if len(configured_classnames) != len(labels):
        raise ValueError("TOP needs one configured classname per bag label")
    upstream_classnames = metadata.get("upstream_classnames")
    if upstream_classnames is None:
        resolved_classnames = configured_classnames
    else:
        resolved_classnames = _non_empty_strings(
            upstream_classnames, label="upstream_classnames")
        if len(resolved_classnames) != len(labels):
            raise ValueError(
                f"{source_path}: TOP needs one upstream classname per bag label")

    role = "bag_level_ctx_init" if mode == "ctx_init" else "bag_level_prompts"
    usage = metadata.get("usage")
    if usage not in _BAG_USAGES:
        raise ValueError(f"{source_path}: TOP bag usage is invalid: {usage!r}")
    if expected_usage is not None and usage != expected_usage:
        raise ValueError(
            f"{source_path}: TOP configured bag usage {expected_usage!r} "
            f"does not match asset usage {usage!r}")
    separator = metadata.get("slot_separator", " ")
    if mode == "ctx_init":
        invalid = [
            label for label, value in zip(labels, values)
            if value.count("*") != TOP_LEARNABLE_SLOT_COUNT]
        if invalid:
            raise ValueError(
                f"{source_path}: TOP ctx_init needs exactly ten learnable "
                f"slots per label; invalid labels {invalid}")
        if metadata.get("learnable_slot_count") != TOP_LEARNABLE_SLOT_COUNT:
            raise ValueError(
                f"{source_path}: TOP ctx_init metadata must declare ten slots")
    else:
        invalid = [label for label, value in zip(labels, values) if "*" in value]
        if invalid:
            raise ValueError(
                f"{source_path}: TOP base bag prompts already contain learnable "
                f"slots for labels {invalid}")
        if separator not in {"", " "}:
            raise ValueError(
                f"{source_path}: TOP bag slot_separator must be empty or one space")

    semantic_prompts = (
        values if mode == "ctx_init" else tuple(
            f"{prompt}{separator}{TOP_LEARNABLE_SLOTS}"
            for prompt in values))
    prompt_digest = top_bag_prompt_bank_sha256(
        mode=mode, labels=labels, prompts=semantic_prompts,
        classnames=resolved_classnames)
    audited = dict(record) if record is not None else _manifest_record(source_path)
    provenance, source = _origin(
        source_path, payload=payload, metadata=metadata, record=audited,
        role=role, usage=usage, expected_provenance=expected_provenance)
    file_digest = _file_sha256(source_path)
    _verify_digests(
        source_path, file_sha256=file_digest,
        prompt_bank_sha256=prompt_digest, record=audited,
        expected_file_sha256=expected_file_sha256,
        expected_prompt_bank_sha256=expected_prompt_bank_sha256)
    return TOPBagPromptBank(
        source_path, labels, values, resolved_classnames, mode, separator,
        file_digest, prompt_digest, provenance, source, role, usage)


def resolve_top_prompt_condition(
    instance: TOPInstancePromptBank,
    bag: TOPBagPromptBank | None,
) -> TOPPromptCondition:
    """Derive the reportable condition from the roles actually selected."""
    if bag is None:
        provenance = f"{instance.provenance}_instance_with_random_classname_bag"
        source = f"top_{provenance}"
    elif (instance.provenance == bag.provenance == "upstream"
          and bag.usage == "standard_upstream_recipe"
          and bag.role == "bag_level_ctx_init"):
        provenance = "upstream"
        source = "top_upstream_code_instance_and_bag_initializers"
    elif (instance.provenance == bag.provenance == "upstream"
          and bag.usage == "alternative_unwired"):
        provenance = "upstream_supplementary_condition"
        source = "top_upstream_supplementary_bag_condition"
    else:
        provenance = (
            f"{instance.provenance}_instance_with_{bag.provenance}_bag")
        source = "top_mixed_instance_and_bag_prompt_banks"
    return TOPPromptCondition(instance, bag, provenance, source)


def load_top_prompt_condition(
    instance_path: str | Path,
    *,
    label_dict: Mapping[str, int],
    classnames: Sequence[str],
    bag_path: str | Path | None = None,
    expected_instance_file_sha256: str | None = None,
    expected_instance_prompt_bank_sha256: str | None = None,
    expected_instance_provenance: str | None = None,
    expected_bag_file_sha256: str | None = None,
    expected_bag_prompt_bank_sha256: str | None = None,
    expected_bag_provenance: str | None = None,
    expected_bag_usage: str | None = None,
    expected_condition_provenance: str | None = None,
    expected_condition_source: str | None = None,
) -> TOPPromptCondition:
    labels = _ordered_labels(label_dict)
    configured_classnames = _non_empty_strings(
        classnames, label="classnames")
    if len(configured_classnames) != len(labels):
        raise ValueError("TOP needs one configured classname per bag label")
    instance = load_top_instance_prompt_bank(
        instance_path,
        expected_file_sha256=expected_instance_file_sha256,
        expected_prompt_bank_sha256=expected_instance_prompt_bank_sha256,
        expected_provenance=expected_instance_provenance)
    bag = None
    if bag_path is not None:
        bag = load_top_bag_prompt_bank(
            bag_path, label_dict=label_dict, classnames=configured_classnames,
            expected_file_sha256=expected_bag_file_sha256,
            expected_prompt_bank_sha256=expected_bag_prompt_bank_sha256,
            expected_provenance=expected_bag_provenance,
            expected_usage=expected_bag_usage)
    condition = resolve_top_prompt_condition(instance, bag)
    for label, expected, actual in (
        ("prompt_provenance", expected_condition_provenance,
         condition.provenance),
        ("prompt_source", expected_condition_source, condition.source),
    ):
        if expected is not None and expected != actual:
            raise ValueError(
                f"TOP {label} {expected!r} contradicts active prompt roles; "
                f"expected {actual!r}")
    return condition


__all__ = [
    "TOP_PROMPT_FORMAT", "TOP_LEARNABLE_SLOT_COUNT", "TOP_LEARNABLE_SLOTS",
    "TOPInstancePromptBank", "TOPBagPromptBank", "TOPPromptCondition",
    "load_top_instance_prompt_bank", "load_top_bag_prompt_bank",
    "load_top_prompt_condition", "resolve_top_prompt_condition",
    "top_instance_prompt_bank_sha256", "top_bag_prompt_bank_sha256",
]
