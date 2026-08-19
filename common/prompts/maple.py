"""Validate MAPLE's order-sensitive multiscale attribute prompt banks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class MAPLEPromptBank:
    """A validated MAPLE bank and its scientific provenance."""

    payload: dict[str, Any]
    classnames: tuple[str, ...]
    provenance: str
    source: str
    entity_counts: tuple[int, int]

    @property
    def digest(self) -> str:
        roles = {level: self.payload[level] for level in ("low", "high")}
        encoded = json.dumps(
            roles, ensure_ascii=False, separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MAPLE {role} must be a non-empty string")
    return value


def _ordered_text_mapping(
    value: Any,
    *,
    role: str,
    classnames: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"MAPLE {role} must be a mapping")
    actual = tuple(value)
    if actual != classnames:
        raise ValueError(
            f"MAPLE {role} class order {list(actual)} does not match "
            f"classifier order {list(classnames)}"
        )
    for classname, prompt in value.items():
        _text(prompt, f"{role}.{classname}")
    return value


def load_maple_prompt_bank(
    path: str | Path,
    *,
    classnames: Sequence[str],
) -> MAPLEPromptBank:
    """Load a MAPLE bank while preserving its logit-defining key order."""
    ordered_classnames = tuple(classnames)
    if (not ordered_classnames
            or any(not isinstance(name, str) or not name.strip()
                   for name in ordered_classnames)
            or len(set(ordered_classnames)) != len(ordered_classnames)):
        raise ValueError("MAPLE classnames must be unique non-empty strings")

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: MAPLE prompt payload must be a mapping")
    unexpected = [
        key for key in payload
        if key not in {"low", "high"} and not str(key).startswith("_")
    ]
    if unexpected:
        raise ValueError(
            f"{source}: MAPLE prompt payload has unknown roles {unexpected}"
        )

    counts: list[int] = []
    for level in ("low", "high"):
        block = payload.get(level)
        if not isinstance(block, dict):
            raise ValueError(f"{source}: MAPLE requires a {level!r} block")
        _text(block.get("tumor"), f"{level}.tumor")
        _ordered_text_mapping(
            block.get("global_info"),
            role=f"{level}.global_info",
            classnames=ordered_classnames,
        )
        entities = block.get("entities")
        if not isinstance(entities, list) or not entities:
            raise ValueError(f"{source}: MAPLE {level}.entities must be non-empty")
        names: list[str] = []
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(
                    f"{source}: MAPLE {level} entity {index} must be a mapping"
                )
            names.append(_text(entity.get("name"), f"{level}.entities[{index}].name"))
            _text(
                entity.get("general_feature"),
                f"{level}.entities[{index}].general_feature",
            )
            _ordered_text_mapping(
                entity.get("attributes"),
                role=f"{level}.entities[{index}].attributes",
                classnames=ordered_classnames,
            )
        if len(set(names)) != len(names):
            raise ValueError(f"{source}: MAPLE {level} entity names must be unique")
        counts.append(len(entities))

    metadata = payload.get("_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: MAPLE _metadata must be a mapping")
    bank = MAPLEPromptBank(
        payload=payload,
        classnames=ordered_classnames,
        provenance=str(payload.get("_provenance", "unknown")),
        source=str(source),
        entity_counts=(counts[0], counts[1]),
    )
    declared_order = metadata.get("classnames")
    if declared_order is not None and tuple(declared_order) != ordered_classnames:
        raise ValueError(
            f"{source}: MAPLE metadata classnames do not match classifier order"
        )
    declared_digest = metadata.get("prompt_bank_sha256")
    if declared_digest is not None and declared_digest != bank.digest:
        raise ValueError(
            f"{source}: MAPLE prompt_bank_sha256 does not match the ordered bank"
        )
    return bank


__all__ = ["MAPLEPromptBank", "load_maple_prompt_bank"]
