"""Load SLIP prompt banks without collapsing their ensemble structure."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class SLIPPromptBank:
    """Normalized text roles consumed by SLIP's two prompt branches."""

    templates: tuple[str, ...]
    slide_classnames: tuple[tuple[str, ...], ...]
    tissue_classnames: tuple[tuple[str, ...], ...]
    provenance: str
    source: str

    @property
    def digest(self) -> str:
        payload = {
            "templates": self.templates,
            "slide_classnames": self.slide_classnames,
            "tissue_classnames": self.tissue_classnames,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def config_values(self) -> dict[str, list[Any]]:
        return {
            "text_templates": list(self.templates),
            "slip_slide_classnames": [
                list(group) for group in self.slide_classnames
            ],
            "tissue_classnames": [
                list(group) for group in self.tissue_classnames
            ],
        }


def _strings(value: Any, role: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"SLIP {role} must be a non-empty string list")
    values = tuple(
        item.strip() if isinstance(item, str) else "" for item in value
    )
    if any(not item for item in values):
        raise ValueError(f"SLIP {role} must contain only non-empty strings")
    return values


def _groups(
    value: Any,
    role: str,
    *,
    pair_separator: str | None = None,
    allow_flat_strings: bool = True,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"SLIP {role} must be a non-empty list of text groups")
    groups: list[tuple[str, ...]] = []
    for index, group in enumerate(value):
        if isinstance(group, str) and pair_separator:
            parts = group.split(pair_separator, 1)
            if len(parts) != 2:
                raise ValueError(
                    f"SLIP {role} entry {index} lacks declared separator "
                    f"{pair_separator!r}"
                )
            group = parts
        elif isinstance(group, str):
            if not allow_flat_strings:
                raise ValueError(
                    f"SLIP {role} entry {index} is a flat string; preserve "
                    "the native nested text group or declare _pair_separator"
                )
            group = [group]
        groups.append(_strings(group, f"{role} entry {index}"))
    if len(set(groups)) != len(groups):
        raise ValueError(f"SLIP {role} contains duplicate text groups")
    return tuple(groups)


def load_slip_prompt_bank(
    path: str | Path,
    *,
    fallback_slide_classnames: Sequence[str],
    labels: Sequence[str] | None = None,
) -> SLIPPromptBank:
    """Load one complete or task-extension SLIP prompt bank.

    Upstream banks retain the original list-of-lists representation because
    each tissue's short name and description are encoded independently and
    averaged. Generated legacy assets may declare ``_pair_separator`` to be
    losslessly expanded into that same runtime structure.
    """
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        tissues = payload.get("tissue_classnames", payload.get("tissues"))
        templates = payload.get("templates", ["{}"])
        slide_classnames = payload.get(
            "slide_classnames", list(fallback_slide_classnames)
        )
        pair_separator = payload.get("_pair_separator")
        provenance = str(payload.get("_provenance", "unknown"))
        metadata = payload.get("_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}: SLIP _metadata must be a mapping")
    else:
        tissues = payload
        templates = ["{}"]
        slide_classnames = list(fallback_slide_classnames)
        pair_separator = None
        provenance = "unknown"
        metadata = {}

    bank = SLIPPromptBank(
        templates=_strings(templates, "templates"),
        slide_classnames=_groups(slide_classnames, "slide_classnames"),
        tissue_classnames=_groups(
            tissues,
            "tissue_classnames",
            pair_separator=pair_separator,
            allow_flat_strings=False,
        ),
        provenance=provenance,
        source=str(source),
    )
    if any(template.count("{}") != 1 for template in bank.templates):
        raise ValueError(
            f"{source}: each SLIP template must contain exactly one '{{}}'"
        )
    if len(bank.slide_classnames) != len(fallback_slide_classnames):
        raise ValueError(
            f"{source}: SLIP slide_classnames has "
            f"{len(bank.slide_classnames)} groups, expected "
            f"{len(fallback_slide_classnames)}"
        )
    declared_order = metadata.get("label_order")
    if (labels is not None and declared_order is not None
            and list(declared_order) != list(labels)):
        raise ValueError(
            f"{source}: SLIP label_order {declared_order} does not match "
            f"classifier order {list(labels)}"
        )
    declared_digest = metadata.get("prompt_bank_sha256")
    if declared_digest is not None and declared_digest != bank.digest:
        raise ValueError(
            f"{source}: SLIP prompt_bank_sha256 does not match the complete "
            "ordered bank"
        )
    if len(bank.tissue_classnames) < len(bank.slide_classnames):
        raise ValueError(
            f"{source}: SLIP tissue vocabulary is smaller than its slide "
            "classifier"
        )
    return bank


__all__ = ["SLIPPromptBank", "load_slip_prompt_bank"]
