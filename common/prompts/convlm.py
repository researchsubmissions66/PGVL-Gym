"""Strict loading for ConVLM prompt and encoded attribute banks.

The ConVLM release does not publish the ``att_splits.mat`` consumed by its
training script.  Consequently, every usable bank in PGVL-Gym is either an
explicitly generated text bank or an external encoded artifact.  These
loaders keep the class order, prompt content, encoder identity, and feature
space bound together instead of accepting an anonymous matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ATTRIBUTE_EMBEDDING_SCHEMA = "pgvl.convlm.attribute_embeddings.v1"
_PROVENANCE = frozenset({"upstream", "derived", "generated"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def convlm_prompt_bank_sha256(
    classnames: Sequence[str], prompts: Mapping[str, Sequence[str]],
) -> str:
    """Hash ordered class-to-prompt content independently of JSON whitespace."""
    ordered = {str(name): list(prompts[str(name)]) for name in classnames}
    encoded = json.dumps(
        ordered, ensure_ascii=False, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest_record(path: Path) -> dict[str, Any]:
    prompt_root = Path(__file__).resolve().parents[2] / "text_prompts"
    try:
        key = str(path.resolve().relative_to(prompt_root.resolve()))
    except ValueError:
        return {}
    try:
        payload = json.loads(
            (prompt_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    record = payload.get("assets", {}).get(key, {}) \
        if isinstance(payload, dict) else {}
    return record if isinstance(record, dict) else {}


def _classnames(value: Sequence[str]) -> tuple[str, ...]:
    result = tuple(value)
    if (not result
            or any(not isinstance(name, str) or not name.strip()
                   for name in result)
            or len(set(result)) != len(result)):
        raise ValueError("ConVLM classnames must be unique non-empty strings")
    return result


@dataclass(frozen=True)
class ConVLMPromptBank:
    """One validated, classifier-ordered ConVLM text attribute bank."""

    path: Path
    classnames: tuple[str, ...]
    prompts: tuple[tuple[str, ...], ...]
    file_sha256: str
    prompt_bank_sha256: str
    provenance: str
    source: str

    @property
    def prompt_counts(self) -> tuple[int, ...]:
        return tuple(len(row) for row in self.prompts)


def load_convlm_prompt_bank(
    path: str | Path,
    *,
    classnames: Sequence[str],
    record: Mapping[str, Any] | None = None,
    expected_file_sha256: str | None = None,
    expected_prompt_bank_sha256: str | None = None,
) -> ConVLMPromptBank:
    """Load a text bank and validate its content and scientific provenance."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: ConVLM attribute prompts must be a mapping")

    ordered = _classnames(classnames)
    prompt_payload = {
        str(key): value for key, value in payload.items()
        if not str(key).startswith("_")
    }
    if tuple(prompt_payload) != ordered:
        raise ValueError(
            f"{source}: ConVLM attribute prompt order must match classnames; "
            f"expected {list(ordered)}, got {list(prompt_payload)}")
    rows: list[tuple[str, ...]] = []
    for classname, values in prompt_payload.items():
        if (not isinstance(values, list) or not values
                or any(not isinstance(item, str) or not item.strip()
                       for item in values)):
            raise ValueError(
                f"{source}: ConVLM attributes for {classname!r} must be a "
                "non-empty string list")
        rows.append(tuple(values))

    file_digest = _sha256(source)
    bank_digest = convlm_prompt_bank_sha256(ordered, prompt_payload)
    audited = dict(record) if record is not None else _manifest_record(source)
    metadata = payload.get("_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: ConVLM _metadata must be a mapping")

    inline_origin = payload.get("_provenance")
    recorded_origin = audited.get("provenance")
    if (inline_origin is not None and recorded_origin is not None
            and inline_origin != recorded_origin):
        raise ValueError(
            f"{source}: ConVLM inline and manifest provenance disagree")
    provenance = str(inline_origin or recorded_origin or "unknown")
    if provenance not in _PROVENANCE:
        raise ValueError(
            f"{source}: ConVLM prompt bank must declare provenance as one of "
            f"{sorted(_PROVENANCE)} inline or in PROVENANCE.json")
    copied = audited.get("copied_from_upstream")
    if provenance == "generated" and copied is not None and copied is not False:
        raise ValueError(
            f"{source}: generated ConVLM bank cannot be marked copied upstream")

    declared_order = metadata.get("classnames", audited.get("classnames"))
    if declared_order is not None and tuple(declared_order) != ordered:
        raise ValueError(
            f"{source}: ConVLM provenance classnames do not match classifier order")
    actual_counts = {name: len(prompt_payload[name]) for name in ordered}
    declared_counts = metadata.get(
        "prompt_counts_per_class", audited.get("prompt_counts_per_class"))
    if declared_counts is not None and declared_counts != actual_counts:
        raise ValueError(
            f"{source}: ConVLM prompt counts do not match provenance")

    declared_file_digest = audited.get("sha256")
    if declared_file_digest is not None and declared_file_digest != file_digest:
        raise ValueError(
            f"{source}: ConVLM file sha256 does not match provenance")
    if expected_file_sha256 is not None and expected_file_sha256 != file_digest:
        raise ValueError(
            f"{source}: ConVLM file sha256 does not match config")
    declared_bank_digest = metadata.get(
        "prompt_bank_sha256", audited.get("ordered_prompt_sha256"))
    if declared_bank_digest is not None and declared_bank_digest != bank_digest:
        raise ValueError(
            f"{source}: ConVLM ordered prompt digest does not match provenance")
    if (expected_prompt_bank_sha256 is not None
            and expected_prompt_bank_sha256 != bank_digest):
        raise ValueError(
            f"{source}: ConVLM ordered prompt digest does not match config")

    return ConVLMPromptBank(
        path=source,
        classnames=ordered,
        prompts=tuple(rows),
        file_sha256=file_digest,
        prompt_bank_sha256=bank_digest,
        provenance=provenance,
        source=str(audited.get("source", metadata.get("source", source))),
    )


@dataclass(frozen=True)
class ConVLMAttributeBank:
    """A metadata-bound precomputed ConVLM attribute matrix."""

    path: Path
    embeddings: Any
    classnames: tuple[str, ...]
    feature_space_id: str
    prompt_bank_sha256: str
    source_prompt_file_sha256: str | None
    prompt_provenance: str
    encoder: Mapping[str, str]


def load_convlm_attribute_embeddings(
    path: str | Path,
    *,
    classnames: Sequence[str],
    feature_space_id: str | None,
    expected_prompt_bank_sha256: str | None = None,
) -> ConVLMAttributeBank:
    """Load an encoded bank, rejecting anonymous tensors and NumPy arrays.

    A raw rank-2 tensor has no reliable way to establish which row is which
    class or which text tower produced its coordinates.  Wrap it in the schema
    documented by :data:`ATTRIBUTE_EMBEDDING_SCHEMA` before using it.
    """
    source = Path(path)
    if source.suffix.lower() == ".npy":
        raise ValueError(
            f"{source}: bare NumPy ConVLM matrices cannot carry provenance; "
            "wrap the embeddings in a metadata-bound .pt artifact")
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{source}: bare ConVLM tensors are unsafe; expected a mapping "
            f"with schema={ATTRIBUTE_EMBEDDING_SCHEMA!r}")
    if payload.get("schema") != ATTRIBUTE_EMBEDDING_SCHEMA:
        raise ValueError(
            f"{source}: ConVLM attribute artifact schema must be "
            f"{ATTRIBUTE_EMBEDDING_SCHEMA!r}")

    ordered = _classnames(classnames)
    recorded_order = payload.get("classnames")
    if not isinstance(recorded_order, (list, tuple)) \
            or tuple(recorded_order) != ordered:
        raise ValueError(
            f"{source}: ConVLM attribute class order must exactly match "
            "classnames")
    embeddings = payload.get("embeddings")
    if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
        raise ValueError(f"{source}: ConVLM embeddings must be a rank-2 tensor")
    if embeddings.shape[0] != len(ordered):
        raise ValueError(
            f"{source}: ConVLM attribute row count must equal classnames")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{source}: ConVLM embeddings contain NaN or infinity")
    if (embeddings.float().norm(dim=-1) == 0).any():
        raise ValueError(f"{source}: ConVLM embeddings contain an all-zero row")

    embedded_space = payload.get("feature_space_id")
    if not isinstance(embedded_space, str) or not embedded_space.strip():
        raise ValueError(
            f"{source}: ConVLM artifact requires feature_space_id")
    if feature_space_id is not None and embedded_space != feature_space_id:
        raise ValueError(
            f"{source}: ConVLM attribute feature space {embedded_space!r} "
            f"does not match {feature_space_id!r}")

    prompt_digest = payload.get("prompt_bank_sha256")
    if not _valid_digest(prompt_digest):
        raise ValueError(
            f"{source}: ConVLM artifact requires a 64-character "
            "prompt_bank_sha256")
    if (expected_prompt_bank_sha256 is not None
            and prompt_digest != expected_prompt_bank_sha256):
        raise ValueError(
            f"{source}: ConVLM source prompt digest does not match config")
    source_file_digest = payload.get("source_prompt_file_sha256")
    if source_file_digest is not None and not _valid_digest(source_file_digest):
        raise ValueError(
            f"{source}: ConVLM source_prompt_file_sha256 must be a "
            "64-character digest")
    prompt_provenance = payload.get("prompt_provenance")
    if prompt_provenance not in _PROVENANCE:
        raise ValueError(
            f"{source}: ConVLM artifact requires prompt_provenance in "
            f"{sorted(_PROVENANCE)}")

    encoder = payload.get("encoder")
    required = {"model_name", "weights", "feature_space_id", "checkpoint_sha256"}
    if not isinstance(encoder, dict) or required.difference(encoder):
        missing = sorted(required.difference(encoder or {})) \
            if isinstance(encoder, dict) else sorted(required)
        raise ValueError(
            f"{source}: ConVLM artifact encoder metadata is missing {missing}")
    if encoder["feature_space_id"] != embedded_space:
        raise ValueError(
            f"{source}: ConVLM encoder and embedding feature spaces disagree")
    if not _valid_digest(encoder["checkpoint_sha256"]):
        raise ValueError(
            f"{source}: ConVLM encoder checkpoint_sha256 must be a "
            "64-character digest")
    for key in ("model_name", "weights", "feature_space_id"):
        if not isinstance(encoder[key], str) or not encoder[key].strip():
            raise ValueError(
                f"{source}: ConVLM encoder.{key} must be a non-empty string")

    return ConVLMAttributeBank(
        path=source,
        embeddings=embeddings,
        classnames=ordered,
        feature_space_id=embedded_space,
        prompt_bank_sha256=prompt_digest,
        source_prompt_file_sha256=source_file_digest,
        prompt_provenance=prompt_provenance,
        encoder=encoder,
    )


__all__ = [
    "ATTRIBUTE_EMBEDDING_SCHEMA", "ConVLMAttributeBank", "ConVLMPromptBank",
    "convlm_prompt_bank_sha256", "load_convlm_attribute_embeddings",
    "load_convlm_prompt_bank",
]
