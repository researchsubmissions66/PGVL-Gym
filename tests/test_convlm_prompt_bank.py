import hashlib
import json
from pathlib import Path

import pytest
import torch

from common.preflight import preflight
from common.prompts import (
    ATTRIBUTE_EMBEDDING_SCHEMA,
    load_convlm_attribute_embeddings,
    load_convlm_prompt_bank,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = REPO_ROOT / "text_prompts" / "convlm"
PROVENANCE = json.loads(
    (REPO_ROOT / "text_prompts" / "PROVENANCE.json").read_text())


EXPECTED = {
    "brca_attributes.json": (2, (4, 4)),
    "camelyon16_attributes.json": (2, (2, 2)),
    "nsclc_attributes.json": (2, (4, 4)),
    "rcc_attributes.json": (3, (4, 4, 4)),
    "ubc_ocean_attributes.json": (5, (4, 4, 4, 4, 4)),
}


def _class_payload(path: Path) -> dict[str, list[str]]:
    return {
        key: value for key, value in json.loads(path.read_text()).items()
        if not key.startswith("_")
    }


def _valid_artifact(path: Path, classnames: list[str]) -> dict:
    return {
        "schema": ATTRIBUTE_EMBEDDING_SCHEMA,
        "embeddings": torch.arange(
            1, len(classnames) * 3 + 1, dtype=torch.float32,
        ).reshape(len(classnames), 3),
        "classnames": classnames,
        "feature_space_id": "test:text-encoder",
        "prompt_bank_sha256": "a" * 64,
        "prompt_provenance": "generated",
        "encoder": {
            "model_name": "test-encoder",
            "weights": str(path.parent / "encoder.pt"),
            "feature_space_id": "test:text-encoder",
            "checkpoint_sha256": "b" * 64,
        },
    }


def test_all_convlm_banks_are_audited_generated_substitutes():
    summary = PROVENANCE["method_summaries"]["convlm"]
    assert summary["upstream_commit"] == \
        "a399e51585eeb4c7974b274174ca9b0360a9120d"
    assert summary["copied_from_upstream"] == []
    assert "att_splits.mat" in summary["upstream_asset_issue"]["training_input"]

    for filename, (class_count, counts) in EXPECTED.items():
        path = PROMPT_ROOT / filename
        payload = _class_payload(path)
        record = PROVENANCE["assets"][f"convlm/{filename}"]
        bank = load_convlm_prompt_bank(path, classnames=list(payload))

        assert len(bank.classnames) == class_count
        assert bank.prompt_counts == counts
        assert bank.provenance == "generated"
        assert record["copied_from_upstream"] is False
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["ordered_prompt_sha256"] == bank.prompt_bank_sha256


def test_convlm_loader_rejects_order_and_unattributed_custom_banks(
    tmp_path: Path,
):
    prompt = tmp_path / "attributes.json"
    prompt.write_text(json.dumps({"B": ["b"], "A": ["a"]}))
    with pytest.raises(ValueError, match="order must match classnames"):
        load_convlm_prompt_bank(prompt, classnames=["A", "B"])

    prompt.write_text(json.dumps({"A": ["a"], "B": ["b"]}))
    with pytest.raises(ValueError, match="must declare provenance"):
        load_convlm_prompt_bank(prompt, classnames=["A", "B"])


def test_convlm_doctor_rejects_generated_bank_reported_as_upstream():
    path = PROMPT_ROOT / "nsclc_attributes.json"
    classnames = list(_class_payload(path))
    report = preflight({
        "method": "convlm",
        "classnames": classnames,
        "attribute_prompt_path": str(path),
        "attribute_feature_space_id": "test:text-encoder",
        "attribute_encoder": {
            "model_name": "test",
            "weights": str(path),
            "feature_space_id": "test:text-encoder",
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "prompt_provenance": "upstream",
        "prompt_source": "convlm_upstream_attribute_prompts",
    }, checks={"prompts"})

    assert any("prompt_provenance contradicts" in item
               for item in report.problems)
    assert any("prompt_source contradicts" in item
               for item in report.problems)


def test_convlm_encoded_bank_requires_complete_metadata(tmp_path: Path):
    bare = tmp_path / "bare.pt"
    torch.save(torch.ones(2, 3), bare)
    with pytest.raises(ValueError, match="bare ConVLM tensors are unsafe"):
        load_convlm_attribute_embeddings(
            bare, classnames=["A", "B"],
            feature_space_id="test:text-encoder")

    artifact_path = tmp_path / "attributes.pt"
    artifact = _valid_artifact(artifact_path, ["A", "B"])
    torch.save(artifact, artifact_path)
    bank = load_convlm_attribute_embeddings(
        artifact_path,
        classnames=["A", "B"],
        feature_space_id="test:text-encoder",
        expected_prompt_bank_sha256="a" * 64,
    )
    assert bank.prompt_provenance == "generated"
    assert bank.encoder["checkpoint_sha256"] == "b" * 64

    report = preflight({
        "method": "convlm",
        "classnames": ["A", "B"],
        "attribute_embeddings": str(artifact_path),
        "attribute_feature_space_id": "test:text-encoder",
        "attribute_prompt_bank_sha256": "a" * 64,
        "prompt_provenance": "generated",
        "prompt_source": "convlm_precomputed_attribute_embeddings",
    }, checks={"prompts"})
    assert report.ok, report.problems

    artifact["encoder"].pop("checkpoint_sha256")
    torch.save(artifact, artifact_path)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        load_convlm_attribute_embeddings(
            artifact_path, classnames=["A", "B"],
            feature_space_id="test:text-encoder")


def test_convlm_prompt_origin_follows_content_not_path_selection():
    from scripts.tcga_benchmark import _prompt_provenance

    value = "text_prompts/convlm/nsclc_attributes.json"
    legacy = {"convlm_attribute_prompt_json": value}
    explicit = {"prompts": {"convlm": value}}

    assert _prompt_provenance(legacy, "convlm") == "generated"
    assert _prompt_provenance(explicit, "convlm") == "generated"
