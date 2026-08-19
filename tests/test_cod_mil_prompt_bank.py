import csv
from pathlib import Path

import pytest
import torch

from common.prompts import (
    PROMPT_FEATURE_SCHEMA,
    load_prompt_bank_csv,
    prompt_feature_metadata,
    validate_prompt_feature_metadata,
)
from methods.cod_mil.adapter import CoDMILMethod
from common.method_provenance import method_provenance


def _write_bank(path: Path, n_classes: int = 2) -> list[str]:
    prompts = (
        [f"class {index} low" for index in range(n_classes)]
        + [f"class {index} high" for index in range(n_classes)]
        + [f"normal tissue {index}" for index in range(15)]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([prompt] for prompt in prompts)
    return prompts


def _payload(path: Path, prompts: list[str], n_classes: int = 2) -> dict:
    embeddings = torch.nn.functional.normalize(
        torch.arange(len(prompts) * 8, dtype=torch.float32).reshape(len(prompts), 8) + 1,
        dim=1,
    )
    return {
        "embeddings": embeddings,
        **prompt_feature_metadata(
            prompts,
            n_classes=n_classes,
            source_path=path,
            feature_space_id="test/paired",
            encoder="test-encoder",
            checkpoint_sha256="0" * 64,
        ),
    }


def test_verified_cod_mil_payload_is_bound_to_exact_csv_order(tmp_path: Path):
    bank = tmp_path / "bank.csv"
    prompts = _write_bank(bank)
    payload = _payload(bank, prompts)

    embeddings = validate_prompt_feature_metadata(
        payload,
        prompts=load_prompt_bank_csv(bank),
        n_classes=2,
        source_path=bank,
        context="test payload",
    )

    assert embeddings.shape == (19, 8)
    assert payload["schema"] == PROMPT_FEATURE_SCHEMA
    assert payload["row_roles"] == {
        "low": [0, 2], "high": [2, 4], "background": [4, 19],
    }


def test_cod_mil_rejects_bare_or_reordered_prompt_features(tmp_path: Path):
    bank = tmp_path / "bank.csv"
    prompts = _write_bank(bank)
    payload = _payload(bank, prompts)

    with pytest.raises(ValueError, match="unverified legacy"):
        validate_prompt_feature_metadata(
            payload["embeddings"], prompts=prompts, n_classes=2,
            source_path=bank, context="legacy.pt")

    payload["prompts"] = [prompts[1], prompts[0], *prompts[2:]]
    with pytest.raises(ValueError, match="prompt order"):
        validate_prompt_feature_metadata(
            payload, prompts=prompts, n_classes=2,
            source_path=bank, context="reordered.pt")


def test_cod_mil_runtime_rejects_upstream_style_bare_tensor(tmp_path: Path):
    bank = tmp_path / "bank.csv"
    prompts = _write_bank(bank)
    tensor = tmp_path / "legacy.pt"
    torch.save(torch.ones((len(prompts), 512)), tensor)
    method = CoDMILMethod({
        "backbone": "plip",
        "feature_dim": 512,
        "feature_space_id": "test/paired",
        "text_feature_space_id": "test/paired",
        "n_classes": 2,
        "text_prompt_bank_csv": str(bank),
        "text_prompt_features": str(tensor),
    }, device="cpu")

    with pytest.raises(ValueError, match="unverified legacy"):
        method._prepare_text_features()


def test_cod_mil_model_width_follows_feature_dim():
    method = CoDMILMethod({
        "backbone": "plip", "feature_dim": 512,
        "feature_space_id": "hf:vinid/plip", "n_classes": 3,
    }, device="cpu")

    model = method.build_model()

    assert model.L == 512
    assert model.attention_block.L == 512
    assert model.fc_text[0].in_features == 512
    assert method_provenance("cod_mil", method.cfg).upstream_fidelity == "partial"
