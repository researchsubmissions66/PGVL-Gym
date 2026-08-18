"""Contract tests for the language-model asset pipeline.

None of these start a server: they check the properties that keep a generated
asset honest -- pinned identity, verified attribution, recorded sampling.
"""
from __future__ import annotations

import pytest

from common.llm import (
    LLMClient, LLMUnavailableError, SamplingParams, get_spec, list_models,
)


@pytest.mark.parametrize("name", list_models())
def test_every_registered_model_pins_an_identity(name):
    spec = get_spec(name)
    assert spec.model_id and "/" in spec.model_id
    assert spec.modality in {"text", "vision_language"}
    assert spec.rationale.strip()
    assert spec.feature_space_id.startswith("hf:")


def test_cached_models_pin_a_revision():
    """A cached model has a resolved snapshot, so its revision is knowable.

    Provenance that names a repository without a revision cannot be reproduced
    once upstream moves.
    """
    for name in list_models():
        spec = get_spec(name)
        if spec.cached:
            assert spec.revision, f"{name} is cached but pins no revision"
            assert "@" in spec.feature_space_id


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError, match="Unknown language model"):
        get_spec("not-a-model")


def test_provenance_records_what_is_needed_to_regenerate():
    client = LLMClient(endpoint="http://localhost:9/v1", model="patho-r1-7b",
                       sampling=SamplingParams(temperature=0.0, seed=7))
    record = client.provenance(prompt_template="wsi_five_answers/v1")

    assert record["_provenance"] == "generated"
    assert record["_model"] == get_spec("patho-r1-7b").feature_space_id
    assert record["_sampling"]["temperature"] == 0.0
    assert record["_sampling"]["seed"] == 7
    assert record["_prompt_template"] == "wsi_five_answers/v1"
    assert record["_generated_at"]


def test_client_refuses_to_attribute_output_to_the_wrong_model(monkeypatch):
    """Serving a different model must fail, not be silently recorded.

    A server started with the wrong checkpoint would otherwise produce text
    stamped with provenance it does not have -- the text-side equivalent of
    mislabelling a feature space.
    """
    client = LLMClient(endpoint="http://localhost:9/v1", model="patho-r1-7b")
    monkeypatch.setattr(client, "served_model", lambda: "qwen2.5-7b-instruct")

    with pytest.raises(LLMUnavailableError, match="does not match"):
        client.verify()


def test_client_accepts_a_matching_served_name(monkeypatch):
    """vLLM reports --served-model-name or a path, so matching is by substring."""
    client = LLMClient(endpoint="http://localhost:9/v1", model="patho-r1-7b")
    for served in ("patho-r1-7b", "WenchuanZhang/Patho-R1-7B",
                   "/cache/hub/models--WenchuanZhang--Patho-R1-7B/snapshots/7a69"):
        monkeypatch.setattr(client, "served_model", lambda s=served: s)
        assert client.verify() == served


def test_unreachable_endpoint_raises_rather_than_returning_empty():
    client = LLMClient(endpoint="http://127.0.0.1:9/v1", model="patho-r1-7b")
    with pytest.raises(LLMUnavailableError, match="cannot reach"):
        client.served_model()
