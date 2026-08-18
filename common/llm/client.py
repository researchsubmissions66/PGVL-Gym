"""HTTP client for a vLLM-served model, with provenance attached to output.

vLLM exposes the OpenAI API, so this needs no vLLM import and no shared
interpreter. Two properties matter more than convenience:

* **Attribution is verified, not asserted.** ``/v1/models`` is queried before
  the first completion and the served identity is compared against the
  requested spec. Recording ``Patho-R1`` provenance on text a different server
  produced would be the same class of error as mislabelling a feature space.
* **Sampling is deterministic by default.** ``temperature=0`` with a fixed seed,
  and every asset carries the settings that produced it, so a generated bank can
  be regenerated rather than merely trusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from .registry import LLMSpec, get_spec


class LLMUnavailableError(RuntimeError):
    """Raised when the endpoint is unreachable or serving another model."""


@dataclass
class SamplingParams:
    """Decoding settings recorded alongside anything generated with them."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 1
    stop: Optional[Sequence[str]] = None

    def as_dict(self) -> dict[str, Any]:
        payload = {"temperature": self.temperature, "top_p": self.top_p,
                   "max_tokens": self.max_tokens, "seed": self.seed}
        if self.stop:
            payload["stop"] = list(self.stop)
        return payload


@dataclass
class LLMClient:
    """Prompt one registered model through an OpenAI-compatible endpoint.

    Args:
        endpoint: Base URL, e.g. ``http://gpu042:8000/v1``.
        model: Registered model name.
        sampling: Decoding settings; deterministic by default.
        timeout: Per-request timeout in seconds.
        api_key: Sent as a bearer token when the server requires one.
    """

    endpoint: str
    model: str
    sampling: SamplingParams = field(default_factory=SamplingParams)
    timeout: float = 300.0
    api_key: str = "EMPTY"
    _spec: LLMSpec = field(init=False, repr=False)
    _served: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._spec = get_spec(self.model)
        self.endpoint = self.endpoint.rstrip("/")

    # ------------------------------------------------------------------
    @property
    def spec(self) -> LLMSpec:
        return self._spec

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def served_model(self) -> str:
        """Return the model id the endpoint reports, querying it once.

        Raises:
            LLMUnavailableError: If the endpoint cannot be reached or lists no
                model.
        """
        if self._served is not None:
            return self._served
        import requests

        try:
            response = requests.get(f"{self.endpoint}/models",
                                    headers=self._headers(), timeout=30)
            response.raise_for_status()
            data = response.json().get("data") or []
        except Exception as error:                                # noqa: BLE001
            raise LLMUnavailableError(
                f"cannot reach {self.endpoint}: {error}") from error
        if not data:
            raise LLMUnavailableError(f"{self.endpoint} lists no served model")
        self._served = str(data[0].get("id", ""))
        return self._served

    def verify(self) -> str:
        """Confirm the endpoint serves the requested model.

        vLLM reports whatever ``--served-model-name`` or path it was started
        with, so an exact string match is too strict; the registered name or
        repository id must appear in it.

        Raises:
            LLMUnavailableError: If the served identity does not match.
        """
        served = self.served_model()
        needles = {self._spec.name.lower(), self._spec.model_id.lower(),
                   self._spec.model_id.split("/")[-1].lower()}
        if not any(needle in served.lower() for needle in needles):
            raise LLMUnavailableError(
                f"{self.endpoint} serves {served!r}, which does not match "
                f"requested model {self._spec.model_id!r}. Refusing to attribute "
                "generated text to a model that did not produce it.")
        return served

    def complete(self, messages: Sequence[Mapping[str, Any]],
                 sampling: Optional[SamplingParams] = None) -> str:
        """Return one chat completion.

        Raises:
            LLMUnavailableError: If the request fails or returns no choice.
        """
        import requests

        self.verify()
        params = (sampling or self.sampling).as_dict()
        payload = {"model": self.served_model(), "messages": list(messages),
                   **params}
        try:
            response = requests.post(
                f"{self.endpoint}/chat/completions", headers=self._headers(),
                data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            choices = response.json().get("choices") or []
        except Exception as error:                                # noqa: BLE001
            raise LLMUnavailableError(
                f"completion failed on {self.endpoint}: {error}") from error
        if not choices:
            raise LLMUnavailableError("endpoint returned no completion choice")
        return choices[0]["message"]["content"]

    # ------------------------------------------------------------------
    def provenance(self, *, prompt_template: Optional[str] = None
                   ) -> dict[str, Any]:
        """Return the record every generated asset must carry.

        Enough to regenerate the asset: which model at which revision, the
        identity the server actually reported, the decoding settings, and the
        template used to build the prompts.
        """
        return {
            "_provenance": "generated",
            "_model": self._spec.feature_space_id,
            "_model_name": self._spec.name,
            "_served_as": self._served,
            "_architecture": self._spec.architecture,
            "_sampling": (self.sampling.as_dict()),
            "_prompt_template": prompt_template,
            "_generated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        }
