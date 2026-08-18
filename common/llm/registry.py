"""Registry of language models this benchmark may prompt.

Text assets that a model generates are experimental inputs, so the same rule
that governs encoders governs these: an asset records exactly what produced it.
A spec therefore pins a repository *and a revision*, and the client refuses to
attribute output to a model the server is not actually serving.

Models are served out of process by vLLM, which speaks the OpenAI API. Nothing
here imports vLLM: it pins torch hard and this environment pins torch for the
methods, so the two must not share an interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LLMSpec:
    """Identity and serving requirements of one registered language model."""

    name: str
    model_id: str
    revision: Optional[str]
    architecture: str
    modality: str                      # "text" or "vision_language"
    context_length: int
    rationale: str
    cached: bool = True

    @property
    def feature_space_id(self) -> str:
        """Exact identity recorded on every asset this model generates."""
        return f"hf:{self.model_id}" + (f"@{self.revision}" if self.revision else "")


_SPECS: Dict[str, LLMSpec] = {
    "patho-r1-7b": LLMSpec(
        name="patho-r1-7b",
        model_id="WenchuanZhang/Patho-R1-7B",
        revision="7a69eb299bde72a4c3b8ec26fe0b17515346ef73",
        architecture="qwen2_5_vl",
        modality="vision_language",
        context_length=32768,
        rationale=("Pathology-tuned reasoning model built on Qwen2.5-VL-7B. The "
                   "default for pathology text assets: it has seen diagnostic "
                   "language a general model has not."),
    ),
    "qwen2.5-7b-instruct": LLMSpec(
        name="qwen2.5-7b-instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        revision=None,
        architecture="qwen2",
        modality="text",
        context_length=32768,
        cached=False,
        rationale=("Non-pathology control at the same parameter count as "
                   "Patho-R1. Generating an asset with both shows whether the "
                   "pathology tuning contributed anything, rather than assuming "
                   "it did."),
    ),
    "quilt-llava-7b": LLMSpec(
        name="quilt-llava-7b",
        model_id="wisdomik/Quilt-Llava-v1.5-7b",
        revision="1bdf5f8b75fb26acc80b08aba7f1979e8e9b12bd",
        architecture="llava",
        modality="vision_language",
        context_length=4096,
        rationale=("Pathology VLM trained on Quilt-1M. Registered because ConVLM "
                   "attributes derive from Quilt-LLaVA descriptions upstream, so "
                   "regenerating them is closest to the published source."),
    ),
    "pathgen-llava": LLMSpec(
        name="pathgen-llava",
        model_id="jamessyx/PathGen-LLaVA",
        revision=None,
        architecture="llava",
        modality="vision_language",
        context_length=4096,
        cached=False,
        rationale="Pathology VLM used to caption WSI patches; not yet cached.",
    ),
}


def list_models() -> list[str]:
    """Return registered model names in stable order."""
    return list(_SPECS)


def get_spec(name: str) -> LLMSpec:
    """Return one model's spec.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    key = str(name).strip().lower()
    if key not in _SPECS:
        raise KeyError(
            f"Unknown language model {name!r}. Registered: {list(_SPECS)}")
    return _SPECS[key]


def register_spec(spec: LLMSpec) -> None:
    """Add a model to the registry.

    Raises:
        ValueError: If the name is already registered.
    """
    if spec.name in _SPECS:
        raise ValueError(f"Language model {spec.name!r} is already registered")
    _SPECS[spec.name] = spec
