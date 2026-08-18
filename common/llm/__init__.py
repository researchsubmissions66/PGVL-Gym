"""Language-model access for generating benchmark text assets."""
from .client import LLMClient, LLMUnavailableError, SamplingParams
from .registry import LLMSpec, get_spec, list_models, register_spec

__all__ = ["LLMClient", "LLMUnavailableError", "SamplingParams", "LLMSpec",
           "get_spec", "list_models", "register_spec"]
