"""Composite WSI/VLM model: pluggable selectors / prompts / aggregators / recipes."""
from .interfaces import PromptBank, PatchSelector, PromptModule, Aggregator, Recipe
from .model import CompositeModel
from . import selectors, prompts, aggregators, recipes, losses

__all__ = ["CompositeModel", "PromptBank", "PatchSelector", "PromptModule",
           "Aggregator", "Recipe", "selectors", "prompts", "aggregators",
           "recipes", "losses"]
