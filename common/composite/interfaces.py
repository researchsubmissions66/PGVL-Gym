"""Pluggable component interfaces for the composite WSI/VLM model.

Four building blocks compose into one model:

    backbone -> SELECTORS (pipeline) -> PROMPTS (bank) -> AGGREGATORS (fused)

Selectors stack sequentially. Prompts fuse into a bank. Aggregators run
in parallel and fuse via one of two modes:

  - "logit_ensemble" (Level 1): each aggregator runs end-to-end and
    produces (C,) logits; logits combine at the very end. Most decoupled,
    most robust if any single aggregator is buggy.
  - "vector_fusion"  (Level 2): each aggregator returns a slide vector
    (D,); vectors are fused into one (D,) and a shared classifier head
    produces (C,). Tighter coupling, often higher capacity.

The user picks the mode in YAML.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
@dataclass
class PromptBank:
    """Output of the prompt block.

    ``text_features`` is the canonical ``[classes, dimension]`` tensor every
    aggregator can consume. ``aux`` is a mapping of optional tensors that
    specific aggregators may want (e.g. SLIP needs tissue prompts,
    MAPLE-graph optionally provides per-entity features).
    """
    text_features: torch.Tensor                  # (C, D)
    aux: Dict[str, torch.Tensor]                 # e.g. {"tissue": (T, D)}


# ---------------------------------------------------------------------------
class PatchSelector(nn.Module, ABC):
    """Filters / re-orders patches before aggregation.

    Implementations are functions (patches, text_features, coords) ->
    patches', where patches' is a (possibly smaller) subset of the input.
    Selectors stack: the output of one feeds the next.
    """
    name: str = "base_selector"

    @abstractmethod
    def forward(self,
                patches: torch.Tensor,                   # (N, D)
                text_features: torch.Tensor,             # (C, D)
                coords: Optional[torch.Tensor] = None    # (N, 2) or None
                ) -> torch.Tensor:                       # (N', D), N' <= N
        ...


# ---------------------------------------------------------------------------
class PromptModule(nn.Module, ABC):
    """Produces text features.

    Each prompt module returns a (C, D) tensor in `text_features`.
    Auxiliary tensors that don't fit that shape (entity attributes,
    tissue prompts, chain-of-diagnosis hierarchy) go in `aux`.

    Many prompt modules are themselves composed of CoOp-style learnable
    context vectors; common code lives in `common.models.coop`.
    """
    name: str = "base_prompts"

    @abstractmethod
    def forward(self) -> PromptBank:
        ...


# ---------------------------------------------------------------------------
class Aggregator(nn.Module, ABC):
    """Pools patch features into a slide-level prediction.

    Each aggregator implements BOTH return modes:

      - `forward_vector`: returns (D,) slide vector (no classification)
      - `forward_logits`: returns (C,) class logits (with internal head)

    The composite model picks whichever mode matches the configured
    fusion strategy.  Subclasses can implement only one of the two
    methods and inherit a default for the other (vector -> linear head;
    logits -> not vector-recoverable, raises).
    """
    name: str = "base_agg"
    out_dim: int = 0     # only meaningful for vector mode; aggregators
                         # whose internal vector dim differs from D set this.

    @abstractmethod
    def forward_logits(self,
                       patches: torch.Tensor,        # (N', D)
                       bank: PromptBank
                       ) -> torch.Tensor:            # (C,)
        ...

    def forward_vector(self,
                       patches: torch.Tensor,
                       bank: PromptBank
                       ) -> torch.Tensor:
        """Return a pre-classification slide vector when supported.

        Raises:
            NotImplementedError: If the aggregator exposes only logits.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose a slide-vector "
            f"output. Use fusion mode 'logit_ensemble' instead.")


# ---------------------------------------------------------------------------
class Recipe(ABC):
    """Bundles optimizer + scheduler + epoch count for one paper's recipe."""
    name: str = "base_recipe"

    @abstractmethod
    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        ...

    def build_scheduler(self, optimizer):
        """Build an optional scheduler; the default recipe uses none."""
        return None

    @property
    @abstractmethod
    def epochs(self) -> int:
        ...
