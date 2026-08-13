"""Method registry.

Every paper in this codebase is exposed through a `BaseMethod` adapter
that has a uniform interface. The unified `train.py` then dispatches
to the right method by looking it up in the registry below.

Adding a new method
-------------------
1. Create a folder `methods/<my_method>/` with a `__init__.py`.
2. Put the method's unique model file(s) there. Re-use anything you
   can from `common/`.
3. Subclass `BaseMethod` (file `methods/<my_method>/adapter.py`) and
   implement at minimum:

       build_model(self, cfg) -> nn.Module
       train_step(self, batch, model, optimizer, loss_fn) -> dict
       eval_step(self, batch, model, loss_fn) -> dict

   Many methods can simply inherit `CLAMScaffoldMethod` (defined below)
   which already wires up the FOCUS/ViLa-MIL training loop.
4. Register the adapter in `methods/__init__.py` `METHODS = {...}`.

That's it -- `train.py --method <my_method> --config configs/<my>.yaml`
will pick it up.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from common.backbones import EncoderBundle, MethodBackboneContract


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------
class BaseMethod(ABC):
    """Define the uniform lifecycle implemented by every method adapter.

    A method adapter holds the *recipe* for a paper:
      - how to instantiate the model from a config dict
      - how a single train step looks
      - how a single eval step looks

    State that survives across steps (e.g. running EMA, prototypes,
    pseudo labels) should live as attributes on `self`.

    Args:
        cfg: Validated run configuration. Adapters may read method-specific
            fields but should preserve common fields such as ``n_classes`` and
            ``feature_dim``.
        device: PyTorch device used for model parameters and input batches.
    """

    name: str = "base"
    backbone_contract: Optional[MethodBackboneContract] = None

    def __init__(self, cfg: Dict[str, Any], device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.backbone_name = (self.backbone_contract.validate_config(cfg)
                              if self.backbone_contract is not None else None)

    @classmethod
    def get_backbone_contract(cls) -> MethodBackboneContract:
        """Return the encoder/feature contract declared by the adapter.

        Raises:
            NotImplementedError: If the adapter omitted its required contract.
        """
        if cls.backbone_contract is None:
            raise NotImplementedError(
                f"Method adapter {cls.__name__} has no backbone contract")
        return cls.backbone_contract

    def load_encoder(self, *, weights_path: Optional[str] = None,
                     **loader_options: Any) -> EncoderBundle:
        """Build and validate the configured encoder in one operation.

        Args:
            weights_path: Optional local checkpoint or model identifier. When
                omitted, ``cfg['backbone_weights']`` is used.
            **loader_options: Family-specific options forwarded to the encoder
                registry, such as a pinned revision or offline-only loading.

        Returns:
            A validated capability-aware encoder bundle.

        Raises:
            RuntimeError: If the adapter has no runtime backbone contract.
            BackboneCompatibilityError: If the bundle violates the method
                contract or declared feature provenance.
        """
        if self.backbone_contract is None or self.backbone_name is None:
            raise RuntimeError(f"Method '{self.name}' has no runtime backbone")
        from common.backbones import build_encoder
        if weights_path is None:
            weights_path = self.cfg.get("backbone_weights")
        bundle = build_encoder(self.backbone_name, weights_path=weights_path,
                               device=self.device, **loader_options)
        return self.backbone_contract.validate_bundle(self.cfg, bundle)

    # ------------------------------------------------------------------
    # required hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Instantiate the trainable model and move it to ``self.device``."""

    @abstractmethod
    def train_step(self, batch, model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module) -> Dict[str, float]:
        """Run one optimization step.

        Returns:
            A mapping containing at least ``loss``, ``logits``, and ``label``.
        """

    @abstractmethod
    def eval_step(self, batch, model: nn.Module,
                  loss_fn: Optional[nn.Module] = None
                  ) -> Dict[str, float]:
        """Run a forward-only validation or test step.

        Returns:
            A mapping containing at least ``loss``, ``logits``, and ``label``.
        """

    # ------------------------------------------------------------------
    # optional hooks (override if your method needs them)
    # ------------------------------------------------------------------
    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build Adam over trainable parameters using the configured recipe."""
        params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.Adam(
            params,
            lr=self.cfg.get("lr", 1e-4),
            weight_decay=self.cfg.get("weight_decay", 1e-5),
        )

    def build_scheduler(self, optimizer):
        """Default: ReduceLROnPlateau(min, factor=0.1, patience=10).

        FOCUS / ViLa-MIL / CoD-MIL use this. PathPT overrides with a
        cosine-with-warmup. SLIP / TOP / WSI-FiVE use their own.
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=10)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Handle an optional callback after a full train/validation epoch."""

    def on_fold_end(self, fold: int, metrics: Dict[str, float]) -> None:
        """Handle an optional callback after a cross-validation fold."""
