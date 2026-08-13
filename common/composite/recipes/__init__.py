"""Training recipes -- one per paper plus mix-and-match.

Each recipe encodes one paper's optimizer / scheduler / epoch budget.
Some are mutually exclusive (you cannot run TOP's lr=0.02 alongside
PathPT's lr=1e-4); the user picks ONE.
"""
from __future__ import annotations
import math
import torch

from common.composite.interfaces import Recipe


class FocusRecipe(Recipe):
    """FOCUS / ViLa-MIL / CoD-MIL: Adam(1e-4), 200 epochs, ReduceLROnPlateau."""
    name = "focus"

    def __init__(self, lr: float = 1e-4, weight_decay: float = 1e-5,
                 epochs: int = 200, patience: int = 10):
        self.lr = lr; self.weight_decay = weight_decay
        self._epochs = epochs; self.patience = patience

    def build_optimizer(self, model):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=self.patience, verbose=True)

    @property
    def epochs(self): return self._epochs


class PathPTRecipe(Recipe):
    """PathPT: Adam(1e-4), 20 epochs, cosine + 10% linear warmup.
    Locked across PLIP/CONCH/KEEP/MUSK by design."""
    name = "pathpt"

    def __init__(self, lr: float = 1e-4, epochs: int = 20):
        self.lr = lr; self._epochs = epochs

    def build_optimizer(self, model):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

    def build_scheduler(self, optimizer):
        warmup = max(1, int(self._epochs * 0.1))
        epochs = self._epochs

        def lr_lambda(cur):
            if cur < warmup:
                return cur / warmup
            return 0.5 * (1.0 + math.cos(
                math.pi * (cur - warmup) / max(1, epochs - warmup)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @property
    def epochs(self): return self._epochs


class TOPRecipe(Recipe):
    """TOP: lr 0.02, epochs ~8000, no scheduler."""
    name = "top"

    def __init__(self, lr: float = 0.02, epochs: int = 8000):
        self.lr = lr; self._epochs = epochs

    def build_optimizer(self, model):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

    @property
    def epochs(self): return self._epochs


class SLIPRecipe(Recipe):
    """SLIP: lr=2e-3, epochs=10."""
    name = "slip"

    def __init__(self, lr: float = 2e-3, epochs: int = 10):
        self.lr = lr; self._epochs = epochs

    def build_optimizer(self, model):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

    @property
    def epochs(self): return self._epochs


class WSIFiVERecipe(Recipe):
    """WSI-FiVE: lr=8e-6, 30 epochs, cosine."""
    name = "wsi_five"

    def __init__(self, lr: float = 8e-6, epochs: int = 30,
                 weight_decay: float = 0.05):
        self.lr = lr; self._epochs = epochs; self.weight_decay = weight_decay

    def build_optimizer(self, model):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._epochs)

    @property
    def epochs(self): return self._epochs


REGISTRY = {
    "focus":    FocusRecipe,
    "pathpt":   PathPTRecipe,
    "top":      TOPRecipe,
    "slip":     SLIPRecipe,
    "wsi_five": WSIFiVERecipe,
}


def build_recipe(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown recipe '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
