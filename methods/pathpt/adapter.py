"""PathPT adapter (Nat. Commun. 2026).

PathPT's signature: PPT*(cfg, classnames_lst, model, tokenizer,
                         device, param, vfeat_dim).

Crucially, PathPT uses the *same* training recipe across all four
foundation backbones (PLIP/CONCH/KEEP/MUSK):

    - Adam(lr=1e-4)
    - cosine schedule with 10% linear warm-up
    - 20 epochs
    - n_ctx=32, prompt_init="a histopathology image of " * 8
    - aux_weight=0.5

That recipe is enforced in `build_optimizer` / `build_scheduler` here.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)
from . import params as pathpt_params


class PathPTMethod(BaseMethod):
    """Adapt PathPT while preserving its encoder-independent training recipe."""
    name = "pathpt"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_BAG,
        swap_policy=SwapPolicy.ALLOWLIST, default_backbone="plip",
        supported_backbones=("plip", "conch", "keep", "musk"),
        feature_dims={"plip": (512,), "conch": (512,),
                      "keep": (768,), "musk": (1024,)},
        required_capabilities=frozenset({Cap.SOFT_PROMPT, Cap.PAIRED_TILE_TEXT}),
        rationale="PathPT retains one native soft-prompt implementation per listed backbone family.")

    def build_model(self) -> nn.Module:
        backbone = self.backbone_name
        encoder = self.load_encoder(weights_path=self.cfg.get("backbone_weights"))
        vfeat_dim = encoder.spec.tile_dim
        if vfeat_dim is None:
            raise ValueError(f"PathPT backbone '{backbone}' has no tile feature dimension")

        if backbone == "keep":
            from .pathpt_models.PathPT_model_KEEP import PPTKEEP as _Cls
        elif backbone == "conch":
            from .pathpt_models.PathPT_model_CONCH import PPTCONCH as _Cls
        elif backbone == "plip":
            from .pathpt_models.PathPT_model_PLIP import PPTPLIP as _Cls
        elif backbone == "musk":
            from .pathpt_models.PathPT_model_MUSK import PPTMUSK as _Cls
        else:
            raise KeyError(f"PathPT does not support backbone '{backbone}'")

        prompt_cfg = pathpt_params.PromptLearnerConfig(
            n_ctx=self.cfg.get("n_ctx", 32),
            init=self.cfg.get("prompt_init", "template"),
        )

        # `param` dict mirrors what PathPT's params.py supplies in their
        # `subtype_params['ucs']`. Kept identical across backbones.
        param = {
            "learnable":   self.cfg.get("learnable", "token"),
            "vision_only": self.cfg.get("vision_only", False),
            "vision_grad": self.cfg.get("vision_grad", True),
            "use_aug":     self.cfg.get("use_aug", False),
            "loss_weight": self.cfg.get("loss_weight", [1.0, 0.5, 0.1]),
        }

        # Native PathPT treats each class entry as a list of synonyms. Accept
        # the unified config's simpler list[str] without changing that logic.
        classnames = [item if isinstance(item, (list, tuple)) else [item]
                      for item in self.cfg["classnames"]]
        kwargs = dict(
            classnames_lst=classnames, model=encoder.raw_model,
            device=self.device, param=param, vfeat_dim=vfeat_dim)
        if backbone == "conch":
            # The official CONCH branch obtains its custom tokenizer internally.
            model = _Cls(prompt_cfg, **kwargs)
        else:
            model = _Cls(prompt_cfg, tokenizer=encoder.raw_tokenizer, **kwargs)
        # Match PathPT's native model_init(): optimize the learned context and
        # spatial-awareness modules, while keeping the foundation VLM frozen.
        trainable_prefixes = ("prompt_learner.ctx", "mlp.", "prompt_", "mil.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
        return model.to(self.device)

    def _slide_logits(self, output: object) -> torch.Tensor:
        """Aggregate native per-patch probabilities into one slide prediction."""
        if isinstance(output, dict):
            candidate = output.get("logits")
        elif isinstance(output, tuple):
            primary = output[0] if output else None
            candidate = primary if torch.is_tensor(primary) else output[1]
        else:
            candidate = output
        if not torch.is_tensor(candidate):
            raise TypeError("PathPT did not return tensor patch or slide predictions")
        if candidate.shape[-1] != self.cfg["n_classes"]:
            raise ValueError(
                f"PathPT prediction width {candidate.shape[-1]} does not "
                f"match n_classes={self.cfg['n_classes']}")
        if candidate.ndim == 1:
            candidate = candidate.unsqueeze(0)
        elif candidate.ndim == 2 and candidate.shape[0] != 1:
            candidate = candidate.mean(dim=0, keepdim=True)
        elif candidate.ndim == 3:
            candidate = candidate.mean(dim=1)
        if candidate.ndim != 2:
            raise ValueError(
                f"PathPT predictions must reduce to [batch, classes], got "
                f"{tuple(candidate.shape)}")
        # Native PathPT returns softmax-normalized patch predictions. Logging
        # their mean gives CrossEntropyLoss a stable slide-level score while
        # preserving gradients through prompt and spatial modules.
        if bool(((candidate.detach() >= 0) & (candidate.detach() <= 1)).all()):
            row_sums = candidate.detach().sum(dim=-1)
            if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
                candidate = candidate.clamp_min(1e-12).log()
        return candidate

    # ------------------------------------------------------------------
    # PathPT-specific recipe (locked across backbones)
    # ------------------------------------------------------------------
    def build_optimizer(self, model):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg.get("lr", 1e-4),
        )

    def build_scheduler(self, optimizer):
        epochs = self.cfg.get("epochs", 20)
        warm_up = max(1, int(epochs * 0.1))

        def lr_lambda(cur):
            if cur < warm_up:
                return cur / warm_up
            return 0.5 * (1.0 + math.cos(
                math.pi * (cur - warm_up) / (epochs - warm_up)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    def train_step(self, batch, model, optimizer, loss_fn):
        feats = batch[0].to(self.device)
        label = batch[-1].to(self.device)

        optimizer.zero_grad()
        out = model(feats)
        logits = self._slide_logits(out)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        feats = batch[0].to(self.device)
        label = batch[-1].to(self.device)
        out = model(feats)
        logits = self._slide_logits(out)
        loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": label}
