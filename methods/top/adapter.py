"""TOP adapter (NeurIPS 2023).

The vendored MIL_CLIP class takes two `PromptLearner` instances --
bag-level and instance-level. We construct them from the classnames
in the config.

TOP's training recipe is very different from the rest:
    lr_TB = lr_IB = 0.02   (vs 1e-4 elsewhere)
    epochs = 8000          (vs 20-200 elsewhere)
    weight_lossA = 25
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import BaseMethod
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)


class TOPMethod(BaseMethod):
    """Adapt TOP's two-level CLIP-RN50 prompt and pooling objective."""
    name = "top"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_BAG,
        swap_policy=SwapPolicy.FIXED, config_key="clip_arch",
        default_backbone="clip-rn50", supported_backbones=("clip-rn50",),
        feature_dims={"clip-rn50": (1024,)},
        required_capabilities=frozenset({Cap.SOFT_PROMPT, Cap.PAIRED_TILE_TEXT}),
        rationale="TOP hardcodes RN50-width instance and bag prompt attention.")

    def build_model(self) -> nn.Module:
        from .learnable_prompt import (
            MIL_CLIP, PromptLearner)

        encoder = self.load_encoder(
            weights_path=self.cfg.get("backbone_weights")).freeze()
        clip_model = encoder.raw_model

        bag_pl = PromptLearner(
            classnames=self.cfg["classnames"],
            clip_model=clip_model,
            n_ctx=self.cfg.get("n_ctx_bag", 4),
            ctx_init=self.cfg.get("ctx_init_bag", ""),
            csc=self.cfg.get("csc", True),
        )
        inst_pl = PromptLearner(
            classnames=self.cfg.get("instance_classnames", self.cfg["classnames"]),
            clip_model=clip_model,
            n_ctx=self.cfg.get("n_ctx_inst", 4),
            ctx_init=self.cfg.get("ctx_init_inst", ""),
            csc=self.cfg.get("csc", True),
        )
        model = MIL_CLIP(bag_pl, inst_pl,
                         clip_model=clip_model,
                         pooling_strategy=self.cfg.get(
                             "pooling_strategy", "learnablePrompt_multi"))
        return model.to(self.device)

    # TOP uses different LRs for the two prompt learners
    def build_optimizer(self, model):
        params = []
        if hasattr(model, "prompt_learner_bagLevel"):
            params.append({"params": model.prompt_learner_bagLevel.parameters(),
                           "lr": self.cfg.get("lr_TB", 0.002)})
        if hasattr(model, "prompt_learner_instanceLevel"):
            params.append({"params": model.prompt_learner_instanceLevel.parameters(),
                           "lr": self.cfg.get("lr_IB", 0.002)})
        # any remaining trainable params (pooling, bag head)
        learned = set()
        for g in params:
            for p in g["params"]:
                learned.add(id(p))
        rest = [p for p in model.parameters()
                if p.requires_grad and id(p) not in learned]
        if rest:
            params.append({"params": rest, "lr": self.cfg.get("lr", 0.002)})
        return torch.optim.Adam(params)

    def build_scheduler(self, optimizer):
        return None

    def _slide_logits(self, output):
        raw = output[0] if isinstance(output, tuple) else output
        auxiliary = None
        if (raw.ndim == 2 and raw.shape ==
                (self.cfg["n_classes"], self.cfg["n_classes"])):
            # ``learnablePrompt_multi`` produces a class-conditioned
            # cross-correlation matrix. Its diagonal is the slide-level
            # class score; row-wise identity classification preserves TOP's
            # two-level alignment objective.
            targets = torch.arange(raw.shape[0], device=raw.device)
            auxiliary = F.cross_entropy(raw, targets)
            raw = torch.diagonal(raw).unsqueeze(0)
        elif raw.ndim == 1:
            raw = raw.unsqueeze(0)
        return raw, auxiliary

    def train_step(self, batch, model, optimizer, loss_fn):
        feats = batch[0].to(self.device)
        label = batch[-1].to(self.device)
        optimizer.zero_grad()
        out = model(feats)
        logits, lossA = self._slide_logits(out)
        loss = loss_fn(logits, label)
        if lossA is not None:
            loss = loss + self.cfg.get("weight_lossA", 25.0) * lossA
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        feats = batch[0].to(self.device)
        label = batch[-1].to(self.device)
        out = model(feats)
        logits, _ = self._slide_logits(out)
        loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": label}
