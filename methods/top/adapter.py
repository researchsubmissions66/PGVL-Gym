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
import hashlib
import json

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

    # Upstream appends ten learnable slots to every initialised prompt. Keep
    # the separator outside this literal: the released TCGA instance recipe
    # concatenates the first ``*`` directly after the final period, whereas
    # its bag recipe (and CAMELYON instance recipe) inserts one space.
    LEARNABLE_SLOTS = " ".join(["*"] * 10)
    BINARY_ONLY_POOLING = frozenset({
        "NoCoOp", "learnablePrompt_noCoOp",
        "learnablePrompt_multi_noCoOp",
    })

    def _instance_ctx_init(self) -> list[str]:
        """Return TOP's instance prototype prompts, or [] to fall back.

        ``instance_prompt_path`` names the prototype bank exported from the
        paper's released ``knowledge_from_chatGPT`` table. Each entry becomes
        ``an H&E stained image of {tissue}, which is {description}`` followed by
        the learnable context slots, matching the upstream initialisation.
        """
        path = self.cfg.get("instance_prompt_path")
        if not path:
            return []
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        prototypes = payload["prototypes"] if isinstance(payload, dict) else payload
        if not isinstance(prototypes, list) or not prototypes:
            raise ValueError(f"{path}: TOP instance bank has no prototypes")
        metadata = payload.get("_metadata", {}) \
            if isinstance(payload, dict) else {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: TOP instance _metadata must be a mapping")
        prompts: list[str] = []
        for index, item in enumerate(prototypes):
            if not isinstance(item, dict) or not isinstance(
                    item.get("prompt"), str) or not item["prompt"].strip():
                raise ValueError(
                    f"{path}: TOP instance prototype {index} needs a prompt")
            prompt = item["prompt"]
            tissue, description = item.get("tissue"), item.get("description")
            if isinstance(tissue, str) and isinstance(description, str):
                expected = (
                    f"an H&E stained image of {tissue}, which is {description}")
                if prompt != expected:
                    raise ValueError(
                        f"{path}: TOP instance prototype {index} does not "
                        "match its tissue/description fields")
            if "*" in prompt:
                raise ValueError(
                    f"{path}: TOP instance prototype {index} already contains "
                    "learnable slots")
            prompts.append(prompt)
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"{path}: TOP instance prompts must be unique")
        declared_count = metadata.get("count")
        if declared_count is not None and declared_count != len(prompts):
            raise ValueError(
                f"{path}: metadata count {declared_count} does not match "
                f"{len(prompts)} prototypes")
        declared_digest = metadata.get("ordered_prompt_sha256")
        digest = hashlib.sha256("\n".join(prompts).encode()).hexdigest()
        if declared_digest is not None and declared_digest != digest:
            raise ValueError(
                f"{path}: ordered_prompt_sha256 does not match the prompt bank")
        separator = self.cfg.get(
            "instance_slot_separator", metadata.get("slot_separator", ""))
        if separator not in {"", " "}:
            raise ValueError(
                "instance_slot_separator must be either an empty string or "
                "one space")
        return [f"{prompt}{separator}{self.LEARNABLE_SLOTS}"
                for prompt in prompts]

    def _bag_ctx_init(self):
        """Return per-class bag prompts when declared, else the configured init."""
        path = self.cfg.get("bag_prompt_path")
        if not path:
            return self.cfg.get("ctx_init_bag", "")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: TOP bag prompt bank must be a mapping")
        # Code-faithful assets store the complete upstream ctx_init literal,
        # including its ten slots. Supplementary/experimental banks store base
        # prompts and let the adapter append the same slots.
        exact_ctx = payload.get("ctx_init")
        prompts = exact_ctx if exact_ctx is not None else payload.get("prompts")
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError(
                f"{path}: TOP bag bank needs a prompts or ctx_init mapping")
        label_dict = self.cfg["label_dict"]
        labels = [label for label, _ in sorted(
            label_dict.items(), key=lambda item: item[1])]
        missing = [label for label in labels if label not in prompts]
        if missing:
            raise KeyError(
                f"{path}: bag prompts missing entries for {missing}; "
                f"file provides {sorted(prompts)}")
        values = [prompts[label] for label in labels]
        if any(not isinstance(value, str) or not value.strip()
               for value in values):
            raise ValueError(f"{path}: TOP bag prompts must be non-empty strings")
        metadata = payload.get("_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: TOP bag _metadata must be a mapping")
        declared_order = metadata.get("label_order")
        if declared_order is not None and declared_order != labels:
            raise ValueError(
                f"{path}: TOP bag label_order {declared_order} does not match "
                f"the classifier order {labels}")
        if exact_ctx is not None:
            invalid = [label for label, value in zip(labels, values)
                       if value.count("*") != 10]
            if invalid:
                raise ValueError(
                    f"{path}: code-faithful TOP ctx_init entries must contain "
                    f"ten learnable slots; invalid labels {invalid}")
            return values
        invalid = [label for label, value in zip(labels, values) if "*" in value]
        if invalid:
            raise ValueError(
                f"{path}: base TOP bag prompts already contain learnable "
                f"slots for labels {invalid}")
        separator = metadata.get("slot_separator", " ")
        if separator not in {"", " "}:
            raise ValueError(
                f"{path}: slot_separator must be empty or one space")
        return [f"{value}{separator}{self.LEARNABLE_SLOTS}"
                for value in values]

    def _bag_classnames(self) -> list[str]:
        """Use the task suffixes paired with an upstream bag initializer.

        ``PromptLearner`` appends these names after the learnable slots, so the
        ctx_init string alone is not the whole prompt. Code-faithful banks carry
        the exact suffix list; task extensions without a published bag bank use
        the benchmark's configured classnames.
        """
        path = self.cfg.get("bag_prompt_path")
        if not path:
            return list(self.cfg["classnames"])
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("_metadata", {}) \
            if isinstance(payload, dict) else {}
        upstream = metadata.get("upstream_classnames") \
            if isinstance(metadata, dict) else None
        if upstream is None:
            return list(self.cfg["classnames"])
        if (not isinstance(upstream, list)
                or len(upstream) != len(self.cfg["label_dict"])
                or any(not isinstance(name, str) or not name.strip()
                       for name in upstream)):
            raise ValueError(
                f"{path}: upstream_classnames must contain one non-empty "
                "suffix per task label")
        return upstream

    def build_model(self) -> nn.Module:
        from .learnable_prompt import (
            MIL_CLIP, PromptLearner)

        encoder = self.load_encoder(
            weights_path=self.cfg.get("backbone_weights")).freeze()
        clip_model = encoder.raw_model

        bag_pl = PromptLearner(
            classnames=self._bag_classnames(),
            clip_model=clip_model,
            n_ctx=self.cfg.get("n_ctx_bag", 4),
            ctx_init=self._bag_ctx_init(),
            csc=self.cfg.get("csc", True),
        )
        # TOP's instance branch is prototype-based: its prompts are tissue
        # phenotypes shared across tasks, not the bag class names. Upstream
        # passes the templated descriptions as ctx_init and uses positional
        # "Prototype i" placeholders as classnames, so the learner sizes itself
        # from the number of prototypes rather than the number of classes.
        instance_ctx = self._instance_ctx_init()
        inst_pl = PromptLearner(
            classnames=[f"Prototype {index}" for index in range(len(instance_ctx))]
            if instance_ctx else self.cfg.get(
                "instance_classnames", self.cfg["classnames"]),
            clip_model=clip_model,
            n_ctx=self.cfg.get("n_ctx_inst", 4),
            ctx_init=instance_ctx or self.cfg.get("ctx_init_inst", ""),
            csc=self.cfg.get("csc", True),
        )
        pooling_strategy = self.cfg.get(
            "pooling_strategy", "learnablePrompt_multi")
        if (pooling_strategy in self.BINARY_ONLY_POOLING
                and len(self.cfg["classnames"]) != 2):
            raise ValueError(
                f"TOP pooling_strategy={pooling_strategy!r} is binary-only in "
                "the upstream implementation; choose a prompt-classifier "
                "pooling strategy for multiclass tasks")
        model = MIL_CLIP(bag_pl, inst_pl,
                         clip_model=clip_model,
                         pooling_strategy=pooling_strategy)
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
        """Reduce TOP's prototype-conditioned output to one slide prediction.

        ``learnablePrompt_multi`` returns ``(n_prototypes, n_classes)``: each
        instance prototype pools the bag into its own representation, which is
        then scored against the bag class prompts. Upstream takes the slide
        probability as the mean over prototypes *after* softmax
        (``bag_prediction.mean(0)`` in train_TCGAFeat_MIL_CLIP.py), so the mean
        is over probabilities, not logits.

        The value returned here is ``log`` of that mean. Because the mean of
        softmax rows already sums to one, ``softmax(log p) == p`` exactly, so
        the benchmark's shared metric code recovers upstream's probabilities and
        cross-entropy against it is the correct negative log-likelihood.

        The auxiliary term is TOP's LossA: an off-diagonal correlation penalty
        on the instance attention scores that pushes prototypes apart. It is
        computed from the attention matrix the model returns alongside the
        logits, not from the logits themselves.
        """
        raw = output[0] if isinstance(output, tuple) else output
        attention = output[1] if isinstance(output, tuple) and len(output) > 1 else None

        auxiliary = None
        if attention is not None and attention.ndim == 2:
            normed = torch.softmax(attention, dim=0)
            auxiliary = torch.triu(normed.T @ normed, diagonal=1).mean()

        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw.ndim == 2 and raw.shape[0] > 1 and raw.shape[1] == self.cfg["n_classes"]:
            probabilities = raw.softmax(dim=1).mean(dim=0, keepdim=True)
            raw = probabilities.clamp_min(1e-12).log()
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
