"""Two-stage SLDPC adapter: CPI then DHNO/SICL prompt refinement."""
from __future__ import annotations

from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import BaseMethod
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)


class SLDPCMethod(BaseMethod):
    """Adapt SLDPC's staged prompt refinement over registered slide vectors."""
    name = "sldpc"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.SLIDE_EMBEDDING,
        swap_policy=SwapPolicy.CAPABILITY, default_backbone="titan",
        required_capabilities=frozenset({Cap.TEXT_ENCODE, Cap.SOFT_PROMPT}),
        feature_dim_key=None, feature_space_key="prompt_feature_space_id",
        require_feature_space=True,
        rationale=(
            "The runtime prompt backbone must expose differentiable soft "
            "text prompts. Offline slide features are independently "
            "registered and use either the backbone's native paired "
            "projection or an explicit trainable adapter."))

    def build_model(self) -> nn.Module:
        expected_epochs = self.cfg.get("stage1_epochs", 50) + self.cfg.get("stage2_epochs", 50)
        self.cfg.setdefault("epochs", expected_epochs)
        if self.cfg.get("epochs", expected_epochs) != expected_epochs:
            raise ValueError(
                "SLDPC epochs must match both stages exactly: set epochs = "
                "stage1_epochs + stage2_epochs")
        if self.cfg.get("early_stopping", True):
            raise ValueError(
                "SLDPC requires early_stopping: false in the unified loop; "
                "each stage tracks and restores its own best prompt internally.")
        loader_options = {}
        weights_path = self.cfg.get("backbone_weights")
        if self.backbone_name == "titan":
            # ``backbone_weights`` records the exact cached snapshot for the
            # benchmark audit. Load through the stable Hub model ID + pinned
            # revision so the runtime bundle retains the same feature-space
            # identifier instead of treating the local directory as a new ID.
            model_reference = (
                self.cfg.get("prompt_encoder_model_id")
                or self.cfg.get("titan_model_id", "MahmoodLab/TITAN"))
            loader_options = {
                "model_id": model_reference,
                "revision": self.cfg.get(
                    "prompt_encoder_revision", self.cfg.get("titan_revision")),
                "local_files_only": self.cfg.get("local_files_only", False),
            }
            weights_path = None
        encoder = self.load_encoder(weights_path=weights_path, **loader_options)
        if encoder.text is None:
            raise TypeError(f"SLDPC backbone '{encoder.spec.name}' has no text interface")
        projection_cfg = dict(self.cfg.get("slide_projection", {}))
        projection_mode = str(projection_cfg.get("mode", "native")).lower()
        feature_dim = int(self.cfg["feature_dim"])
        if projection_mode == "native":
            encoder.require(
                Cap.SLIDE_PROJECT, Cap.PAIRED_SLIDE_TEXT,
                consumer="SLDPC native projection")
            if self.cfg["feature_space_id"] != encoder.spec.feature_space_id:
                raise ValueError(
                    "SLDPC native projection requires the offline slide "
                    "feature space to exactly match the prompt backbone: "
                    f"{self.cfg['feature_space_id']!r} != "
                    f"{encoder.spec.feature_space_id!r}. Use a learned "
                    "slide_projection when swapping slide encoders.")
            if (encoder.spec.slide_input_dim is not None
                    and feature_dim != encoder.spec.slide_input_dim):
                raise ValueError(
                    "SLDPC native projection expects slide width "
                    f"{encoder.spec.slide_input_dim}, got {feature_dim}")
            slide_adapter = None
        elif projection_mode in {"linear", "mlp"}:
            from common.models.slide_alignment import (
                build_slide_embedding_adapter,
            )
            output_dim = encoder.spec.shared_dim
            if output_dim is None:
                raise ValueError(
                    f"SLDPC prompt backbone '{encoder.spec.name}' does not "
                    "declare a shared text embedding dimension")
            declared_output = projection_cfg.get("output_dim")
            if (declared_output is not None
                    and int(declared_output) != int(output_dim)):
                raise ValueError(
                    "SLDPC slide_projection.output_dim must match the prompt "
                    f"space width {output_dim}, got {declared_output}")
            slide_adapter = build_slide_embedding_adapter(
                projection_cfg,
                input_dim=feature_dim,
                output_dim=int(output_dim),
            ).to(self.device)
        else:
            raise ValueError(
                "SLDPC slide_projection.mode must be native, linear, or mlp")

        from .model import PromptedSlideTextModel, SLDPCPromptLearner
        prompt = SLDPCPromptLearner(
            self.cfg["classnames"], encoder.text, n_ctx=self.cfg.get("n_ctx", 8),
            ctx_init=self.cfg.get("ctx_init"), csc=self.cfg.get("csc", False),
            class_token_position=self.cfg.get("class_token_position", "end"),
            omega=self.cfg.get("omega", 0.8)).to(self.device)
        self._phase = "stage1"
        self._cache: dict[str, tuple[torch.Tensor, int]] = {}
        self._bank_by_class: dict[int, torch.Tensor] = {}
        self._base_text = None
        self._validation_logits: list[torch.Tensor] = []
        self._validation_labels: list[torch.Tensor] = []
        self._best_scores = {"stage1": float("-inf"),
                             "stage2": float("-inf")}
        self._best_contexts: dict[str, torch.Tensor | None] = {
            "stage1": None, "stage2": None}
        self._best_projection_state: dict[str, torch.Tensor] | None = None
        return PromptedSlideTextModel(
            encoder, prompt, slide_adapter=slide_adapter,
            slide_input_dim=feature_dim).to(self.device)

    def build_optimizer(self, model):
        optimizer = torch.optim.AdamW(
            filter(lambda item: item.requires_grad, model.parameters()),
            lr=self.cfg.get("stage1_lr", self.cfg.get("lr", 1e-3)),
            weight_decay=self.cfg.get("weight_decay", 0.0))
        self._optimizer = optimizer
        return optimizer

    def build_scheduler(self, optimizer):
        return None

    def _cache_batch(self, batch) -> None:
        ids = batch.get("slide_id", [str(index) for index in range(len(batch["label"]))])
        labels = torch.as_tensor(batch["label"])
        for index, slide_id in enumerate(ids):
            self._cache.setdefault(str(slide_id), (batch["feat"][index].detach().cpu(),
                                                    int(labels[index])))

    def _prepare_stage2(self, model) -> None:
        if not self._cache:
            raise RuntimeError("SLDPC did not observe Stage-1 training embeddings")
        features = torch.stack([entry[0] for entry in self._cache.values()]).to(self.device)
        labels = torch.tensor([entry[1] for entry in self._cache.values()], device=self.device)
        with torch.no_grad():
            bank = model.project_slide(features).detach()
            self._base_text = model.prompt_learner(mode="base").detach()
        grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
        for index, label in enumerate(labels.tolist()):
            grouped[int(label)].append(bank[index])
        self._bank_by_class = {label: torch.stack(values) for label, values in grouped.items()}
        missing = sorted(set(range(self.cfg["n_classes"])) - self._bank_by_class.keys())
        if missing:
            raise RuntimeError(
                "SLDPC Stage-2 feature bank has no training examples for "
                f"classes {missing}")
        # The upstream trainer owns a deterministic per-stage sampler. Use
        # PyTorch's fold-specific initial seed without coupling draws to other
        # data-loader or augmentation RNG calls.
        self._stage2_rng = random.Random(torch.initial_seed())

    def _hard_negative_extension(self, projected: torch.Tensor, labels: torch.Tensor,
                                 task_text: torch.Tensor):
        assert self._base_text is not None
        del task_text  # the frozen base prompt P performs DHNO retrieval
        k = min(max(1, int(self.cfg.get("topk", 8))), self.cfg["n_classes"])
        if k == 1:
            return projected, labels

        # Match upstream DHNO: retrieve with frozen P, exclude the current
        # positive and other ground-truth classes already present in the
        # mini-batch, then uniformly fill any exhausted candidate slots.
        base_scores = projected @ self._base_text.t()
        top_classes = base_scores.topk(k, dim=-1).indices.tolist()
        base_ids = list(range(self.cfg["n_classes"]))
        batch_ground_truth = set(labels.tolist())
        all_features, all_labels = [], []
        for row, label in enumerate(labels.tolist()):
            label = int(label)
            selected = [label]
            taken = {label}
            forbidden = batch_ground_truth - {label}
            for candidate in top_classes[row]:
                candidate = int(candidate)
                if candidate not in taken and candidate not in forbidden:
                    selected.append(candidate)
                    taken.add(candidate)
                if len(selected) == k:
                    break
            while len(selected) < k:
                pool = [candidate for candidate in base_ids
                        if candidate not in taken]
                if not pool:
                    pool = base_ids
                candidate = self._stage2_rng.choice(pool)
                selected.append(candidate)
                taken.add(candidate)

            all_features.append(projected[row])
            all_labels.append(label)
            for candidate in selected[1:]:
                candidate_bank = self._bank_by_class[candidate]
                sample = self._stage2_rng.randrange(candidate_bank.shape[0])
                all_features.append(candidate_bank[sample])
                all_labels.append(candidate)
        return torch.stack(all_features), torch.tensor(all_labels, device=self.device)

    @staticmethod
    def _symmetric_info_nce(features: torch.Tensor, text: torch.Tensor, tau: float) -> torch.Tensor:
        logits = features @ text.t() / tau
        targets = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))

    def train_step(self, batch, model, optimizer, loss_fn):
        self._remember_model(model)
        features = batch["feat"].to(self.device)
        labels = torch.as_tensor(batch["label"], device=self.device, dtype=torch.long)
        if self._phase == "stage1":
            self._cache_batch(batch)
        optimizer.zero_grad()
        projected = model.project_slide(features)
        text = model.prompt_learner(mode="task")
        logits = projected @ text.t()
        if self._phase == "stage1":
            divisor = self.cfg.get("tau", 0.07) if self.cfg.get("stage1_apply_tau", False) else 1.0
            loss = loss_fn(logits / divisor, labels)
        else:
            extended, extended_labels = self._hard_negative_extension(projected, labels, text)
            paired_text = text[extended_labels]
            loss = self._symmetric_info_nce(extended.detach(), paired_text, self.cfg.get("tau", 0.07))
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": labels}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        self._remember_model(model)
        features = batch["feat"].to(self.device)
        labels = torch.as_tensor(batch["label"], device=self.device, dtype=torch.long)
        mode = "fused" if self._phase == "stage2" else "task"
        logits = model(features, mode=mode)
        loss = loss_fn(logits, labels).item() if loss_fn is not None else 0.0
        self._validation_logits.append(logits.detach().cpu())
        self._validation_labels.append(labels.detach().cpu())
        return {"loss": loss, "logits": logits, "label": labels}

    def _validation_score(self, metrics) -> float:
        if not self._validation_logits:
            return float("nan")
        logits = torch.cat(self._validation_logits)
        labels = torch.cat(self._validation_labels).long()
        self._validation_logits.clear()
        self._validation_labels.clear()
        monitor = str(self.cfg.get("monitor_metric", "F1")).upper()
        if monitor in {"LOSS", "VAL_LOSS"}:
            return -float(metrics["val_loss"])

        predictions = logits.argmax(dim=-1)
        if monitor == "ACC":
            return float(predictions.eq(labels).float().mean().item() * 100.0)
        if monitor == "F1":
            scores = []
            for class_index in range(self.cfg["n_classes"]):
                positive = labels.eq(class_index)
                predicted = predictions.eq(class_index)
                true_positive = (positive & predicted).sum().item()
                false_positive = ((~positive) & predicted).sum().item()
                false_negative = (positive & (~predicted)).sum().item()
                denominator = 2 * true_positive + false_positive + false_negative
                scores.append(0.0 if denominator == 0 else
                              2.0 * true_positive / denominator)
            return 100.0 * sum(scores) / len(scores)
        if monitor == "AUC":
            from sklearn.metrics import roc_auc_score
            probabilities = logits.softmax(dim=-1).numpy()
            targets = labels.numpy()
            try:
                if self.cfg["n_classes"] == 2:
                    return float(100.0 * roc_auc_score(
                        targets, probabilities[:, 1]))
                return float(100.0 * roc_auc_score(
                    targets, probabilities,
                    labels=list(range(self.cfg["n_classes"])),
                    multi_class="ovr", average="macro"))
            except ValueError:
                return float("nan")
        raise ValueError(
            "SLDPC monitor_metric must be one of F1, ACC, AUC, or val_loss")

    @torch.no_grad()
    def _update_best_prompt(self, model, metrics) -> None:
        score = self._validation_score(metrics)
        if score == score and score > self._best_scores[self._phase]:
            self._best_scores[self._phase] = score
            self._best_contexts[self._phase] = (
                model.prompt_learner.ctx_learnable.detach().clone())
            slide_adapter = getattr(model, "slide_adapter", None)
            if self._phase == "stage1" and slide_adapter is not None:
                self._best_projection_state = {
                    key: value.detach().clone()
                    for key, value in slide_adapter.state_dict().items()
                }

    @torch.no_grad()
    def _restore_best_prompt(self, model, phase: str) -> None:
        context = self._best_contexts[phase]
        if context is None:
            raise RuntimeError(
                f"SLDPC could not select a best {phase} prompt from validation")
        model.prompt_learner.ctx_learnable.copy_(context)
        slide_adapter = getattr(model, "slide_adapter", None)
        if (phase == "stage1" and slide_adapter is not None
                and self._best_projection_state is not None):
            slide_adapter.load_state_dict(self._best_projection_state)

    def on_epoch_end(self, epoch: int, metrics) -> None:
        model = getattr(self, "_model_for_transition", None)
        # train.py stores no model on adapters; capture it lazily from the
        # optimizer's parameter owner on the first stage instead.
        if model is None:
            model = self._last_model
        self._update_best_prompt(model, metrics)
        if self._phase == "stage2":
            if epoch + 1 == self.cfg["epochs"]:
                # The upstream Stage-2 trainer restores its best validation
                # prompt before the final test evaluation.
                self._restore_best_prompt(model, "stage2")
            return
        if epoch + 1 != self.cfg.get("stage1_epochs", 50):
            return
        # Upstream restores the best Stage-1 prompt before CPI cloning.
        self._restore_best_prompt(model, "stage1")
        # CPI: preserve the learned prompt as P, then start P' from P.
        model.prompt_learner.clone_learnable_to_frozen()
        model.prompt_learner.reinit_learnable_from_frozen()
        self._prepare_stage2(model)
        self._phase = "stage2"
        # Upstream constructs a fresh AdamW optimizer for Stage-2. Clearing
        # moments here preserves that behavior while retaining the unified
        # training loop's optimizer object.
        self._optimizer.state.clear()
        for group in self._optimizer.param_groups:
            group["lr"] = self.cfg.get("stage2_lr", self.cfg.get("lr", 1e-3))

    def on_fold_end(self, fold: int, metrics) -> None:
        self._cache = {}

    def on_checkpoint_loaded(self, model, checkpoint_kind: str, fold: int) -> None:
        if checkpoint_kind != "final":
            raise ValueError(
                "SLDPC standalone evaluation requires the final checkpoint; "
                "the unified two-stage run does not write an early-stopping "
                "best checkpoint")
        # The phase is adapter state rather than a model parameter. Final
        # checkpoints are written only after the mandatory Stage-2 transition.
        self._phase = "stage2"
        self._last_model = model

    # Keep the model reference only for the automatic Stage-1 -> Stage-2
    # transition. It is not a registered module and does not affect checkpoints.
    def _remember_model(self, model) -> None:
        self._last_model = model
