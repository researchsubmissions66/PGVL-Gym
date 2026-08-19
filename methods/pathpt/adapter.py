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

That recipe is enforced in `build_optimizer` / `build_scheduler` here. Native
mode additionally restores the released prompt selector, synthetic Normal
patch class, PatchSSLoss, and vote-based WSI inference. The historical
slide-cross-entropy integration remains available as an explicit legacy mode.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import tempfile

import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import (
    BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy)
from . import params as pathpt_params
from .loss import PatchSSLoss
from .native import (
    choose_prompt_embedding,
    extract_patch_scores,
    generate_patch_targets,
    select_prompt_embedding,
    vote_slide_probabilities,
)
from .prompts import PathPTPromptBank, resolve_prompt_bank


class PathPTMethod(BaseMethod):
    """Adapt PathPT while preserving its encoder-independent training recipe."""
    name = "pathpt"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_BAG,
        swap_policy=SwapPolicy.ALLOWLIST, default_backbone="plip",
        supported_backbones=("plip", "conch", "keep", "musk"),
        feature_dims={"plip": (512,), "conch": (512,),
                      "keep": (768,), "musk": (1024,)},
        required_capabilities=frozenset({
            Cap.TEXT_ENCODE, Cap.SOFT_PROMPT, Cap.PAIRED_TILE_TEXT}),
        rationale="PathPT retains one native soft-prompt implementation per listed backbone family.")

    LEGACY_MODE = "simplified_slide_ce"
    NATIVE_MODE = "upstream_patch_ssl"

    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        self.training_mode = str(
            self.cfg.get("training_mode", self.LEGACY_MODE)).strip().lower()
        if self.training_mode not in {self.LEGACY_MODE, self.NATIVE_MODE}:
            raise ValueError(
                "PathPT training_mode must be 'simplified_slide_ce' or "
                "'upstream_patch_ssl'")
        self.prompt_bank: PathPTPromptBank | None = None
        self._epoch = 0
        self._fold = 0
        self._encoder = None

    @property
    def native_mode(self) -> bool:
        return self.training_mode == self.NATIVE_MODE

    def build_model(self) -> nn.Module:
        backbone = self.backbone_name
        encoder = self.load_encoder(weights_path=self.cfg.get("backbone_weights"))
        self._encoder = encoder
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

        if self.native_mode:
            if self.cfg.get("vision_only", False):
                raise ValueError(
                    "PathPT upstream_patch_ssl requires vision_only=false; "
                    "the released vision-only head does not expose the "
                    "synthetic-Normal patch classifier")
            self.prompt_bank = resolve_prompt_bank(self.cfg)
            classnames = [list(row) for row in self.prompt_bank.class_synonyms]
        else:
            # Legacy integration treats each configured class as a singleton
            # synonym list and has no synthetic patch class.
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
        if self.native_mode:
            # Define the buffer at construction so evaluation-only processes
            # can strictly load a training checkpoint without rerunning prompt
            # selection or touching the train split.
            model.register_buffer(
                "pathpt_selected_prompt_embedding",
                torch.zeros(
                    self.patch_class_count, vfeat_dim,
                    device=self.device, dtype=torch.float32),
                persistent=True)
        return model.to(self.device)

    @property
    def patch_class_count(self) -> int:
        if self.native_mode:
            if self.prompt_bank is None:
                raise RuntimeError("PathPT prompt bank has not been built")
            return len(self.prompt_bank.class_synonyms)
        return int(self.cfg["n_classes"])

    def _write_prompt_trace(self, selection) -> None:
        if self.prompt_bank is None or not self.cfg.get("results_dir"):
            return
        output = Path(self.cfg["results_dir"])
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"fold{self._fold}_pathpt_prompt_selection.json"
        payload = {
            "schema_version": 1,
            "fold": self._fold,
            "task": self.prompt_bank.task,
            "prompt_provenance": self.prompt_bank.provenance,
            "prompt_source": self.prompt_bank.source,
            "prompt_note": self.prompt_bank.note,
            "synthetic_normal": self.prompt_bank.synthetic_normal,
            "templates": 22,
            "prompt_counts": [len(row) for row in self.prompt_bank.prompts],
            "top_classifier_indices": list(selection.top_classifier_indices),
            "balanced_accuracies": list(selection.balanced_accuracies),
            "candidate_prompt_indices": [
                list(row) for row in selection.prompt_indices],
        }
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output,
            prefix=f".{target.name}.", suffix=".tmp", delete=False)
        try:
            with handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, target)
        except Exception:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise

    def prepare_fold(self, fold, model, train_loader) -> None:
        """Select the zero-shot patch classifier from training slides only."""
        if not self.native_mode:
            return
        if self.prompt_bank is None or self._encoder is None:
            raise RuntimeError("PathPT model must be built before prepare_fold")
        self._fold = int(fold)
        prompt_select = bool(self.cfg.get("prompt_select", True))
        classifier_count = int(self.cfg.get("prompt_classifier_count", 200))
        select_count = int(self.cfg.get("prompt_select_count", 100))
        # Upstream uses a second, unsampled WSI loader for prompt selection.
        # Reuse the fold loader without duplicating feature files, temporarily
        # disabling only its per-epoch patch cap while selector workers exist.
        selection_seed = int(self.cfg.get("seed", 1)) + int(fold)
        if not prompt_select:
            selection = choose_prompt_embedding(
                self._encoder.encode_text, self.prompt_bank.prompts,
                device=torch.device(self.device), seed=selection_seed)
        else:
            dataset = getattr(train_loader, "dataset", None)
            saved_patch_num = getattr(dataset, "patch_num", None)
            if dataset is not None and hasattr(dataset, "patch_num"):
                dataset.patch_num = None
            try:
                selection = select_prompt_embedding(
                    self._encoder.encode_text,
                    train_loader,
                    self.prompt_bank.prompts,
                    n_slide_classes=int(self.cfg["n_classes"]),
                    synthetic_normal=self.prompt_bank.synthetic_normal,
                    device=torch.device(self.device),
                    classifier_count=classifier_count,
                    select_count=select_count,
                    top_patches=int(self.cfg.get("prompt_top_patches", 100)),
                    classifier_batch_size=int(
                        self.cfg.get("prompt_classifier_batch_size", 16)),
                    text_batch_size=int(
                        self.cfg.get("prompt_text_batch_size", 128)),
                )
            finally:
                if dataset is not None and hasattr(dataset, "patch_num"):
                    dataset.patch_num = saved_patch_num
        selected = selection.embedding.detach().to(
            device=model.pathpt_selected_prompt_embedding.device,
            dtype=model.pathpt_selected_prompt_embedding.dtype)
        if selected.shape != model.pathpt_selected_prompt_embedding.shape:
            raise ValueError(
                "PathPT selected prompt embedding shape changed from "
                f"{tuple(model.pathpt_selected_prompt_embedding.shape)} to "
                f"{tuple(selected.shape)}")
        model.pathpt_selected_prompt_embedding.copy_(selected)
        self._write_prompt_trace(selection)

    def on_train_epoch_start(self, epoch: int, model: nn.Module) -> None:
        self._epoch = int(epoch)

    def on_checkpoint_loaded(
        self, model: nn.Module, checkpoint_kind: str, fold: int,
    ) -> None:
        if self.native_mode and not hasattr(
                model, "pathpt_selected_prompt_embedding"):
            raise RuntimeError(
                "native PathPT checkpoint is missing its selected prompt "
                "classifier")

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

    def _native_slide_logits(self, patch_scores: torch.Tensor) -> torch.Tensor:
        assert self.prompt_bank is not None
        probabilities = vote_slide_probabilities(
            patch_scores,
            n_classes=int(self.cfg["n_classes"]),
            synthetic_normal=self.prompt_bank.synthetic_normal)
        return probabilities.clamp_min(1e-12).log()

    def _native_eval_patch_scores(
        self, model: nn.Module, feats: torch.Tensor,
    ) -> torch.Tensor:
        """Match upstream's bounded 50k-patch inference chunks."""
        patch_count = feats.shape[1] if feats.ndim == 3 else feats.shape[0]
        chunk_size = int(self.cfg.get("eval_patch_batch_size", 50_000))
        if chunk_size <= 0:
            raise ValueError("PathPT eval_patch_batch_size must be positive")
        pieces = []
        for start in range(0, patch_count, chunk_size):
            block = (feats[:, start:start + chunk_size]
                     if feats.ndim == 3 else feats[start:start + chunk_size])
            pieces.append(extract_patch_scores(
                model(block), self.patch_class_count))
        return torch.cat(pieces, dim=0)

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
        decay_epochs = max(1, epochs - warm_up)

        def lr_lambda(cur):
            # LambdaLR evaluates lambda(0) during construction. Using ``cur``
            # directly therefore made the first training epoch run at LR=0.
            step = cur + 1
            if step <= warm_up:
                return step / warm_up
            progress = min(max((step - warm_up) / decay_epochs, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    def train_step(self, batch, model, optimizer, loss_fn):
        feats = batch[0].to(self.device)
        label = batch[-1].to(self.device)

        optimizer.zero_grad()
        out = model(feats)
        if self.native_mode:
            if self.prompt_bank is None or not hasattr(
                    model, "pathpt_selected_prompt_embedding"):
                raise RuntimeError(
                    "PathPT native prompt selection was not prepared")
            patch_scores = extract_patch_scores(out, self.patch_class_count)
            feature_rows = feats.squeeze(0) if feats.ndim == 3 else feats
            text = model.pathpt_selected_prompt_embedding.to(
                device=feature_rows.device, dtype=feature_rows.dtype)
            zero_shot_scores = torch.softmax(
                feature_rows.float() @ text.float().t(), dim=-1)
            patch_targets = generate_patch_targets(
                zero_shot_scores, label,
                synthetic_normal=self.prompt_bank.synthetic_normal,
                threshold=float(self.cfg.get("logits_thd", 0.0)))
            details = PatchSSLoss(
                patch_scores,
                patch_targets,
                epoch=self._epoch,
                total_epoch=int(self.cfg.get("epochs", 20)),
                weights=self.cfg.get("loss_weight", [1.0, 0.5, 0.1]),
                balance=bool(self.cfg.get("balance", True)),
                vision_only=False,
                pseudo_loss=bool(self.cfg.get("enable_pseudo", True)))
            loss = details["loss"]
            if not torch.isfinite(loss):
                raise ValueError("PathPT PatchSSLoss became non-finite")
            loss.backward()
            optimizer.step()
            logits = self._native_slide_logits(patch_scores.detach())
            return {
                "loss": loss.item(), "logits": logits, "label": label,
                "labeled_loss": float(details["labeled_loss"].detach()),
                "pseudo_loss": float(details["pseudo_loss"].detach()),
                "candidate_loss": float(details["candidate_loss"].detach()),
            }
        logits = self._slide_logits(out)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        feats = batch[0].to(self.device)
        label = batch[-1].to(self.device)
        if self.native_mode:
            patch_scores = self._native_eval_patch_scores(model, feats)
            logits = self._native_slide_logits(patch_scores)
            loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
            return {"loss": loss, "logits": logits, "label": label}
        out = model(feats)
        logits = self._slide_logits(out)
        loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": label}
