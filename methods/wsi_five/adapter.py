"""WSI-FiVE adapter (CVPR 2024).

FiVE consumes a *precomputed* patch bag, not raw tiles: with the release's
shipped `IS_IMG_PTH: True` its vision tower is `nn.Identity()` and features are
read from disk. What the method owns is the text side -- a LoRA-adapted
BioClinicalBERT tower whose encoded clinical questions condition patch
aggregation through cross-attention.

The contract therefore declares a patch bag with a fixed native width, not an
encoder-owning architecture. See `docs/design-decisions.md` for the evidence and
for the deviations that remain.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import FeatureLevel, MethodBackboneContract, SwapPolicy
from common.prompts import (
    WSI_FIVE_PROMPT_FORMAT,
    load_wsi_five_answer_bank,
    load_wsi_five_evaluation_bank,
    load_wsi_five_question_bank,
)
from .prompts import (
    ANSWER_FIELD_COUNT,
    augment_answer_bank,
    normalize_answer_fields,
)


class WSIFiVEMethod(BaseMethod):
    """Adapt WSI-FiVE patch bags, questions, and answer-bank supervision."""

    name = "wsi_five"
    NATIVE_MODE = "upstream_answer_bank"
    SIMPLIFIED_MODE = "simplified_classnames"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_BAG,
        swap_policy=SwapPolicy.PRECOMPUTED, default_backbone="wsi-five-vit",
        supported_backbones=("wsi-five-vit",), feature_dims={"wsi-five-vit": (512,)},
        rationale=(
            "FiVE reads a precomputed 512-wide patch bag; its own tower is the "
            "LoRA-adapted BioClinicalBERT text encoder, not a vision encoder."))

    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        self.training_mode = str(self.cfg.get(
            "training_mode", self.SIMPLIFIED_MODE)).strip().lower()
        if self.training_mode not in {
                self.NATIVE_MODE, self.SIMPLIFIED_MODE}:
            raise ValueError(
                "WSI-FiVE training_mode must be 'upstream_answer_bank' or "
                "'simplified_classnames'")
        if self.cfg.get("wsi_prompt_format") != WSI_FIVE_PROMPT_FORMAT:
            raise ValueError(
                "WSI-FiVE requires wsi_prompt_format=" +
                WSI_FIVE_PROMPT_FORMAT)
        for key in (
            "clinical_questions", "wsi_question_file_sha256",
            "wsi_question_bank_sha256", "wsi_question_provenance",
            "prompt_provenance", "prompt_source",
        ):
            if not self.cfg.get(key):
                raise ValueError(f"WSI-FiVE requires {key}")
        self.question_bank = load_wsi_five_question_bank(
            self.cfg["clinical_questions"],
            expected_file_sha256=self.cfg["wsi_question_file_sha256"],
            expected_prompt_bank_sha256=self.cfg["wsi_question_bank_sha256"],
            expected_provenance=self.cfg["wsi_question_provenance"],
        )
        if self.native_mode:
            for key in (
                "report_csv", "wsi_answer_file_sha256",
                "wsi_answer_bank_sha256", "wsi_answer_provenance",
                "evaluation_prompt_path", "wsi_evaluation_file_sha256",
                "wsi_evaluation_bank_sha256", "wsi_evaluation_provenance",
            ):
                if not self.cfg.get(key):
                    raise ValueError(
                        f"WSI-FiVE upstream_answer_bank requires {key}")
            self.answer_bank = load_wsi_five_answer_bank(
                self.cfg["report_csv"],
                expected_file_sha256=self.cfg["wsi_answer_file_sha256"],
                expected_answer_bank_sha256=self.cfg["wsi_answer_bank_sha256"],
                expected_provenance=self.cfg["wsi_answer_provenance"],
            )
            evaluation = load_wsi_five_evaluation_bank(
                self.cfg["evaluation_prompt_path"], self.cfg["label_dict"],
                expected_file_sha256=self.cfg["wsi_evaluation_file_sha256"],
                expected_prompt_bank_sha256=(
                    self.cfg["wsi_evaluation_bank_sha256"]),
                expected_provenance=self.cfg["wsi_evaluation_provenance"],
            )
            self.evaluation_prompts = evaluation.prompts
            expected_prompt_provenance = (
                f"{self.question_bank.provenance}_questions_with_"
                f"{self.answer_bank.provenance}_answer_and_"
                f"{evaluation.provenance}_evaluation_banks")
            expected_prompt_source = "wsi_five_derived_upstream_text_assets"
        else:
            self.answer_bank = None
            self.evaluation_prompts = tuple(self.cfg.get("classnames", ()))
            expected_prompt_provenance = (
                f"{self.question_bank.provenance}_questions_with_"
                "classname_comparison")
            expected_prompt_source = "wsi_five_simplified_classname_baseline"
        if self.cfg["prompt_provenance"] != expected_prompt_provenance:
            raise ValueError(
                "WSI-FiVE prompt_provenance does not match active text roles")
        if self.cfg["prompt_source"] != expected_prompt_source:
            raise ValueError(
                "WSI-FiVE prompt_source does not match active text roles")
        if len(self.evaluation_prompts) != int(self.cfg["n_classes"]):
            raise ValueError(
                "WSI-FiVE evaluation prompt count must match n_classes")
        self._train_answer_bank: tuple[tuple[str, ...], ...] | None = None
        self._answer_to_index: dict[tuple[str, ...], int] = {}
        self._fold = 0

    @property
    def native_mode(self) -> bool:
        return self.training_mode == self.NATIVE_MODE

    def build_model(self) -> nn.Module:
        from .model import WSIFiVEModel

        model = WSIFiVEModel(
            self.cfg["classnames"],
            self.cfg["clinicalbert_weights"],
            feature_dim=self.cfg.get("feature_dim", 512),
            num_frames=self.cfg.get("num_frames", 2048),
            context_length=self.cfg.get("prompt_context_length", 308),
            learnable_prompts=self.cfg.get("learnable_prompts", 16),
            lora_targets=self.cfg.get("lora_targets", "query,key,value,dense"),
            logit_scale=self.cfg.get("logit_scale", 300.0),
            prompt_list=list(self.question_bank.questions),
        )
        return model.to(self.device)

    @staticmethod
    def _unpack(batch):
        """Return bag, text supervision, patch metadata, and slide label."""
        features, supervision, patch_info = batch[0], batch[1], batch[2]
        return features, supervision, patch_info, batch[-1]

    def _to_device(self, patch_info: dict) -> dict:
        return {
            "patch_inds": patch_info["patch_inds"].to(self.device),
            "patch_pub_cnt": patch_info["patch_pub_cnt"].to(self.device),
            "sample_range": patch_info["sample_range"],
        }

    def _write_answer_bank_trace(self) -> None:
        if self._train_answer_bank is None or not self.cfg.get("results_dir"):
            return
        output = Path(self.cfg["results_dir"])
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"fold{self._fold}_wsi_five_answer_bank.json"
        serialized = json.dumps(
            self._train_answer_bank, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")
        payload = {
            "schema_version": 1,
            "fold": self._fold,
            "training_mode": self.training_mode,
            "training_candidate_count": len(self._train_answer_bank),
            "training_bank_sha256": hashlib.sha256(serialized).hexdigest(),
            "answer_source": self.cfg.get("report_csv"),
            "question_source": self.cfg.get("clinical_questions"),
            "evaluation_prompt_source": self.cfg.get(
                "evaluation_prompt_path"),
            "bank_scope": "training_fold_only",
            "upstream_repository": "https://github.com/ls1rius/WSI_FiVE",
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
        """Build the answer candidate bank from this training fold only."""
        if not self.native_mode:
            return
        if len(model.prompt_list) != ANSWER_FIELD_COUNT:
            raise ValueError(
                "WSI-FiVE native mode requires the six upstream questions")
        dataset = getattr(train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "native_answer_bank"):
            raise TypeError(
                "WSI-FiVE native mode requires its structured-answer dataset")
        self._fold = int(fold)
        self._train_answer_bank = dataset.native_answer_bank()
        self._answer_to_index = {
            fields: index for index, fields in enumerate(self._train_answer_bank)}
        self._write_answer_bank_trace()

    def train_step(self, batch, model, optimizer, loss_fn):
        features, supervision, patch_info, label = self._unpack(batch)
        features, label = features.to(self.device), label.to(self.device)
        optimizer.zero_grad(set_to_none=True)
        patch_info = self._to_device(patch_info)
        if self.native_mode:
            if self._train_answer_bank is None:
                raise RuntimeError(
                    "WSI-FiVE native answer bank was not prepared for this fold")
            if len(supervision) != 1:
                raise ValueError("WSI-FiVE native training requires batch_size=1")
            sample_fields = normalize_answer_fields(supervision[0])
            original_target = self._answer_to_index.get(sample_fields)
            if original_target is None:
                raise ValueError(
                    "WSI-FiVE sample answer is absent from the training-fold bank")
            drop_count = random.randint(0, ANSWER_FIELD_COUNT - 1)
            dropped = random.sample(range(ANSWER_FIELD_COUNT), drop_count)
            augmented = augment_answer_bank(
                self._train_answer_bank,
                drop_question_indices=dropped,
                rng=random,
            )
            slide = model.encode_slide(
                features, patch_info, augmented.kept_question_indices)
            answer_logits = model.compare(
                slide,
                augmented.texts,
                partial_text_gradients=True,
                text_gradient_sample_size=int(
                    self.cfg.get("text_gradient_sample_size", 96)),
            )
            answer_target = torch.tensor(
                [augmented.original_to_augmented[original_target]],
                dtype=torch.long, device=label.device)
            loss = loss_fn(answer_logits, answer_target)
            # The unified trainer needs task-class logits for its metrics. They
            # do not contribute to the native answer-bank objective.
            with torch.no_grad():
                logits = model.compare(
                    slide.detach(), self.evaluation_prompts)
        else:
            logits = model(
                features, patch_info, self.evaluation_prompts)
            loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        features, _supervision, patch_info, label = self._unpack(batch)
        features, label = features.to(self.device), label.to(self.device)
        logits = model(
            features, self._to_device(patch_info), self.evaluation_prompts)
        loss = loss_fn(logits, label).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": label}
