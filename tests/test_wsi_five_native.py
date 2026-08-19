from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn

from methods.wsi_five.adapter import WSIFiVEMethod
from methods.wsi_five.dataset import WSI_FiVE_Dataset
from methods.wsi_five import dataset as wsi_five_dataset
from methods.wsi_five.prompts import (
    augment_answer_bank,
    load_evaluation_prompts,
)


def _answers(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix} differentiation", "Unknown", f"{prefix} vascular",
        f"{prefix} pleural", f"{prefix} adjacent", f"{prefix} margins",
    )


def _write_eval_prompts(path):
    path.write_text(json.dumps({
        "_provenance": "upstream",
        "prompts": {"LUAD": "upstream LUAD", "LUSC": "upstream LUSC"},
    }))


def _native_cfg(eval_path, **updates):
    cfg = {
        "method": "wsi_five",
        "backbone": "wsi-five-vit",
        "feature_dim": 512,
        "n_classes": 2,
        "classnames": ["lung adenocarcinoma", "lung squamous carcinoma"],
        "label_dict": {"LUAD": 0, "LUSC": 1},
        "training_mode": "upstream_answer_bank",
        "evaluation_prompt_path": str(eval_path),
    }
    cfg.update(updates)
    return cfg


def test_answer_augmentation_drops_questions_unknowns_and_hashes_duplicates():
    first = _answers("a")
    second = _answers("b")
    duplicate = tuple(first)

    result = augment_answer_bank(
        (first, second, duplicate),
        drop_question_indices=(2, 4),
        rng=random.Random(7),
    )

    assert result.kept_question_indices == (0, 1, 3, 5)
    assert result.original_to_augmented == (0, 1, 0)
    assert len(result.texts) == 2
    assert all("Unknown" not in text for text in result.texts)
    assert all("vascular" not in text and "adjacent" not in text
               for text in result.texts)


def test_evaluation_prompts_follow_label_index_order(tmp_path):
    path = tmp_path / "eval.json"
    _write_eval_prompts(path)

    prompts = load_evaluation_prompts(path, {"LUSC": 1, "LUAD": 0})

    assert prompts == ("upstream LUAD", "upstream LUSC")


def test_dataset_builds_native_bank_from_training_rows_only(tmp_path):
    report_path = tmp_path / "answers.csv"
    rows = []
    for case_id, prefix in (("case-a", "a"), ("case-b", "b")):
        fields = _answers(prefix)
        rows.append({
            "case_id": case_id, "answer": "; ".join(fields),
            **{f"q{index}": value
               for index, value in enumerate(fields, start=1)},
        })
    pd.DataFrame(rows).to_csv(report_path, index=False)
    split = pd.DataFrame([
        {"slide_id": "slide-a", "case_id": "case-a", "label": "LUAD"},
        {"slide_id": "slide-b", "case_id": "case-b", "label": "LUSC"},
        {"slide_id": "slide-a2", "case_id": "case-a", "label": "LUAD"},
    ])
    dataset = WSI_FiVE_Dataset(
        split, tmp_path, report_path, {"LUAD": 0, "LUSC": 1},
        supervision_mode="upstream_answer_bank")

    assert dataset.native_answer_bank() == (_answers("a"), _answers("b"))


def test_native_validation_loader_does_not_load_answer_asset(
        tmp_path, monkeypatch):
    split = pd.DataFrame([
        {"slide_id": "slide-a", "case_id": "case-a", "label": "LUAD"},
    ])
    monkeypatch.setattr(
        "common.datasets.split_tables.load_phase_table",
        lambda _cfg, _phase, _fold: split)
    seen_report_paths = []
    original_init = WSI_FiVE_Dataset.__init__

    def capture_init(self, *args, **kwargs):
        seen_report_paths.append(kwargs.get("report_csv"))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(WSI_FiVE_Dataset, "__init__", capture_init)
    cfg = {
        "training_mode": "upstream_answer_bank",
        "report_csv": str(tmp_path / "held-out-answers.csv"),
        "label_dict": {"LUAD": 0},
        "num_workers": 0,
    }

    loader = wsi_five_dataset.build_wsi_five_loader(
        cfg, split="val", shuffle=False, fold=0)

    assert loader.dataset.reports == {}
    assert loader.dataset.answer_fields == {}
    assert seen_report_paths == [None]


class _DummyFiVE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector = nn.Parameter(torch.tensor([[0.25, -0.5]]))
        self.prompt_list = [f"q{index}" for index in range(6)]
        self.last_questions = None
        self.last_texts = None

    def encode_slide(self, _features, _patch_info, question_indices=None):
        self.last_questions = question_indices
        return self.vector

    def compare(self, slide, texts, **_kwargs):
        self.last_texts = tuple(texts)
        weights = torch.stack([
            torch.tensor([float(index + 1), float(1 - index)])
            for index in range(len(texts))
        ]).to(slide)
        return slide @ weights.t()

    def forward(self, features, patch_info, comparison_texts):
        return self.compare(
            self.encode_slide(features, patch_info), comparison_texts)


def _batch(supervision):
    return (
        torch.ones((1, 3, 512)), [supervision],
        {
            "patch_inds": torch.arange(3).reshape(1, 3),
            "patch_pub_cnt": torch.tensor([3.0]),
            "sample_range": [3],
        },
        torch.tensor([0]),
    )


def test_native_adapter_uses_answer_loss_but_returns_class_logits(
        tmp_path, monkeypatch):
    eval_path = tmp_path / "eval.json"
    _write_eval_prompts(eval_path)
    method = WSIFiVEMethod(_native_cfg(eval_path), device="cpu")
    bank = (_answers("a"), _answers("b"))
    dataset = SimpleNamespace(native_answer_bank=lambda: bank)
    model = _DummyFiVE()
    method.prepare_fold(0, model, SimpleNamespace(dataset=dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(random, "randint", lambda _low, _high: 0)

    output = method.train_step(
        _batch(_answers("a")), model, optimizer, nn.CrossEntropyLoss())

    assert output["loss"] > 0
    assert output["logits"].shape == (1, 2)
    assert model.last_questions == (0, 1, 2, 3, 4, 5)
    # The final comparison is for metric logits, never the per-slide answers.
    assert model.last_texts == ("upstream LUAD", "upstream LUSC")


def test_native_eval_ignores_per_slide_answers(tmp_path):
    eval_path = tmp_path / "eval.json"
    _write_eval_prompts(eval_path)
    method = WSIFiVEMethod(_native_cfg(eval_path), device="cpu")
    model = _DummyFiVE()

    output = method.eval_step(_batch(_answers("a")), model)

    assert output["logits"].shape == (1, 2)
    assert model.last_questions is None
    assert model.last_texts == ("upstream LUAD", "upstream LUSC")


def test_wsi_five_native_provenance_describes_restored_text_objective(tmp_path):
    from common.method_provenance import method_provenance

    path = tmp_path / "eval.json"
    _write_eval_prompts(path)
    provenance = method_provenance("wsi_five", _native_cfg(path))

    assert provenance.implementation.endswith("native_text_objective")
    assert provenance.upstream_fidelity == "partial"
    assert "label hashing" in provenance.note
