from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from common.datasets.dataset_generic import _load_feature_tensor
from common.utils.core_utils import Accuracy_Logger
from methods.focus.adapter import _set_trainable_scope
from train import classification_metrics


def test_dual_scale_loader_reads_native_hdf5_feature_bag(tmp_path: Path):
    path = tmp_path / "slide.h5"
    expected = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=expected.numpy())

    actual = _load_feature_tensor(path)

    assert torch.equal(actual, expected)


def test_accuracy_logger_reports_micro_accuracy():
    logger = Accuracy_Logger(n_classes=3)
    logger.log_batch([0, 0, 2, 1], [0, 1, 2, 1])

    assert logger.get_overall_summary() == (0.75, 3, 4)


def test_soft_context_scope_freezes_every_other_parameter():
    class PromptLearner(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx = nn.Parameter(torch.ones(2, 3))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_learner = PromptLearner()
            self.classifier = nn.Linear(3, 2)

    model = Model()
    _set_trainable_scope(model, "soft_context")

    trainable = [name for name, value in model.named_parameters()
                 if value.requires_grad]
    assert trainable == ["prompt_learner.ctx"]


def test_unified_classification_metrics_include_calibration_and_macro_scores():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    labels = np.array([0, 1, 0, 1])

    metrics = classification_metrics(probabilities, labels, n_classes=2)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["auroc_ovr"] == 1.0
    assert metrics["nll"] > 0.0
    assert metrics["ece"] > 0.0
