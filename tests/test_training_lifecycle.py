from __future__ import annotations

from pathlib import Path
import argparse
import json

import torch

import train
from methods.base import BaseMethod


class _Writer:
    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class _ToyMethod(BaseMethod):
    name = "toy"

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        self.loaded = []

    def build_model(self):
        return torch.nn.Linear(2, 2).to(self.device)

    def build_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.01)

    def build_scheduler(self, optimizer):
        return None

    def train_step(self, batch, model, optimizer, loss_fn):
        features, labels = batch
        features, labels = features.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        return {"loss": loss, "logits": logits.detach(), "label": labels}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        features, labels = batch
        features, labels = features.to(self.device), labels.to(self.device)
        logits = model(features)
        loss = loss_fn(logits, labels) if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": labels}

    def on_checkpoint_loaded(self, model, checkpoint_kind, fold):
        self.loaded.append((checkpoint_kind, fold))


def test_adapter_receives_a_private_nested_config_copy():
    cfg = {"nested": {"value": 1}}
    method = _ToyMethod(cfg)

    method.cfg["nested"]["value"] = 2
    method.cfg["derived"] = True

    assert cfg == {"nested": {"value": 1}}


def test_best_checkpoint_reload_notifies_adapter(tmp_path: Path, monkeypatch):
    batch = (torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
             torch.tensor([0, 1]))
    monkeypatch.setattr(
        train, "build_loaders", lambda *_args, **_kwargs: ([batch], [batch], [batch]))
    cfg = {
        "seed": 1,
        "n_classes": 2,
        "epochs": 1,
        "early_stopping": True,
        "es_patience": 2,
        "es_stop_epoch": 0,
        "evaluate_test": True,
        "results_dir": str(tmp_path),
    }
    method = _ToyMethod(cfg)

    result = train.train_one_fold(0, cfg, method, _Writer())

    assert result["test_acc"] is not None
    assert method.loaded == [("best", 0)]


def test_main_constructs_a_fresh_adapter_for_each_fold(
    tmp_path: Path, monkeypatch,
):
    instances = []

    class Method:
        def __init__(self, cfg, device="cuda"):
            self.cfg = cfg
            self.device = device
            instances.append(self)

        def on_fold_end(self, fold, metrics):
            pass

    class Report:
        ok = True
        warnings = []
        problems = []

    cfg = {
        "method": "focus",
        "results_dir": str(tmp_path / "results"),
        "seed": 7,
        "k_start": 0,
        "k_end": 2,
        "evaluate_test": True,
    }
    monkeypatch.setattr(train, "parse_args", lambda: type("Args", (), {
        "method": "focus", "config": "unused.yaml", "device": "cpu",
        "seed": None, "rerun": False,
    })())
    monkeypatch.setattr(train, "load_yaml_config", lambda _path: dict(cfg))
    monkeypatch.setattr(train, "preflight", lambda _cfg: Report())
    monkeypatch.setattr(train, "get_method", lambda _name: Method)
    monkeypatch.setattr(train, "SummaryWriter", lambda **_kwargs: _Writer())
    monkeypatch.setattr(
        train, "train_one_fold",
        lambda fold, _cfg, method, _writer: {
            "test_acc": 0.5,
            "best_val_loss": 1.0,
            "adapter_id": id(method),
        },
    )

    assert train.main() == 0
    assert len(instances) == 2
    assert instances[0] is not instances[1]


def test_completed_documentation_upgrade_persists_new_identity(
    tmp_path: Path, monkeypatch,
):
    results = tmp_path / "results"
    results.mkdir()
    original = {
        "method": "focus", "results_dir": str(results),
        "k_start": 0, "k_end": 1, "seed": 1,
    }
    upgraded = {
        **original,
        "implementation_provenance": "vendored",
        "upstream_fidelity": "upstream",
        "fidelity_note": "documentation only",
    }
    (results / "config.json").write_text(json.dumps(original))
    train._write_metrics(results / "metrics.json", "focus", original, [{
        "fold": 0, "best_val_loss": 1.0, "test_acc": 0.5,
    }])
    monkeypatch.setattr(train, "get_method", lambda _name: object)
    monkeypatch.setattr(train, "SummaryWriter", lambda **_kwargs: _Writer())

    result = train._run_ready_experiment(
        argparse.Namespace(rerun=False, device="cpu"),
        upgraded, "focus", results)

    assert result == 0
    state = json.loads((results / "metrics.json").read_text())
    assert state["run_identity"] == train._run_identity("focus", upgraded)
    assert json.loads((results / "config.json").read_text()) == upgraded
