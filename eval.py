"""Unified evaluation entry point.

Loads a previously-trained checkpoint and runs only the test phase.

Usage
-----
    python eval.py --method focus --config configs/focus_ubc.yaml \\
                   --ckpt_dir results/focus
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from methods import get_method
from common.utils.utils import get_split_loader
from common.utils.core_utils import Accuracy_Logger
from common.datasets.dataset_generic import Generic_MIL_Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    method = get_method(args.method)(cfg, device=args.device)
    dataset = Generic_MIL_Dataset(
        csv_path=cfg["dataset_csv"],
        data_dir_s=cfg.get("data_folder_s"),
        data_dir_l=cfg.get("data_folder_l"),
        feature_path_column_s=cfg.get("feature_path_column_s"),
        feature_path_column_l=cfg.get("feature_path_column_l"),
        feature_key=cfg.get("feature_key", "features"),
        mode=cfg.get("loader_mode", "transformer"),
        shuffle=False,
        seed=cfg.get("seed", 1),
        print_info=True,
        label_dict=cfg["label_dict"],
        patient_strat=False, ignore=[],
        label_col=cfg.get("label_column"))

    fold_accs = []
    for fold in range(cfg.get("k", 5)):
        ckpt = Path(args.ckpt_dir) / f"fold{fold}_best.pt"
        if not ckpt.exists():
            print(f"Skipping fold {fold} -- no checkpoint at {ckpt}")
            continue

        _, _, test_split = dataset.return_splits(
            from_id=False,
            csv_path=os.path.join(cfg["split_dir"], f"splits_{fold}.csv"))
        loader = get_split_loader(test_split, mode=cfg.get("loader_mode", "transformer"))

        model = method.build_model()
        model.load_state_dict(
            torch.load(ckpt, map_location=args.device, weights_only=True))
        model.eval()

        logger = Accuracy_Logger(n_classes=cfg["n_classes"])
        for batch in loader:
            out = method.eval_step(batch, model)
            preds = out["logits"].argmax(dim=1)
            logger.log_batch(preds.cpu().numpy(), out["label"].cpu().numpy())
        acc = logger.get_overall_summary()[0]
        assert acc is not None
        print(f"  fold {fold}: test acc = {acc:.4f}")
        fold_accs.append(acc)

    if fold_accs:
        print(f"\nMean test acc: {np.mean(fold_accs):.4f} "
              f"+- {np.std(fold_accs):.4f}")


if __name__ == "__main__":
    main()
