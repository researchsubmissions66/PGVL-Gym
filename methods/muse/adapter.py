"""Adapter and text-prior loading for MUSE."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn

from methods.base import BaseMethod
from common.backbones import (
    BackboneCapability as Cap, BackboneCompatibilityError, FeatureLevel,
    MethodBackboneContract, SwapPolicy, get_spec)


def _csv_descriptions(path: str | Path) -> list[str]:
    """Read MUSE's index-plus-description CSVs, including their blank header."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    descriptions = []
    for row in rows:
        # The MUSE release uses `,0` as the header and keeps the narrative
        # in the longest non-empty cell of each subsequent row.
        values = [str(value).strip() for value in row if str(value).strip()]
        if not values or values == ["0"]:
            continue
        text = max(values, key=len)
        if len(text) > 8:
            descriptions.append(text)
    if not descriptions:
        raise ValueError(f"No descriptions found in MUSE prompt CSV: {path}")
    return descriptions


class MUSEMethod(BaseMethod):
    """Adapt MUSE with independent offline patch and runtime prompt encoders."""
    name = "muse"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_BAG,
        swap_policy=SwapPolicy.CAPABILITY, default_backbone="conch",
        required_capabilities=frozenset({Cap.TEXT_ENCODE}),
        feature_dim_key=None, feature_space_key="prompt_feature_space_id",
        require_feature_space=True,
        rationale=(
            "MUSE independently registers its offline patch encoder and "
            "runtime prompt encoder. The learned visual adapter maps any "
            "declared patch width into the prompt encoder's shared space."))

    def _load_prompt_bank(self) -> torch.Tensor:
        cached = getattr(self, "_prompt_bank", None)
        if cached is not None:
            return cached
        path = self.cfg.get("prompt_features")
        prompt_spec = get_spec(self.backbone_name)
        embedded_space = None
        if path:
            # A saved prompt bank avoids allocating the text model, but it is
            # still an artifact of the selected encoder's shared space.
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict):
                embedded_space = payload.get("feature_space_id")
                if "embeddings" in payload:
                    bank = torch.as_tensor(payload["embeddings"])
                else:
                    rows = [
                        torch.as_tensor(payload[name])
                        for name in self.cfg["classnames"]
                    ]
                    bank = torch.stack(rows)
            else:
                bank = torch.as_tensor(payload)
        else:
            prompt_csvs = self.cfg.get("prompt_csvs")
            if not prompt_csvs:
                raise KeyError(
                    "MUSE requires either prompt_features or prompt_csvs. Run "
                    "scripts/import_upstream_assets.py --download to obtain the official CSVs.")
            if not isinstance(prompt_csvs, dict):
                raise TypeError("prompt_csvs must map each classname to its published MUSE CSV")
            all_descriptions = [_csv_descriptions(prompt_csvs[name])
                                for name in self.cfg["classnames"]]
            weights_path = self.cfg.get("backbone_weights")
            encoder = self.load_encoder(weights_path=weights_path).freeze()
            vectors = []
            text_batch_size = self.cfg.get("text_batch_size", 64)
            with torch.no_grad():
                for descriptions in all_descriptions:
                    encoded_batches = []
                    for start in range(0, len(descriptions), text_batch_size):
                        encoded_batches.append(encoder.encode_text(
                            descriptions[start:start + text_batch_size],
                            normalize=True).cpu())
                    vectors.append(torch.cat(encoded_batches, dim=0))
            if (encoder.spec.shared_dim is not None and
                    encoder.spec.shared_dim != vectors[0].shape[-1]):
                raise RuntimeError(
                    f"{encoder.spec.name} returned text width {vectors[0].shape[-1]}, "
                    f"expected {encoder.spec.shared_dim}")
            # Each class may have a different number of prompts in custom
            # experiments. MUSE's retrieval bank is rectangular, so pad by
            # repeating its last valid description rather than adding zeros.
            width = max(item.shape[0] for item in vectors)
            bank = torch.stack([torch.cat([item, item[-1:].expand(width - item.shape[0], -1)])
                                if item.shape[0] < width else item for item in vectors])
        if (bank.ndim != 3 or bank.shape[0] != self.cfg["n_classes"]
                or bank.shape[1] == 0):
            raise ValueError("MUSE prompt features must be [n_classes, descriptions, embedding_dim]")
        if not torch.isfinite(bank).all():
            raise ValueError("MUSE prompt features contain NaN or infinity")
        if (prompt_spec.shared_dim is not None and
                bank.shape[-1] != prompt_spec.shared_dim):
            raise BackboneCompatibilityError(
                f"MUSE prompt bank width {bank.shape[-1]} does not match "
                f"backbone '{prompt_spec.name}' shared width {prompt_spec.shared_dim}.")
        prompt_space = self.cfg.get("prompt_feature_space_id")
        if prompt_space != prompt_spec.feature_space_id:
            raise BackboneCompatibilityError(
                f"MUSE prompt bank feature space '{prompt_space}' does not "
                f"match backbone '{prompt_spec.name}' "
                f"({prompt_spec.feature_space_id}).")
        if embedded_space is not None and embedded_space != prompt_space:
            raise BackboneCompatibilityError(
                f"MUSE prompt artifact space '{embedded_space}' does not "
                f"match configured prompt space '{prompt_space}'.")
        self._prompt_bank = bank.float()
        return self._prompt_bank

    def build_model(self) -> nn.Module:
        from .model import MUSEModel
        prompt_bank = self._load_prompt_bank()
        return MUSEModel(
            input_dim=self.cfg.get("feature_dim", 512), n_classes=self.cfg["n_classes"],
            prompt_bank=prompt_bank, embed_dim=self.cfg.get("embed_dim", prompt_bank.shape[-1]),
            num_heads=self.cfg.get("num_heads", 8), num_experts=self.cfg.get("num_experts", 8),
            num_selected=self.cfg.get("num_selected", 2), retrieval_k=self.cfg.get("retrieval_k", 8),
            dropout=self.cfg.get("dropout", 0.25)).to(self.device)

    def train_step(self, batch, model, optimizer, loss_fn):
        features, labels = batch[0].to(self.device), batch[-1].to(self.device)
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(), "label": labels}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        features, labels = batch[0].to(self.device), batch[-1].to(self.device)
        logits = model(features)
        loss = loss_fn(logits, labels).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": labels}
