"""Adapter for the official ConVLM patch-level zero-shot protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import BaseMethod
from common.backbones import FeatureLevel, MethodBackboneContract, SwapPolicy


class ConVLMMethod(BaseMethod):
    """Adapt ConVLM's patch-bag, attribute-conditioned zero-shot protocol."""
    name = "convlm"
    backbone_contract = MethodBackboneContract(
        method=name, feature_level=FeatureLevel.PATCH_BAG,
        swap_policy=SwapPolicy.PRECOMPUTED, default_backbone="convlm-vit",
        supported_backbones=("convlm-vit",),
        rationale=(
            "ConVLM trains on precomputed patch features -- upstream extracts "
            "them with UNI and exposes the embedding width as a config value. "
            "Attribute injection and token pruning are the method's own; the "
            "visual encoder is not."))

    def _attributes(self) -> torch.Tensor:
        if hasattr(self, "_attribute_bank"):
            return self._attribute_bank
        path = self.cfg.get("attribute_embeddings")
        if path:
            source = Path(path)
            if source.suffix == ".npy":
                attributes = torch.from_numpy(np.load(source))
            else:
                attributes = torch.load(
                    source, map_location="cpu", weights_only=True)
            if isinstance(attributes, dict):
                attributes = attributes.get(
                    "embeddings", attributes.get("attributes"))
        else:
            prompt_path = self.cfg.get("attribute_prompt_path")
            encoder_cfg = self.cfg.get("attribute_encoder")
            if not prompt_path or not isinstance(encoder_cfg, dict):
                raise KeyError(
                    "ConVLM requires attribute_embeddings or an "
                    "attribute_prompt_path plus attribute_encoder.")
            with Path(prompt_path).open(encoding="utf-8") as handle:
                prompt_bank = json.load(handle)
            if list(prompt_bank) != list(self.cfg["classnames"]):
                raise ValueError(
                    "ConVLM attribute prompt order must match classnames")

            import open_clip

            text_model = open_clip.create_model(
                encoder_cfg["model_name"],
                pretrained=str(encoder_cfg["weights"]),
                device=self.device,
            ).eval()
            text_model.requires_grad_(False)
            tokenizer = open_clip.get_tokenizer(encoder_cfg["model_name"])
            rows = []
            with torch.inference_mode():
                for classname in self.cfg["classnames"]:
                    tokens = tokenizer(prompt_bank[classname]).to(self.device)
                    encoded = F.normalize(
                        text_model.encode_text(tokens).float(), dim=-1)
                    rows.append(F.normalize(encoded.mean(dim=0), dim=0))
            attributes = torch.stack(rows).cpu()
        if not isinstance(attributes, torch.Tensor) or attributes.ndim != 2:
            raise ValueError("attribute_embeddings must be a rank-2 tensor or contain 'embeddings'")
        if attributes.shape[0] != self.cfg["n_classes"]:
            raise ValueError("attribute_embeddings row count must equal n_classes")
        attributes = attributes.float()
        if not torch.isfinite(attributes).all():
            raise ValueError("attribute_embeddings contain NaN or infinity")
        if (attributes.norm(dim=-1) == 0).any():
            raise ValueError("attribute_embeddings contain an all-zero class row")
        self._attribute_bank = F.normalize(attributes.to(self.device), dim=-1)
        return self._attribute_bank

    def _seen_indices(self, device: torch.device) -> torch.Tensor:
        values = self.cfg.get("seen_class_indices", list(range(self.cfg["n_classes"])))
        indices = torch.as_tensor(values, device=device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError("seen_class_indices must be a one-dimensional list")
        if (indices.numel() == 0 or indices.min() < 0
                or indices.max() >= self.cfg["n_classes"]):
            raise ValueError("seen_class_indices must be non-empty valid global class indices")
        if indices.unique().numel() != indices.numel():
            raise ValueError("seen_class_indices must not contain duplicates")
        return indices

    def build_model(self) -> nn.Module:
        from .model import AttributeConVLM
        attributes = self._attributes()
        model = AttributeConVLM(
            attr_dim=attributes.shape[1],
            feature_dim=self.cfg.get("feature_dim", 1024),
            max_patches=self.cfg.get("max_patches", 4096),
            width=self.cfg.get("embed_dim", 768),
            depth=self.cfg.get("depth", 12), heads=self.cfg.get("num_heads", 12),
            drop=self.cfg.get("drop", 0.0), keep_rate=self.cfg.get("keep_rate", 0.7),
        )
        return model.to(self.device)

    def build_optimizer(self, model):
        return torch.optim.Adam(
            model.parameters(), lr=self.cfg.get("lr", 1e-4),
            weight_decay=self.cfg.get("weight_decay", 1e-5))

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.get("lr_milestones", [10, 20, 30]), gamma=0.5)

    @staticmethod
    def _global_logits(embedding: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        return embedding @ attributes.t()

    @staticmethod
    def _flatten_tiles(features: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Normalise a feature bag to ``[batch, patches, dim]``.

        The bag is already the unit ConVLM encodes, so unlike the raw-tile form
        there is nothing to flatten -- each slide is one sequence.
        """
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(
                "ConVLM expects [batch, patches, feature_dim], got "
                f"{list(features.shape)}")
        return features, features.shape[0], 1

    @staticmethod
    def _slide_embeddings(tile_embeddings: torch.Tensor, batch_size: int,
                          tile_count: int) -> torch.Tensor:
        embeddings = tile_embeddings.view(batch_size, tile_count, -1).mean(dim=1)
        return F.normalize(embeddings, dim=-1)

    def train_step(self, batch, model, optimizer, loss_fn):
        images, labels = batch[0].to(self.device), batch[-1].to(self.device)
        images, batch_size, tile_count = self._flatten_tiles(images)
        attributes = self._attributes()
        seen = self._seen_indices(labels.device)
        to_local = torch.full((self.cfg["n_classes"],), -1, device=labels.device, dtype=torch.long)
        to_local[seen] = torch.arange(seen.numel(), device=labels.device)
        local_labels = to_local[labels]
        if (local_labels < 0).any():
            raise ValueError("ConVLM train splits may only contain seen_class_indices")

        optimizer.zero_grad()
        tile_conditions = attributes[seen][local_labels].repeat_interleave(
            tile_count, dim=0)
        output = model(images, tile_conditions)
        slide_embeddings = self._slide_embeddings(
            output["embedding"], batch_size, tile_count)
        seen_logits = self._global_logits(slide_embeddings, attributes[seen])
        loss = loss_fn(seen_logits, local_labels)
        if output["aux_vis"]:
            alignment = torch.stack([F.l1_loss(item, output["global"]) for item in output["aux_vis"]]).mean()
            loss = loss + self.cfg.get("loss_global_alignment", 1.0) * alignment
        if output["aux_attr"]:
            reconstruction = torch.stack([
                F.l1_loss(item, tile_conditions) for item in output["aux_attr"]
            ]).mean()
            loss = loss + self.cfg.get("loss_sr", 1.0) * reconstruction
        loss.backward()
        optimizer.step()

        logits = torch.full((labels.shape[0], self.cfg["n_classes"]), -1e9,
                            device=labels.device)
        logits[:, seen] = seen_logits.detach()
        return {"loss": loss.item(), "logits": logits, "label": labels}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        images, labels = batch[0].to(self.device), batch[-1].to(self.device)
        images, batch_size, tile_count = self._flatten_tiles(images)
        attributes = self._attributes()
        # At inference ConVLM compares each patch representation with all
        # candidate attributes, including unseen classes.
        output = model(images, None)
        slide_embeddings = self._slide_embeddings(
            output["embedding"], batch_size, tile_count)
        logits = self._global_logits(slide_embeddings, attributes)
        loss = loss_fn(logits, labels).item() if loss_fn is not None else 0.0
        return {"loss": loss, "logits": logits, "label": labels}
