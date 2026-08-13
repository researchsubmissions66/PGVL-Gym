"""MAPLE attribute-graph prompts (simplified for the composite system).

The full MAPLE GCN over entities lives in
`methods/maple/maple_model/model.py`. Here we expose a lightweight
adapter: load attribute lists per class from JSON, average their
encoded text embeddings to form `text_features`, and stash the
per-attribute embeddings in `aux["attributes"]`. Aggregators that
want the full graph treatment can use the original MAPLE module
directly via the legacy adapter.

Expected JSON shape (same as MAPLE's `templete/maple/<DATASET>_attributes.json`):
    {
      "HGSC": ["papillary architecture", "marked atypia", ...],
      "LGSC": ["uniform nuclei", ...],
      ...
    }

Reference:
    Zhou et al., "MAPLE: Multi-scale Attribute-enhanced Prompt
    Learning", NeurIPS 2025.
"""
from __future__ import annotations
import json
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import PromptModule, PromptBank


class MAPLEGraphPrompts(PromptModule):
    name = "maple_graph"

    def __init__(self,
                 classnames: List[str],
                 backbone: nn.Module,
                 tokenizer,
                 text_path: str,
                 templates=None,
                 encoder=None):
        super().__init__()
        if not os.path.isfile(text_path):
            raise FileNotFoundError(
                f"MAPLEGraphPrompts requires a JSON of per-class "
                f"attributes at {text_path}.")
        with open(text_path) as f:
            self.attributes = json.load(f)
        self.classnames = classnames
        self.templates = templates or ["a histopathology image showing {}."]
        self.encoder = encoder

        with torch.no_grad():
            class_feats, attr_feats, attr_class_idx = self._encode(backbone, tokenizer)
        self.register_buffer("class_features", class_feats)         # (C, D)
        self.register_buffer("attribute_features", attr_feats)      # (A, D)
        self.register_buffer("attribute_class_index", attr_class_idx)  # (A,)

    def _encode_batch(self, backbone, tokenizer, prompts):
        if self.encoder is not None:
            return self.encoder.encode_text(prompts, normalize=True)
        device = next(backbone.parameters()).device
        tokenized = tokenizer(prompts) if callable(tokenizer) \
            else tokenizer(prompts, return_tensors="pt",
                           padding=True, truncation=True)["input_ids"]
        if isinstance(tokenized, dict):
            tokenized = tokenized["input_ids"]
        embed = backbone.encode_text(tokenized.to(device))
        return F.normalize(embed, dim=-1)

    def _encode(self, backbone, tokenizer):
        per_class = []
        all_attrs = []
        all_idx = []
        for ci, c in enumerate(self.classnames):
            attrs = self.attributes.get(c, [c])
            prompts = [t.format(a) for a in attrs for t in self.templates]
            attr_e = self._encode_batch(backbone, tokenizer, prompts)
            per_class.append(F.normalize(attr_e.mean(0), dim=-1))
            all_attrs.append(attr_e)
            all_idx.extend([ci] * attr_e.size(0))
        class_feats = F.normalize(torch.stack(per_class), dim=-1)
        attr_feats = torch.cat(all_attrs, dim=0)
        idx = torch.tensor(all_idx, dtype=torch.long, device=class_feats.device)
        return class_feats, attr_feats, idx

    def forward(self) -> PromptBank:
        return PromptBank(
            text_features=self.class_features,
            aux={"attributes": self.attribute_features,
                 "attribute_class_index": self.attribute_class_index})
