"""CoD-MIL chain-of-diagnosis prompts.

Loads a hierarchy of broad and specific diagnostic prompts from a
JSON file and encodes them with the frozen text encoder. The
`text_features` exposed to aggregators is the per-class average; the
broad/specific tensors are passed in `aux` so aggregators that want
hierarchy-aware behavior can use them.

Expected JSON shape:
    {
      "HGSC": {
        "broad":   ["malignant ovarian cancer"],
        "specific":["high-grade serous carcinoma with marked atypia",
                    "papillary architecture", ...]
      },
      "LGSC": {...},
      ...
    }

Reference:
    Shi et al., "Chain-of-Diagnosis Prompting MIL", IEEE TMI 2024.
"""
from __future__ import annotations
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.composite.interfaces import PromptModule, PromptBank


class CoDChainPrompts(PromptModule):
    name = "cod_chain"

    def __init__(self,
                 classnames,
                 backbone: nn.Module,
                 tokenizer,
                 hierarchy_path: str,
                 templates=None,
                 encoder=None):
        super().__init__()
        if not os.path.isfile(hierarchy_path):
            raise FileNotFoundError(
                f"CoDChainPrompts requires a hierarchy JSON at "
                f"{hierarchy_path}. See docstring for format.")
        with open(hierarchy_path) as f:
            self.hierarchy = json.load(f)
        self.classnames = classnames
        self.templates = templates or ["a histopathology image of {}."]
        self.encoder = encoder

        with torch.no_grad():
            broad, specific, fused = self._encode(backbone, tokenizer)
        self.register_buffer("broad_features", broad)
        self.register_buffer("specific_features", specific)
        self.register_buffer("fused_features", fused)

    def _encode_one(self, backbone, tokenizer, prompts):
        if self.encoder is not None:
            return self.encoder.encode_text(prompts, normalize=True)
        device = next(backbone.parameters()).device
        tokenized = tokenizer(prompts) if callable(tokenizer) \
            else tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
        if isinstance(tokenized, dict):
            tokenized = tokenized["input_ids"]
        embed = backbone.encode_text(tokenized.to(device))
        return F.normalize(embed, dim=-1)

    def _encode(self, backbone, tokenizer):
        broad_per_class, spec_per_class = [], []
        for c in self.classnames:
            entry = self.hierarchy.get(c, {"broad": [c], "specific": [c]})
            broad_p = [t.format(b) for b in entry["broad"]   for t in self.templates]
            spec_p  = [t.format(s) for s in entry["specific"] for t in self.templates]
            broad_per_class.append(self._encode_one(backbone, tokenizer, broad_p).mean(0))
            spec_per_class .append(self._encode_one(backbone, tokenizer, spec_p ).mean(0))
        broad   = F.normalize(torch.stack(broad_per_class), dim=-1)    # (C, D)
        spec    = F.normalize(torch.stack(spec_per_class), dim=-1)     # (C, D)
        fused   = F.normalize(0.5 * (broad + spec), dim=-1)            # (C, D)
        return broad, spec, fused

    def forward(self) -> PromptBank:
        return PromptBank(
            text_features=self.fused_features,
            aux={"broad": self.broad_features,
                 "specific": self.specific_features})
