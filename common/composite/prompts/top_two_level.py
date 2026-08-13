"""TOP two-level prompts.

Bag-level CoOp prompts (one per slide class) and instance-level CoOp
prompts (one per fine-grained pathological finding). The bag-level
features become `text_features`; instance-level features go in
`aux["instance"]`.

Reference:
    Qu et al., "TOP: Two-level Prompt Learning", NeurIPS 2023.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

from common.composite.interfaces import PromptModule, PromptBank


class TOPTwoLevelPrompts(PromptModule):
    name = "top_two_level"

    def __init__(self,
                 classnames: List[str],
                 backbone: nn.Module,
                 tokenizer=None,
                 instance_classnames: List[str] = None,
                 n_ctx_bag: int = 4,
                 n_ctx_inst: int = 4,
                 csc: bool = True):
        super().__init__()
        from common.models.coop import PromptLearner, TextEncoder

        self.bag_pl = PromptLearner(
            classnames=classnames, clip_model=backbone,
            n_ctx=n_ctx_bag, ctx_init="", csc=csc, tokenizer_fn=tokenizer)

        # Instance-level: defaults to per-class fine-grained findings.
        # If user doesn't supply, just reuse the bag-level classnames.
        instance_classnames = instance_classnames or classnames
        self.inst_pl = PromptLearner(
            classnames=instance_classnames, clip_model=backbone,
            n_ctx=n_ctx_inst, ctx_init="", csc=csc, tokenizer_fn=tokenizer)

        self.text_encoder = TextEncoder(backbone)

    def forward(self) -> PromptBank:
        bag_p = self.bag_pl()
        inst_p = self.inst_pl()
        bag_text = self.text_encoder(bag_p, self.bag_pl.tokenized_prompts)
        inst_text = self.text_encoder(inst_p, self.inst_pl.tokenized_prompts)
        return PromptBank(
            text_features=bag_text,
            aux={"instance": inst_text})
