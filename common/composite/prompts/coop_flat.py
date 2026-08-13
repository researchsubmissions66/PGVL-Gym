"""CoOp flat learnable prompts.

The canonical (C, D) per-class learnable prompt with `n_ctx` context
tokens. Wraps `common.models.coop.PromptLearner` and exposes the
PromptBank interface.
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn

from common.composite.interfaces import PromptModule, PromptBank


class CoOpFlatPrompts(PromptModule):
    name = "coop_flat"

    def __init__(self,
                 classnames: List[str],
                 backbone: nn.Module,
                 tokenizer=None,
                 n_ctx: int = 16,
                 ctx_init: str = "",
                 csc: bool = False):
        super().__init__()
        # Lazy import so users without OpenAI CLIP don't break here.
        from common.models.coop import PromptLearner, TextEncoder
        self.prompt_learner = PromptLearner(
            classnames=classnames,
            clip_model=backbone,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            csc=csc,
            tokenizer_fn=tokenizer)
        self.text_encoder = TextEncoder(backbone)

    def forward(self) -> PromptBank:
        prompts = self.prompt_learner()
        text_features = self.text_encoder(
            prompts, self.prompt_learner.tokenized_prompts)
        return PromptBank(text_features=text_features, aux={})
