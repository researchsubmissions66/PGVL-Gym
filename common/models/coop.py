"""Canonical CoOp `PromptLearner` and `TextEncoder` shared across methods.

Source: extracted from FOCUS / ViLa-MIL / SLIP — these classes are
byte-for-byte the same in those repos and originate from
KaiyangZhou/CoOp. Methods that need a slightly different variant
(e.g. CSC, dual-scale, CONCH-specific tokenizer) should import this
class as a base or reuse the helper functions below.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Optional


class TextEncoder(nn.Module):
    """CoOp-style text encoder. Takes pre-embedded prompts and the
    tokenized prompt indices, returns the EOT-position projected feature.

    Works with any CLIP-style model that exposes:
        .transformer, .positional_embedding, .ln_final, .text_projection,
        .dtype
    """

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)            # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)            # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # take features from the EOT position
        eot_indices = tokenized_prompts.to(x.device).argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device),
              eot_indices] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Vanilla CoOp PromptLearner.

    Args:
        classnames: list of class names (e.g. ["LUAD", "LUSC"])
        clip_model: a CLIP-style model with `.token_embedding`, `.dtype`,
                    `.ln_final`, `.visual.input_resolution`
        n_ctx: number of learnable context tokens (default 16)
        ctx_init: optional natural-language string used to initialise the
                  context vectors (e.g. "a histopathology image of a")
        csc: class-specific context. If True, every class gets its own
             learnable context vector.
        class_token_position: "end" / "front" / "middle"
        tokenizer_fn: callable(str) -> LongTensor used for tokenisation.
                      Defaults to OpenAI CLIP's `clip.tokenize`.
    """

    def __init__(self,
                 classnames: List[str],
                 clip_model: nn.Module,
                 n_ctx: int = 16,
                 ctx_init: str = "",
                 csc: bool = False,
                 class_token_position: str = "end",
                 tokenizer_fn=None):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        embedding_device = clip_model.token_embedding.weight.device

        if tokenizer_fn is None:
            from clip import clip as _clip
            tokenizer_fn = _clip.tokenize

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenizer_fn(ctx_init).to(embedding_device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if csc:
                ctx_vectors = torch.empty(
                    n_cls, n_ctx, ctx_dim, dtype=dtype,
                    device=embedding_device)
            else:
                ctx_vectors = torch.empty(
                    n_ctx, ctx_dim, dtype=dtype, device=embedding_device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + c + "." for c in classnames]
        tokenized_prompts = torch.cat(
            [tokenizer_fn(p) for p in prompts]).to(embedding_device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])           # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])   # CLS, EOS
        self.register_buffer(
            "tokenized_prompts", tokenized_prompts, persistent=False)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = class_token_position

    def forward(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            raise NotImplementedError(
                f"class_token_position={self.class_token_position} "
                "not implemented in the shared PromptLearner. "
                "Methods needing 'front' or 'middle' should subclass.")
        return prompts
