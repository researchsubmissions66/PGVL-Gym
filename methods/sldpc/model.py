"""Promptable slide-text model for SLDPC.

The CPI/DHNO/SICL algorithm remains method-owned. Encoder-specific text and
slide operations arrive through the common capability interfaces.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from common.backbones import EncoderBundle, TokenBatch, get_spec
from common.backbones.factory import TitanPromptableText
from common.models.slide_alignment import TrainableSlideAdapter


def _token_attr(tokenizer: Any, *names: str, default: int = 0) -> int:
    for owner in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                return int(value)
    return default


class SLDPCPromptLearner(nn.Module):
    """Class-unified/class-specific context with SLDPC CPI and WFM."""

    def __init__(self, classnames: list[str], text_encoder: Any, n_ctx: int = 8,
                 ctx_init: str | None = None, csc: bool = False,
                 class_token_position: str = "end", omega: float = 0.8):
        super().__init__()
        # Compatibility for direct external use with a raw TITAN module.
        if (hasattr(text_encoder, "text_encoder") and
                not hasattr(text_encoder, "embed_tokens")):
            text_encoder = TitanPromptableText(text_encoder)
        required = ("tokenize", "embed_tokens", "encode_embedded",
                    "token_width", "dtype", "device", "tokenizer")
        missing = [name for name in required if not hasattr(text_encoder, name)]
        if missing:
            raise TypeError(
                "SLDPC requires a PromptableTextEncoder; missing " +
                ", ".join(missing))

        self.n_cls, self.n_ctx = len(classnames), n_ctx
        self.csc, self.omega = csc, omega
        self.position = class_token_position
        object.__setattr__(self, "_text_encoder", text_encoder)
        self.tokenizer = text_encoder.tokenizer
        self.dtype = text_encoder.dtype
        device = text_encoder.device
        self.ctx_dim = int(text_encoder.token_width)

        ctx_shape = ((self.n_cls, n_ctx, self.ctx_dim)
                     if csc else (n_ctx, self.ctx_dim))
        context = torch.empty(ctx_shape, device=device, dtype=self.dtype)
        nn.init.normal_(context, std=0.02)
        if ctx_init:
            initial_tokens = text_encoder.tokenize(
                [ctx_init.replace("_", " ")]).to(device)
            embedded = text_encoder.embed_tokens(initial_tokens)
            if initial_tokens.eot_indices is not None:
                eot_index = int(initial_tokens.eot_indices[0])
            elif initial_tokens.attention_mask is not None:
                eot_index = int(initial_tokens.attention_mask[0].sum().item()) - 1
            else:
                pad = _token_attr(self.tokenizer, "pad_token_id")
                eot_index = int(
                    initial_tokens.input_ids[0].ne(pad).sum().item()) - 1
            # Slice BOS < context < EOS using the semantic EOT position.
            # TITAN returns a fixed 128-token tensor, so [1:-1] would turn
            # padding into 126 learnable context tokens.
            initial = embedded[0, 1:eot_index]
            if initial.shape[0] == 0:
                raise ValueError("SLDPC ctx_init contains no non-special tokens")
            self.n_ctx = initial.shape[0]
            context = (initial.unsqueeze(0).expand(self.n_cls, -1, -1).clone()
                       if csc else initial.clone())
        self.ctx_learnable = nn.Parameter(context.clone())
        self.ctx_frozen = nn.Parameter(context.clone(), requires_grad=False)

        names = [name.replace("_", " ") for name in classnames]
        name_tokens = text_encoder.tokenize(names).to(device)
        name_ids = name_tokens.input_ids
        name_mask = name_tokens.attention_mask
        if name_mask is None:
            pad_value = _token_attr(self.tokenizer, "pad_token_id")
            name_mask = name_ids.ne(pad_value).long()
        name_lens = (name_mask.sum(dim=-1) - 2).clamp_min(0).tolist()
        max_name = int(max(name_lens))
        total = 1 + self.n_ctx + max_name + 1
        max_length = _token_attr(
            self.tokenizer, "model_max_length", "context_length", default=0)
        if max_length and total > max_length:
            raise ValueError("SLDPC prompt exceeds the text context length")

        pad = _token_attr(self.tokenizer, "pad_token_id")
        bos = _token_attr(
            self.tokenizer, "cls_token_id", "bos_token_id",
            default=int(name_ids[0, 0]))
        eos = _token_attr(self.tokenizer, "eos_token_id", "sep_token_id")
        ids = torch.full((self.n_cls, total), pad,
                         dtype=torch.long, device=device)
        ids[:, 0] = bos
        for index, length in enumerate(name_lens):
            ids[index, 1 + self.n_ctx:1 + self.n_ctx + length] = \
                name_ids[index, 1:1 + length]
        ids[:, -1] = eos
        positions = torch.full((self.n_cls,), total - 1,
                               dtype=torch.long, device=device)
        with torch.no_grad():
            embedded = text_encoder.embed_tokens(
                TokenBatch(ids, ids.ne(pad).long(), positions)).to(self.dtype)
        self.register_buffer("prefix", embedded[:, :1], persistent=False)
        self.register_buffer("suffix", embedded[:, 1 + self.n_ctx:], persistent=False)
        self.register_buffer("token_ids", ids, persistent=False)
        self.name_lens = [int(item) for item in name_lens]

    @torch.no_grad()
    def clone_learnable_to_frozen(self) -> None:
        self.ctx_frozen.copy_(self.ctx_learnable)
        self.ctx_frozen.requires_grad_(False)

    @torch.no_grad()
    def reinit_learnable_from_frozen(self) -> None:
        self.ctx_learnable.copy_(self.ctx_frozen)

    def _assemble(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim == 2:
            context = context.unsqueeze(0).expand(self.n_cls, -1, -1)
        if self.position == "end":
            return torch.cat([self.prefix, context, self.suffix], dim=1)
        assembled = []
        for index, length in enumerate(self.name_lens):
            name = self.suffix[index:index + 1, :length]
            eos = self.suffix[index:index + 1, length:]
            if self.position == "front":
                assembled.append(torch.cat([
                    self.prefix[index:index + 1], name,
                    context[index:index + 1], eos], dim=1))
            elif self.position == "middle":
                if self.n_ctx % 2:
                    raise ValueError("SLDPC middle context requires even n_ctx")
                half = self.n_ctx // 2
                assembled.append(torch.cat([
                    self.prefix[index:index + 1],
                    context[index:index + 1, :half], name,
                    context[index:index + 1, half:], eos], dim=1))
            else:
                raise ValueError(f"Unknown class_token_position: {self.position}")
        return torch.cat(assembled, dim=0)

    def _encode(self, context: torch.Tensor) -> torch.Tensor:
        prompts = self._assemble(context).to(dtype=self.dtype)
        pad = _token_attr(self.tokenizer, "pad_token_id")
        eos = _token_attr(self.tokenizer, "eos_token_id", "sep_token_id")
        positions = self.token_ids.eq(eos).float().argmax(dim=-1).long()
        tokens = TokenBatch(
            self.token_ids, self.token_ids.ne(pad).long(), positions)
        return self._text_encoder.encode_embedded(
            prompts, tokens, normalize=True)

    def forward(self, mode: str = "task",
                omega: float | None = None) -> torch.Tensor:
        if mode in {"train", "task"}:
            context = self.ctx_learnable
        elif mode == "base":
            context = self.ctx_frozen
        elif mode == "fused":
            weight = self.omega if omega is None else omega
            context = weight * self.ctx_learnable + (1.0 - weight) * self.ctx_frozen
        else:
            raise ValueError(f"Unknown SLDPC prompt mode: {mode}")
        return self._encode(context)


class PromptedSlideTextModel(nn.Module):
    """Prompt learning with either native or learned slide projection."""

    def __init__(self, encoder: EncoderBundle,
                 prompt_learner: SLDPCPromptLearner,
                 slide_adapter: TrainableSlideAdapter | None = None,
                 slide_input_dim: int | None = None):
        super().__init__()
        encoder.freeze()
        # The foundation model remains non-owning, as in the original wrapper,
        # so optimizer and checkpoint state contain only prompt parameters.
        object.__setattr__(self, "encoder", encoder)
        self.prompt_learner = prompt_learner
        self.slide_adapter = slide_adapter
        self.slide_input_dim = (
            int(slide_input_dim) if slide_input_dim is not None
            else encoder.spec.slide_input_dim)

    def project_slide(self, features: torch.Tensor) -> torch.Tensor:
        expected = self.slide_input_dim
        if expected is not None and (
                features.ndim != 2 or features.shape[-1] != expected):
            raise ValueError(
                "SLDPC slide feature source expects raw "
                f"slide embeddings shaped [batch, {expected}], got "
                f"{list(features.shape)}.")
        if self.slide_adapter is not None:
            return self.slide_adapter(features)
        return self.encoder.project_slide(features, normalize=True)

    def forward(self, features: torch.Tensor,
                mode: str = "task") -> torch.Tensor:
        return self.project_slide(features) @ self.prompt_learner(mode=mode).t()


# Preserve external import names while routing through the new interfaces.
TitanPromptLearner = SLDPCPromptLearner


class _LegacyTitanSlideProjector:
    def __init__(self, titan: nn.Module):
        self.titan = titan

    def project_slide(self, raw_embeddings: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
        projection = self.titan.vision_encoder.proj
        raw_embeddings = raw_embeddings.to(
            device=projection.device, dtype=projection.dtype)
        output = raw_embeddings @ projection
        return torch.nn.functional.normalize(output.float(), dim=-1) \
            if normalize else output


class PromptedTitan(PromptedSlideTextModel):
    """Compatibility constructor accepting the former raw TITAN argument."""

    def __init__(self, titan_or_encoder: Any,
                 prompt_learner: SLDPCPromptLearner):
        if isinstance(titan_or_encoder, EncoderBundle):
            encoder = titan_or_encoder
        else:
            text = TitanPromptableText(titan_or_encoder)
            encoder = EncoderBundle(
                raw_model=titan_or_encoder, raw_tokenizer=text.tokenizer,
                spec=get_spec("titan"), text=text,
                slide=_LegacyTitanSlideProjector(titan_or_encoder))
        super().__init__(encoder, prompt_learner)
