"""Report-conditioned patch fusion for the unified WSI-FiVE adapter."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ClinicalTextTower(nn.Module):
    def __init__(self, model_path: str, output_dim: int = 512,
                 freeze_base: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True)
        self.base = AutoModel.from_pretrained(
            model_path, local_files_only=True)
        if freeze_base:
            self.base.requires_grad_(False)
            self.base.eval()
        width = int(self.base.config.hidden_size)
        self.projection = nn.Linear(width, output_dim, bias=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.base.parameters()):
            self.base.eval()
        return self

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256,
            return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        output = self.base(**tokens).last_hidden_state
        mask = tokens["attention_mask"].to(output.dtype).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(self.projection(pooled).float(), dim=-1)


class WSIFiVEClassifier(nn.Module):
    """Fuse a WSI patch sequence with its report, then compare to classes."""

    def __init__(self, classnames: list[str], clinicalbert_path: str,
                 feature_dim: int = 512, num_heads: int = 8,
                 max_frames: int = 2048, freeze_text_base: bool = True):
        super().__init__()
        self.classnames = list(classnames)
        self.text = ClinicalTextTower(
            clinicalbert_path, feature_dim, freeze_base=freeze_text_base)
        self.position = nn.Parameter(torch.empty(1, max_frames, feature_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=feature_dim * 4, batch_first=True,
            activation="gelu", norm_first=True)
        self.patch_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.report_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True)
        self.fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def forward(self, features: torch.Tensor,
                reports: list[str]) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[-1] != self.position.shape[-1]:
            raise ValueError(
                "WSI-FiVE expects [batch, patches, 512], got "
                f"{list(features.shape)}")
        if features.shape[1] > self.position.shape[1]:
            raise ValueError(
                f"WSI-FiVE received {features.shape[1]} patches, but max_frames "
                f"is {self.position.shape[1]}")
        if len(reports) != features.shape[0]:
            raise ValueError("WSI-FiVE needs one report per slide")

        device = features.device
        encoded = self.patch_encoder(
            features.float() + self.position[:, :features.shape[1]])
        report_features = self.text(reports, device).unsqueeze(1)
        attended, _ = self.report_attention(
            report_features, encoded, encoded, need_weights=False)
        pooled = encoded.mean(dim=1)
        slide = F.normalize(
            self.fusion(torch.cat([pooled, attended[:, 0]], dim=-1)), dim=-1)
        class_prompts = [
            f"a histopathology slide diagnostic of {name}"
            for name in self.classnames
        ]
        class_features = self.text(class_prompts, device)
        return self.logit_scale.exp().clamp(max=100) * slide @ class_features.t()
