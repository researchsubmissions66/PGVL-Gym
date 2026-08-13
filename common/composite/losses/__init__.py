"""Composite losses (CE + optional SLIP contrastive + MAPLE attribute alignment)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def slip_contrastive_loss(cross_corr, label, temperature: float = 0.01):
    n_cls = cross_corr.size(0)
    flat = cross_corr.view(-1)
    pos_idx = int(label.item() * n_cls + label.item())
    pos_logit = flat[pos_idx:pos_idx + 1]
    neg_idx = [i for i in range(n_cls * n_cls) if i != pos_idx]
    neg_logits = flat[neg_idx]
    logits = torch.cat([pos_logit, neg_logits]) / temperature
    return F.cross_entropy(logits, torch.tensor(0, device=logits.device))


def attribute_alignment_loss(slide_vec, attribute_features,
                             attribute_class_index, label):
    if slide_vec.numel() == 0 or attribute_features.numel() == 0:
        return torch.tensor(0.0, device=slide_vec.device)
    sv = F.normalize(slide_vec, dim=-1)
    af = F.normalize(attribute_features, dim=-1)
    sims = af @ sv
    pos_mask = (attribute_class_index == label.squeeze())
    if pos_mask.sum() == 0 or (~pos_mask).sum() == 0:
        return torch.tensor(0.0, device=slide_vec.device)
    pos = sims[pos_mask].mean()
    neg = sims[~pos_mask].mean()
    return F.relu(neg - pos + 0.2)


def composite_loss(logits, label, *,
                   ce_weight: float = 1.0,
                   slip_extras=None,
                   slip_contrastive_weight: float = 0.0,
                   maple_extras=None,
                   aux_attribute_weight: float = 0.0):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if label.dim() == 0:
        label = label.unsqueeze(0)
    loss = ce_weight * F.cross_entropy(logits, label)

    if slip_extras is not None and slip_contrastive_weight > 0:
        cc, temp = slip_extras
        loss = loss + slip_contrastive_weight * \
            slip_contrastive_loss(cc, label[0], temperature=temp)

    if maple_extras is not None and aux_attribute_weight > 0:
        sv, af, idx = maple_extras
        loss = loss + aux_attribute_weight * \
            attribute_alignment_loss(sv, af, idx, label[0])

    return loss
