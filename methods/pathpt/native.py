"""PathPT-native prompt selection, patch targets, and WSI voting."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PromptSelection:
    """Selected zero-shot classifier and reproducibility trace."""

    embedding: torch.Tensor
    top_classifier_indices: tuple[int, ...]
    balanced_accuracies: tuple[float, ...]
    prompt_indices: tuple[tuple[int, ...], ...]


@torch.no_grad()
def choose_prompt_embedding(
    encode_text: Callable[[Sequence[str]], torch.Tensor],
    prompts: Sequence[Sequence[str]],
    *,
    device: torch.device,
    seed: int,
) -> PromptSelection:
    """Choose one prompt per class as upstream does when selection is off."""
    rng = random.Random(seed)
    indices = tuple(rng.randrange(len(class_prompts))
                    for class_prompts in prompts)
    texts = [prompts[class_index][prompt_index]
             for class_index, prompt_index in enumerate(indices)]
    embedding = F.normalize(
        encode_text(texts).detach().float().to(device), dim=-1)
    return PromptSelection(
        embedding=embedding,
        top_classifier_indices=(0,),
        balanced_accuracies=(),
        prompt_indices=(indices,),
    )


def _patch_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError(
            f"PathPT expected [patches, dimension], got {tuple(value.shape)}")
    return value


def extract_patch_scores(output: object, patch_class_count: int) -> torch.Tensor:
    """Extract the per-patch output from every vendored backbone branch."""
    candidates: list[Any]
    if isinstance(output, dict):
        candidates = [output.get("patch_logits"), output.get("logits")]
    elif isinstance(output, tuple):
        candidates = list(output)
    else:
        candidates = [output]
    for candidate in candidates:
        if not torch.is_tensor(candidate) or candidate.ndim < 1:
            continue
        if candidate.shape[-1] != patch_class_count:
            continue
        if candidate.ndim == 3 and candidate.shape[0] == 1:
            candidate = candidate.squeeze(0)
        if candidate.ndim == 1:
            candidate = candidate.unsqueeze(0)
        if candidate.ndim == 2:
            return candidate
    raise ValueError(
        "PathPT model returned no [patches, patch_classes] tensor with "
        f"width {patch_class_count}")


def generate_patch_targets(
    zero_shot_scores: torch.Tensor,
    slide_label: torch.Tensor,
    *,
    synthetic_normal: bool,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Generate upstream normal/subtype/candidate patch labels."""
    scores = _patch_matrix(zero_shot_scores)
    if slide_label.numel() != 1:
        raise ValueError("PathPT native training requires batch_size=1")
    label = int(slide_label.item())

    if synthetic_normal:
        tumour_index = label + 1
        if tumour_index >= scores.shape[1]:
            raise ValueError(
                f"slide label {label} maps outside {scores.shape[1]} patch classes")
        targets = torch.full(
            (scores.shape[0],), -tumour_index,
            dtype=torch.long, device=scores.device)
        maxima = scores.argmax(dim=1)
        normal = (scores[:, 0] > threshold) & maxima.eq(0)
        tumour = ((scores[:, tumour_index] > threshold)
                  & maxima.eq(tumour_index))
        targets[normal] = 0
        targets[tumour] = tumour_index
        return targets

    # Binary CAMELYON adaptation: a normal WSI has no tumour candidate;
    # every patch is known normal. A tumour WSI retains {Normal, Tumour}
    # candidates exactly like the upstream subtype selector.
    if label == 0:
        return torch.zeros(scores.shape[0], dtype=torch.long,
                           device=scores.device)
    if label >= scores.shape[1]:
        raise ValueError(
            f"slide label {label} maps outside {scores.shape[1]} patch classes")
    targets = torch.full(
        (scores.shape[0],), -label, dtype=torch.long, device=scores.device)
    maxima = scores.argmax(dim=1)
    normal = (scores[:, 0] > threshold) & maxima.eq(0)
    tumour = (scores[:, label] > threshold) & maxima.eq(label)
    targets[normal] = 0
    targets[tumour] = label
    return targets


def vote_slide_probabilities(
    patch_scores: torch.Tensor,
    *,
    n_classes: int,
    synthetic_normal: bool,
) -> torch.Tensor:
    """Apply PathPT's patch vote, excluding the synthetic Normal class."""
    scores = _patch_matrix(patch_scores)
    predictions = scores.argmax(dim=1)
    counts = torch.bincount(predictions, minlength=scores.shape[1]).float()

    if not synthetic_normal:
        if scores.shape[1] != n_classes:
            raise ValueError("binary PathPT vote width does not match n_classes")
        total = counts.sum()
        return (counts / total).unsqueeze(0)

    if scores.shape[1] != n_classes + 1:
        raise ValueError(
            "PathPT synthetic-Normal vote requires n_classes + 1 patch scores")
    tumour_counts = counts[1:]
    all_normal = bool(tumour_counts.sum().eq(0))
    tied = bool(tumour_counts.eq(tumour_counts.max()).sum().gt(1))
    if all_normal or tied:
        tumour_predictions = scores[:, 1:].argmax(dim=1)
        tumour_counts = torch.bincount(
            tumour_predictions, minlength=n_classes).float()
    return (tumour_counts / tumour_counts.sum()).unsqueeze(0)


def _encode_prompt_classes(
    encode_text: Callable[[Sequence[str]], torch.Tensor],
    prompts: Sequence[Sequence[str]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    encoded: list[torch.Tensor] = []
    for class_prompts in prompts:
        pieces = []
        for start in range(0, len(class_prompts), batch_size):
            result = encode_text(class_prompts[start:start + batch_size])
            pieces.append(result.detach().float().to(device))
        features = torch.cat(pieces, dim=0)
        encoded.append(F.normalize(features, dim=-1))
    return encoded


@torch.no_grad()
def select_prompt_embedding(
    encode_text: Callable[[Sequence[str]], torch.Tensor],
    train_loader: Iterable[Any],
    prompts: Sequence[Sequence[str]],
    *,
    n_slide_classes: int,
    synthetic_normal: bool,
    device: torch.device,
    classifier_count: int = 200,
    select_count: int = 100,
    top_patches: int = 100,
    classifier_batch_size: int = 16,
    text_batch_size: int = 128,
) -> PromptSelection:
    """Reproduce PathPT's training-split WSI prompt selector."""
    if classifier_count <= 0 or select_count <= 0 or top_patches <= 0:
        raise ValueError("PathPT prompt-selection counts must be positive")
    encoded = _encode_prompt_classes(
        encode_text, prompts, device=device, batch_size=text_batch_size)
    prompt_indices = []
    classifiers = []
    for classifier_index in range(classifier_count):
        # Upstream resets Python's RNG to the classifier index, making the 200
        # candidate classifiers identical across folds. Preserve that detail.
        rng = random.Random(classifier_index)
        indices = tuple(rng.randrange(len(rows)) for rows in encoded)
        prompt_indices.append(indices)
        classifiers.append(torch.stack([
            encoded[class_index][prompt_index]
            for class_index, prompt_index in enumerate(indices)
        ]))
    classifier_tensor = torch.stack(classifiers)

    predictions: list[list[int]] = [[] for _ in range(classifier_count)]
    labels: list[int] = []
    for batch in train_loader:
        features = _patch_matrix(batch[0]).to(device=device, dtype=torch.float32)
        label = batch[-1]
        if not torch.is_tensor(label) or label.numel() != 1:
            raise ValueError("PathPT prompt selection requires batch_size=1 labels")
        labels.append(int(label.item()))
        keep = min(top_patches, features.shape[0])
        for start in range(0, classifier_count, classifier_batch_size):
            block = classifier_tensor[start:start + classifier_batch_size]
            logits = torch.einsum("nd,mcd->mnc", features, block)
            pooled = logits.topk(keep, dim=1).values.sum(dim=1)
            if synthetic_normal:
                pooled = pooled[:, 1:]
            block_predictions = pooled.argmax(dim=1).cpu().tolist()
            for offset, prediction in enumerate(block_predictions):
                predictions[start + offset].append(int(prediction))
    if not labels:
        raise ValueError("PathPT prompt selection received an empty train split")

    present = sorted(set(labels))
    if any(label < 0 or label >= n_slide_classes for label in present):
        raise ValueError("PathPT train split contains an out-of-range label")
    scores = []
    for candidate_predictions in predictions:
        recalls = []
        for class_index in present:
            positions = [i for i, value in enumerate(labels)
                         if value == class_index]
            recalls.append(sum(
                candidate_predictions[i] == class_index for i in positions
            ) / len(positions))
        scores.append(sum(recalls) / len(recalls))
    ranking = sorted(range(classifier_count), key=lambda i: (-scores[i], i))
    selected = tuple(ranking[:min(select_count, classifier_count)])
    embedding = F.normalize(
        classifier_tensor[list(selected)].mean(dim=0), dim=-1)
    return PromptSelection(
        embedding=embedding,
        top_classifier_indices=selected,
        balanced_accuracies=tuple(float(value) for value in scores),
        prompt_indices=tuple(prompt_indices),
    )
