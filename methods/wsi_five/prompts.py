"""Native WSI-FiVE text-supervision utilities.

The release uses two different text banks for two different purposes:

* six clinical questions condition patch aggregation; and
* semicolon-separated answers form the training comparison bank.

At evaluation time the answer bank is replaced by one published diagnostic
description per class.  Keeping those roles separate prevents per-slide report
text from becoming privileged test-time input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from common.prompts.wsi_five import (
    ANSWER_FIELD_COUNT,
    load_wsi_five_evaluation_bank,
)



@dataclass(frozen=True)
class AugmentedAnswerBank:
    """One question-dropped answer bank and its remapped targets."""

    texts: tuple[str, ...]
    original_to_augmented: tuple[int, ...]
    kept_question_indices: tuple[int, ...]


def normalize_answer_fields(fields: Sequence[Any]) -> tuple[str, ...]:
    """Validate and normalize one six-answer WSI-FiVE supervision record."""
    if len(fields) != ANSWER_FIELD_COUNT:
        raise ValueError(
            "WSI-FiVE native supervision requires exactly six answers, got "
            f"{len(fields)}")
    normalized = tuple(str(value).strip() for value in fields)
    if any(not value or value.casefold() == "nan" for value in normalized):
        raise ValueError("WSI-FiVE native supervision contains a blank answer")
    return normalized


def _is_unknown(value: str) -> bool:
    # Upstream removes an answer segment when its first token is the BERT token
    # for ``Unknown``.  The checked-in GPT sheets spell that response exactly,
    # but accepting punctuation keeps the string-level reconstruction faithful.
    first = value.lstrip().split(maxsplit=1)[0].rstrip(".,;:()[]{}").casefold()
    return first == "unknown"


def augment_answer_bank(
    answers: Sequence[Sequence[Any]],
    *,
    drop_question_indices: Sequence[int],
    rng: random.Random | Any = random,
) -> AugmentedAnswerBank:
    """Apply upstream question dropout, answer filtering, and label hashing.

    Deduplication is stable rather than relying on Python set iteration.  This
    does not alter the loss because targets are remapped with the bank; it makes
    reruns reproducible across ``PYTHONHASHSEED`` values.
    """
    normalized = tuple(normalize_answer_fields(row) for row in answers)
    if not normalized:
        raise ValueError("WSI-FiVE native training received an empty answer bank")
    dropped = set()
    for raw_index in drop_question_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError("WSI-FiVE dropped question indices must be integers")
        if raw_index < 0 or raw_index >= ANSWER_FIELD_COUNT:
            raise ValueError(
                f"WSI-FiVE question index {raw_index} is outside [0, 6)")
        dropped.add(raw_index)
    if len(dropped) >= ANSWER_FIELD_COUNT:
        raise ValueError("WSI-FiVE must retain at least one clinical question")

    reduced: list[tuple[str, ...]] = []
    for row in normalized:
        reduced.append(tuple(
            answer for index, answer in enumerate(row)
            if index not in dropped and not _is_unknown(answer)))

    unique: list[tuple[str, ...]] = []
    index_by_fields: dict[tuple[str, ...], int] = {}
    remapped: list[int] = []
    for fields in reduced:
        if fields not in index_by_fields:
            index_by_fields[fields] = len(unique)
            unique.append(fields)
        remapped.append(index_by_fields[fields])

    texts: list[str] = []
    for fields in unique:
        shuffled = list(fields)
        rng.shuffle(shuffled)
        # Empty text is intentional when all retained answers are Unknown; the
        # released token pipeline produces a start/end-only candidate too.
        texts.append("; ".join(shuffled))
    return AugmentedAnswerBank(
        texts=tuple(texts),
        original_to_augmented=tuple(remapped),
        kept_question_indices=tuple(
            index for index in range(ANSWER_FIELD_COUNT)
            if index not in dropped),
    )


def load_evaluation_prompts(
    path: str | Path,
    label_dict: Mapping[str, int],
) -> tuple[str, ...]:
    """Load released evaluation descriptions in class-index order."""
    return load_wsi_five_evaluation_bank(path, label_dict).prompts
