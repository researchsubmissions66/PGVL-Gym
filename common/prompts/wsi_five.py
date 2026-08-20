"""Strict loading for WSI-FiVE's three text-supervision roles."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


WSI_FIVE_PROMPT_FORMAT = (
    "six_questions_structured_answers_and_class_descriptions")
ANSWER_FIELD_COUNT = 6
_PROVENANCE = frozenset({"upstream", "derived", "generated"})


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_record(path: Path) -> dict[str, Any]:
    prompt_root = Path(__file__).resolve().parents[2] / "text_prompts"
    try:
        key = str(path.resolve().relative_to(prompt_root.resolve()))
    except ValueError:
        return {}
    try:
        payload = json.loads(
            (prompt_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    record = payload.get("assets", {}).get(key, {}) \
        if isinstance(payload, dict) else {}
    return record if isinstance(record, dict) else {}


def _origin(
    path: Path,
    *,
    record: Mapping[str, Any],
    inline: str | None = None,
    expected: str | None = None,
) -> tuple[str, str]:
    provenance = record.get("provenance", inline or expected)
    if provenance not in _PROVENANCE:
        raise ValueError(
            f"{path}: WSI-FiVE asset needs upstream/derived/generated provenance")
    source = record.get("source", "")
    return str(provenance), str(source)


def _verify(
    path: Path,
    *,
    file_sha256: str,
    bank_sha256: str,
    provenance: str,
    record: Mapping[str, Any],
    manifest_bank_key: str,
    expected_file_sha256: str | None,
    expected_bank_sha256: str | None,
    expected_provenance: str | None,
) -> None:
    checks = (
        ("manifest file sha256", record.get("sha256"), file_sha256),
        ("manifest bank sha256", record.get(manifest_bank_key), bank_sha256),
        ("configured file sha256", expected_file_sha256, file_sha256),
        ("configured bank sha256", expected_bank_sha256, bank_sha256),
        ("configured provenance", expected_provenance, provenance),
    )
    for label, expected, actual in checks:
        if expected is not None and expected != actual:
            raise ValueError(
                f"{path}: WSI-FiVE {label} mismatch: expected {expected}, "
                f"got {actual}")


@dataclass(frozen=True)
class WSIFiVEQuestionBank:
    path: Path
    questions: tuple[str, ...]
    file_sha256: str
    prompt_bank_sha256: str
    provenance: str
    source: str

    def config_values(self) -> dict[str, str]:
        return {
            "wsi_question_file_sha256": self.file_sha256,
            "wsi_question_bank_sha256": self.prompt_bank_sha256,
            "wsi_question_provenance": self.provenance,
        }


@dataclass(frozen=True)
class WSIFiVEAnswerRecord:
    case_id: str
    cancer_type: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class WSIFiVEAnswerBank:
    path: Path
    records: tuple[WSIFiVEAnswerRecord, ...]
    file_sha256: str
    answer_bank_sha256: str
    provenance: str
    source: str

    def config_values(self) -> dict[str, str]:
        return {
            "wsi_answer_file_sha256": self.file_sha256,
            "wsi_answer_bank_sha256": self.answer_bank_sha256,
            "wsi_answer_provenance": self.provenance,
        }


@dataclass(frozen=True)
class WSIFiVEEvaluationBank:
    path: Path
    class_names: tuple[str, ...]
    prompts: tuple[str, ...]
    file_sha256: str
    prompt_bank_sha256: str
    provenance: str
    source: str

    def config_values(self) -> dict[str, str]:
        return {
            "wsi_evaluation_file_sha256": self.file_sha256,
            "wsi_evaluation_bank_sha256": self.prompt_bank_sha256,
            "wsi_evaluation_provenance": self.provenance,
        }


def load_wsi_five_question_bank(
    path: str | Path,
    *,
    record: Mapping[str, Any] | None = None,
    expected_file_sha256: str | None = None,
    expected_prompt_bank_sha256: str | None = None,
    expected_provenance: str | None = None,
) -> WSIFiVEQuestionBank:
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if (not isinstance(questions, list) or len(questions) != ANSWER_FIELD_COUNT
            or any(not isinstance(value, str) or not value.strip()
                   for value in questions)):
        raise ValueError(
            f"{source_path}: WSI-FiVE requires exactly six non-empty questions")
    values = tuple(value.strip() for value in questions)
    metadata = dict(record) if record is not None else _manifest_record(source_path)
    provenance, source = _origin(
        source_path, record=metadata,
        inline=payload.get("_provenance") if isinstance(payload, dict) else None,
        expected=expected_provenance)
    file_digest = _file_sha256(source_path)
    bank_digest = _json_sha256(list(values))
    _verify(
        source_path, file_sha256=file_digest, bank_sha256=bank_digest,
        provenance=provenance, record=metadata,
        manifest_bank_key="prompt_bank_sha256",
        expected_file_sha256=expected_file_sha256,
        expected_bank_sha256=expected_prompt_bank_sha256,
        expected_provenance=expected_provenance,
    )
    return WSIFiVEQuestionBank(
        source_path, values, file_digest, bank_digest, provenance, source)


def load_wsi_five_answer_bank(
    path: str | Path,
    *,
    record: Mapping[str, Any] | None = None,
    expected_file_sha256: str | None = None,
    expected_answer_bank_sha256: str | None = None,
    expected_provenance: str | None = None,
) -> WSIFiVEAnswerBank:
    source_path = Path(path)
    required = [
        "case_id", "cancer_type", "answer",
        *(f"q{index}" for index in range(1, ANSWER_FIELD_COUNT + 1)),
    ]
    with source_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if fields != required:
        raise ValueError(
            f"{source_path}: WSI-FiVE answer columns must be exactly {required}")
    if not rows:
        raise ValueError(f"{source_path}: WSI-FiVE answer bank is empty")
    records: list[WSIFiVEAnswerRecord] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        case_id = str(row["case_id"]).strip()
        cancer_type = str(row["cancer_type"]).strip()
        answers = tuple(
            str(row[f"q{index}"]).strip()
            for index in range(1, ANSWER_FIELD_COUNT + 1))
        if not case_id or not cancer_type or any(not value for value in answers):
            raise ValueError(
                f"{source_path}: WSI-FiVE answer row {row_number} has blanks")
        if case_id in seen:
            raise ValueError(
                f"{source_path}: duplicate WSI-FiVE case_id {case_id!r}")
        seen.add(case_id)
        if str(row["answer"]).strip() != "; ".join(answers):
            raise ValueError(
                f"{source_path}: answer row {row_number} does not equal q1..q6")
        records.append(WSIFiVEAnswerRecord(case_id, cancer_type, answers))
    canonical = [{
        "case_id": item.case_id,
        "cancer_type": item.cancer_type,
        "answers": list(item.answers),
    } for item in records]
    metadata = dict(record) if record is not None else _manifest_record(source_path)
    provenance, source = _origin(
        source_path, record=metadata, expected=expected_provenance)
    file_digest = _file_sha256(source_path)
    bank_digest = _json_sha256(canonical)
    _verify(
        source_path, file_sha256=file_digest, bank_sha256=bank_digest,
        provenance=provenance, record=metadata,
        manifest_bank_key="answer_bank_sha256",
        expected_file_sha256=expected_file_sha256,
        expected_bank_sha256=expected_answer_bank_sha256,
        expected_provenance=expected_provenance,
    )
    return WSIFiVEAnswerBank(
        source_path, tuple(records), file_digest, bank_digest,
        provenance, source)


def load_wsi_five_evaluation_bank(
    path: str | Path,
    label_dict: Mapping[str, int],
    *,
    record: Mapping[str, Any] | None = None,
    expected_file_sha256: str | None = None,
    expected_prompt_bank_sha256: str | None = None,
    expected_provenance: str | None = None,
) -> WSIFiVEEvaluationBank:
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, dict) or set(prompts) != set(label_dict):
        raise ValueError(
            f"{source_path}: evaluation prompt labels must exactly match label_dict")
    indices = list(label_dict.values())
    if (any(isinstance(index, bool) or not isinstance(index, int)
            for index in indices)
            or sorted(indices) != list(range(len(indices)))):
        raise ValueError("WSI-FiVE label_dict values must be contiguous indices")
    labels = tuple(label for label, _ in sorted(
        label_dict.items(), key=lambda item: item[1]))
    values = tuple(prompts[label].strip() if isinstance(prompts[label], str)
                   else "" for label in labels)
    if any(not value for value in values):
        raise ValueError(f"{source_path}: WSI-FiVE evaluation prompt is blank")
    metadata = dict(record) if record is not None else _manifest_record(source_path)
    provenance, source = _origin(
        source_path, record=metadata,
        inline=payload.get("_provenance") if isinstance(payload, dict) else None,
        expected=expected_provenance)
    file_digest = _file_sha256(source_path)
    bank_digest = _json_sha256({
        "class_names": list(labels), "prompts": list(values)})
    _verify(
        source_path, file_sha256=file_digest, bank_sha256=bank_digest,
        provenance=provenance, record=metadata,
        manifest_bank_key="prompt_bank_sha256",
        expected_file_sha256=expected_file_sha256,
        expected_bank_sha256=expected_prompt_bank_sha256,
        expected_provenance=expected_provenance,
    )
    return WSIFiVEEvaluationBank(
        source_path, labels, values, file_digest, bank_digest,
        provenance, source)


__all__ = [
    "ANSWER_FIELD_COUNT", "WSI_FIVE_PROMPT_FORMAT", "WSIFiVEAnswerBank",
    "WSIFiVEAnswerRecord", "WSIFiVEEvaluationBank", "WSIFiVEQuestionBank",
    "load_wsi_five_answer_bank", "load_wsi_five_evaluation_bank",
    "load_wsi_five_question_bank",
]
