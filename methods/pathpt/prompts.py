"""Task prompt banks used by PathPT's native supervision pipeline.

The BRCA, UBC-OCEAN, and CAMELYON wording and the 22 templates are imported
from the vendored PathPT ``utils.py`` copy without correcting its content.
NSCLC and RCC were not present in that upstream prompt module; their compact
synonym sets are explicit PGVL-Gym task extensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._utils import brca_names, camelyon_names, templates, ubc_names


@dataclass(frozen=True)
class PathPTPromptBank:
    """Ordered synonyms and rendered prompts for one slide task."""

    task: str
    class_synonyms: tuple[tuple[str, ...], ...]
    prompts: tuple[tuple[str, ...], ...]
    synthetic_normal: bool
    provenance: str
    source: str
    note: str = ""


_GENERATED_BANKS: dict[str, dict[str, list[str]]] = {
    "nsclc": {
        "Normal": [
            "normal lung tissue", "lung normal tissue",
            "lung non-cancerous tissue",
        ],
        "LUAD": ["lung adenocarcinoma", "adenocarcinoma of the lung"],
        "LUSC": [
            "lung squamous cell carcinoma",
            "squamous cell carcinoma of the lung",
        ],
    },
    "rcc": {
        "Normal": [
            "normal kidney tissue", "kidney normal tissue",
            "kidney non-cancerous tissue",
        ],
        "CCRCC": [
            "clear cell renal cell carcinoma",
            "renal clear cell carcinoma",
        ],
        "PRCC": [
            "papillary renal cell carcinoma",
            "renal papillary carcinoma",
        ],
        "CHRCC": [
            "chromophobe renal cell carcinoma",
            "renal chromophobe carcinoma",
        ],
    },
}


def _canonical_task(value: Any) -> str:
    task = str(value or "").strip().lower().replace("-", "_")
    return {
        "tcga_brca": "brca",
        "ubc": "ubc_ocean",
        "ubc_ocean": "ubc_ocean",
        "camelyon": "camelyon16",
        "cam16": "camelyon16",
        "tcga_nsclc": "nsclc",
        "lung": "nsclc",
        "tcga_rcc": "rcc",
    }.get(task, task)


def _render(synonyms: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(
        template.replace("CLASSNAME", name)
        for name in class_names for template in templates
    ) for class_names in synonyms)


def _validate_count(cfg: Mapping[str, Any], synonyms: Sequence[Sequence[str]],
                    synthetic_normal: bool) -> None:
    expected = int(cfg["n_classes"]) + int(synthetic_normal)
    if len(synonyms) != expected:
        raise ValueError(
            f"PathPT prompt bank has {len(synonyms)} patch classes, expected "
            f"{expected} for n_classes={cfg['n_classes']} and "
            f"synthetic_normal={synthetic_normal}")


def resolve_prompt_bank(cfg: Mapping[str, Any]) -> PathPTPromptBank:
    """Resolve the ordered PathPT bank and its content provenance."""
    task = _canonical_task(cfg.get("task", cfg.get("cohort")))
    custom = cfg.get("pathpt_prompt_synonyms")
    if custom is not None:
        if not isinstance(custom, list) or not custom or not all(
                isinstance(row, list) and row and all(
                    isinstance(item, str) and item.strip() for item in row)
                for row in custom):
            raise ValueError(
                "pathpt_prompt_synonyms must be a non-empty list of "
                "non-empty string lists")
        synthetic = bool(cfg.get("pathpt_synthetic_normal", True))
        synonyms = tuple(tuple(row) for row in custom)
        _validate_count(cfg, synonyms, synthetic)
        return PathPTPromptBank(
            task or "custom", synonyms, _render(synonyms), synthetic,
            "explicit", "config:pathpt_prompt_synonyms")

    if task == "brca":
        order = ("Normal", "Invasive Ductal Carcinoma",
                 "Invasive Lobular Carcinoma")
        synonyms = tuple(tuple(brca_names[key]) for key in order)
        synthetic = True
        provenance = "upstream"
        source = "MAGIC-AI4Med/PathPT utils.py:brca_names"
        note = "Verbatim upstream synonyms and templates."
    elif task == "ubc_ocean":
        order = ("Normal", "CC", "EC", "HGSC", "LGSC", "MC")
        synonyms = tuple(tuple(ubc_names[key]) for key in order)
        synthetic = True
        provenance = "upstream"
        source = "MAGIC-AI4Med/PathPT utils.py:ubc_names"
        note = "Verbatim upstream synonyms and templates."
    elif task == "camelyon16":
        # CAMELYON is a normal-vs-tumour slide task, not PathPT's tumour-only
        # subtyping setup. Its upstream bank already contains Normal, so adding
        # a second synthetic Normal row would corrupt the label mapping.
        order = ("Normal", "Tumor")
        synonyms = tuple(tuple(camelyon_names[key]) for key in order)
        synthetic = False
        provenance = "upstream"
        source = "MAGIC-AI4Med/PathPT utils.py:camelyon_names"
        note = (
            "Verbatim upstream bank, including its concatenated synonym; "
            "slide-level binary supervision is a local adaptation.")
    elif task in _GENERATED_BANKS:
        bank = _GENERATED_BANKS[task]
        label_dict = cfg.get("label_dict", {})
        label_order = ([key for key, _value in sorted(
            label_dict.items(), key=lambda item: item[1])]
            if isinstance(label_dict, Mapping) else [])
        if not label_order:
            raise ValueError(
                f"PathPT generated {task} bank requires ordered label_dict")
        order = ("Normal", *label_order)
        missing = [key for key in order if key not in bank]
        if missing:
            raise ValueError(
                f"PathPT {task} prompt bank has no synonyms for {missing}")
        synonyms = tuple(tuple(bank[key]) for key in order)
        synthetic = True
        provenance = "generated"
        source = "PGVL-Gym methods/pathpt/prompts.py"
        note = "Task extension; no task-matched bank exists in upstream PathPT."
    else:
        classnames = cfg.get("classnames")
        if not isinstance(classnames, list) or not classnames:
            raise ValueError(
                "PathPT native mode requires task or non-empty classnames")
        synonyms = (("normal non-cancerous tissue",), *(
            (str(classname),) for classname in classnames))
        synthetic = True
        provenance = "generated"
        source = "PGVL-Gym classname fallback"
        note = "Generic local fallback; not an upstream PathPT prompt bank."

    _validate_count(cfg, synonyms, synthetic)
    return PathPTPromptBank(
        task, synonyms, _render(synonyms), synthetic,
        provenance, source, note)
