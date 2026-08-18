"""Generate benchmark text assets with a served language model.

Some methods need text a paper published for one cohort and not for others:
WSI-FiVE's answers to its clinical questions exist for TCGA-Lung alone, and its
question set is lung-specific. Rather than leave those cohorts unrunnable or
quietly substitute class names, this script generates the missing text and
records exactly what produced it.

Every asset written here carries `_provenance: generated` plus the model,
revision, decoding settings and prompt template, so it can be regenerated and
so a result computed from it is never mistaken for the published condition.

    scripts/llm_server.sh patho-r1-7b
    python scripts/generate_llm_assets.py --task wsi_five_answers \
        --cohort rcc --model patho-r1-7b --endpoint http://gpu042:8000/v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.llm import LLMClient, LLMUnavailableError, SamplingParams  # noqa: E402

# Bumped whenever the wording changes, so assets built with different
# instructions are distinguishable rather than silently mixed.
ANSWER_TEMPLATE_ID = "wsi_five_answers/v1"

ANSWER_SYSTEM = (
    "You are a pathologist reading a diagnostic report. Answer each numbered "
    "question from the report alone. If the report does not state the answer, "
    "reply exactly 'Unknown' for that question. Never infer beyond the text."
)
ANSWER_USER = (
    "Report:\n{report}\n\nQuestions:\n{questions}\n\n"
    "Answer all {count} questions in order, as one line, separated by '; '. "
    "Give only the answers, no numbering and no commentary."
)


def _questions(cohort_questions: Path) -> list[str]:
    payload = json.loads(cohort_questions.read_text(encoding="utf-8"))
    return list(payload["questions"] if isinstance(payload, dict) else payload)


def _reports(report_csv: Path, cases: set[str]) -> dict[str, str]:
    reports: dict[str, str] = {}
    with report_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = next((row[key] for key in
                               ("patient_filename", "case_id", "slide_id")
                               if key in row and row[key]), None)
            text = next((row[key] for key in ("text", "report")
                         if key in row and row[key]), None)
            if not identifier or not text:
                continue
            case = str(identifier).split(".")[0]
            if not cases or case in cases:
                reports.setdefault(case, str(text))
    return reports


def _cohort_cases(cohort: str) -> tuple[set[str], dict[str, str]]:
    """Return the cohort's case ids and its declared WSI-FiVE assets."""
    import yaml

    for protocol in sorted(REPO_ROOT.glob("benchmarks/*/protocol.yaml")):
        payload = yaml.safe_load(protocol.read_text(encoding="utf-8"))
        if cohort not in payload.get("cohorts", {}):
            continue
        cfg = payload["cohorts"][cohort]
        manifest = protocol.parent / "data" / cohort / "manifest.csv"
        cases: set[str] = set()
        if manifest.is_file():
            with manifest.open(newline="", encoding="utf-8") as handle:
                cases = {row["case_id"] for row in csv.DictReader(handle)}
        return cases, cfg
    raise KeyError(f"no protocol declares cohort {cohort!r}")


def generate_wsi_five_answers(args: argparse.Namespace) -> Path:
    """Answer WSI-FiVE's clinical questions from each case's report."""
    cases, cfg = _cohort_cases(args.cohort)
    questions_path = REPO_ROOT / cfg["wsi_five_questions_json"]
    report_path = REPO_ROOT / cfg["wsi_report_csv"]
    questions = _questions(questions_path)
    reports = _reports(report_path, cases)
    if not reports:
        # A cohort whose report CSV already holds answers has no free-text
        # column, which is the NSCLC case: the authors published answers there,
        # so regenerating them would replace the one paper-faithful asset with
        # a generated one.
        raise SystemExit(
            f"{report_path} has no free-text report column for cohort "
            f"{args.cohort}. If this cohort already ships answers, it does not "
            "need generating -- generated text would displace published text.")

    client = LLMClient(endpoint=args.endpoint, model=args.model,
                       sampling=SamplingParams(temperature=args.temperature,
                                               max_tokens=args.max_tokens,
                                               seed=args.seed))
    client.verify()
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))

    output = REPO_ROOT / "text_prompts" / "wsi_five" / f"{args.cohort}_report_answers.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = client.provenance(prompt_template=ANSWER_TEMPLATE_ID)

    columns = ["case_id", "answer"] + [f"q{i}" for i in range(1, len(questions) + 1)]
    written = failed = 0
    items = sorted(reports.items())
    if args.limit:
        items = items[: args.limit]

    with output.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"# {json.dumps(provenance)}\n")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, (case, report) in enumerate(items, 1):
            prompt = ANSWER_USER.format(report=report[: args.report_chars],
                                        questions=numbered,
                                        count=len(questions))
            try:
                answer = client.complete([
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": prompt}]).strip()
            except LLMUnavailableError as error:
                print(f"  ! {case}: {error}", file=sys.stderr)
                failed += 1
                continue
            parts = [p.strip() for p in answer.split(";")]
            # A short answer means the model did not respond per question; pad
            # with Unknown so the column count stays fixed and the shortfall is
            # visible rather than shifting later answers into the wrong slot.
            parts = (parts + ["Unknown"] * len(questions))[: len(questions)]
            row = {"case_id": case, "answer": "; ".join(parts)}
            row.update({f"q{i}": p for i, p in enumerate(parts, 1)})
            writer.writerow(row)
            written += 1
            if index % 25 == 0:
                print(f"  {index}/{len(items)} cases", flush=True)

    print(f"wrote {output} ({written} cases, {failed} failed)")
    print(f"  provenance: {provenance['_model']}")
    return output


TASKS = {"wsi_five_answers": generate_wsi_five_answers}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--model", required=True,
                        help="registered model name; see common/llm/registry.py")
    parser.add_argument("--endpoint", required=True,
                        help="vLLM OpenAI endpoint, e.g. http://gpu042:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--report-chars", type=int, default=12000,
                        help="truncate each report to fit the context window")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N cases; use for a smoke run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TASKS[args.task](args)


if __name__ == "__main__":
    main()
