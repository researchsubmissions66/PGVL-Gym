import hashlib
import json
from pathlib import Path

from common.preflight import preflight
from common.prompts.maple import load_maple_prompt_bank


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPLE_ROOT = REPO_ROOT / "text_prompts" / "maple"


def test_released_maple_banks_are_pinned_exact_copies():
    expected = {
        "LUNG_attributes.json": (
            ["lung adenocarcinoma", "lung squamous cell carcinoma"],
            "e35fb110800949ac9e7067098ed9bb6ebcd5534fa79b27ac882ae86dd1eb435f",
        ),
        "RCC_attributes.json": (
            [
                "clear cell renal cell carcinoma",
                "papillary renal cell carcinoma",
                "chromophobe renal cell carcinoma",
            ],
            "b963535ef8143e4c957aa25ed34ab68299d74cfd770c0bdf5bd2badd5acca8cc",
        ),
        "BRCA_attributes.json": (
            ["invasive ductal carcinoma", "invasive lobular carcinoma"],
            "2edf3587a5e7e96f6cd06a06863ae66d82f0b40aeb969183a2901eb6415683d6",
        ),
    }
    for name, (classnames, digest) in expected.items():
        path = MAPLE_ROOT / name
        bank = load_maple_prompt_bank(path, classnames=classnames)
        assert bank.entity_counts == (8, 8)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_generated_ubc_bank_is_explicit_and_complete():
    classnames = ["HGSC", "LGSC", "EC", "CC", "MC"]
    bank = load_maple_prompt_bank(
        MAPLE_ROOT / "UBC_attributes.json", classnames=classnames)

    assert bank.provenance == "generated"
    assert bank.classnames == tuple(classnames)
    assert bank.entity_counts == (1, 1)
    assert bank.digest == (
        "700a091433fb3c5b654cbcf3082cd5757bf42f77c29079ca93aa95e8b7135e60"
    )


def test_doctor_rejects_maple_class_key_reordering(tmp_path: Path):
    source = MAPLE_ROOT / "LUNG_attributes.json"
    payload = json.loads(source.read_text())
    for level in ("low", "high"):
        payload[level]["global_info"] = dict(
            reversed(list(payload[level]["global_info"].items())))
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(payload))

    report = preflight({
        "method": "maple",
        "text_prompt_path": str(path),
        "classnames": [
            "lung adenocarcinoma", "lung squamous cell carcinoma"
        ],
    }, checks={"prompts"})

    assert any("class order" in problem for problem in report.problems)


def test_doctor_rejects_maple_provenance_drift():
    report = preflight({
        "method": "maple",
        "text_prompt_path": str(MAPLE_ROOT / "UBC_attributes.json"),
        "classnames": ["HGSC", "LGSC", "EC", "CC", "MC"],
        "prompt_provenance": "upstream",
    }, checks={"prompts"})

    assert any("prompt_provenance contradicts" in problem
               for problem in report.problems)
