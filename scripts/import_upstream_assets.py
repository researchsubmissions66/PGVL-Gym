"""Import prompt and report assets supplied by the original method repos.

The upstream repositories remain the source of truth.  This utility copies
only their public, non-model assets into the layout used by this repository:

* CoD-MIL RCC prompt CSV and CLIP text embeddings;
* MAPLE Lung/RCC attribute JSONs (plus a flattened composite-compatible view);
* MSCPT GPT description JSONs;
* SLIP tissue lists converted from its Python constant to JSON; and
* WSI-FiVE report-preprocessing files, including ``TCGA_Reports.csv``;
* MUSE's released class-description CSVs; and
* SLDPC's zero-shot prompt YAMLs.

Examples
--------
Clone the sources once, then import from them::

    python scripts/import_upstream_assets.py --source-root /path/to/upstreams

Or let the script obtain shallow temporary clones itself::

    python scripts/import_upstream_assets.py --download

``--download`` requires network access and never writes a clone into this
repository.  Existing destination files are left untouched unless
``--overwrite`` is supplied.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAMS = {
    "cod-mil": "https://github.com/Jiangbo-Shi/CoD-MIL.git",
    "maple": "https://github.com/JJ-ZHOU-Code/MAPLE.git",
    "mscpt": "https://github.com/Hanminghao/MSCPT.git",
    "slip": "https://github.com/LTS5/SLIP.git",
    "wsi-five": "https://github.com/ls1rius/WSI_FiVE.git",
    "muse": "https://github.com/JiahaoXu-god/CVPR2026_MUSE.git",
    "sldpc": "https://github.com/linlu2022/SLDPC.git",
}


def copy_file(source: Path, destination: Path, overwrite: bool, dry_run: bool) -> bool:
    if not source.is_file():
        raise FileNotFoundError(f"Expected upstream asset is absent: {source}")
    if destination.exists() and not overwrite:
        print(f"skip  {destination.relative_to(REPO_ROOT)} (already exists)")
        return False
    print(f"copy  {source.name} -> {destination.relative_to(REPO_ROOT)}")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return True


def write_json(payload, destination: Path, overwrite: bool, dry_run: bool) -> bool:
    if destination.exists() and not overwrite:
        print(f"skip  {destination.relative_to(REPO_ROOT)} (already exists)")
        return False
    print(f"write {destination.relative_to(REPO_ROOT)}")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n")
    return True


def import_cod_mil(source_root: Path, overwrite: bool, dry_run: bool) -> None:
    prompt_dir = source_root / "cod-mil" / "prompt"
    target = REPO_ROOT / "text_prompts" / "cod_mil"
    copy_file(prompt_dir / "text_prompt_kidney_v2.csv",
              target / "rcc_chain_of_diagnosis.csv", overwrite, dry_run)
    # Named for the CLIP variant that produced the embeddings, not just "clip":
    # the registry distinguishes clip-rn50 from clip-vitb, and the cohort
    # protocols reference this exact filename. A bare "clip" name here would
    # leave `cod_prompt_features` dangling after a clean re-import.
    copy_file(prompt_dir / "text_prompt_feature_kidney_v2_clip.pt",
              target / "rcc_text_prompt_features_clip_rn50.pt", overwrite, dry_run)
    copy_file(prompt_dir / "text_prompt_feature_kidney_v2_plip.pt",
              target / "rcc_text_prompt_features_plip.pt", overwrite, dry_run)
    copy_file(prompt_dir / "text_prompt_feature_kidney_v2_quiltnet.pt",
              target / "rcc_text_prompt_features_quiltnet.pt", overwrite, dry_run)

    with (prompt_dir / "text_prompt_kidney_v2.csv").open(newline="") as handle:
        prompts = [row[0] for row in csv.reader(handle) if row]
    classes = [
        "clear cell renal cell carcinoma",
        "papillary renal cell carcinoma",
        "chromophobe renal cell carcinoma",
    ]
    if len(prompts) < 2 * len(classes):
        raise ValueError("CoD-MIL RCC prompt CSV has an unexpected layout")
    hierarchy = {
        name: {"broad": [prompts[index]], "specific": [prompts[index + len(classes)]]}
        for index, name in enumerate(classes)
    }
    write_json(hierarchy, target / "rcc_chain_of_diagnosis.json", overwrite, dry_run)


def flatten_maple_attributes(payload: dict) -> dict:
    """Create the class -> attributes view used by the composite prompt module."""
    result: dict[str, list[str]] = {}
    for level in ("low", "high"):
        for entity in payload.get(level, {}).get("entities", []):
            for class_name, description in entity.get("attributes", {}).items():
                result.setdefault(class_name, []).append(description)
        for class_name, description in payload.get(level, {}).get("global_info", {}).items():
            result.setdefault(class_name, []).append(description)
    return result


def import_maple(source_root: Path, overwrite: bool, dry_run: bool) -> None:
    base = source_root / "maple" / "templete" / "maple"
    target = REPO_ROOT / "text_prompts" / "maple"
    for upstream_name, local_name in (("lung", "LUNG"), ("rcc", "RCC"), ("brca", "BRCA")):
        source = base / upstream_name / f"{upstream_name}_8.json"
        copy_file(source, target / f"{local_name}_attributes.json", overwrite, dry_run)
        payload = json.loads(source.read_text())
        write_json(flatten_maple_attributes(payload),
                   target / f"{local_name}_composite_attributes.json",
                   overwrite, dry_run)


def import_mscpt(source_root: Path, overwrite: bool, dry_run: bool) -> None:
    source = source_root / "mscpt" / "train_data" / "gpt"
    target = REPO_ROOT / "train_data" / "gpt"
    for file in sorted(source.rglob("*.json")):
        copy_file(file, target / file.relative_to(source), overwrite, dry_run)


def load_slip_prompts(source: Path) -> dict:
    """Safely read the literal ``PROMPTS`` assignment without importing code."""
    module = ast.parse(source.read_text())
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PROMPTS"
                for target in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError(f"PROMPTS assignment not found in {source}")


def import_slip(source_root: Path, overwrite: bool, dry_run: bool) -> None:
    prompts = load_slip_prompts(source_root / "slip" / "datasets" / "prompt.py")
    target = REPO_ROOT / "text_prompts" / "slip"
    for dataset, data in prompts.items():
        # The composite prompt module accepts a simple list of strings.  Keep
        # each tissue name and its description together so no information is
        # lost during conversion.
        tissues = [f"{name}: {description}"
                   for name, description in data["tissue_classnames"]]
        write_json(tissues, target / f"{dataset}_tissues.json", overwrite, dry_run)


def import_wsi_five(source_root: Path, overwrite: bool, dry_run: bool) -> None:
    source = source_root / "wsi-five" / "gpt_preprocess"
    target = REPO_ROOT / "methods" / "wsi_five" / "gpt_preprocess"
    names = (
        "TCGA_Reports.csv",
        "LUAD_report.csv",
        "LUSC_report.csv",
        "LUAD_report_answer_v7_471.xlsx",
        "LUSC_report_answer_v1_468.xlsx",
        "luad_tcga_pub_clinical_data.tsv",
        "gpt_deal_wsi.ipynb",
    )
    for name in names:
        copy_file(source / name, target / name, overwrite, dry_run)


def import_muse(source_root: Path, overwrite: bool, dry_run: bool) -> None:
    """Import the prompt banks used by MUSE's supported public tasks."""
    source = source_root / "muse" / "text_prompt"
    target = REPO_ROOT / "text_prompts" / "muse"
    tasks = {
        "camelyon_all": ("generated_new_0.csv", "generated_new_1.csv"),
        "tcga_nsclc": ("generated_new_0.csv", "generated_new_1.csv"),
        "tcga_brca": ("generated_new_0.csv", "generated_new_1.csv"),
    }
    for task, names in tasks.items():
        for name in names:
            copy_file(source / task / name, target / task / name, overwrite, dry_run)


def import_sldpc(source_root: Path, overwrite: bool, dry_run: bool) -> None:
    """Keep the released SLDPC zero-shot textual templates beside its configs."""
    source = source_root / "sldpc" / "data" / "datasets" / "zero_shot_prompts"
    target = REPO_ROOT / "text_prompts" / "sldpc"
    for prompt in sorted(source.glob("*.yaml")):
        copy_file(prompt, target / prompt.name, overwrite, dry_run)


def clone_sources(root: Path) -> None:
    for name, url in UPSTREAMS.items():
        print(f"clone {url}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(root / name)], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        help="Directory containing cod-mil/, maple/, mscpt/, slip/, wsi-five/, muse/, and sldpc/")
    parser.add_argument("--download", action="store_true",
                        help="Create shallow temporary clones of the official public repositories")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace already-imported destination files")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.source_root) == bool(args.download):
        raise SystemExit("Specify exactly one of --source-root or --download")

    if args.download:
        with tempfile.TemporaryDirectory(prefix="unified_wsi_assets_") as raw_root:
            source_root = Path(raw_root)
            clone_sources(source_root)
            run_imports(source_root, args.overwrite, args.dry_run)
    else:
        run_imports(args.source_root, args.overwrite, args.dry_run)


def run_imports(source_root: Path, overwrite: bool, dry_run: bool) -> None:
    required = [source_root / name for name in UPSTREAMS]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing upstream checkout(s): " + ", ".join(missing))
    import_cod_mil(source_root, overwrite, dry_run)
    import_maple(source_root, overwrite, dry_run)
    import_mscpt(source_root, overwrite, dry_run)
    import_slip(source_root, overwrite, dry_run)
    import_wsi_five(source_root, overwrite, dry_run)
    import_muse(source_root, overwrite, dry_run)
    import_sldpc(source_root, overwrite, dry_run)


if __name__ == "__main__":
    main()
