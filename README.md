# Unified WSI Vision-Language Codebase

A single, registry-based codebase that runs twelve recent few-shot
whole-slide-image (WSI) vision-language methods through one entry
point.

Systematic protocols cover TCGA NSCLC/BRCA/RCC plus
[CAMELYON16 and UBC-OCEAN](benchmarks/additional_tasks/README.md).

## Configure local paths

The committed protocols and generated examples use anonymous placeholders such
as `/path/to/PGVL-Gym`, `/path/to/features`, `/path/to/metadata`, and
`/path/to/model-cache`. Before generating or launching runs, update the two
protocol files for your own storage layout:

- `benchmarks/tcga/protocol.yaml`
- `benchmarks/additional_tasks/protocol.yaml`

Then regenerate manifests, splits, run configs, and readiness reports with
`python scripts/tcga_benchmark.py all --protocol <protocol>`. Do not infer a
cohort from whatever feature files happen to exist.

| Method     | Paper venue          | Validated/default encoder boundary |
| ---------- | -------------------- | ---------------------------------- |
| FOCUS      | CVPR 2025            | CONCH                              |
| ViLa-MIL   | CVPR 2024            | CLIP RN50                          |
| CoD-MIL    | TMI 2024             | precomputed CLIP RN50 space        |
| MAPLE      | NeurIPS 2025         | Hugging Face CLIP / PLIP           |
| MSCPT      | TMI 2025             | Hugging Face CLIP / PLIP / CONCH   |
| PathPT     | Nat. Commun. 2026    | PLIP / CONCH / KEEP / MUSK         |
| TOP        | NeurIPS 2023         | CLIP RN50                          |
| SLIP       | ISBI 2025            | CLIP / BiomedCLIP / PLIP           |
| WSI-FiVE   | CVPR 2024            | method-owned ViT + report tower    |
| MUSE       | CVPR 2026            | CONCH default; `text_encode` API   |
| ConVLM     | —                    | method-owned ViT + QuiltNet attributes |
| SLDPC      | —                    | TITAN default; paired slide-text API |

## Layout

```
unified_wsi_vlm/
├── train.py                # one entry point for all methods
├── eval.py                 # evaluation-only entry point
├── preprocess.py           # CLAM patching script
├── extract_features.py     # CLAM feature extraction script
├── clip/                   # OpenAI CLIP source (shared by TOP/SLIP/WSI-FiVE)
├── common/
│   ├── wsi_core/           # CLAM tissue segmentation & patching
│   ├── datasets/           # Generic_MIL_Dataset + bag_features
│   ├── utils/              # core_utils (EarlyStopping, Accuracy_Logger), file/loss utils
│   ├── models/
│   │   ├── coop.py             # canonical PromptLearner + TextEncoder
│   │   ├── transmil.py         # TransLayer + PPEG (Nyström attention)
│   │   ├── _clam_blocks.py     # Attn_Net, Attn_Net_Gated, MultiheadAttention
│   │   └── mil_baselines.py    # MIL_fc, MIL_fc_mc
│   └── backbones/
│       ├── interfaces.py       # capabilities, feature provenance, method contracts
│       └── factory.py          # build_encoder(name) + legacy build_backbone(name)
├── methods/                 # one folder per paper, each with adapter.py + vendored model files
│   ├── base.py                  # BaseMethod abstract class
│   ├── __init__.py              # registry
│   ├── focus/
│   ├── vila_mil/
│   ├── cod_mil/
│   ├── maple/
│   ├── mscpt/
│   ├── pathpt/
│   ├── top/
│   ├── slip/
│   ├── wsi_five/
│   ├── muse/
│   ├── convlm/
│   └── sldpc/
└── configs/                 # one YAML per method (and per backbone for PathPT)
```

## What's shared vs. unique

The codebase explicitly keeps the **common parts together** and
**isolates the unique parts**:

* **CLAM scaffolding** (`wsi_core/`, `Generic_MIL_Dataset`,
  `EarlyStopping`, `collate_MIL`, `get_split_loader`,
  `Attn_Net_Gated`) – byte-identical across FOCUS / ViLa-MIL /
  CoD-MIL / MAPLE in their original repos. Stored once in `common/`.
* **OpenAI CLIP source** (tokenizer, vocab file, ViT/RN50 model code) –
  byte-identical between TOP, SLIP, and WSI-FiVE. Stored once at
  `./clip/`.
* **CoOp prompt-learning blocks** (`PromptLearner`, `TextEncoder`) –
  recur in seven of the original nine repos with the same structure. Stored
  once at `common/models/coop.py`.
* **TransMIL Nyström attention** – byte-identical between SLIP and
  PathPT. Stored once at `common/models/transmil.py`.
* **Backbone loading and validation** – CLIP / PLIP / CONCH / MUSK /
  KEEP / BiomedCLIP / TITAN are exposed as capability-aware encoder bundles
  from `common/backbones/`. The old `(model, tokenizer, info)` loader remains
  available to vendored code that needs it.

The unique parts (e.g. the MAPLE entity GCN, the FOCUS visual
compression module, the PathPT per-backbone forward, the WSI-FiVE
MedCLIP / X-CLIP vision tower) live under their own `methods/<name>/`
folder, and are wired in through a thin `adapter.py` that implements
a uniform `BaseMethod` interface (`build_model`, `train_step`,
`eval_step`, optionally `build_optimizer` / `build_scheduler`).

## Backbone interfaces and safe swaps

Backbone compatibility is now explicit rather than inferred from a tensor
width or model name. `BackboneSpec` describes an encoder's dimensions,
feature-space provenance, and capabilities; `EncoderBundle` exposes only the
uniform operations the encoder actually supports while retaining
`raw_model`/`raw_tokenizer` for a paper's native implementation. Every method
adapter declares a `MethodBackboneContract`, which validates the config before
the model is constructed.

This interface layer does not rewrite the paper architectures. Allowlisted
methods still enter their original family-specific branches, and fixed methods
remain fixed. Capability-swappable methods accept another registered bundle
only when it implements every required operation. Inspect all boundaries
without loading a checkpoint:

```bash
python scripts/list_backbone_compatibility.py
python scripts/list_backbone_compatibility.py --method sldpc --json
```

SLDPC keeps native paired projection as its default, while explicitly selected
linear/MLP variants can align another registered slide-vector source to its
prompt space. MUSE independently registers an offline patch encoder and a
runtime prompt encoder: its learned visual adapter maps any declared static
patch width to the selected black-box text encoder's output width. See
[Backbone interfaces and swap boundaries](docs/BACKBONE_INTERFACES.md)
for the complete method matrix and extension examples.

## Training

```bash
# Configs are organised as configs/<method>/<dataset>[_<backbone>].yaml
# 42 configs covering 9 methods x 3 datasets (UBC-OCEAN, TCGA-Lung, TCGA-RCC)
# See configs/README.md for the full matrix.

# FOCUS on UBC-OCEAN
python train.py --method focus    --config configs/focus/ubc.yaml

# Fresh standard FOCUS+CONCH on patient-disjoint TCGA-RCC fold 0
python train.py --method focus \
  --config configs/focus/rcc_conch_fold0.yaml --device cuda:0

# PathPT with each foundation model -- IDENTICAL hyperparams
python train.py --method pathpt   --config configs/pathpt/ubc_keep.yaml
python train.py --method pathpt   --config configs/pathpt/ubc_conch.yaml
python train.py --method pathpt   --config configs/pathpt/ubc_musk.yaml
python train.py --method pathpt   --config configs/pathpt/ubc_plip.yaml

# Other methods (each on whatever dataset you choose)
python train.py --method vila_mil --config configs/vila_mil/lung.yaml
python train.py --method cod_mil  --config configs/cod_mil/rcc.yaml
python train.py --method maple    --config configs/maple/rcc.yaml
python train.py --method mscpt    --config configs/mscpt/ubc_conch.yaml
python train.py --method top      --config configs/top/lung.yaml
python train.py --method slip     --config configs/slip/lung.yaml
python train.py --method wsi_five --config configs/wsi_five/lung.yaml
python train.py --method muse     --config configs/muse/lung.yaml
python train.py --method convlm   --config configs/convlm/lung_zsl.yaml
python train.py --method sldpc    --config configs/sldpc/lung.yaml
```

The dual-scale CLAM loader also accepts native HDF5 bags directly from a
manifest. Set `feature_path_column_s`, `feature_path_column_l`, and optionally
`feature_key`; no intermediate `.pt` conversion or duplicated feature tree is
required.

## Dummy-feature smoke tests

Build every experiment variant from its generated configuration and run a
finite-logit forward pass without requiring dataset features:

```bash
# One representative 4-shot config for all 20 variants on CAMELYON16
conda run -n trident python -u scripts/smoke_test.py \
  --matrix benchmarks/additional_tasks/run_matrix.csv \
  --cohort camelyon16 --device cuda:0

# The same check for a TCGA cohort
conda run -n trident python -u scripts/smoke_test.py \
  --matrix benchmarks/tcga/run_matrix.csv \
  --cohort rcc --device cuda:0
```

Each model runs in an isolated subprocess with a configurable `--timeout`.
The report is written beside the selected matrix as
`smoke_report_<cohort>.json` and records the failure stage, traceback, logits
shape, runtime, trainable parameter count, and peak CUDA allocation.

## Documentation website

The documentation site combines curated guides with API reference generated
directly from the stable Python docstrings:

```bash
python -m pip install -r requirements-docs.txt
python scripts/check_docstrings.py
python -m mkdocs build --strict
python -m mkdocs serve
```

Start with [the documentation home](docs/index.md). The strict build and
public-API docstring audit run automatically in
`.github/workflows/docs.yml`; pushes to `main` or `master` publish the built
site through GitHub Pages.

## Evaluation

```bash
python eval.py --method pathpt \
               --config configs/pathpt_ubc_keep.yaml \
               --ckpt_dir results/pathpt_ubc_keep_10shot
```

## Hyperparameter Recipes

Each `configs/*.yaml` mirrors the **exact hyperparameters from the
original paper's released training script**.  Notable differences:

| Method   | LR        | Epochs    | Optimizer | Scheduler                       |
| -------- | --------- | --------- | --------- | ------------------------------- |
| PathPT   | 1e-4      | 20        | Adam      | cosine + 10% warmup (locked across PLIP/CONCH/KEEP/MUSK) |
| FOCUS    | 1e-4      | 200 max   | Adam      | ReduceLROnPlateau               |
| ViLa-MIL | 1e-4      | 200 max   | Adam      | ReduceLROnPlateau               |
| MAPLE    | 2e-4      | 200 max   | Adam      | (none)                          |
| MSCPT    | 1e-4      | 50–100*   | Adam      | configurable                    |
| TOP      | **0.02**  | **8000**  | Adam      | (none)                          |
| SLIP     | 2e-3      | 10        | Adam      | (none)                          |
| WSI-FiVE | 8e-6      | 30        | Adam      | cosine                          |
| MUSE     | 1e-4      | 200 max   | Adam      | ReduceLROnPlateau               |
| ConVLM   | 1e-4      | 40        | Adam      | MultiStep (10/20/30)            |
| SLDPC    | 1e-3      | 50 + 50   | AdamW     | none                            |

*MSCPT epochs vary by backbone: CLIP=100, PLIP=50, CONCH=50.*

## Why PathPT's recipe is locked across backbones

PathPT's main contribution is a **fair benchmark of foundation
models** for rare-cancer subtyping. Holding optimizer, LR, epochs,
prompt length, and loss weights constant ensures a backbone's number
reflects the encoder, not training-recipe variance. The PathPT
adapter (`methods/pathpt/adapter.py`) overrides
`build_optimizer` and `build_scheduler` to enforce the locked recipe
no matter which YAML you pass.

## Adding a new method

1. Create `methods/<my_method>/` with `__init__.py`.
2. Drop your model file(s) inside.
3. Write `methods/<my_method>/adapter.py` subclassing `BaseMethod` with
   `build_model`, `train_step`, `eval_step` (and optionally
   `build_optimizer` / `build_scheduler`).
4. Declare a `MethodBackboneContract` describing the input feature level,
   required capabilities, and whether the native code is capability-swappable,
   allowlisted, fixed, or based on precomputed aligned features.
5. Add a branch to `methods/__init__.py::get_method()`.
6. Write `configs/<my_method>_<dataset>.yaml`.
7. Run `python train.py --method <my_method> --config <yaml>`.

## Installation

```bash
git clone <this-repo>
cd unified_wsi_vlm
pip install -e .              # uses pyproject.toml
# or:
pip install -r requirements.txt
```

Optional, per-backbone:

* CONCH: `pip install git+https://github.com/Mahmood-Lab/CONCH.git`
  + HF token for `MahmoodLab/conch`
* MUSK: vendor the original `musk/` package or install upstream
* KEEP: `transformers >= 4.40` (loads via HF `Astaxanthin/KEEP`)
* PLIP: `transformers` + HF model `vinid/plip`

## Preprocessing pipeline

```bash
# 1. Tile WSIs into patches (CLAM)
python preprocess.py \
    --source /path/to/raw_wsi \
    --save_dir /path/to/patches \
    --patch_size 256 --step_size 256 --seg --patch

# 2. Extract features per patch using your chosen backbone
python extract_features.py \
    --data_h5_dir /path/to/patches \
    --csv_path /path/to/dataset.csv \
    --feat_dir /path/to/features \
    --model_name conch          # or plip / clip / keep / musk
```

## Importing upstream prompt and report assets

The source repos distribute some of the text priors and reports separately
from this consolidated tree.  Import the published assets with:

```bash
python scripts/import_upstream_assets.py --download
```

This installs the published CoD-MIL RCC prompts and CLIP embeddings, MAPLE
Lung/RCC/BRCA attribute JSONs, MSCPT GPT descriptions, SLIP tissue lists,
WSI-FiVE's raw TCGA report CSV, MUSE class-description CSVs, and SLDPC
zero-shot prompt templates. For new tasks, the benchmark prompt compiler can
derive method-native prompt files from one declarative task profile. CoD-MIL
can encode a generated chain at runtime, and ConVLM can encode generated
attributes with its configured QuiltNet tower. Data-derived assets such as
CoD-MIL cross-magnification maps must still be generated from patch coordinates.

## Method-specific data contracts

Paper-faithful MUSE uses a single bag of 512-dimensional CONCH patch features
and encodes the published per-class GPT descriptions with CONCH. Experimental
variants may instead use MUSK, KEEP, or another registered static patch source:
`patch_encoder` records its provenance, and the trainable visual adapter maps
`feature_dim` to the independently declared prompt encoder's `embed_dim`. A
different registered text encoder can also be selected when its shared output
width is used as `embed_dim`.

ConVLM is a patch-image zero-shot classifier, not a WSI-bag method. Its split
CSVs contain `image_path,label`; training labels must be a subset of
`seen_class_indices`, while evaluation compares against all attribute vectors.
Supply your own `[n_classes, attribute_dim]` QuiltNet embeddings from the
paper's Quilt-LLaVA descriptions—the release does not distribute a reusable
attribute bank.

Methods that declare `FeatureLevel.SLIDE_EMBEDDING` consume one registered
vector per slide through the shared loader. HDF5, torch, and stacked pickle
stores use the same exact-key and exact-width validation; the offline slide
encoder's checkpoint, feature space, resolution, and path are recorded
independently from any runtime prompt backbone. SLDPC is currently the only
registered method at this feature level. Its default `MahmoodLab/TITAN` path
uses TITAN's matching native projection. A different offline slide encoder may
instead select an explicit trainable `linear` or `mlp` alignment into the
prompt text space. Equal vector widths alone never establish compatibility,
and an adapter is never inserted implicitly.
Its run is deliberately two-stage: continuous prompt initialization (CPI),
then hard-negative sampling and symmetric InfoNCE (SICL) prompt refinement,
with the best validation prompt restored at each stage boundary.

Generate CoD-MIL maps from paired CLAM coordinate files before training:

```bash
python scripts/generate_cross_magnification_maps.py \
    --low-h5-dir /path/to/10x/h5_files \
    --high-h5-dir /path/to/20x/h5_files \
    --output-dir maps/TCGA_RCC_10x_to_20x \
    --low-patch-size 512 --high-patch-size 256
```

Patch and stride values must use the level-0 coordinate system stored in the
H5 files; add `--low-step-size` when the low-resolution patches overlap.

## Acknowledgments

This codebase consolidates code from the following twelve repositories.
All copyright remains with the original authors.

* [dddavid4real/FOCUS](https://github.com/dddavid4real/focus)
* [Jiangbo-Shi/ViLa-MIL](https://github.com/Jiangbo-Shi/ViLa-MIL)
* [Jiangbo-Shi/CoD-MIL](https://github.com/Jiangbo-Shi/CoD-MIL)
* [JJ-ZHOU-Code/MAPLE](https://github.com/JJ-ZHOU-Code/MAPLE)
* [Hanminghao/MSCPT](https://github.com/Hanminghao/MSCPT)
* [MAGIC-AI4Med/PathPT](https://github.com/MAGIC-AI4Med/PathPT)
* [miccaiif/TOP](https://github.com/miccaiif/TOP)
* [LTS5/SLIP](https://github.com/LTS5/SLIP)
* [ls1rius/WSI_FiVE](https://github.com/ls1rius/WSI_FiVE)
* [JiahaoXu-god/CVPR2026_MUSE](https://github.com/JiahaoXu-god/CVPR2026_MUSE)
* [BasitAlawode/ConVLM](https://github.com/BasitAlawode/ConVLM)
* [linlu2022/SLDPC](https://github.com/linlu2022/SLDPC)

The shared scaffolding is derived from
[mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) and
[KaiyangZhou/CoOp](https://github.com/KaiyangZhou/CoOp).
