# Contributing & Repo Map

## Provenance map

This codebase consolidates code from nine upstream repos. For
auditability, this table records exactly which file came from where
and whether it was modified.

### `common/` (deduplicated CLAM/CoOp scaffolding)

| Unified path                             | Origin                              | Modifications                                   |
| ---------------------------------------- | ----------------------------------- | ----------------------------------------------- |
| `common/wsi_core/WholeSlideImage.py`     | CLAM via `focus/wsi_core/`          | Imports patched to `common.*`                   |
| `common/wsi_core/batch_process_utils.py` | CLAM via `focus/wsi_core/`          | none                                            |
| `common/wsi_core/util_classes.py`        | CLAM via `focus/wsi_core/`          | none                                            |
| `common/wsi_core/wsi_utils.py`           | CLAM via `focus/wsi_core/`          | none                                            |
| `common/datasets/dataset_generic.py`     | CLAM via `focus/datasets/`          | imports patched                                 |
| `common/datasets/dataset_h5.py`          | CLAM via `focus/datasets/`          | none                                            |
| `common/datasets/wsi_dataset.py`         | CLAM via `focus/datasets/`          | none                                            |
| `common/datasets/BatchWSI.py`            | CLAM via `focus/datasets/`          | none                                            |
| `common/datasets/bag_features.py`        | NEW (this codebase)                 | n/a                                             |
| `common/utils/core_utils.py`             | CLAM via `focus/utils/`             | imports patched                                 |
| `common/utils/utils.py`                  | CLAM via `focus/utils/`             | none                                            |
| `common/utils/file_utils.py`             | CLAM via `focus/utils/`             | none                                            |
| `common/utils/loss_utils.py`             | CLAM via `focus/utils/`             | none                                            |
| `common/utils/eval_utils.py`             | CLAM via `focus/utils/`             | imports patched                                 |
| `common/models/_clam_blocks.py`          | `focus/models/model_utils.py`       | renamed                                         |
| `common/models/mil_baselines.py`         | `focus/models/model_mil.py`         | renamed                                         |
| `common/models/coop.py`                  | NEW (canonical CoOp)                | merged from FOCUS, ViLa-MIL, SLIP               |
| `common/models/transmil.py`              | NEW (canonical TransMIL block)      | merged from SLIP, PathPT                        |
| `common/backbones/factory.py`            | NEW                                 | replaces per-repo backbone loaders              |

### `clip/` (verbatim OpenAI CLIP source, shared by TOP/SLIP/WSI-FiVE)

* `clip/clip.py`, `clip/model.py`, `clip/simple_tokenizer.py`,
  `clip/bpe_simple_vocab_16e6.txt.gz` – from `slip/clip/` (the
  variant that has the prompt-tuning hooks).

### `methods/` (per-paper adapters + vendored model files)

| Unified path                                | Origin                                                |
| ------------------------------------------- | ----------------------------------------------------- |
| `methods/base.py`                           | NEW                                                   |
| `methods/__init__.py`                       | NEW (registry)                                        |
| `methods/focus/model.py`                    | `dddavid4real/FOCUS/models/model_FOCUS.py`            |
| `methods/focus/model_utils.py`              | NEW shim → `common.models._clam_blocks`               |
| `methods/focus/adapter.py`                  | NEW                                                   |
| `methods/vila_mil/model.py`                 | `Jiangbo-Shi/ViLa-MIL/models/model_ViLa_MIL.py`       |
| `methods/vila_mil/model_utils.py`           | NEW shim                                              |
| `methods/vila_mil/adapter.py`               | NEW                                                   |
| `methods/cod_mil/model.py`                  | `Jiangbo-Shi/CoD-MIL/models/model_CoT.py`             |
| `methods/cod_mil/model_utils.py`            | `Jiangbo-Shi/CoD-MIL/models/model_utils.py`           |
| `methods/cod_mil/{configs,resnet_custom,vision_transformer}.py` | `Jiangbo-Shi/CoD-MIL/models/` |
| `methods/cod_mil/adapter.py`                | NEW                                                   |
| `methods/maple/maple_model/`                | `JJ-ZHOU-Code/MAPLE/models/maple/`                    |
| `methods/maple/_orig_trainer/`              | `JJ-ZHOU-Code/MAPLE/trainer/` (kept for reference)    |
| `methods/maple/adapter.py`                  | NEW                                                   |
| `methods/mscpt/mscpt_model/`                | `Hanminghao/MSCPT/model/`                             |
| `methods/mscpt/dataset.py`                  | NEW                                                   |
| `methods/mscpt/adapter.py`                  | NEW                                                   |
| `methods/pathpt/pathpt_models/`             | `MAGIC-AI4Med/PathPT/models/`                         |
| `methods/pathpt/subtyping/`                 | `MAGIC-AI4Med/PathPT/subtyping/` (kept for reference) |
| `methods/pathpt/wsi_selecters/`             | `MAGIC-AI4Med/PathPT/wsi_selecters/`                  |
| `methods/pathpt/{params,loss,evaluation,WSI_dataset}.py` | `MAGIC-AI4Med/PathPT/`                  |
| `methods/pathpt/dataset.py`                 | NEW                                                   |
| `methods/pathpt/adapter.py`                 | NEW (locks recipe across backbones)                   |
| `methods/top/learnable_prompt.py`           | `miccaiif/TOP/models/learnable_prompt.py`             |
| `methods/top/{_util,_utliz}.py`             | `miccaiif/TOP/`                                       |
| `methods/top/adapter.py`                    | NEW                                                   |
| `methods/slip/{networks,methods}/`          | `LTS5/SLIP/{networks,methods}/`                       |
| `methods/slip/_datasets/`                   | `LTS5/SLIP/datasets/` (renamed to avoid shadowing)    |
| `methods/slip/adapter.py`                   | NEW                                                   |
| `methods/wsi_five/wsi_five_models/`         | `ls1rius/WSI_FiVE/models/`                            |
| `methods/wsi_five/_{datasets,utils,configs}/`| `ls1rius/WSI_FiVE/{datasets,utils,configs}/`         |
| `methods/wsi_five/gpt_preprocess/`          | `ls1rius/WSI_FiVE/gpt_preprocess/` (large CSV dropped) |
| `methods/wsi_five/dataset.py`               | NEW                                                   |
| `methods/wsi_five/adapter.py`               | NEW                                                   |

### Top-level entry points

* `train.py`, `eval.py` – NEW (this codebase)
* `preprocess.py` – `Jiangbo-Shi/ViLa-MIL/create_patches_fp.py` (CLAM)
* `extract_features.py` – `Jiangbo-Shi/CoD-MIL/extract_features_fp.py` (CLAM)
* `scripts/create_splits_seq.py` – CLAM via ViLa-MIL
* `scripts/create_splits_fewshot.py` – CLAM via ViLa-MIL (NUL bytes stripped)
* `scripts/sanity_check.py` – NEW

## Style

* All Python ≥ 3.9.
* Black-compatible formatting (line length 88, no enforcement).
* Run `python scripts/sanity_check.py` before opening a PR.

## Open issues / TODOs

The following items are intentionally left as **stub / best-effort**
in this consolidation pass; PRs welcome:

1. PathPT's vendored `PromptLearner*` classes expect a `param` dict
   with several keys (`learnable`, `vision_only`, ...) that are
   passed unchanged from the YAML; not all combinations have been
   exercised end-to-end.
2. CoD-MIL uses external CLIP text-prompt features stored on disk
   (`map_10x_20x_files/`); the adapter ships a placeholder
   `_prepare_text_features` that the user should replace with the
   correct precomputation logic (see CoD-MIL's `prompt/` folder).
3. WSI-FiVE's `build_model` helper looks for an upstream `build.py`
   that wasn't part of the original release; the adapter falls back
   to direct `FiVE(...)` instantiation with default args.
4. SLIP's `compute_loss` is contrastive over slide-level prototypes;
   the train loop uses `compute_loss` directly and ignores the
   generic `loss_fn`. This matches the upstream behaviour but means
   you cannot swap loss functions for SLIP via the YAML.
