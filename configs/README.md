# Configs

Each method has its own subfolder; each YAML inside is one runnable
experiment.

```
configs/
├── _defaults.yaml          # documentation of common keys (not used at runtime)
├── focus/{ubc,lung,rcc}.yaml
├── vila_mil/{ubc,lung,rcc}.yaml
├── cod_mil/{ubc,lung,rcc}.yaml
├── maple/{ubc,lung,rcc}.yaml
├── mscpt/{ubc,lung,rcc}_{clip,plip,conch}.yaml         # 3 datasets × 3 backbones = 9
├── pathpt/{ubc,lung,rcc}_{plip,conch,keep,musk}.yaml   # 3 datasets × 4 backbones = 12
├── top/{ubc,lung,rcc}.yaml
├── slip/{ubc,lung,rcc}.yaml
├── wsi_five/{ubc,lung,rcc}.yaml
├── muse/lung.yaml
├── convlm/lung_zsl.yaml
└── sldpc/lung.yaml
```

45 configs in total. Run any with:

```bash
python train.py --method pathpt --config configs/pathpt/ubc_keep.yaml
```

## Recipe philosophy

Base YAMLs mirror the training recipe from the original paper's release script
for that method. Generalized benchmark configs keep the method recipe but may
compile task prompts for cohorts not released by the authors; those configs
record `prompt_provenance` and must be reported as extensions rather than exact
prompt-asset reproductions.

ViLa-MIL configs use the released headerless one-column layout: all low-scale
class prompts, then all high-scale prompts. Lung and RCC point to pinned exact
upstream copies; UBC-OCEAN is a generated extension and declares its positional
file-class order and hashes explicitly. FOCUS uses the same native positional
layout but keeps method-owned assets and provenance: Lung and UBC-OCEAN are
exact upstream copies, while RCC is a generated task extension. The two methods'
files are not interchangeable merely because their schemas match.

TOP configs bind the task-agnostic instance bank and any task-specific bag
initializer independently with file and semantic hashes. They also declare the
bag usage (`standard_upstream_recipe` versus an explicit alternative); runtime
and doctor derive the combined provenance from those validated roles.

| Method   | Reference script in upstream repo                       |
| -------- | ------------------------------------------------------- |
| FOCUS    | `UBC-OCEAN.sh`, `LUAD_LUSC.sh`, `camelyon.sh`           |
| ViLa-MIL | `ViLa-MIL.sh` from `Jiangbo-Shi/ViLa-MIL`               |
| CoD-MIL  | `main.py` and `prompt/` from `Jiangbo-Shi/CoD-MIL`       |
| MAPLE    | `run.sh` from `JJ-ZHOU-Code/MAPLE`                      |
| MSCPT    | `scripts/mscpt/train_my_*.sh` from `Hanminghao/MSCPT`   |
| PathPT   | `train.py` defaults from `MAGIC-AI4Med/PathPT`          |
| TOP      | `exp_TCGA.sh` from `miccaiif/TOP`                       |
| SLIP     | `main.py` defaults from `LTS5/SLIP`                     |
| WSI-FiVE | `configs/wsi/fix_pth.yaml` from `ls1rius/WSI_FiVE`      |
| MUSE     | `models/model_text_retrevial.py` from `JiahaoXu-god/CVPR2026_MUSE` |
| ConVLM   | `convlm.py` and `train_function.py` from `BasitAlawode/ConVLM` |
| SLDPC    | `sldpc/trainers/{stage1,stage2}_trainer.py` from `linlu2022/SLDPC` |

For methods that benchmark several backbones with **identical
hyperparameters** (PathPT), every backbone variant is byte-equivalent
except for `backbone` and `feature_root` -- this is the paper's
locked fair-comparison recipe.

For methods that **vary recipe per backbone** (MSCPT: CLIP=100 epochs,
PLIP/CONCH=50 epochs), each variant captures the original.

## Regenerating

If you want to change the dataset class names, paths, or recipe in
bulk, edit `scripts/generate_configs.py` and rerun:

```bash
python scripts/generate_configs.py --force
```

The script regenerates the entire matrix from a single source of
truth.

## Notable hyperparameter contrasts

| Method   | LR      | Epochs   | Optimizer | Scheduler                   |
| -------- | ------- | -------- | --------- | --------------------------- |
| PathPT   | 1e-4    | 20       | Adam      | cosine + 10% warmup (locked across backbones) |
| FOCUS    | 1e-4    | 200 max  | Adam      | ReduceLROnPlateau           |
| ViLa-MIL | 1e-4    | 200 max  | Adam      | ReduceLROnPlateau           |
| MAPLE    | 2e-4    | 200 max  | Adam      | (none)                      |
| MSCPT    | 1e-4    | 50–100*  | Adam      | configurable                |
| TOP      | **0.02**| **8000** | Adam      | (none)                      |
| SLIP     | 2e-3    | 10       | Adam      | (none)                      |
| WSI-FiVE | 8e-6    | 30       | Adam      | cosine                      |
| MUSE     | 1e-4    | 200 max  | Adam      | ReduceLROnPlateau           |
| ConVLM   | 1e-4    | 40       | Adam      | MultiStep (10/20/30)        |
| SLDPC    | 1e-3    | 50 + 50  | AdamW     | none                        |

*MSCPT: CLIP=100, PLIP/CONCH=50.*
