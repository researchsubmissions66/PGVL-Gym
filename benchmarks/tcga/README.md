# TCGA NSCLC/BRCA/RCC benchmark

This benchmark freezes one patient-disjoint data protocol for every method and
encoder:

- five stratified outer folds;
- 4, 8, and 16 labeled patients per class for both training and validation;
- nested shot sets (`4-shot` is a subset of `8-shot`, which is a subset of
  `16-shot`);
- the same held-out patients and slides at every shot level;
- one deterministic slide per selected train/validation patient and all
  annotated slides for held-out patients;
- seed 1 for data selection and training.

Fold construction is feature-agnostic. It uses only cohort annotations,
patient IDs, and subtype labels. The frozen universes are 1,043 NSCLC slides
from 946 patients, 1,054 BRCA slides from 991 patients, and 873 RCC slides from
832 patients. Adding, removing, or relocating a feature set cannot change a
patient fold or shot selection.

## Resolution-aware feature registry

Each atomic `feature_sources` entry in `protocol.yaml` declares:

- its resolution, such as `5x`, `10x`, or `20x`;
- an encoder/backbone and exact feature-space identifier;
- the exact local encoder checkpoint used for runtime prompt/text encoding;
- expected tensor width and HDF5/torch feature key;
- one `{slide_id}` path template.

Experiments bind semantic roles independently. Single-scale methods bind
`bag`; dual-scale methods bind `low` and `high`. For example, FOCUS can bind
5x/20x while another experiment binds 10x/20x, without changing any fold. The
generator maps those roles into each method's native configuration and rejects
incompatible backbone, feature-space, width, key, or role combinations before
loading a model. It supports HDF5 and torch bags without copying or renaming
files.

The current registry covers all 13 method families in PGVL-Gym: FOCUS, MAPLE,
MSCPT, MUSE, PathPT, ViLa-MIL, CoD-MIL, TOP, SLIP, WSI-FiVE, ConVLM, SLDPC,
and Composite. PathPT includes separate CONCH, MUSK, and KEEP variants. Every
method is generated across NSCLC, BRCA, and RCC at the same three shot levels.
More encoders can be added by registering a feature set and pointing an
experiment at it; folds remain unchanged.

Inputs are typed rather than assumed to be one HDF5 layout:

| Input kind | Methods | Registered source |
| --- | --- | --- |
| Patch bag | FOCUS, MAPLE, MSCPT, MUSE, PathPT, ViLa-MIL, CoD-MIL, TOP, SLIP, Composite | Resolution-specific HDF5 or torch features |
| Slide embedding | SLDPC currently; any method declaring `FeatureLevel.SLIDE_EMBEDDING` | Registered per-slide vector store |
| Patch sequence | WSI-FiVE | Method-owned MedCLIP-style sequence plus clinical report |
| Raw tile directory | ConVLM | Per-slide RGB tile directory |

Future-facing entries remain valid configurations even when their source data
has not been extracted. In particular, ImageNet ResNet-50 features are not
relabeled as OpenAI CLIP-RN50 features, and unrelated UCCA checkpoints or
features are never used as substitutes.

The slide-embedding loader and learned alignment layers are framework-level
components under `common/`, not SLDPC-owned utilities. Unified training routes
every current or future `FeatureLevel.SLIDE_EMBEDDING` adapter through the same
HDF5/torch/pickle, feature-key, dimension, and provenance checks. Methods that
consume patch bags or construct a slide representation internally are not
misclassified as cached slide-vector consumers.

For example, a new resolution-specific feature store requires only an atomic
source plus a compatible experiment binding:

```yaml
feature_sources:
  musk_20x_new_store:
    resolution: 20x
    backbone: musk
    feature_key: features
    feature_dim: 1024
    feature_space_id: hf:xiangjx/musk
    path_template: /path/to/features/{slide_id}.h5

experiments:
  pathpt_musk20x:
    method: pathpt
    features: {bag: musk_20x_new_store}
    epochs: 20
```

Dual-scale experiments bind sources independently:

```yaml
experiments:
  focus_5x20x:
    method: focus
    features: {low: conch_v1_5x, high: conch_v1_20x}
    epochs: 200
```

The low and high sources may use any declared resolutions, but must satisfy
the selected method's backbone, feature-space, tensor-width, and feature-key
contract. Numeric magnifications are checked so `low` cannot exceed `high`.

### Generated-config guarantees

Every generated YAML contains all runtime provenance rather than relying on
method defaults:

- `encoder.name`, `encoder.weights`, `encoder.feature_space_id`, and
  `encoder.feature_dim`;
- role-specific `feature_sources`, `feature_resolutions`, path columns, and
  source directories;
- an explicit prompt source and the exact prompt file or class-template mode;
- the cohort manifest, frozen split directory, class order, label mapping,
  shot count, fold count, seed, and results directory.

Generation fails if an encoder checkpoint is missing, an encoder space or
dimension disagrees with its registered feature source, a prompt path is
missing, or a prompt schema/class order disagrees with the cohort. In
particular, MSCPT descriptions are checked against the actual subtype labels;
the TCGA-BRCA benchmark uses an IDC/ILC description bank and cannot silently
fall back to the unrelated High/Low grade prompts.

`run_matrix.csv` exposes `encoder_ready`, `auxiliary_ready`, `config_valid`,
`prompt_source`, and `prompt_asset` in addition to feature availability.
`missing_feature_files` and `missing_auxiliary_files` are counted separately.
This matters for methods such as CoD-MIL, which needs both feature bags and a
cross-scale map, and WSI-FiVE, which needs both patch sequences and reports.

### Dimension handling

Feature width is declared by each atomic source, copied into the generated
model configuration, and checked against an actual feature bag during
validation. Loaders themselves accept any non-empty two-dimensional patch
tensor and do not reshape its last dimension.

Dimension adaptation remains method-specific:

- MUSE independently registers its static `patch_encoder` and runtime prompt
  encoder. Its learned `feature_dim -> embed_dim` visual adapter supports
  CONCH, MUSK, KEEP, or another declared patch width without relabeling the
  input feature space. `embed_dim` remains the prompt encoder's shared width.
- FOCUS has a learned `feature_dim -> 512` input projection, while retaining
  its CONCH text-space requirements.
- PathPT selects a native backbone branch and accepts that branch's declared
  width: PLIP/CONCH 512, KEEP 768, or MUSK 1024.
- MAPLE and MSCPT currently require their native 512-wide feature contracts.
- ViLa-MIL and CoD-MIL use the declared 1024-wide CLIP-RN50 feature space;
  CoD-MIL additionally validates its class prompt tensor and cross-scale map.
- TOP, SLIP, and Composite consume their declared CLIP-RN50 patch bags and
  reject look-alike ResNet feature spaces.
- SLDPC consumes one registered slide embedding rather than a patch bag. Its
  offline `slide_encoder` and runtime `prompt_encoder` are independent. TITAN
  uses its frozen native paired projection; another slide encoder may emit any
  declared width and use a trainable `linear` or `mlp` projection into the
  prompt text space.
- WSI-FiVE projects its declared 512-wide patch sequence into a
  report-conditioned transformer space.
- ConVLM encodes RGB tiles and averages normalized tile embeddings into one
  slide representation, so evaluation still reports one prediction per slide.

The TCGA MSCPT adapter explicitly runs in
`precomputed_shared_features` mode: both low- and high-resolution inputs are
512-wide bags already projected by the declared paired VLM. The raw-tile
deep-vision-prompt branch is disabled in this mode; the multi-scale text and
graph prompt components remain trainable. This avoids treating projected HDF5
features as raw RGB tiles.

No generic projection is inserted for a method that does not define one. An
unsupported width or feature-space pairing is rejected rather than coerced.

### Swapping the SLDPC slide encoder

Register the new offline embedding source without adding it to the runtime
backbone factory. `runtime_encoder: false` means training reads its cached
slide vectors but does not load that encoder:

```yaml
feature_sources:
  my_slide_encoder_5x:
    input_kind: slide_embedding
    runtime_encoder: false
    resolution: 5x
    backbone: my-slide-encoder
    encoder_weights: /models/my-slide-encoder/checkpoint.pt
    feature_key: slide_vector
    feature_dim: 1536
    feature_space_id: local:my-slide-encoder@checkpoint-id
    path_template: /features/my-slide-encoder/{slide_id}.pt
```

Point an SLDPC experiment at that source and select an explicit learned
alignment. The prompt encoder is configured separately and supplies the
differentiable text tower:

```yaml
experiments:
  sldpc_my_slide_encoder:
    method: sldpc
    features: {bag: my_slide_encoder_5x}
    prompt_encoder:
      name: titan
      weights: /models/titan/snapshot
      model_id: MahmoodLab/TITAN
      revision: exact-hub-revision
      feature_space_id: hf:MahmoodLab/TITAN@exact-hub-revision
      local_files_only: true
    slide_projection:
      mode: mlp             # native, linear, or mlp
      hidden_dim: 1536      # optional for mlp
      dropout: 0.1
    epochs: 100
```

HDF5 and torch `.pt`/`.pth` embeddings are supported. The loader enforces the
registered key and width on every sample. `native` is accepted only when the
slide and prompt feature spaces match exactly and the prompt backbone exposes
its paired slide projector.

Readiness is strict: every slide in the annotation-defined universe must have
every input required by the feature set. For example, the seven missing BRCA
bags are reported rather than silently dropping slides. CONCH-v1.5,
Virchow2, or UNI-v2 features are never substituted for another declared space.

## Commands

Regenerate and validate all manifests, folds, configs, and the run matrix:

```bash
conda run -n trident python scripts/tcga_benchmark.py all
```

Generate the validated CoD-MIL CLIP-RN50 prompt tensors and ConVLM QuiltNet
attribute tensors from the locally cached encoders:

```bash
conda run -n trident python scripts/generate_tcga_text_features.py
```

Inspect `feature_coverage.csv` for per-input coverage and `run_matrix.csv` for
experiment readiness. A row is launchable only when `ready` is `true`; missing
files are counted explicitly. The current generator produces 180 validated
configs: 13 method families across three cohorts and three shot levels, with
additional backbone/resolution variants where registered, including separate
MUSE-CONCH, MUSE-MUSK, and MUSE-KEEP patch-source experiments. SLDPC is currently
input-complete for NSCLC. Configs that depend on future CLIP-RN50 bags,
CoD-MIL maps, WSI-FiVE sequences/reports, ConVLM RGB tile directories, or
missing TITAN slide embeddings remain `ready: false` instead of silently
dropping cases.

Each method configuration can be launched with its command in the matrix. For
example:

```bash
conda run -n trident python train.py --method focus \
  --config benchmarks/tcga/configs/focus/brca_4shot.yaml
```

Every completed fold writes slide-level and patient-level accuracy, balanced
accuracy, macro-F1, per-class recall, one-vs-rest AUROC, negative
log-likelihood, calibration error, and a slide prediction CSV. Aggregate all
completed jobs with:

```bash
conda run -n trident python scripts/tcga_benchmark.py aggregate
```

Do not modify generated split CSVs per method or encoder. Register features and
experiments in `protocol.yaml`, then run a full regeneration and validation.
