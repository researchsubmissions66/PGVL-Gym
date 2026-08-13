# CAMELYON16 and UBC-OCEAN benchmark

This protocol extends the shared, feature-agnostic benchmark engine beyond
TCGA. It fixes the same comparison controls for every generated run:

- seed `1`, five outer folds, and nested `4/8/16`-shot train and validation
  subsets;
- one deterministic training/validation slide per case and all test slides;
- task membership defined from annotations, never from feature availability;
- task-qualified feature paths, so equal slide names in different datasets
  cannot collide;
- explicit encoder, feature-space, resolution, prompt, and weight provenance.

## Current inventory

CAMELYON16 uses the 270 official-train and 129 official-test annotations found
in the non-UCCA FiVE checkout. Their original partition is retained in the
manifest as `source_partition`; the systematic protocol then creates a new
five-fold outer evaluation over all 399 slides (239 normal, 160 tumor). Every
slide is treated as one case because CAMELYON16 has one lymph-node slide per
identifier in these annotations.

UBC-OCEAN is registered with the canonical five-class order `CC, EC, HGSC,
LGSC, MC`, but its annotation table is not present on this machine. Put the
official WSI training metadata into `metadata/ubc_ocean.csv` with columns
`image_id,label,is_tma`. TMA composites are deliberately filtered out. Until
that file contains rows, UBC configs remain valid but their run-matrix rows
have `metadata_ready=false`, `split_ready=false`, and `ready=false`.

No UCCA checkpoint, annotation, prompt, or feature path is used.

## Method coverage

FOCUS is original-domain coverage rather than a speculative extension: its
upstream repository publishes launch scripts and few-shot splits for CAMELYON
and UBC-OCEAN. The paper's CAMELYON experiment draws from both CAMELYON16 and
CAMELYON17; this benchmark currently uses CAMELYON16 only, so it must not be
reported as an exact reproduction of FOCUS's combined CAMELYON cohort.

| Method family | CAMELYON16 | UBC-OCEAN |
|---|---:|---:|
| PathPT (CONCH, MUSK, KEEP) | yes | yes |
| MUSE (CONCH, MUSK, KEEP patch bags) | yes | yes |
| FOCUS (10x/20x, 5x/20x) | yes | yes |
| MSCPT (10x/20x, 5x/20x) | yes | yes |
| MAPLE (10x/20x, 5x/20x) | yes | yes |
| ViLa-MIL | yes | yes |
| CoD-MIL | yes | yes |
| TOP | yes | yes |
| SLIP | yes | yes |
| WSI-FiVE | yes | yes |
| SLDPC with TITAN slide embeddings | yes | yes |
| ConVLM | yes | yes |
| Composite classname baseline | yes | yes |

CoD-MIL's upstream `main.py` contains a CAMELYON16 task, but not a UBC-OCEAN
task. Its public prompt directory currently contains only the kidney/RCC prompt
CSV and precomputed CLIP/PLIP/QuiltNet tensors. The generalized configs compile
a chain from the dataset prompt profile and encode it once with the configured
CLIP-RN50 tower at runtime. Cross-magnification maps are still data-derived
auxiliaries and must be generated from patch coordinates.

ConVLM likewise encodes the task's generated attribute descriptions with the
configured cached QuiltNet tower when a precomputed tensor is not supplied.
WSI-FiVE normally consumes a slide-specific report. Because these datasets do
not provide a clinical report corpus, its extension uses one class-agnostic
task context for every slide; it never derives text from the ground-truth
label. These modes are valid generalized-framework evaluations, but must be
reported as extensions rather than upstream-released prompt replications.

The protocol emits 120 validated configs: 20 experiment variants, two tasks,
and three shot settings. `prompt_provenance` in every config distinguishes
generated/user-defined prompts from upstream assets.

## Adding a dataset

No model code should change. Add only:

1. A cohort entry in `protocol.yaml` defining metadata columns, ordered
   `labels`, ordered human-readable `classnames`, and a `prompt_spec` path.
2. A version-1 prompt profile defining the same labels, low- and
   high-resolution descriptions, and optional aliases/tissues/attributes. A
   released `small_mag`/`big_mag` JSON can be referenced with
   `description_source` instead of copying its text.
3. Feature-source path templates and dimensions for whichever encoders will be
   evaluated. Feature availability never changes the split universe.

The prompt compiler writes native assets for FOCUS/ViLa-MIL, MUSE, MSCPT,
MAPLE, CoD-MIL, SLIP, SLDPC, and ConVLM. PathPT, TOP, and the composite baseline
consume the ordered classnames directly; WSI-FiVE consumes either a configured
report table or the profile's class-agnostic context. Experiments without a
`tasks` restriction automatically expand to the new cohort.

## Feature layout

The default registry expects files under:

```text
/path/to/features/additional_tasks/<task>/
  5x_256px_0px_overlap/features_conch_v1/<slide_id>.h5
  10x_256px_0px_overlap/features_conch_v1/<slide_id>.h5
  20x_512px_0px_overlap/features_conch_v1/<slide_id>.h5
  10x_384px_0px_overlap/features_musk/<slide_id>.h5
  20x_256px_0px_overlap/features_keep/<slide_id>.h5
  20x_512px_0px_overlap/slide_features_titan/<slide_id>.h5
  20x_224px_0px_overlap/features_wsi_five_medclip/<slide_id>.pt
  20x_448px_0px_overlap/raw_tiles_convlm/<slide_id>/
  ...
```

Each source can be replaced independently in `protocol.yaml`; dimensions and
feature-space provenance are checked before configs are generated.

## Prepare and validate

From the repository root:

```bash
conda run -n trident python scripts/tcga_benchmark.py all \
  --protocol benchmarks/additional_tasks/protocol.yaml
```

The shared script retains its historical filename for compatibility. Outputs
include `feature_coverage.csv`, canonical manifests, patient-disjoint splits,
`run_matrix.csv`, `config_audit.csv`, and `validation_report.json`.

Launch only rows where `ready=true`, using the exact command in the run matrix.
For example:

```bash
conda run -n trident python train.py --method muse \
  --config benchmarks/additional_tasks/configs/muse/camelyon16_4shot.yaml
```
