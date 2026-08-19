# Configuration

There are two configuration levels:

- a **protocol** describes cohorts, feature registries, experiment variants,
  prompts, folds, and shot counts;
- a generated **run config** is the immutable input to one method invocation.

Every generated run records independent provenance axes:

| Field | Meaning |
| --- | --- |
| `encoder_provenance` | Whether the encoder boundary is native or bridged by an adapter |
| `prompt_provenance` | Where the actual text/prompt artifact originated |
| `implementation_provenance` | Whether the executed method code is vendored, mixed, or locally reconstructed |
| `upstream_fidelity` | `upstream`, `partial`, or `local_baseline` |

Set `require_upstream_fidelity: true` when a campaign must reject partial
implementations rather than merely emit a doctor warning.

## Dataset definition

A task entry defines stable label order and prompt semantics:

```yaml
tasks:
  my_task:
    labels: [class_a, class_b]
    classnames:
      - class A diagnosis
      - class B diagnosis
    prompt_spec: benchmarks/my_benchmark/prompts/my_task.yaml
```

Label order is significant. It determines numeric labels, logit columns, and
method-native prompt ordering.

## Feature registry

Feature sources are named independently from model methods:

```yaml
feature_registry:
  conch_5x:
    encoder: conch
    feature_space_id: hf:MahmoodLab/conch
    input_kind: patch_bag
    feature_dim: 512
    resolution: 5x
    path_template: /data/{cohort}/5x/features_conch
    feature_key: features
```

The important fields are:

| Field | Meaning |
| --- | --- |
| `encoder` | producer or slide encoder name |
| `feature_space_id` | exact checkpoint/model provenance |
| `input_kind` | patch bag, slide embedding, sequence, or raw tiles |
| `feature_dim` | last tensor dimension expected by the adapter |
| `resolution` | semantic magnification such as `5x`, `10x`, or `20x` |
| `path_template` | cohort-specific storage location |
| `feature_key` | exact tensor key inside HDF5/mapping payloads |

Low and high resolution are roles, not fixed magnifications. A dual-scale
experiment may map `low` to `5x` and `high` to `20x`, or use another declared
pair, as long as the method and feature registry agree.

## Method and encoder selection

Generated configs record both the runtime prompt encoder and the offline
feature source where applicable:

```yaml
method: muse
backbone: conch
backbone_weights: /models/conch
feature_sources:
  bag: musk_10x
feature_dim: 1024
feature_space_id: hf:xiangjx/musk
prompt_feature_space_id: hf:MahmoodLab/conch
```

MUSE can learn an adapter between these declared widths. A method that compares
patch and text embeddings directly cannot do so unless its architecture
explicitly defines such a projection.

For SLDPC, `slide_encoder` identifies cached slide vectors while `backbone`
identifies the runtime prompt tower. `slide_projection_mode: native` requires
the paired slide projector; `linear` and `mlp` are explicit learned-alignment
variants and must be reported separately.

## Path expansion

Configuration loading recursively expands environment variables and leading
`~/` notation in nested mappings and lists, not only top-level path fields.
Use `${PGVL_REPO_ROOT}`, `${PGVL_STORAGE_ROOT}`, and `${PGVL_USER_ROOT}` for
portable benchmark configs. An undefined variable remains visible to the
doctor and is a configuration error rather than being interpreted as a literal
filesystem name.

Resolved YAML must also round-trip through JSON because `config.json` and the
resume fingerprint are the persistent experiment identity. Quote date-like or
numeric mapping keys; YAML dates, sets, binary values, non-string keys, and
`.nan`/`.inf` numbers are rejected during loading instead of failing after a
results directory has been created.
Duplicate mapping keys are rejected at any nesting level instead of silently
keeping the last value. YAML merge keys retain their standard explicit
override behavior.

SLDPC configs must state `epochs` explicitly and set it to
`stage1_epochs + stage2_epochs`. The unified loop does not infer a private
adapter default for this schedule because the two upstream training stages are
part of the experiment identity.

## Split layouts

Runtime loaders and the doctor share three fold representations:

- generated `foldN/{train,val,test}.csv` files with complete manifest rows;
- CLAM-style `splits_N.csv` files with `train`, `val`, and `test` slide-ID
  columns; and
- upstream-style `foldN.csv` files, which may also carry `train_label`,
  `val_label`, and `test_label` columns.

For a wide table without phase-label columns, `dataset_csv` must contain unique
`slide_id` and `label` columns. The loader joins the requested IDs back to that
manifest and retains its exact feature-path and case-ID fields. A label present
in both sources must agree. Direct `train.csv`, `val.csv`, and `test.csv` files
at the split root are supported only for a single-fold run; multi-fold configs
must scope them under `foldN/` so the same split is never counted repeatedly.

## Prompt sources

Most upstream repositories ship their prompts as an explicit per-task file, and
that file is part of the published method: FOCUS reads a
`class_name,low_res_prompt,high_res_prompt` CSV, MSCPT a GPT description JSON,
MUSE per-class description CSVs, SLIP a tissue-name JSON. Which file a run
embeds is a scientific parameter, so state it in the cohort rather than leave it
to resolution order:

```yaml
cohorts:
  ubc_ocean:
    prompts:
      focus: text_prompts/focus/UBC_OCEAN_two_scale_text_prompt.csv
      vila_mil: text_prompts/focus/UBC_OCEAN_two_scale_text_prompt.csv
      mscpt: train_data/gpt/description/UBC-OCEAN.json
      slip: text_prompts/slip/ubc_ocean_tissues.json
      sldpc: text_prompts/sldpc/ubc_ocean.yaml
      muse:                       # a method may name several files
        - text_prompts/muse/ubc_ocean/generated_new_0.csv
        - text_prompts/muse/ubc_ocean/generated_new_1.csv
```

Paths are repository-relative or absolute. A named file that does not exist is
an error, never a silent fallback: a prompt the author asked for and did not get
would change what the model reads without saying so.

A method with no entry falls back, in order:

1. the cohort's published per-method key (`focus_prompt_csv`,
   `mscpt_prompt_json`, `muse_prompt_csvs`, …), then
2. the asset compiled from `prompt_spec`.

`prompt_precedence` inverts those two for a cohort, or protocol-wide:

| Value | Meaning |
| --- | --- |
| `upstream` (default) | the paper's published asset wins; the compiler is the fallback for tasks that have no published prompts |
| `generated` | prefer prompts compiled from `prompt_spec`, for instance to compare every method under one uniform prompt style |

WSI-FiVE is an exception to this generic fallback. Its six aligned questions,
training answers, and evaluation descriptions have different roles and cannot
be inferred from a class-description `prompt_spec`; configure them explicitly.

Each generated config records which source was actually used in
`prompt_provenance`, and the run matrix carries it as `prompt_source` and
`prompt_asset`. Check those before reporting: a cohort declaring a `prompt_spec`
alongside a published CSV will silently read one of the two, and the two are not
the same experiment.

!!! warning "Class names are prompts for some methods"

    `classnames` is not merely a label ordering. TOP, SLDPC, SLIP,
    WSI-FiVE's `simplified_classnames` mode, and PathPT's legacy
    `simplified_slide_ce` mode build text from it, so a bare study code such as
    `KIRC` or `CC` becomes the prompt. Native PathPT instead resolves its
    audited task synonym bank. Native WSI-FiVE trains against a fold-local
    answer bank and evaluates against `evaluation_prompt_path`; `label_dict`
    still fixes that bank's class order. Keep diagnosis names and label order
    aligned even when a method reads an explicit bank.

## Reproducibility fields

Every comparable generated run records at least:

- `shots`, `seed`, `k_start`, `k_end`, and `split_dir`;
- `n_classes`, `classnames`, and `label_dict`;
- method/backbone identities and weight locations;
- feature roles, dimensions, spaces, and resolutions;
- prompt source and provenance;
- optimizer, scheduler, epoch, and early-stopping settings;
- the experiment-specific results directory.

Do not edit one generated YAML in isolation for a benchmark change. Update the
protocol and regenerate the matrix so validation and provenance reports remain
synchronized.

## Preflight coverage policy

Run configs require every manifest row to have every referenced feature file.
Preflight checks non-empty files and, for multi-input methods, the intersection
of rows across all required feature columns. This strict default prevents two
methods from reporting results over different accidental subsets of a cohort.

An intentionally partial exploratory run must state its minimum acceptable
coverage explicitly as a fraction:

```yaml
min_feature_coverage: 0.95
```

Such a run emits a coverage warning and should not be mixed with complete-cohort
benchmark results. A zero-file source and a zero-row joint intersection remain
fatal regardless of the configured threshold.
