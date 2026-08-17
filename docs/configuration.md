# Configuration

There are two configuration levels:

- a **protocol** describes cohorts, feature registries, experiment variants,
  prompts, folds, and shot counts;
- a generated **run config** is the immutable input to one method invocation.

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

Each generated config records which source was actually used in
`prompt_provenance`, and the run matrix carries it as `prompt_source` and
`prompt_asset`. Check those before reporting: a cohort declaring a `prompt_spec`
alongside a published CSV will silently read one of the two, and the two are not
the same experiment.

!!! warning "Class names are prompts for some methods"

    `classnames` is not merely a label ordering. PathPT's
    `pathpt_classname_template`, TOP, SLDPC, WSI-FiVE, and SLIP build the text
    they embed from it, so a bare study code such as `KIRC` or `CC` becomes the
    prompt. Declare the diagnosis as it would be written in a report
    (`clear cell renal cell carcinoma`), and keep the order aligned with
    `labels`. Methods that read an explicit prompt file — FOCUS, MSCPT, MUSE —
    use `classnames` only as a key, and are unaffected.

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
