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
