# Architecture

PGVL-Gym separates experiment orchestration from paper-specific model code.
The unified trainer knows only the method interface and loader contract; the
adapter owns the method's model construction and optimization recipe.

```text
Protocol YAML
    │
    ├── dataset labels, prompts, folds, shots
    ├── feature sources and resolutions
    └── experiment registry
            │
            ▼
Generated run config ──► MethodBackboneContract validation
            │
            ├──► feature loader ──► method-specific batch
            └──► BaseMethod adapter ──► paper model
                                      ├── build_model
                                      ├── train_step
                                      └── eval_step
```

## Stable layers

### Protocol and configuration layer

`scripts/tcga_benchmark.py` compiles a protocol into manifests, deterministic
patient-disjoint folds, method-native prompt assets, validated YAML configs,
and a run matrix. Dataset names are not hardcoded in model adapters.

### Method layer

Every registered adapter subclasses `methods.base.BaseMethod`. The common
trainer calls the same lifecycle for every method:

1. validate the encoder contract;
2. build the model;
3. construct the optimizer and scheduler;
4. pass loader batches to `train_step` or `eval_step`;
5. collect the normalized result mapping.

Vendored model files remain under `methods/<method>/`. Cross-method utilities
belong under `common/` only when their semantics are genuinely shared.

### Encoder layer

`EncoderBundle` wraps the native model and tokenizer while exposing narrow
capabilities such as text encoding, tile encoding, soft prompting, or native
slide projection. `MethodBackboneContract` then restricts how an adapter may
consume those capabilities.

See [Backbone interfaces](BACKBONE_INTERFACES.md) for the compatibility matrix
and registration API.

### Data layer

Loader selection is based on tensor level, not on a paper name:

| Tensor level | Typical shape | Shared loader responsibility |
| --- | --- | --- |
| Patch bag | `[patches, dim]` | exact feature key and width |
| Dual-scale patch bag | two variable-length bags | independent low/high sources |
| Slide embedding | `[dim]` | source type, key, width, slide ID |
| Patch sequence + report | `[frames, dim]` plus text | sequence/report pairing |
| Raw tile directory | image tiles | tile sampling and transforms |

## Dependency direction

`train.py` and `eval.py` may depend on `methods` and `common`. Adapters may
depend on `common` and their own vendored implementation. `common` must never
import a concrete method adapter. This keeps registry imports lazy and makes
the generated API documentation safe to build without loading model weights.
