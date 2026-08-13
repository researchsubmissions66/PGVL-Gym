# Reproducibility

Reproducibility in PGVL-Gym covers the split universe, feature provenance,
prompt provenance, runtime configuration, and environment. Matching only the
method name is not enough.

## Frozen comparison unit

A fair comparison keeps the following fixed across methods:

- annotation-defined patients and slides;
- outer folds and held-out test cases;
- nested 4/8/16-shot train and validation subsets;
- class order and metric implementation;
- seed policy and fold range.

Features may differ when a model requires a native encoder, but every difference
must appear in the feature and resolution signatures. Do not call unlike
feature spaces an encoder ablation unless the rest of the protocol remains
identical.

## Seed behavior

The protocol seed controls fold and few-shot selection. During training, fold
`k` uses `configured_seed + k` for Python, NumPy, and PyTorch. A command-line
`--seed` changes the training seed in the saved effective config; it does not
regenerate frozen split membership.

## Capture the environment

Archive the following alongside benchmark outputs:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
nvidia-smi
sha256sum benchmarks/tcga/protocol.yaml
```

Also record the exact local checkpoint revision or model-cache snapshot. A
mutable model name such as `conch` is insufficient provenance without a
revision, checkpoint hash, or stable feature-space identifier.

## Before launching GPUs

- Regenerate configs from the committed protocol.
- Run protocol validation and review every invalid row.
- Confirm the intended matrix rows have `ready=true`.
- Inspect feature coverage for the full annotation universe.
- Run one isolated dummy-feature smoke test per experiment variant.
- Verify that trainable parameter names match the intended paper recipe.
- Reserve the test set for the final selected configuration.

## After each run

- Keep the effective `config.json`, best checkpoint, metrics, and predictions.
- Do not replace a failed fold with another seed without reporting it.
- Aggregate from run directories rather than manually transcribing values.
- Confirm every comparison row has the same expected fold count.
- Preserve validation-selection criteria independently from final test metrics.

## Reproduction versus extension

A framework configuration can be valid without being an exact paper
reproduction. Mark it as an **extension** when the task, prompts, feature
encoder, report source, resolution pairing, projection, or optimization recipe
was not released by the upstream method. Cite the upstream paper and describe
the changed boundary explicitly.
