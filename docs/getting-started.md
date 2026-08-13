# Getting started

## Install

Use the existing `trident` environment on the benchmark server:

```bash
conda activate trident
python -m pip install -e ".[all]"
```

For documentation-only work, install the small documentation dependency set:

```bash
python -m pip install -r requirements-docs.txt
```

## Inspect the registry

List every method/encoder boundary without allocating a foundation model:

```bash
python scripts/list_backbone_compatibility.py
python scripts/list_backbone_compatibility.py --method sldpc --json
```

The output comes from each adapter's machine-checkable
`MethodBackboneContract`, not from a separate documentation table.

## Validate generated protocols

```bash
# TCGA NSCLC, BRCA, and RCC
python scripts/tcga_benchmark.py validate

# CAMELYON16 and UBC-OCEAN
python scripts/tcga_benchmark.py validate \
  --protocol benchmarks/additional_tasks/protocol.yaml
```

Validation checks config structure, prompt assets, encoder contracts, feature
roles, dimensions, and provenance. Missing future feature files are reported
separately from invalid configurations.

## Run a dummy-feature smoke test

```bash
python -u scripts/smoke_test.py \
  --matrix benchmarks/tcga/run_matrix.csv \
  --cohort rcc \
  --device cuda:0
```

The harness selects one 4-shot config per experiment variant and runs each in
an isolated subprocess. It builds the configured model, loads cached encoder
weights and prompt assets, sends method-appropriate dummy features through
`eval_step`, and verifies finite `[batch, classes]` logits.

## Launch a generated run

Use the exact command stored in a run matrix row whose `ready` field is true:

```bash
python train.py --method focus \
  --config benchmarks/tcga/configs/focus/nsclc_4shot.yaml \
  --device cuda:0
```

`ready: false` is intentional: it means at least one declared feature,
metadata, split, or auxiliary asset is unavailable. The framework does not
substitute another feature space.

!!! success "Next step"

    Continue with the [end-to-end tutorial](tutorial.md) to validate a ready
    PathPT configuration, run an isolated model forward, launch training, and
    interpret the generated metrics.
