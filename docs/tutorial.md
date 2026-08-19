# End-to-end tutorial

This tutorial follows one real experiment from validation to aggregate output:
**PathPT with CONCH patch features on the 4-shot TCGA-NSCLC task**. The same
workflow applies to other methods after their different input contracts pass
validation.

<div class="pgvl-tutorial-meta">
  <div><span>Method</span><strong>PathPT</strong></div>
  <div><span>Task</span><strong>TCGA-NSCLC</strong></div>
  <div><span>Input</span><strong>CONCH · 10x</strong></div>
  <div><span>Protocol</span><strong>4-shot · 5 folds</strong></div>
</div>

## What you will do

By the end, you will know how to:

- validate a benchmark protocol without loading a model;
- confirm that one run has every required asset;
- smoke-test the model with method-appropriate dummy features;
- distinguish a quick wiring check from a reportable benchmark;
- launch the frozen five-fold run;
- inspect and aggregate its outputs.

## 1. Activate the environment

From the repository root:

```bash
conda activate pgvl-gym
python --version
python -m pip check
```

If the environment does not exist yet, create it from the repository root with
`conda env create --file environment.yml`. Follow the
[environment guide](environment.md) for a smaller method-specific installation.

This tutorial assumes the locally registered CONCH checkpoint and feature bags
exist. Do not download or substitute another encoder merely to make a row
appear ready.

## 2. Validate the protocol

Run static validation before allocating a GPU:

```bash
python scripts/tcga_benchmark.py validate
```

Validation checks the cohort labels, prompt assets, generated configs, encoder
contracts, feature roles, dimensions, and sampled stored tensors. The command
updates `benchmarks/tcga_brca/validation_report.json`.

!!! tip "Validity is not readiness"

    A config can be structurally valid while a future feature store is absent.
    The next step checks the files needed for this specific run.

## 3. Confirm the selected run is ready

Print the PathPT/NSCLC/4-shot matrix row:

```bash
python - <<'PY'
import pandas as pd

matrix = pd.read_csv("benchmarks/tcga_brca/run_matrix.csv")
row = matrix[
    (matrix["experiment"] == "pathpt")
    & (matrix["cohort"] == "nsclc")
    & (matrix["shots"] == 4)
]
print(row.T.to_string(header=False))
PY
```

Continue only when `config_valid` and `ready` are both true. If not, use the
reported readiness fields rather than editing the generated config around the
missing asset.

The selected config is:

```text
benchmarks/tcga_nsclc/configs/pathpt/nsclc_4shot.yaml
```

Important fields to inspect are:

```yaml
method: pathpt
training_mode: upstream_patch_ssl
backbone: conch
feature_sources: {bag: conch_v1_10x}
feature_resolutions: {bag: 10x}
feature_dim: 512
feature_space_id: hf:MahmoodLab/conch
shots: 4
k_start: 0
k_end: 5
seed: 1
```

These fields form one reportable experiment identity. Do not change the path
to a same-width embedding produced by another model.

## 4. Run the isolated smoke test

```bash
python -u scripts/smoke_test.py \
  --config benchmarks/tcga_nsclc/configs/pathpt/nsclc_4shot.yaml \
  --device cuda:0 \
  --timeout 300 \
  --result-json /tmp/pgvl_pathpt_smoke.json
```

A pass means the adapter built the configured model, loaded the required local
weights and prompts, accepted a 512-wide dummy patch bag, and returned finite
logits with shape `[batch, 2]`.

It does **not** measure accuracy, read the real feature store, or establish
convergence.

## 5. Optional two-epoch learning run

For a quick training-loop check, copy the generated config to a temporary
learning config:

```bash
cp benchmarks/tcga_nsclc/configs/pathpt/nsclc_4shot.yaml \
  /tmp/pathpt_nsclc_tutorial.yaml
```

Edit only the temporary file:

```yaml
k_start: 0
k_end: 1
epochs: 2
evaluate_test: false
results_dir: ./results/tutorial/pathpt_nsclc
```

Then run:

```bash
python train.py \
  --method pathpt \
  --config /tmp/pathpt_nsclc_tutorial.yaml \
  --device cuda:0
```

This verifies the real loaders, optimizer, forward/backward pass, validation,
and checkpoint writing. Because it changes the fold range, epoch budget, and
test policy, its output is a tutorial artifact—not a benchmark result.

## 6. Launch the frozen benchmark

Use the unmodified generated config for the reportable run:

```bash
python train.py \
  --method pathpt \
  --config benchmarks/tcga_nsclc/configs/pathpt/nsclc_4shot.yaml \
  --device cuda:0
```

The config runs folds 0 through 4. Each fold uses the same frozen split files,
selects PathPT's zero-shot patch classifier from those training slides only,
trains on four labeled patients per class with `PatchSSLoss`, selects its
checkpoint with the validation partition, and evaluates once on the held-out
test partition using patch voting.

During model construction, confirm that the printed trainable parameter names
match the intended PathPT recipe. An unexpectedly frozen or fully trainable
foundation encoder is a configuration problem, not a harmless detail.

## 7. Inspect the outputs

The standard config writes to:

```text
results/tcga_benchmark/pathpt/nsclc/4shot/
```

Each fold also writes `foldN_pathpt_prompt_selection.json`, which records the
active bank's source/provenance, candidate prompt indices, selection scores,
and winning classifiers.

Inspect the effective config and fold metrics:

```bash
python -m json.tool \
  results/tcga_benchmark/pathpt/nsclc/4shot/config.json

python -m json.tool \
  results/tcga_benchmark/pathpt/nsclc/4shot/metrics.json
```

Expected artifacts include best checkpoints, per-fold prediction CSVs,
`config.json`, and `metrics.json`. Prediction CSVs contain slide/case IDs and
per-class probabilities when the loader supplies complete metadata.

The primary metrics are accuracy, balanced accuracy, macro F1, AUROC OVR,
negative log-likelihood, expected calibration error, and per-class recall.
Patient-level metrics average slide probabilities within each case before
scoring.

## 8. Aggregate the benchmark table

```bash
python scripts/tcga_benchmark.py aggregate
```

This scans result locations referenced by `run_matrix.csv` and writes:

```text
benchmarks/tcga_brca/fold_results.csv
benchmarks/tcga_brca/aggregate_results.csv
```

Filter the aggregate table to this tutorial run:

```bash
python - <<'PY'
import pandas as pd

results = pd.read_csv("benchmarks/tcga_brca/aggregate_results.csv")
rows = results[
    (results["experiment"] == "pathpt")
    & (results["cohort"] == "nsclc")
    & (results["shots"] == 4)
]
print(rows.to_string(index=False))
PY
```

Verify that every reported metric has the expected five folds. An incomplete
mean is not a final comparison.

## 9. Switch to another method

Do not start by changing the PathPT config's `method` field. Each adapter owns
different prompt, feature, and loader requirements.

Instead:

1. inspect the [method support matrix](support-matrix.md);
2. select that method's generated row from `run_matrix.csv`;
3. confirm `config_valid=true` and `ready=true`;
4. smoke-test its own generated config;
5. launch the exact matrix command.

For example, `pathpt_musk` is a separate experiment variant because MUSK has a
different checkpoint, feature width, and feature space. Its fold membership is
identical, but its representation provenance is not.

## Tutorial complete

You now have the complete PGVL-Gym flow:

```text
protocol → validation → readiness → smoke test → training → fold metrics → aggregate report
```

Continue with [Configuration](configuration.md) before registering another
feature source, or [Extending the framework](extending.md) before adding a new
dataset or adapter.
