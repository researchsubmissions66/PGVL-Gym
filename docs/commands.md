# Commands and run lifecycle

PGVL-Gym separates protocol preparation from model execution. Run commands
from the repository root in the same environment that contains the model
dependencies.

## Protocol lifecycle

The benchmark command accepts the following stages:

| Stage | What it produces |
| --- | --- |
| `inventory` | Cohort manifests and feature-coverage counts |
| `prepare` | Manifests, coverage reports, and deterministic split files |
| `configs` | Per-run YAML files, the run matrix, and configuration audit |
| `validate` | Static contract and sampled tensor validation report |
| `aggregate` | Fold-level and aggregate result tables from completed runs |
| `all` | Inventory, preparation, config generation, and validation |

Run the complete TCGA preparation pipeline:

```bash
python scripts/tcga_benchmark.py all
```

Use the same engine with another protocol:

```bash
python scripts/tcga_benchmark.py all \
  --protocol benchmarks/additional_tasks/protocol.yaml
```

`--output-dir` may be used to write generated artifacts somewhere other than
the protocol directory. Keep the protocol and generated output together when
archiving an experiment.

## Inspect compatibility

The compatibility command reads adapter contracts without loading model
weights:

```bash
python scripts/list_backbone_compatibility.py
python scripts/list_backbone_compatibility.py --method pathpt
python scripts/list_backbone_compatibility.py --json > compatibility.json
```

The live command is the source of truth when it differs from a manually copied
table.

## Smoke-test a configuration

Test one generated configuration:

```bash
python -u scripts/smoke_test.py \
  --config benchmarks/tcga/configs/maple/nsclc_4shot.yaml \
  --device cuda:0
```

Test one representative 4-shot run per experiment variant:

```bash
python -u scripts/smoke_test.py \
  --matrix benchmarks/tcga/run_matrix.csv \
  --cohort nsclc \
  --device cuda:0 \
  --timeout 300 \
  --result-json benchmarks/tcga/smoke_report_nsclc.json
```

Matrix entries run in isolated subprocesses so a previous model cannot retain
GPU memory. A smoke pass proves that the configured model builds and produces
finite class logits; it does not prove that training will converge.

## Train

Launch a row whose `ready` field is true using the exact method and config in
the run matrix:

```bash
python train.py \
  --method pathpt \
  --config benchmarks/tcga/configs/pathpt_musk/rcc_4shot.yaml \
  --device cuda:0
```

Options:

| Option | Meaning |
| --- | --- |
| `--method` | Registered adapter name |
| `--config` | Generated or user-authored run YAML |
| `--device` | PyTorch device, such as `cuda:0` or `cpu` |
| `--seed` | Optional runtime override of the configured training seed |

Training iterates from `k_start` through `k_end - 1`. With a configured seed
of `s`, fold `k` is initialized with `s + k`.

## Evaluate and aggregate

The unified training loop evaluates the selected best checkpoint on the test
partition when `evaluate_test: true` and writes metrics into the configured
results directory. This is the recommended evaluation path for every adapter.

`eval.py` is a legacy convenience command built around the generic dual-scale
loader. Do not use it for slide embeddings, report-conditioned sequences, raw
tile directories, or method-specific loaders. For those methods, rely on the
adapter-aware test phase in `train.py`.

After runs finish, aggregate every available `metrics.json` referenced by the
matrix:

```bash
python scripts/tcga_benchmark.py aggregate
python scripts/tcga_benchmark.py aggregate \
  --protocol benchmarks/additional_tasks/protocol.yaml
```

Missing runs are skipped rather than imputed. Always inspect the reported fold
count before comparing means.
