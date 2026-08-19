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

Command help remains available in an incomplete bootstrap environment. Running
a benchmark command without a core dependency exits with status 2 and points
to `scripts/preflight.py --system` instead of failing with an import traceback.

Use the same engine with another protocol:

```bash
python scripts/tcga_benchmark.py all \
  --protocol benchmarks/tcga_brca/protocol.yaml
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

## Diagnose a run with the doctor

The preflight CLI behaves like a read-only doctor: it diagnoses each failure,
suggests a repair, and ends with a campaign-friendly summary. Normal checks do
not import PyTorch, construct a model, or open feature tensors. Explicit
`--deep` mode loads feature payloads for semantic validation.

The doctor identifies:

- missing, empty, unreadable, unresolved, and wrong-type configured input paths;
- unsafe, non-directory, or non-writable results paths, while accepting a
  missing directory whose existing parent can create it;
- absent method-required feature roots/columns, prompt banks, reports,
  cross-scale maps, and encoder assets;
- malformed FOCUS/ViLa-MIL prompt tables and MAPLE, MSCPT, TOP, SLIP,
  CoD-MIL, MUSE, ConVLM, or SLDPC prompt graphs/banks, including
  class-order/cardinality drift;
- incompatible variable-length bag batch sizes and invalid batch-failure
  thresholds, optimizer values, epochs, staged-training controls, class schemas,
  and sampling limits;
- malformed manifests, including blank or duplicate headers, wrong-width
  rows, missing/blank/repeated slide IDs, duplicate feature references, and
  unresolved feature references;
- per-input and joint feature coverage across manifest rows;
- missing, empty, or malformed nested phase CSVs, CLAM-style
  `splits_<fold>.csv`, and upstream `fold<fold>.csv` tables, including missing
  phase labels when the dataset manifest cannot supply them;
- blank, repeated, or overlapping split identities and incorrect nested
  `partition` values;
- slide leakage in flat or nested splits and patient leakage in both layouts
  (flat splits inherit case IDs from the manifest), including a failure when
  case IDs are absent and leakage cannot be ruled out;
- nested split rows whose case, label, or configured feature path has drifted
  from the dataset manifest;
- contradictory implementation provenance and partial upstream fidelity;
- malformed YAML and undefined environment variables; and
- when requested with `--system`, unsupported Python versions, missing core
  packages or supported dependency versions, mismatched Torch/torchvision
  releases, invalid PGVL roots, and malformed or incomplete `.env` setup.

```bash
python scripts/preflight.py \
  benchmarks/tcga_brca/configs/focus/brca_4shot.yaml
```

With no selection flags the command checks all aspects. Checks can be combined
to answer narrower questions:

```bash
# Feature roots, per-slide availability, and joint multi-input coverage
python scripts/preflight.py run.yaml --features

# Open every referenced payload and validate key/rank/width/finite values;
# shared pickle stores are also checked against manifest slide IDs
python scripts/preflight.py run.yaml --features --deep

# Prompt assets and encoder checkpoints only
python scripts/preflight.py run.yaml --prompts --encoders

# Several configs with output suitable for automation
python scripts/preflight.py configs/focus/*.yaml --all --json

# An explicitly partial exploratory health check
python scripts/preflight.py run.yaml --features --min-feature-coverage 0.95

# Fast login-node check: validate roots but skip per-slide filesystem stats
python scripts/preflight.py run.yaml --quick

# Diagnose Python, base packages, and PGVL root variables without a run config
python scripts/preflight.py --system

# Make warnings fail an automated readiness gate
python scripts/preflight.py run.yaml --strict --json
```

### Doctor options

| Option | Behavior |
| --- | --- |
| `--assets` | Check dataset manifests, other general inputs, and whether `results_dir` is a safe writable/creatable directory. |
| `--features` | Check feature roots and, normally, every manifest feature reference plus their joint coverage. |
| `--prompts` | Check text prompts, description banks, prompt references, and related prompt collections. |
| `--encoders` | Check top-level and nested encoder checkpoint paths. |
| `--splits` | Validate every configured fold, CSV structure, phase values, and partition leakage. |
| `--all` | Run all configuration checks; this is also the default when no selector is supplied. |
| `--system` | Additionally check Python 3.10–3.11, core dependencies and versions, the repository, `.env`, and PGVL roots. It can be used without a config. |
| `--min-feature-coverage N` | Override the run's required coverage with a validated fraction from 0 through 1. |
| `--quick` | Skip per-slide feature stats and inspect roots only; equivalent to `--no-feature-scan`. |
| `--deep` | Open every available feature payload and validate its key, shape, width, and finite values; validate shared-pickle ID structure, uniqueness, and manifest coverage. |
| `--strict` | Treat warnings as unhealthy, which is useful for CI and campaign gates. |
| `--json` | Suppress prose and emit the versioned JSON contract described below. |
| `--verbose` | Show healthy paths as well as failed paths, including resolved names, types, and file sizes. |
| `--quiet` | Hide healthy detail and print only findings plus the summary. |
| `--no-color` | Force plain text; color is already disabled for redirected output or when `NO_COLOR` is set. |

Multiple config paths are checked independently, so one malformed YAML does
not prevent diagnosis of the remaining files.

### Exit status and JSON

Exit status is zero only when every requested check is healthy. Missing or
unreadable assets, malformed splits, environment failures, and warnings under
`--strict` return one. Argument errors return argparse's standard status two.
The CLI also handles broken output pipes without a traceback.

JSON output is intended as a stable automation boundary. It includes:

- `schema_version`, currently `1`;
- top-level `healthy` and `strict` values;
- a `summary` containing config, problem, warning, host-failure, and duration
  counts;
- optional `system` diagnostics with package versions and repair guidance; and
- one entry per config containing selected checks, resolved path details,
  feature coverage, row counts, warnings, problems, and elapsed time.

## Smoke-test a configuration

Test one generated configuration:

```bash
python -u scripts/smoke_test.py \
  --config benchmarks/tcga_nsclc/configs/maple/nsclc_4shot.yaml \
  --device cuda:0
```

Test one representative 4-shot run per experiment variant:

```bash
python -u scripts/smoke_test.py \
  --matrix benchmarks/tcga_brca/run_matrix.csv \
  --cohort nsclc \
  --device cuda:0 \
  --timeout 300 \
  --result-json benchmarks/tcga_nsclc/smoke_report_nsclc.json
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
  --config benchmarks/tcga_rcc/configs/pathpt_musk/rcc_4shot.yaml \
  --device cuda:0
```

Options:

| Option | Meaning |
| --- | --- |
| `--method` | Registered adapter name |
| `--config` | Generated or user-authored run YAML |
| `--device` | PyTorch device, such as `cuda:0` or `cpu` |
| `--seed` | Optional runtime override of the configured training seed |
| `--rerun` | Archive previous metrics/config state and restart at fold 0 |

Training iterates from `k_start` through `k_end - 1`. With a configured seed
of `s`, fold `k` is initialized with `s + k`.

The CLI method and the YAML `method` must name the same adapter. Supported
aliases are canonicalized (for example, `vila` and `vila-mil` become
`vila_mil`); a real mismatch exits with status 2 before creating the results
directory. A preflight failure exits with status 3 and writes `skipped.json`
when the configured results directory is usable.
After config health passes, `--device` is checked without allocating a model;
invalid targets, unavailable CUDA, and GPU indices outside the visible device
count exit with status 2 before output state is changed.
Only one trainer may own a results directory at a time. A concurrent launch
exits with status 2 and reports the PID recorded in `.run.lock`; the lock is
released automatically on normal exit, exceptions, and process termination.

The campaign launcher validates the same method/executable-config fingerprint
(or compatible legacy `config.json` snapshot) as the trainer and counts the exact unique fold
indices in `metrics.json`, so a state containing folds 0 and 2 still schedules
missing fold 1. Foreign, corrupt, duplicate, non-integer, and out-of-range
resume state is reported as an error rather than treated as fresh or complete.
A completed fold must also contain a finite best validation loss and a valid
test accuracy (or an explicit null when holdout evaluation is disabled).
New documentation-only fidelity fields may be added to an older snapshot
without invalidating it; changes to data, model, optimizer, or evaluation fields
remain fatal. A validated legacy or documentation-only migration rewrites the
metrics fingerprint even when all folds are already complete, so later reads
see a self-consistent snapshot.

## Launch a campaign

The launcher plans the full matrix, protects against duplicate queued jobs, and
writes `benchmarks/launch_report.csv` atomically:

```bash
./launch_pgvl.sh --dry-run
./launch_pgvl.sh --cohort brca --shots 4 --limit 3
./launch_pgvl.sh
```

`--rerun` forwards a real restart to every selected job; the trainer archives
the old metrics, config, checkpoints, predictions, and TensorBoard state before
starting fold 0. A failed queue
query stops real submissions unless `--force` explicitly disables duplicate-job
protection. Negative limits and missing-feature allowances are rejected, and a
successful `sbatch` call is accepted only when its job ID can be parsed.
Corrupt matrix booleans/counts, duplicate derived job names, and rows sharing a
results directory are reported as errors before submission. Duplicate,
incomplete, or wrong-width headers are rejected, as is a `ready=true` row that
contradicts any component readiness flag or nonzero missing-asset count. Fold
indices must be YAML integers and must form a non-empty range; an empty or
reversed range is an error, never an already-complete run.

Best/final model checkpoints, JSON state, and prediction CSVs are written by
atomic replacement. The canonical path therefore always names a complete prior
or new artifact, never an interrupted partial serialization.
Reported epoch loss is weighted by the number of samples in each batch, which
keeps a short final batch from being overrepresented.
Every fold constructs a new adapter from a private copy of the resolved config.
Method caches and staged-training state cannot cross fold boundaries, and an
adapter which fills a derived default cannot change the config fingerprint used
for resume validation. Reloading a best checkpoint also invokes the adapter's
checkpoint hook before holdout inference.

Row, regeneration, or submission errors make the launcher return non-zero after
writing its report. `--best-effort` is the explicit override for automation
that wants a zero exit despite those errors.
Before the first `sbatch`, a real launch atomically writes a provisional report
and creates the log directory. An unusable destination therefore fails before
jobs enter the queue. `--report` is also refused when it resolves to a selected
config, benchmark protocol/run matrix, or run-state JSON file.

## Evaluate and aggregate

The unified training loop evaluates the selected best checkpoint on the test
partition when `evaluate_test: true` and writes metrics into the configured
results directory. This is the recommended evaluation path for every adapter.

`eval.py` uses the same method-specific loader dispatch, logits contract,
checkpoint identity validation, and slide/patient metrics as `train.py`. It
requires every configured checkpoint instead of silently skipping folds:

```bash
python eval.py --method focus --config run.yaml \
  --ckpt_dir results/focus/run
```

The default `--checkpoint auto` selects `best` when early stopping is enabled
and `final` otherwise. Explicit `--checkpoint best` and `--checkpoint final`
remain available. SLDPC owns its best-prompt selection inside the mandatory
two-stage schedule, so auto selects its final checkpoint and restores the
adapter's Stage-2 fused-prompt mode.
Evaluation JSON includes the same run-identity fingerprint as training state.
When `--output` is supplied, the evaluator refuses destinations that would
overwrite the input YAML, checkpoint config/metrics state, or a selected model
checkpoint.

After runs finish, aggregate every available `metrics.json` referenced by the
matrix:

```bash
python scripts/tcga_benchmark.py aggregate
python scripts/tcga_benchmark.py aggregate \
  --protocol benchmarks/tcga_brca/protocol.yaml
```

Missing runs are skipped rather than imputed. Always inspect the reported fold
count before comparing means. Existing metrics are aggregated only after their
saved method/config fingerprint passes the same validation used for resume and
standalone evaluation. The reported fold spread is the population standard
deviation (`ddof=0`), matching the trainer's completion summary.
