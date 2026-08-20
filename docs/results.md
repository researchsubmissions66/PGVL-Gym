# Results and reporting

Each generated config owns a unique `results_dir`. The training entry point
stores enough information to audit individual folds and to build a common
benchmark table.

## Run artifacts

| Artifact | Contents |
| --- | --- |
| `.run.lock` | Advisory process lock and last trainer PID; prevents concurrent writers in one experiment directory |
| `config.json` | Effective configuration, including command-line seed override |
| `fold<K>_best.pt` | Best validation checkpoint when checkpointing is active |
| `fold<K>_predictions.csv` | Slide and case IDs, label, predicted class, and per-class probabilities when batch metadata is available |
| `metrics.json` | Fold metrics and validation loss for one experiment |
| `logs/<method>/` | TensorBoard training and validation loss events |

Generated benchmark aggregation additionally writes:

- `fold_results.csv`, one normalized row per completed fold;
- `aggregate_results.csv`, mean, standard deviation, and observed fold count
  for every metric and experiment signature.

TensorBoard logs are grouped by method name and are convenient for monitoring,
but the experiment-specific results directory is the archival source of truth.

## Reported metrics

The common training loop computes the same classification metrics for every
method:

| Metric | Interpretation |
| --- | --- |
| Accuracy | Fraction of correct predictions |
| Balanced accuracy | Mean recall across classes |
| Macro F1 | Unweighted mean of per-class F1 scores |
| AUROC OVR | Binary AUROC or macro one-vs-rest AUROC; null when undefined |
| NLL | Mean negative log-likelihood of the true class |
| ECE | Ten-bin expected calibration error |
| Per-class recall | Recall indexed by the frozen numeric class order |

Slide-level metrics are always emitted. Patient-level metrics are emitted when
the loader returns case IDs for every prediction; per-slide probabilities are
averaged within each case before patient-level scoring.

## Reading aggregate tables

Never compare only the `mean` column. A valid comparison should match on:

- cohort and shot count;
- feature and resolution signatures;
- fold count;
- prompt provenance and encoder checkpoint;
- slide-level versus patient-level metric namespace.

`aggregate_results.csv` includes the observed `folds` count. A row with fewer
folds than the protocol requested is incomplete, even if its mean is high.

## Minimum reporting checklist

For every result table, state:

1. dataset version, task labels, and patient/slide counts;
2. shot definition, number of folds, seed, and split provenance;
3. method variant and whether it is an upstream reproduction or framework
   extension;
4. patch/slide encoder checkpoint, feature space, magnification, and tile size;
5. prompt source and whether prompts were optimized;
6. mean and standard deviation across the same completed folds;
7. primary accuracy metric plus at least one imbalance-sensitive metric;
8. any failed, skipped, or unavailable runs.
