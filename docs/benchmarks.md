# Benchmark protocols

The benchmark generator provides two protocol families with identical
configuration semantics.

## TCGA

The TCGA protocol covers:

- NSCLC: lung adenocarcinoma vs. lung squamous cell carcinoma;
- BRCA: invasive ductal vs. invasive lobular carcinoma;
- RCC: clear-cell, papillary, and chromophobe renal cell carcinoma.

It generates 96 validated configs across the cohort-specific protocol matrices:
14 BRCA, 10 NSCLC, and 8 RCC experiment variants, each at 4/8/16 shots. See the repository's
`benchmarks/tcga_brca/README.md` for cohort construction, feature coverage, and
readiness details.

## Additional tasks

The additional-task protocol covers CAMELYON16 and UBC-OCEAN with the same
fold, shot, encoder, prompt, and feature-provenance machinery. It generates 42
validated configs: 7 experiment variants per task at 4/8/16 shots. Together,
the five current protocol matrices generate 138 configs. See
`benchmarks/tcga_brca/README.md` for task-specific
metadata expectations.

## Fair-comparison controls

- one seed and the same outer-fold range for every experiment;
- 4, 8, and 16 labeled patients per class;
- nested few-shot subsets where metadata permits;
- patient/case-disjoint train, validation, and test partitions;
- identical split files across methods;
- method variants named separately when the encoder, resolution pair, or
  learned projection changes;
- accuracy reported independently from any continuous optimization objective.

## Readiness versus validity

`config_valid` means the configuration is internally consistent and satisfies
the adapter's declared contract. `ready` means every file needed to launch that
specific run currently exists. This distinction allows future RCC or
additional-resolution features to be registered before extraction finishes,
without presenting those runs as executable.

Feature production is asynchronous bookkeeping, not protocol generation.
`scripts/run_benchmark.py` refreshes per-source coverage and each row's
`missing_feature_files` before planning, using the paths already frozen in the
manifest. It changes only feature-derived coverage/readiness cells; prompt,
encoder, auxiliary, metadata, split, and config validity remain independent
gates. The refresh also expands portable `${PGVL_*}` paths before checking the
filesystem. A full `tcga_benchmark.py all` regeneration is therefore unnecessary
when a backfill merely adds the previously expected feature files.

## Generated artifacts

Each protocol directory contains:

| Artifact | Purpose |
| --- | --- |
| `protocol.yaml` | source-of-truth registry |
| `run_matrix.csv` | one row and launch command per run |
| `config_audit.csv` | normalized configuration comparison |
| `feature_coverage.csv` | per-source missing/available counts |
| `validation_report.json` | static validation result |
| `smoke_report_<cohort>.json` | isolated model-build/forward result |
| `configs/` | generated run YAML files |
| `splits/` | deterministic few-shot fold files |
