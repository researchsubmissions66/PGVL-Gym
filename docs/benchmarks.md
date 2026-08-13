# Benchmark protocols

The benchmark generator provides two protocol families with identical
configuration semantics.

## TCGA

The TCGA protocol covers:

- NSCLC: lung adenocarcinoma vs. lung squamous cell carcinoma;
- BRCA: invasive ductal vs. invasive lobular carcinoma;
- RCC: clear-cell, papillary, and chromophobe renal cell carcinoma.

It generates 180 validated configs: 20 experiment variants, three cohorts,
and 4/8/16-shot settings. See the repository's
`benchmarks/tcga/README.md` for cohort construction, feature coverage, and
readiness details.

## Additional tasks

The additional-task protocol covers CAMELYON16 and UBC-OCEAN with the same
fold, shot, encoder, prompt, and feature-provenance machinery. It generates 120
validated configs. See `benchmarks/additional_tasks/README.md` for task-specific
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
