# Datasets and labels

The protocol, not the feature directory, defines the task universe. Class
order is frozen because it controls integer labels, prompt order, and logit
columns.

## Registered tasks

| Task | Ordered labels | Classes | Current protocol status |
| --- | --- | --- | --- |
| TCGA-NSCLC | `LUAD`, `LUSC` | Lung adenocarcinoma; lung squamous cell carcinoma | Registered with frozen metadata universe |
| TCGA-BRCA | `IDC`, `ILC` | Invasive ductal carcinoma; invasive lobular carcinoma | Registered with frozen metadata universe |
| TCGA-RCC | `CCRCC`, `PRCC`, `CHRCC` | Clear-cell, papillary, and chromophobe RCC | Registered with frozen metadata universe |
| CAMELYON16 | `normal`, `tumor` | Normal lymph node; metastatic lymph node | Registered from official train/test annotations |
| UBC-OCEAN | `CC`, `EC`, `HGSC`, `LGSC`, `MC` | Five ovarian carcinoma subtypes | Registered; local official metadata is required |

The frozen TCGA universes contain 1,043 NSCLC slides from 946 patients, 1,054
BRCA slides from 991 patients, and 873 RCC slides from 832 patients.
CAMELYON16 contains 399 slides in the registered manifest. These counts come
from annotations and do not shrink when a feature source is incomplete.

## Metadata contract

Every cohort needs columns that resolve to:

- a stable slide identifier;
- a patient or case identifier;
- one label from the ordered task label list;
- optional source-partition or filtering fields.

Patient identity is mandatory for patient-disjoint cancer-cohort folds.
CAMELYON16 treats each slide identifier as one case under its current
annotation contract.

For UBC-OCEAN, place the official training metadata at the path declared by
the protocol. The default additional-task protocol expects columns
`image_id`, `label`, and `is_tma`; TMA composites are excluded. Until the
metadata has rows, generated configs can be valid while `metadata_ready`,
`split_ready`, and `ready` remain false.

## Data and feature separation

The metadata universe determines who belongs in a fold. A feature registry
determines whether a particular method can run for those already-selected
slides. Never construct a cohort by listing files in a feature directory:
doing so changes the test population for each encoder and invalidates a fair
comparison.

## Adding a dataset safely

Before generating runs, confirm that:

1. label aliases have been normalized into one canonical order;
2. patient/case IDs cannot leak between partitions;
3. exclusions are driven by metadata and recorded explicitly;
4. the prompt profile has exactly the same labels and order;
5. every feature path template includes a dataset-qualified namespace;
6. licensing permits local use of the data and derived features.

See [Extending the framework](extending.md) for the registration procedure and
[Benchmark protocols](benchmarks.md) for the shared fold controls.
