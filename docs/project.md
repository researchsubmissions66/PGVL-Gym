# Project use and citation

PGVL-Gym is a research benchmarking framework. It combines new framework code
with adapted or vendored components from multiple upstream research projects.

## Intended use

The framework is intended for reproducible research on representation,
prompting, and few-shot classification methods in computational pathology. It
is not a clinical device and must not be used as the sole basis for diagnosis,
treatment, or patient triage.

Benchmark accuracy does not establish clinical validity. External validation,
site-shift analysis, subgroup analysis, calibration, failure review, and
appropriate regulatory processes remain necessary for any clinical study.

## Data and privacy

PGVL-Gym does not grant rights to TCGA, CAMELYON, UBC-OCEAN, model checkpoints,
or derived features. Users are responsible for the original dataset and model
licenses, access terms, attribution, and restrictions on redistribution.

Do not commit protected clinical text, patient identifiers, access tokens,
private model credentials, or restricted slide data. Generated reports should
use de-identified case IDs and aggregate statistics.

## Citation

Until a versioned archive or project DOI is published, cite:

1. the exact PGVL-Gym repository commit used for the experiment;
2. every upstream method paper represented in the comparison;
3. every dataset and foundation model according to its own citation guidance;
4. any prompt-optimization method used to select prompts.

Do not cite PGVL-Gym as if a framework extension were an upstream-released
experiment. Label reproductions and extensions separately in the paper.

!!! note "Release metadata"

    Before a public release, add a `CITATION.cff` containing the final project
    authors, repository URL, version, and DOI. Those details are intentionally
    not invented in this documentation.

## Licensing and provenance

The Python package metadata declares Apache-2.0 for framework-owned code.
Vendored files may carry separate upstream obligations. Review the repository
provenance map in `CONTRIBUTING.md` and the original project licenses before
redistributing a combined release.

## Contribution expectations

A method or dataset contribution should include:

- provenance and citation of upstream code/assets;
- a machine-readable adapter contract;
- explicit feature and prompt provenance;
- unit tests for valid and invalid pairings;
- a dummy-feature smoke configuration;
- documentation and public API docstrings;
- clear disclosure of untested or approximate behavior.

Run the documentation and test checks before proposing a change. See
[Extending the framework](extending.md) and
[Writing documentation](contributing-docs.md) for the concrete workflow.
