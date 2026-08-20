# Troubleshooting

Start with the generated reports before loading a large model. Most failures
are provenance or readiness problems that can be diagnosed without a GPU.

## Fast diagnostic sequence

```bash
python scripts/tcga_benchmark.py validate
python scripts/list_backbone_compatibility.py --method <method>
python -u scripts/smoke_test.py --config <config> --device cuda:0
```

Then inspect the matching row in `run_matrix.csv`, `config_audit.csv`, and
`feature_coverage.csv`.

## Common failures

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| `ready=false` | At least one required asset is absent | Read the matrix readiness columns and missing-file counts |
| `config_valid=false` | Contract, prompt, class-order, or provenance mismatch | Open the corresponding audit error and regenerate after fixing the protocol |
| Feature file not found | Wrong path template, slide ID normalization, or incomplete extraction | Compare the manifest's resolved path with the registry template |
| HDF5/torch key error | `feature_key` does not match the stored payload | Inspect one representative file and update the atomic source definition |
| Expected width differs from tensor | Wrong encoder output or registry dimension | Verify the producer checkpoint; never insert an undeclared projection |
| Feature-space mismatch | Same width but different semantic embedding space | Correct `feature_space_id` and re-extract or select a compatible adapter |
| Every train/eval batch failed | Loader and method contract disagree | Run the isolated smoke test and inspect the first underlying batch error |
| Encoder weights unavailable offline | Cache path or revision is wrong | Check the declared local weights and `local_files_only` model snapshot |
| CUDA out of memory | Model and bag do not fit simultaneously | Use one isolated process per variant, verify no stale GPU process remains, and follow method-native sampling rather than silently changing evaluation |
| UBC-OCEAN has no ready rows | Official local metadata has not been populated | Provide the declared metadata columns and rerun `all` |
| Aggregate table is empty | No referenced result directory contains `metrics.json` | Train ready rows first and confirm each config's `results_dir` |
| Fewer aggregate folds than expected | Some folds failed, were skipped, or used a different directory | Inspect `metrics.json` and do not compare incomplete means as final results |

## Environment failures

These surface as job failures but are never modelling problems. `scripts/pgvl_job.sh`
exits with status **78** for all of them, and a run that skipped for a missing
asset exits **3** and records `skipped.json`; neither should be read as a result.

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| `We couldn't connect to 'https://huggingface.co' … couldn't find them in the cached files` | Compute node is offline and `HF_HOME` points at an empty cache | Export `HF_HOME` to the shared project cache and `HF_HUB_OFFLINE=1`; download weights from a login node first |
| `ModuleNotFoundError` for `h5py`, `ftfy`, `torch_geometric`, `conch` | Running against a site PyTorch module instead of the project environment | Set `PGVL_CONDA_ENV` to the environment built from `environment.yml` |
| `PGVL_CONDA_ENV is unset; refusing to submit` | The campaign has no explicit compute environment | Set it in the ignored `.env` file or export it before a real launch; dry-run planning remains available |
| A completed feature backfill is still shown as missing | The plan was run with `--no-refresh-readiness`, or the manifest references a different path | Run `./launch_pgvl.sh --dry-run` normally; if it remains missing, inspect the manifest's `feature__*` path and `${PGVL_STORAGE_ROOT}` value rather than regenerating unrelated prompt/split artifacts |
| `PermissionError: [Errno 13] … '/path'` | Config still carries the committed `/path/to/...` placeholders | Regenerate configs from a protocol whose paths match local storage |
| `KeyError: "Unknown method 'pathpt_keep'"` | A method name was taken from a config directory name | Submit through `./launch_pgvl.sh`, which reads the matrix's `method` column |
| Job appears hung before any output | First import of an environment on Lustre or GPFS from a cold node | Wait; the cost is per node, not per job. Stage the environment to node-local storage if it recurs |
| `cannot activate conda env` (exit 78) | `PGVL_CONDA_ENV` is wrong or the environment was never built | `conda activate` the same path by hand to see the underlying error |

## Prompt and class-order problems

Prompts must use the same label order as the cohort. If logits appear swapped
or one method reports implausible performance, compare:

- protocol `labels` and `classnames`;
- generated `label_dict`;
- prompt-profile label order;
- the ordering of any precomputed prompt tensor;
- prediction probability column indices.

Never repair a class-order mismatch by relabeling output columns after test
evaluation.

## Docs build problems

Install the documentation dependencies and run the same strict build used by
continuous integration:

```bash
python -m pip install -r requirements-docs.txt
python scripts/check_docstrings.py
python -m mkdocs build --strict
```

A strict-build failure usually identifies a broken navigation target, invalid
reference, or API symbol that can no longer be resolved.

## Useful issue report

When asking for help, include the config path, method, exact command, full
traceback, relevant run-matrix row, validation report excerpt, feature tensor
shape/key, Python and PyTorch versions, and GPU model. Do not attach patient
identifiers, clinical reports, model credentials, or restricted data.
