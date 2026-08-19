# UBC-OCEAN benchmark

Five-class ovarian carcinoma subtyping, run through the same experiment
contract as the TCGA cohorts. This is generalization evidence: a different
institution set, a different scanner population, five classes instead of two or
three, and a strong class imbalance.

The protocol fixes the same comparison controls as every other cohort:

- seed `1`, five outer folds, nested `4/8/16`-shot train and validation subsets;
- one deterministic training and validation slide per case, all test slides;
- task membership defined from annotations, never from feature availability;
- explicit encoder, feature-space, resolution, prompt, and weight provenance.

## Cohort

Canonical five-class order `CC, EC, HGSC, LGSC, MC`. The annotation table is
built by `scripts/build_ubc_ocean_metadata.py` from the read-only release table
at `${PGVL_STORAGE_ROOT}/metadata/RAW_DATA_Cleaned/UBC-OCEAN/metadata.csv`:

| | slides | CC | EC | HGSC | LGSC | MC |
|---|---:|---:|---:|---:|---:|---:|
| whole slides (used) | 513 | 94 | 119 | 217 | 42 | 41 |
| TMA cores (filtered out) | 25 | | | | | |

Tissue microarray cores are excluded by the protocol's `filters: {is_tma: false}`.
`is_tma` is not in the release table; it is derived from the recorded
magnification, where 40x / mpp 0.25 identifies a TMA core and 20x / mpp 0.5 a
whole slide. That split is exact across all 538 images and matches the 25 TMAs
the release documents. It must be written as `True`/`False` so pandas infers a
boolean column — lowercase strings match nothing and silently empty the cohort.

Every image is its own case: `patient_id` is unique across all 538 rows, so
slide-level and case-level evaluation coincide here.

## Feature coverage

Features live under `${PGVL_STORAGE_ROOT}/UBC-OCEAN/<resolution>/features_<encoder>/`.
Coverage is uneven, and the run matrix reflects it rather than hiding it:

| feature source | slides with features | experiments that need it |
|---|---:|---|
| `conch_v1_5x` | 513 / 513 | `mscpt_5x20x` |
| `conch_v1_20x` | 513 / 513 | `mscpt_5x20x`, `focus`, `mscpt` |
| `conch_v1_10x` | **1 / 513** | `pathpt`, `muse`, `focus`, `mscpt` |
| `keep_20x` | **1 / 513** | `pathpt_keep`, `muse_keep` |

Only `focus` and `mscpt_5x20x` are runnable today. The other six
experiments need CONCH re-extracted at 10x and KEEP at 20x; both directories
exist but contain a single slide, which is an interrupted extraction rather than
a missing one.

## Regenerate

```bash
python scripts/build_ubc_ocean_metadata.py --check   # validate the annotations
python scripts/build_ubc_ocean_metadata.py           # write metadata/ubc_ocean.csv
python scripts/tcga_benchmark.py all --protocol benchmarks/ubc_ocean/protocol.yaml
```

Then submit through the resume-aware launcher, which skips the experiments whose
features are absent and records why:

```bash
./launch_pgvl.sh --dry-run --benchmarks ubc_ocean
```
