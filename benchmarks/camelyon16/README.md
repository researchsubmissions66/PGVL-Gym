# CAMELYON16 benchmark

Lymph-node metastasis detection, run through the same experiment contract as
every other cohort. It is the one binary *detection* task in the collection:
the signal is a small tumor region inside an otherwise normal node, so it tests
whether a method's aggregation can find sparse evidence rather than classify
global morphology the way subtyping does.

## Status: declared, not yet runnable

**No CAMELYON16 data is present on this machine** — neither the annotation
tables nor any extracted features. The cohort is therefore declared
`metadata_availability: future` and every feature source is marked
`availability: future`, so `scripts/tcga_benchmark.py` skips it cleanly and the
launcher reports it instead of submitting jobs that would fail on a GPU.

This directory exists so the cohort is a first-class benchmark the moment its
data lands, rather than something to reconstruct later.

## What is needed to activate it

1. **Annotations.** The protocol expects the two official partition tables:

   ```
   ${PGVL_STORAGE_ROOT}/metadata/RAW_DATA_Cleaned/CAMELYON16/CAM16_data_train.csv
   ${PGVL_STORAGE_ROOT}/metadata/RAW_DATA_Cleaned/CAMELYON16/CAM16_data_test.csv
   ```

   with `source_filename` as the slide id, `patient_id` as the case id, and
   `cancer_type` as the label (lowercased to `normal` / `tumor`). The original
   partition is retained in the manifest as `source_partition`; the protocol
   then builds its own five-fold outer evaluation over all slides, so the
   official split is provenance, not the evaluation.

   The published inventory is 270 official-train plus 129 official-test slides,
   399 total (239 normal, 160 tumor), one slide per identifier — so every slide
   is its own case.

2. **Features**, extracted to the same layout as UBC-OCEAN:

   ```
   ${PGVL_STORAGE_ROOT}/CAMELYON16/<resolution>_<px>px_0px_overlap/features_<encoder>/<slide_id>.h5
   ```

   CONCH at 5x/10x/20x and KEEP at 20x cover all eight registered experiments.

3. Drop the `metadata_availability: future` key from the cohort and the
   `availability: future` keys from the feature sources, then regenerate:

   ```bash
   python scripts/tcga_benchmark.py all --protocol benchmarks/camelyon16/protocol.yaml
   ./launch_pgvl.sh --dry-run --benchmarks camelyon16
   ```

## Scope note for reporting

FOCUS's upstream repository publishes launch scripts and few-shot splits for
CAMELYON, but its paper draws from CAMELYON16 *and* CAMELYON17. This protocol
covers CAMELYON16 only, so results here must not be reported as an exact
reproduction of the paper's combined CAMELYON cohort.

CoD-MIL's upstream `main.py` has a CAMELYON16 task, but its public prompt
directory ships only the kidney/RCC prompt CSV and precomputed tensors, so the
generalized config compiles a chain from the dataset prompt profile and encodes
it at runtime with the configured CLIP-RN50 tower.
