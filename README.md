# 🏋️ PGVL-Gym

PGVL-Gym is a reproducible benchmark for few-shot and zero-shot whole-slide
pathology vision-language methods. It standardizes datasets, feature
provenance, patient-disjoint folds, training, and reporting while explicitly
recording whether each adapter is vendored, mixed, or a partial local
reimplementation.

The normal workflow is:

```text
configure local paths → generate run YAMLs → preflight → train → aggregate
```

## 📦 1. Install the environment

The supported Python range is 3.10–3.11. Create the base environment with
Conda:

```bash
conda env create -f environment.yml
conda activate pgvl-gym
pip install -e .
```

Install only the optional dependencies needed by your method:

```bash
pip install -e '.[cod-mil]'
pip install -e '.[pathpt-musk]'
pip install -e '.[convlm]'
```

Available extras are listed in `pyproject.toml`. Foundation-model checkpoints
are not downloaded automatically; benchmark runs are designed for local or
offline model caches.

## ⚙️ 2. Configure machine-local paths

Copy the environment template and edit the ignored local file:

```bash
cp .env.example .env
```

```dotenv
PGVL_REPO_ROOT=/path/to/PGVL-Gym
PGVL_USER_ROOT=/path/to/user-root
PGVL_STORAGE_ROOT=/path/to/storage-root
PGVL_CONDA_ENV=/path/to/project/envs/pgvl-gym
```

Committed protocols, manifests, splits, and configs use references such as
`${PGVL_REPO_ROOT}` and `${PGVL_STORAGE_ROOT}`. Python commands load `.env`
automatically. The launch shell scripts source it as well. Existing process
environment variables take precedence, which is useful on a cluster.

Do not commit `.env`; it is intentionally ignored.

## 🧬 3. Generate experiment YAMLs

Each cohort has one source-of-truth protocol:

```text
benchmarks/<cohort>/protocol.yaml
```

Edit its cohort metadata, feature registry, checkpoints, methods, folds, and
shot counts, then generate manifests, splits, method configs, and the run
matrix:

```bash
python scripts/tcga_benchmark.py all \
  --protocol benchmarks/tcga_brca/protocol.yaml
```

Useful stages are `inventory`, `prepare`, `configs`, `validate`, `aggregate`,
and `all`. Generated configs appear under
`benchmarks/<cohort>/configs/<experiment>/`; runnable rows are indexed in
`benchmarks/<cohort>/run_matrix.csv`.

Feature extraction may finish after those artifacts are generated. Campaign
planning refreshes `feature_coverage.csv`, `missing_feature_files`, and the
feature-derived `ready` state directly from the existing manifests before every
plan. It does not rebuild prompts, manifests, splits, or configs. Thus a pending
feature set remains a clean skip, then becomes runnable automatically once all
of its referenced files arrive. Use `--no-refresh-readiness` only when you
deliberately need to inspect the frozen matrix cells.

Check one generated run before allocating a GPU:

```bash
python scripts/preflight.py \
  benchmarks/tcga_brca/configs/focus/brca_4shot.yaml

python scripts/preflight.py run.yaml --features
python scripts/preflight.py run.yaml --features --deep
python scripts/preflight.py run.yaml --prompts --encoders
python scripts/preflight.py run.yaml --quick
python scripts/preflight.py --system
python scripts/preflight.py run.yaml --strict --json
```

The preflight command is a read-only doctor: failures include suggested fixes.
Normal mode does not construct a model or load a feature tensor; explicit
`--deep` mode opens feature payloads to verify keys, shapes, widths, and finite
values. Shared pickle stores are also checked for key/ID alignment, duplicate
normalized IDs, and coverage of every slide in the manifest. It checks configured assets for
missing, empty, unreadable, unresolved, and wrong-type paths; verifies that the results
directory is safe and writable or creatable; rejects malformed manifests and
duplicate slide rows or feature aliases; measures individual and joint feature coverage; validates
fold CSV structure and identities; and detects slide or patient leakage between
train, validation, and test partitions. Method-aware checks also catch omitted
feature, prompt, report, map, and encoder inputs before model construction, and
validate FOCUS and ViLa-MIL's native positional low-then-high prompt banks,
and the MAPLE, MSCPT, PathPT, TOP, SLIP, and CoD-MIL
schemas, WSI-FiVE's six-question/structured-answer/evaluation banks,
plus the MUSE and ConVLM prompt banks. For SLDPC it hashes the ordered class
tokens actually embedded by Stage 1/2 separately from any optional TITAN
zero-shot synonym YAML, rather than accepting a merely present JSON, YAML, or
CSV or attributing an unused bank to training. Flat `splits_<fold>.csv` and upstream
`fold<fold>.csv` tables are checked against the same phase/label contract used
at runtime; one unscoped phase table cannot be silently reused across folds.
It rejects non-unit batches for variable-length bag methods, invalid optimizer
or staged-training values, and invalid batch-failure tolerances as configuration
errors. Runtime checks also validate CoD-MIL correspondence-map shape, integer
type, coverage, and index bounds.

Useful doctor modes are:

| Option | Purpose |
| --- | --- |
| `--system` | Check Python 3.10–3.11, all core packages (including the supported and mutually compatible Torch/torchvision releases), `.env`, and all PGVL root directories; a run YAML is optional. |
| `--quick` | Check feature roots without statting every manifest row; equivalent to `--no-feature-scan`. |
| `--deep` | Open every available referenced feature and validate its payload; incompatible with `--quick`. |
| `--min-feature-coverage N` | Temporarily override the configured coverage threshold with a fraction from 0 through 1. |
| `--strict` | Turn warnings, including explicitly allowed partial coverage, into a failing readiness gate. |
| `--json` | Emit schema-versioned JSON with summaries, timings, host diagnostics, and per-config results. |
| `--verbose` | Include successful resolved paths, asset types, and file sizes. |
| `--quiet` | Print only findings and the final diagnosis. |
| `--no-color` | Disable ANSI color explicitly; redirected output disables it automatically. |

Selectors `--assets`, `--features`, `--prompts`, `--encoders`, and `--splits`
can be combined; `--all` or omitting selectors runs every check. Multiple YAMLs
or shell globs are accepted. A healthy diagnosis exits zero, diagnosed failures
exit one, and invalid command arguments exit two. See
[Commands and run lifecycle](docs/commands.md#diagnose-a-run-with-the-doctor)
for the full interface and JSON contract.

Prompt provenance is tracked by method and by role in
[`text_prompts/PROVENANCE.json`](text_prompts/PROVENANCE.json). In particular,
TOP's standard NSCLC declaration uses the released 26-instance code bank and
the exact released lung bag initializers; its longer supplementary lung
descriptions are preserved as an explicitly unwired alternative. TOP tasks for
which the authors published no bag initializer are labeled
`upstream_instance_with_random_classname_bag`, rather than being presented as a
fully upstream prompt condition. The doctor validates TOP prompt structure,
ordered labels, learnable-slot counts, role/usage declarations, and both file
and semantic prompt-bank hashes. Runtime, benchmark generation, and the doctor
share that loader and derive `prompt_provenance`/`prompt_source` from the active
instance and bag roles, so a supplementary or modified bank cannot retain the
standard upstream identity.

SLIP's TCGA-NSCLC condition now uses the complete released bank: the exact
template, slide-class groups, and all 17 nested tissue name/description pairs.
The older flattened `Name: description` conversions are unwired because SLIP
embeds and averages the two texts separately. CAMELYON16, TCGA-BRCA, TCGA-RCC,
and UBC-OCEAN use clearly labeled generated task extensions in the same runtime
shape. The exact copied/generated inventory is recorded in
[`text_prompts/PROVENANCE.json`](text_prompts/PROVENANCE.json).

MAPLE's Lung, RCC, and BRCA attribute graphs are byte-exact upstream copies;
UBC-OCEAN and CAMELYON16 are labeled task extensions because MAPLE released no
banks for them. MAPLE class mappings are order-sensitive, so the doctor and
runtime reject reordered keys and verify the pinned upstream hashes. The local
runtime also corrects the released entity-major/class-major reshape mismatch
that otherwise associates attribute text with the wrong logits. This deviation
is disclosed in the provenance ledger and
[`docs/design-decisions.md`](docs/design-decisions.md#maple-prompt-origins-and-ordering).

MUSE's CAMELYON, TCGA-NSCLC, and TCGA-BRCA knowledge banks are byte-exact
copies of the authors' six 300-description CSVs. MUSE released no RCC or
UBC-OCEAN bank; those benchmark conditions are explicitly labeled generated
MUSE-schema extensions even though their underlying descriptions came from
released MSCPT assets. The doctor validates the native `,0` header, sequential
row indices, class-to-file binding, row counts, hashes, and declared
provenance. See
[`docs/design-decisions.md`](docs/design-decisions.md#muse--upstream-and-generated-prompt-banks).

ConVLM publishes neither the `att_splits.mat` consumed by training nor a usable
attribute-bank builder. Consequently, none of the five checked-in ConVLM JSON
banks is described as upstream: all are generated, hashed, class-order-bound,
and unwired by default. The doctor rejects prompt provenance drift and anonymous
attribute tensors; encoded banks must include their source-prompt digest,
encoder feature-space ID, and checkpoint hash. The local precomputed-patch-bag
adapter is also disclosed as a reconstruction because the released training
path consumes RGB images. See
[`docs/design-decisions.md`](docs/design-decisions.md#convlm--missing-upstream-attributes-and-a-local-feature-bag-reconstruction).

## 🚀 4. Run a configuration

```bash
python train.py \
  --method focus \
  --config benchmarks/tcga_brca/configs/focus/brca_4shot.yaml \
  --device cuda:0
```

For a campaign:

```bash
./launch_pgvl.sh --dry-run
./launch_pgvl.sh --cohort brca --shots 4 --limit 3
./launch_pgvl.sh
```

Dry-run planning intentionally does not import Torch, h5py, or method models,
so it can inspect the campaign from a login-node bootstrap Python. Real
submission requires `PGVL_CONDA_ENV`; the compute wrapper activates that exact
environment and the launcher refuses to fall back to a partial site module.

The launcher refreshes feature readiness, skips unavailable assets, avoids queued/completed runs, and
resumes only when the saved method and executable resolved config match. It
validates the same fingerprint as `train.py`, counts exact completed fold
indices (including states with holes), and requires finite validation loss plus
a valid test result for each completed fold. It reports corrupt, duplicate, or
out-of-range resume state instead of marking it done.
`train.py` also requires the CLI method and YAML `method` to resolve to the
same registered adapter; a mismatch exits as a configuration error before it
creates output files.
It likewise rejects an invalid/unavailable `--device` or an out-of-range CUDA
index before creating run state.
`--rerun` is a real from-scratch run: the trainer archives existing metrics,
config, checkpoints, prediction CSVs, and TensorBoard state before fold 0.
Campaign row, regeneration, and submission errors produce a non-zero exit after
the full report is written;
`--best-effort` is the explicit automation override.
The launcher proves the log/report destinations are writable before the first
submission and refuses a report path that would overwrite campaign inputs or
run state.
Malformed matrix booleans/counts, duplicate SLURM job names, and shared results
directories are errors rather than implicit skips or duplicate submissions.
The launcher also rejects incomplete/duplicate matrix headers and a `ready=true`
row whose component readiness evidence or missing-asset counts contradict it.
Best/final checkpoints, metrics, config snapshots, and prediction CSVs are
replaced atomically, so a preemption cannot leave a partial canonical artifact.
Epoch losses are sample-weighted, so a shorter final batch cannot receive the
same influence as a full batch.
Each fold also receives a fresh adapter and a private config copy. Prompt banks,
staged-training state, optimizer references, and derived adapter defaults
therefore cannot leak across folds or change the saved resume fingerprint.
Each results directory is also protected by a process lock, so duplicate
launches cannot write checkpoints, TensorBoard events, or metrics concurrently.
Result aggregation revalidates that fingerprint before reading any fold and
uses the same population-standard-deviation convention as the trainer.

Generated results also carry `implementation_provenance` and
`upstream_fidelity`. These are independent of `encoder_provenance`: a backbone
can be natively supported while the local objective remains partial. Set
`require_upstream_fidelity: true` to make the doctor reject partial adapters.

FOCUS now uses byte-exact upstream prompt CSVs for CAMELYON16, TCGA-NSCLC, and
UBC-OCEAN, pinned to commit `66c4015d5ba09657f4c8183bc06947faecd5b01f`.
TCGA-BRCA and TCGA-RCC remain generated because FOCUS released no
matching banks. FOCUS's actual format is headerless and positional—not the
earlier local named table—and runtime and doctor checks enforce its file-class
binding, provenance, file hash, and ordered prompt-bank hash. UBC's released
class order is explicitly reordered to the benchmark label order. The exact
upstream UBC quoting defect is preserved and disclosed in the provenance file.

ViLa-MIL is kept separate from FOCUS even though both native loaders use the
same positional schema and two magnifications.
The exact headerless upstream Lung and RCC CSVs are copied from commit
`68a11cf0d5cf092dd980f0da1cb38ccac8747a82`; BRCA, UBC-OCEAN, and CAMELYON16
are generated task extensions in the same native low-then-high layout. The
released RCC spelling `CRCC` is preserved and explicitly bound by position to
the benchmark label `CHRCC`. Runtime and doctor checks enforce file order,
format, provenance, and both file and class-bound prompt hashes.

MSCPT uses copied upstream banks for TCGA-NSCLC, TCGA-RCC, and UBC-OCEAN,
while the TCGA-BRCA IDC/ILC and CAMELYON16 banks are local task extensions and
report `prompt_provenance: generated`. The original `Lung.json` is preserved,
but provenance records the nine LUSC entries that describe
adenocarcinoma-associated morphology; this known upstream content issue must be
disclosed with NSCLC results. MSCPT's released `BRCA.json` is retained as an
upstream asset but is not substituted because it represents a different
High/Low recurrence/grade task.

PathPT generated benchmark configs now use
`training_mode: upstream_patch_ssl`: prompts are selected on the training fold only, a
synthetic `Normal` patch class is added for subtype tasks, patch supervision
uses the vendored `PatchSSLoss`, and evaluation uses the released patch-voting
rule. `simplified_slide_ce` remains available only to reproduce older local
runs. BRCA and UBC-OCEAN use PathPT's upstream 22-template synonym banks;
NSCLC and RCC use explicitly generated extensions. CAMELYON uses the upstream
Normal/Tumor bank but is marked partial because binary slide classification is
an adaptation of PathPT's tumour-subtyping protocol. Its upstream malformed
concatenated Normal synonym is preserved and disclosed in
`text_prompts/PROVENANCE.json`.

SLIP preserves prompt-bank structure as well as wording. TCGA-NSCLC uses the
pinned upstream TCGA bank verbatim; its tissue short names and descriptions are
encoded independently and averaged exactly as released. The four benchmark
cohorts for which SLIP published no matching bank use generated extensions and
report `prompt_provenance: generated`. Legacy flattened upstream conversions
remain available only for audit and are marked `derived`, not upstream.

MAPLE uses exact released prompt graphs for TCGA-NSCLC, TCGA-RCC, and
TCGA-BRCA. Its UBC example now selects an explicit generated extension rather
than the previously missing `UBC_attributes.json`. Prompt dictionary order is
validated against classifier order because MAPLE turns insertion order directly
into logit order; the corrected runtime also preserves the entity-major order
in which the prompt learner emits class attributes.

CoD-MIL treats the unchanged upstream RCC CSV as its canonical ordered prompt
bank. The released CLIP-RN50 tensor is retained for audit only because it has 30
rows against the CSV's 27 and shifts the model's positional prompt groups. RCC
therefore selects a verified 27-row RN50 re-encoding whose payload records the
source text, hashes, feature space, and row roles; the compiler, doctor, and
runtime reject unbound legacy tensors. This does not lock the bank to CLIP:
`scripts/build_cod_mil_prompt_features.py` also supports upstream's PLIP and
QuiltNet families, provided prompt and patch features use the same encoder
space. Those width-parameterized paths are labelled partial implementation
extensions because upstream model code hardcodes RN50's width. Full provenance
is recorded in `text_prompts/PROVENANCE.json` and the reasoning is documented
in `docs/design-decisions.md`.

WSI-FiVE's native NSCLC mode now preserves the three distinct text roles in
the official release. The six upstream questions, stored in a derived JSON
container, condition patch aggregation; a complete 939-case answer bank forms
the training-fold-only candidate bank for the native contrastive objective. It
contains 912 nonblank upstream six-answer records reproducibly normalized from
the two released workbooks plus 27 explicitly generated conservative
completions for blank upstream cells;
and the two upstream LUAD/LUSC diagnostic descriptions, stored in a derived
JSON container, are the only comparison text used for
validation and test classification. Per-slide answers are never inference
inputs. RCC and UBC examples remain explicitly labelled
`simplified_classnames`, because WSI-FiVE did not publish native answer and
evaluation banks for those tasks. Asset-level sources and this distinction are
recorded at pinned commit `07344c9ac6eef919fcd1440877ea796feef7445a` in
`text_prompts/PROVENANCE.json` and `docs/design-decisions.md`.
`scripts/build_wsi_five_prompt_assets.py --check` reproduces the answer CSV;
runtime and the doctor reject any byte, ordering, schema, or provenance drift
across the question, answer, and evaluation roles.

## 📄 Example run YAML

Generated YAMLs are preferred, but this shows the core contract:

```yaml
method: focus
backbone: conch
backbone_weights: ${PGVL_STORAGE_ROOT}/models/conch.bin
feature_space_id: hf:MahmoodLab/conch
feature_dim: 512
implementation_provenance: vendored
upstream_fidelity: upstream

dataset_csv: ${PGVL_REPO_ROOT}/benchmarks/tcga_brca/data/brca/manifest.csv
split_dir: ${PGVL_REPO_ROOT}/benchmarks/tcga_brca/splits/brca/4shot
feature_path_column: feature__conch_v1_20x
feature_path_column_l: feature__conch_v1_20x
feature_key: features
min_feature_coverage: 1.0

n_classes: 2
classnames:
  - invasive ductal carcinoma
  - invasive lobular carcinoma
label_dict:
  IDC: 0
  ILC: 1
text_prompt_path: ${PGVL_REPO_ROOT}/text_prompts/focus/TCGA_BRCA_two_scale_text_prompt.csv

shots: 4
k: 5
k_start: 0
k_end: 5
seed: 1
epochs: 200
batch_size: 1
lr: 0.0001
weight_decay: 0.00001
early_stopping: true
results_dir: ${PGVL_REPO_ROOT}/results/focus/brca/4shot
```

Feature provenance and dimensions are part of the experiment identity. Do not
change a generated YAML in place and reuse its results directory.

## 🔗 Register a backbone

A backbone registration declares its real capabilities and returns an
`EncoderBundle`. Registration does not automatically make every method
compatible; each method's `MethodBackboneContract` still decides whether the
combination is native, adaptable, or blocked.

```python
import torch
from common.backbones import (
    BackboneCapability,
    BackboneSpec,
    EncoderBundle,
    register_backbone,
)

SPEC = BackboneSpec(
    name="my-backbone",
    family="my-family",
    feature_space_id="org/my-backbone@revision",
    capabilities=frozenset({BackboneCapability.TILE_ENCODE}),
    tile_dim=768,
    revision="commit-or-checksum",
)

def build_my_backbone(*, weights_path=None, device="cpu", **kwargs):
    model = load_my_model(weights_path).to(device)  # your implementation
    tile_encoder = MyTileEncoder(model)              # implements encode_tiles
    return EncoderBundle(
        raw_model=model,
        spec=SPEC,
        tile=tile_encoder,
        metadata={"weights_path": weights_path},
    )

register_backbone(SPEC, build_my_backbone)
```

For a permanent built-in registration, place the spec and loader in
`common/backbones/factory.py`, export any wrapper from `common/backbones/`, and
add interface tests. Inspect compatibility without loading weights:

```bash
python scripts/list_backbone_compatibility.py
python scripts/list_backbone_compatibility.py --method pathpt --json
```

See `docs/BACKBONE_INTERFACES.md` for capability definitions and method swap
boundaries.

## 🗺️ Repository map

```text
train.py                  unified training and reporting
common/                   datasets, backbone contracts, shared model blocks
methods/<name>/           paper-specific model plus BaseMethod adapter
benchmarks/<cohort>/      protocol and generated experiment artifacts
scripts/tcga_benchmark.py protocol compiler
scripts/preflight.py      filesystem and feature health check
configs/                  small hand-authored examples
docs/                     detailed design and method documentation
```

Run tests with `pytest -q`. Contribution and extension guidance lives in
`CONTRIBUTING.md` and `docs/extending.md`.

## 🙏 Acknowledgments

| Repository | Method/Role | License |
| --- | --- | --- |
| [dddavid4real/FOCUS](https://github.com/dddavid4real/focus) | FOCUS | 🔒 [Apache-2.0](https://opensource.org/licenses/Apache-2.0) |
| [Jiangbo-Shi/ViLa-MIL](https://github.com/Jiangbo-Shi/ViLa-MIL) | ViLa-MIL | 🔒 [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Assumed) |
| [Jiangbo-Shi/CoD-MIL](https://github.com/Jiangbo-Shi/CoD-MIL) | CoD-MIL | 🔒 [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Assumed) |
| [JJ-ZHOU-Code/MAPLE](https://github.com/JJ-ZHOU-Code/MAPLE) | MAPLE | 🔒 [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Assumed) |
| [Hanminghao/MSCPT](https://github.com/Hanminghao/MSCPT) | MSCPT | 🔒 [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Assumed) |
| [MAGIC-AI4Med/PathPT](https://github.com/MAGIC-AI4Med/PathPT) | PathPT | 🔒 [MIT](https://opensource.org/licenses/MIT) |
| [miccaiif/TOP](https://github.com/miccaiif/TOP) | TOP | 🔒 [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Assumed) |
| [LTS5/SLIP](https://github.com/LTS5/SLIP) | SLIP | 🔒 [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Assumed) |
| [ls1rius/WSI_FiVE](https://github.com/ls1rius/WSI_FiVE) | WSI-FiVE | 🔒 [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Assumed) |
| [JiahaoXu-god/CVPR2026_MUSE](https://github.com/JiahaoXu-god/CVPR2026_MUSE) | MUSE | 🔒 [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Assumed) |
| [BasitAlawode/ConVLM](https://github.com/BasitAlawode/ConVLM) | ConVLM | 🔒 [MIT](https://opensource.org/licenses/MIT) |
| [linlu2022/SLDPC](https://github.com/linlu2022/SLDPC) | SLDPC | 🔒 [Apache-2.0](https://opensource.org/licenses/Apache-2.0) |

| Repository | Method/Role | License |
| --- | --- | --- |
| [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) | CLAM Scaffold | 🔒 [GPL-3.0](https://opensource.org/licenses/GPL-3.0) |
| [KaiyangZhou/CoOp](https://github.com/KaiyangZhou/CoOp) | CoOp Blocks | 🔒 [MIT](https://opensource.org/licenses/MIT) |
