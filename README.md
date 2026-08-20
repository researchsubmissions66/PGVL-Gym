<p align="center">
  <img src="docs/assets/logo_gym_no_text.png" alt="PGVL-Gym Logo" width="560">
</p>

<h1 align="center">PGVL-Gym</h1>

<p align="center">
  <a href="https://researchsubmissions66.github.io/PGVL-Gym/project/"><img src="https://img.shields.io/badge/Project-Website-6b21a8?style=for-the-badge" alt="Project website"></a>
  <a href="https://researchsubmissions66.github.io/PGVL-Gym/"><img src="https://img.shields.io/badge/Documentation-6b21a8?style=for-the-badge" alt="Documentation"></a>
  <a href="https://github.com/researchsubmissions66/PGVL-Gym"><img src="https://img.shields.io/badge/Source-1e1b4b?style=for-the-badge&logo=github" alt="Source"></a>
</p>

---

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

The preflight command is a read-only doctor. Failures include suggested fixes. Normal mode does not construct a model or load a feature tensor.

**Key Checks:**
- **Feature Payloads (via `--deep`):** Verifies keys, shapes, widths, and finite values.
- **Data Integrity:** Checks shared pickle stores for key/ID alignment, duplicate IDs, and coverage.
- **Asset Validation:** Scans for missing, empty, unreadable, or wrong-type paths.
- **Safety:** Verifies that the results directory is safe and writable.
- **Leakage Prevention:** Detects slide or patient leakage between train, validation, and test partitions.
- **Method-Specific Schemas:** Validates prompt banks across all methods (FOCUS, ViLa-MIL, MAPLE, MSCPT, PathPT, TOP, SLIP, CoD-MIL, WSI-FiVE, MUSE, ConVLM, SLDPC).
- **Configuration & Runtime:** Rejects non-unit batches for variable-length methods, unscoped phase tables, and invalid staged-training values. Validates CoD-MIL map shapes and bounds.

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

### 📝 Prompt Provenance

Prompt provenance is tracked by method and by role in [`text_prompts/PROVENANCE.json`](text_prompts/PROVENANCE.json).

- **TOP:** Standard NSCLC uses the released 26-instance code bank and lung bag initializers. Unwired alternatives are explicitly labeled. The doctor validates prompt structure, ordered labels, and hashes.
- **SLIP:** TCGA-NSCLC uses the complete released bank. Missing cohort banks are clearly labeled as generated task extensions.
- **MAPLE:** Lung, RCC, and BRCA use byte-exact upstream copies. The local runtime corrects a released reshape mismatch.
- **MUSE:** CAMELYON, TCGA-NSCLC, and TCGA-BRCA use byte-exact copies. RCC and UBC-OCEAN are labeled as generated MUSE-schema extensions.
- **ConVLM:** Due to missing upstream attribute builders, all JSON banks are generated, hashed, and unwired by default. The local patch-bag adapter is disclosed as a reconstruction.

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

### 📋 Campaign Planning & Execution

Dry-run planning inspects the campaign directly from a bootstrap Python without importing heavy libraries. Real submissions require `PGVL_CONDA_ENV`.

**Launcher & Trainer Features:**
- **Smart Resumption:** Skips unavailable assets, avoids queued runs, and resumes only on exact config fingerprint matches.
- **State Validation:** Requires finite validation loss and valid test results for each fold. Corrupt or out-of-range states are rejected.
- **Safety First:** Proves log destinations are writable, locks results directories, and replaces checkpoints atomically to prevent partial writes.
- **Isolation:** Each fold receives a fresh adapter and private config copy, preventing state leakage.
- **Strict Checks:** Rejects mismatched adapters, invalid devices, duplicate job names, and contradictory readiness headers.

Use `--rerun` for a clean slate run (archives existing state) and `--best-effort` for explicit automation override.

Generated results also carry `implementation_provenance` and
`upstream_fidelity`. These are independent of `encoder_provenance`: a backbone
can be natively supported while the local objective remains partial. Set
`require_upstream_fidelity: true` to make the doctor reject partial adapters.

### 🧩 Method Details & Upstream Fidelity

- **FOCUS**: Uses byte-exact upstream CSVs for CAMELYON16, TCGA-NSCLC, and UBC-OCEAN. Enforces file-class binding and provenance checks.
- **ViLa-MIL**: Uses exact headerless upstream Lung and RCC CSVs. Missing cohorts are generated task extensions matching native layouts.
- **MSCPT**: Uses upstream banks for NSCLC, RCC, and UBC-OCEAN. Upstream content defects (like LUSC morphology entries) are preserved but disclosed.
- **PathPT**: Uses `upstream_patch_ssl` training mode with synthetic patch classes and vendored losses.
- **SLIP**: Preserves prompt-bank structure natively (averaging separate texts), diverging from older flattened conversions.
- **MAPLE**: Preserves entity-major order and validates prompt dictionary against classifier order.
- **CoD-MIL**: Uses RCC CSV as the canonical bank, avoiding arbitrary feature tensors. Fully supports PLIP/QuiltNet via a verified re-encoder.
- **WSI-FiVE**: Preserves three distinct text roles (questions, answers, evaluation), avoiding per-slide answers as inference inputs.

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
