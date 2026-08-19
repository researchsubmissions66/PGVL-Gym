# Design decisions and fidelity trade-offs

Every benchmark that consolidates published methods makes choices the original
papers never had to make. This page records those choices for PGVL-Gym: what was
decided, what the alternative was, and — where a decision costs fidelity — what
a reader must not conclude from the resulting number.

It is deliberately written as a ledger rather than a rationale. A reader
comparing our numbers to a paper's needs to know exactly where the two diverge.

---

## 1. Provenance is recorded, never inferred

The framework's central commitment: **a number carries the conditions that
produced it.** Provenance fields travel from the protocol into the
generated config, the run matrix, and `aggregate_results.csv`.

| Field | Values | Answers |
| --- | --- | --- |
| `encoder_provenance` | `native`, `adapted` | Did the method's own code support this encoder, or does a trainable projection bridge it? |
| `prompt_provenance` | `upstream`, `generated`, `classname_template`, or a role-qualified mixed condition such as `upstream_instance_with_random_classname_bag` | Where did the embedded text for every prompt role come from? |
| `prompt_source` | per-method, e.g. `cod_chain_runtime_clip_rn50` | Which asset and encoding path? |
| `implementation_provenance` | `vendored`, `mixed`, `reimplemented`, or a precise partial variant | Which method implementation actually ran? |
| `upstream_fidelity` | `upstream`, `partial`, `local_baseline` | May this number be described as an upstream reproduction? |

All provenance fields are **derived from the declared contract and the files on
disk**, not set by hand, so they cannot drift from what the run actually does.

### FOCUS prompt origins

No FOCUS CSV currently checked into `text_prompts/` is a verbatim upstream
copy. The distinction is recorded per asset in `text_prompts/PROVENANCE.json`
and propagated into generated configs:

| Local bank | Origin | Upstream status |
| --- | --- | --- |
| CAMELYON16 | generated/reworded | FOCUS publishes a CAMELYON bank, but the local wording is a shorter rewrite |
| TCGA-NSCLC | generated/reworded | FOCUS publishes a TCGA-Lung bank, but the local wording is expanded and rewritten |
| UBC-OCEAN | generated/reworded | FOCUS publishes a UBC-OCEAN bank, but the local wording is condensed and rewritten |
| TCGA-BRCA | generated | FOCUS publishes no BRCA prompt CSV |
| TCGA-RCC and RCC-GEPA | generated | FOCUS publishes no RCC prompt CSV |

Accordingly, these runs report `prompt_provenance: generated`. An explicit YAML
path describes how an asset was selected; it does not turn locally authored
text into upstream text.

### MSCPT prompt origins

MSCPT prompt provenance follows the embedded text, not whether the protocol
selected its path through a legacy field or an explicit `prompts:` mapping:

| Benchmark bank | Origin | Fidelity note |
| --- | --- | --- |
| TCGA-NSCLC `Lung.json` | upstream copy | Preserved verbatim, including the released LUSC-content issue described below |
| TCGA-RCC `RCC.json` | upstream copy | Task-matched released bank, 10 `small_mag` and 30 `big_mag` prompts per class |
| UBC-OCEAN `UBC-OCEAN.json` | upstream copy | Task-matched released bank, 10 `small_mag` and 30 `big_mag` prompts per class |
| TCGA-BRCA IDC/ILC | generated | MSCPT's released `BRCA.json` is a different High/Low recurrence/grade task; the local bank has 10 prompts at each scale |
| CAMELYON16 | generated | MSCPT releases no CAMELYON16 bank; this one is compiled from the local canonical prompt profile |

The upstream `Lung.json` places nine adenocarcinoma-associated descriptions in
the `LUSC` block: `small_mag` indices 0 and 2, and `big_mag` indices 0, 1, 3,
8, 9, 28, and 29 (zero-based). They mention glandular/acinar architecture,
mucin, lepidic growth, signet-ring cells, bronchioloalveolar carcinoma, or
micropapillary morphology. MSCPT embeds these descriptions into the LUSC text
representation, so this can affect NSCLC results. The upstream file remains
unchanged: silently correcting it would create a new prompt condition while
still appearing to be an upstream reproduction. The exact indices and policy
are machine-readable in `text_prompts/PROVENANCE.json`.

Generated configs therefore report `upstream` for NSCLC, RCC, and UBC-OCEAN,
and `generated` for the task-extended BRCA and CAMELYON16 banks. The separate
`upstream_fidelity: partial` implementation label still applies because the
feature-only integration bypasses MSCPT's selected-5x raw-image visual-prompt
branch.

### MAPLE prompt origins and ordering

MAPLE releases complete two-scale attribute graphs for TCGA-Lung, TCGA-RCC,
and TCGA-BRCA only. The checked-in `LUNG_attributes.json`,
`RCC_attributes.json`, and `BRCA_attributes.json` files are byte-exact copies
from commit `c38d5d5d55deba3a44e9384c0efeee98e1aec36b`, with file hashes recorded in
the provenance manifest. MAPLE publishes no UBC-OCEAN or CAMELYON16 bank:
those are explicitly generated task extensions. The standalone UBC example
now has a real generated asset instead of pointing to a nonexistent file.

Class-key order is semantic, not cosmetic. MAPLE iterates `global_info` and
every entity's `attributes` mapping to build its class logits. The earlier
doctor compared only key sets, so a JSON reordering could silently permute
predictions relative to numeric labels. The shared loader, doctor, benchmark
validator, and runtime now require every mapping to match classifier order
exactly and verify registered upstream file hashes.

The released runtime also has an attribute-alignment defect. `PromptLearner`
appends prompts entity-major—every class for entity 0, then every class for
entity 1—but `obtain_entities_attr` reshapes that sequence as class-major.
Most entity/class scores therefore consume another pair's description. PGVL
restores the emitted entity-major shape directly. This is a disclosed upstream
bug fix: the prompt files remain unchanged, but corrected runs are not claimed
to reproduce that accidental permutation.

### TOP prompt origins

TOP has two distinct prompt roles. Its instance learner uses 26 task-agnostic
tissue prototypes; its bag learner uses task-specific class initializers. The
standard assets now follow the active literals in the authors' released code:

| Cohort/asset | Instance prototypes | Bag initializer | Reported condition |
| --- | --- | --- | --- |
| TCGA-NSCLC | 26 ordered entries copied from `knowledge_from_chatGPT` | Exact two `bagPrompt_ctx_init` strings from `train_TCGAFeat_MIL_CLIP.py` | `upstream` |
| CAMELYON16 | Same released 26-entry bank | Exact active `normal`/`tumor` initializers from `train_CAMELYONFeat_MIL_CLIP.py` | `upstream` when TOP is enabled |
| TCGA-BRCA | Same released 26-entry bank | No upstream BRCA initializer; random learned context plus IDC/ILC classnames | `upstream_instance_with_random_classname_bag` |
| TCGA-RCC / UBC-OCEAN examples | Same released 26-entry bank | No upstream task initializer; random learned context plus task classnames | `upstream_instance_with_random_classname_bag` |

The longer NSCLC class descriptions in supplementary Figure 4 are also copied
and retained as `top/tcga_nsclc_bag_prompts.json`, but are marked
`alternative_unwired`. They are a legitimate ablation condition, not the
released training-script condition. The adapter accepts either complete
code-faithful `ctx_init` literals or base prompts, checks class-index order and
ten-slot placement, and no longer inserts a second period between an instance
description and its learnable slots. It also preserves the release's tiny
recipe difference: TCGA concatenates the first instance slot directly, while
CAMELYON inserts a space. The instance asset records the pinned upstream commit
and a digest over all 26 ordered rendered prompts.

### SLIP prompt origins

SLIP assigns three separate roles to its released prompt bank: format
templates, slide-class prompts, and tissue-routing prompts. A tissue is itself
a text ensemble. The authors encode its short name and its description as two
independent prompts, normalize both embeddings, average them, and normalize
again. Consequently, joining the pair into one `Name: description` sentence
does not reproduce the released routing vector even though it preserves every
word.

| Benchmark bank | Origin | Selected condition |
| --- | --- | --- |
| TCGA-NSCLC | upstream copy | Exact TCGA template (`{}`), two nested slide-class groups, and all 17 ordered two-text tissue groups |
| CAMELYON16 | generated | Local tissue extension, expanded to the native two-text runtime shape |
| TCGA-BRCA | generated | Local tissue extension, expanded to the native two-text runtime shape |
| TCGA-RCC | generated | Local tissue extension, expanded to the native two-text runtime shape |
| UBC-OCEAN | generated | Local tissue extension, expanded to the native two-text runtime shape |

The complete released DHMC and PatchGastricADC22 banks are also copied for
their original datasets, including the gastric-specific template. The old
`*_tissues.json` conversions of those banks and TCGA are retained only as
`derived`, unwired audit artifacts. Generated configs and the runtime load a
complete bank, while the doctor verifies template arity, slide-class order,
nested tissue structure, and the digest of upstream banks. The importer now
preserves the complete source structure, so refreshing from upstream cannot
silently recreate the flattening bug.

### Why `native` and `adapted` must not share a results table

`pathpt_keep` and `muse_musk` can report identical accuracy and mean different
things. PathPT ships `PathPT_model_KEEP.py`, so its number measures KEEP under
the published code. MUSE with MUSK features keeps CONCH as the text tower and
learns a projection between them, so its number measures MUSE **plus that
projection** — the encoder cannot be credited alone.

Across the benchmark, 6 experiment entries (18 rows) are `adapted`; everything
else is `native`.

### Trade-off accepted

`encoder_provenance` has two values, not four. Prompt provenance is a separate
axis already carried by `prompt_source` and `prompt_provenance`; folding it in
would duplicate state and let the two drift.

---

## 2. Implementation provenance

The adapter registry is the source of truth for implementation fidelity. The
compiler writes its values into every config, matrix row, fold result, and
aggregate grouping; the doctor rejects contradictory declarations.

| Method | Upstream LOC used | Local model LOC | Status |
| --- | ---: | ---: | --- |
| `cod_mil`, `top` | 1,195–6,301 | 0 | fully vendored |
| `maple` | vendored model and objective | attribute-order correction | partial: fixes the released entity-major/class-major reshape defect |
| `mscpt` | vendored model | adapter only | partial: feature-only path bypasses raw-image visual prompting |
| `pathpt` | vendored model, loss, and prompt banks | lifecycle adapter | upstream in `upstream_patch_ssl` subtype mode; legacy slide-CE and CAMELYON adaptation are partial |
| `focus`, `vila_mil` | 245–409 | 3 | vendored (trivial shim) |
| `slip` | 4,411 | 114 | mixed, mostly vendored |
| `wsi_five` | ~1,000 used of 4,071 | 175 | rebuilt from vendored components; `FiVE.py` orchestrator is unusable |
| `muse` | 0 | 105 | reimplemented (self-declared: "feature-space portion") |
| `convlm` | 0 | 124 | reimplemented (self-declared) |
| `sldpc` | 0 | 246 | reimplemented (CPI/DHNO/SICL written here) |

`wsi_five` was the most misleading case and has since been rebuilt: the adapter
now drives the vendored `PatchFusionTransformer`, `MedCLIPTextModel` and
`LoraWrap`, restores the released answer-candidate objective, and reserves the
released diagnostic descriptions for evaluation. What remains unused is
`FiVE.py` (500 lines) and
`_datasets/pipeline.py` (1,950 lines) — and `FiVE.py` is not merely unused but
**unrunnable**, failing both import and construction (see §7). The remaining
limitations are the rebuilt orchestration and feature provenance, not a
missing prompt pipeline.

Partial adapters emit doctor warnings. A publication-oriented protocol can set
`require_upstream_fidelity: true` to turn those warnings into a hard gate.

---

## 3. Vision-language only, by construction

Every registered encoder is a dual-tower vision-language model. Vision-only
pathology foundation models — UNI, GigaPath, Virchow, Phikon, CTransPath,
H-optimus, ResNet50 — are **deliberately absent**, even though features for
several are extracted and available.

A method that learns or injects text prompts has no text tower to attach to.
Including them would produce cells that cannot run, not cells that are blocked
for an interesting reason.

Of 117 method × encoder combinations: 43 native, 29 adaptable, 45 blocked. See
[Compatibility analysis](compatibility-analysis.md) for the cause of every
blocked cell.

---

## 4. Skip visibly, never silently degrade

A benchmark matrix always contains configurations whose assets do not exist yet.
Three mechanisms keep those from producing numbers:

1. **Config generation** records unbuildable experiments in
   `skipped_configs.csv` with the specific missing asset, and does not emit a
   config.
2. **`common/preflight.py`** checks one resolved config against the filesystem
   before any model is built. Failure writes `skipped.json` and exits **3** —
   distinct from 0 (completed) and 78 (environment).
3. **`train.py`** fails on the first sample error by default
   (`max_batch_failure_rate: 0`). A run may explicitly opt into a non-zero
   sample-failure ceiling, but past that fraction it fails rather than reporting
   a number computed on a larger accidental subset. Counts land in
   `metrics.json` as sample counts under `sample_failures`, with the exact
   identities under `failed_slide_ids`.

### Trade-off accepted

Any non-zero ceiling changes the evaluated population and is therefore an
explicit protocol decision, not a hidden resilience default. Comparable
benchmark protocols keep the fail-fast value.

### Stale artifacts are pruned

Generated configs the protocol no longer produces are **deleted** after each
generation pass. A config left behind still names a results directory and still
looks valid to anything walking the config tree, so the protocol is treated as
the sole source of truth.

---

## 5. Encoder loading: pinned provenance over convenience

`feature_space_id` identifies the exact producer of a feature tensor, and
equal widths never establish compatibility. No adapter is inserted implicitly.

### PLIP below torch 2.6

`transformers` refuses `torch.load` on a `.bin` checkpoint under torch 2.6
(CVE-2025-32434), and `vinid/plip`'s default revision publishes only
`pytorch_model.bin`.

**Rejected:** pinning the repository's safetensors-only revision. That revision
carries no `config.json` and no tokenizer files, so the load fails on the
missing config instead — and offline compute nodes cannot fetch them.

**Chosen:** assemble one complete directory (config + tokenizer from the
default revision, `model.safetensors` from the other) and point `PLIP_CKPT` at
it. Verified first that all 400 tensors are bit-identical between the two
revisions. `transformers` prefers safetensors and that path never calls
`torch.load`. `scripts/pgvl_job.sh` exports it, guarded on the file existing so
an unprepared site degrades to the HF cache with a warning.

---

## 6. Per-cohort benchmark directories

Each cohort owns `benchmarks/<cohort>/` holding its `protocol.yaml` and
everything generated from it. Cohorts are independent so one whose data is not
ready cannot hold back the others.

**Trade-off:** shared feature sources are declared repeatedly across protocols.
Accepted, because a single protocol would couple every cohort's readiness.

---

## 7. Method-specific fidelity notes

### CoD-MIL — cross-magnification maps are reconstructed

The released code loads `map_10x_20x_files/<slide>.pt` but the repository ships
**neither the maps nor a generator for them**. `create_patches_fp.py` is stock
CLAM at a single magnification; `datasets/` is stock CLAM; `scripts/` holds only
split creation.

`scripts/generate_cross_magnification_maps.py` reconstructs them from patch
coordinates.

| Decision | Basis |
| --- | --- |
| **10x → 20x** | Confirmed — Table I reports 10x and 20x patches for all three of the paper's datasets, and matches the hardcoded `map_10x_20x_files` |
| **Centre containment** | *Inferred.* The paper introduces an alignment matrix but does not state the geometric rule. Robust to tissue filtering in a way index arithmetic is not |
| **Per-slide patch size** | `patch_size_level0` is read from each slide's H5 attributes, not a global constant |
| **Built from feature files, not the patch store** | The map's indices must address the bag the model loads; building from the feature files' own `coords` makes that true by construction |

The per-slide sizing is not a refinement — BRCA spans three scanner base
magnifications:

| base | 10x | 20x | slides |
| --- | ---: | ---: | ---: |
| 40x | 896 | 448 | 913 |
| 30x | 1344 | 672 | 8 |
| 20x | 448 | 224 | 39 |

A single global size would have bound roughly four times too many high-power
patches per row for 47 of 960 slides — silently, with no error.

Validation: all 960 maps are `(N_low, 4)`, zero empty rows, mean 3.90–3.94
matches per low patch, each high patch in exactly one parent. The 4-way nesting
also reconciles the code's `topk(A, 16)` with the paper's sampling sweep:
**16 low-power regions × 4 children ≈ 64 high-power instances**, landing on the
K≈64 the paper's Fig. 6 favours. The code's 16 and the paper's K are different
quantities.

!!! warning "Not comparable to the paper's numbers"
    Upstream `create_patches_fp.py` defaults to 256 px patches at stride 256;
    this benchmark's store holds 224 px patches. The grids differ, so a 10x tile
    covers a different tissue area and instance counts do not match Table I. The
    *method* is reproduced; the *numbers* are not directly comparable.

### CoD-MIL — prompt banks

The published bank is `C` low-power class prompts + `C` high-power class prompts
+ **a normal-tissue corpus** (21 rows for kidney: 6 organ structures + 15
organ-independent phenotypes). The auxiliary contrastive branch masks the most
discriminative low-power instances and contrasts them against *normal tissue* —
so the corpus is the objective, not padding.

The 15 organ-independent rows are reused **verbatim** across cohorts, including
upstream's `"tpithelial"` typo, because correcting it would change the embedding
and break correspondence with the released bank.

Upstream publishes a normal-tissue bank for **kidney only**. BRCA's six normal
breast structures are authored for this benchmark and marked
`_provenance: generated`; the config reports
`prompt_provenance: upstream_chain_with_generated_normal_tissue` so the mixed
bank is never reported as fully upstream.

!!! danger "The released CLIP RN50 prompt tensor does not match the published CSV"
    `text_prompt_kidney_v2.csv` has 27 rows. `..._plip.pt` and `..._quiltnet.pt`
    are exact 27-row encodings of it. **`..._clip_rn50.pt` has 30 rows.**

    Cosine alignment shows `csv[i] → released[i+3]`: the three extra rows are at
    the **front**, and released rows 3–9 match the CSV only at 0.92–0.97 (the
    class prompts were reworded), while rows 10–29 match at exactly 1.0000.

    So the CSV is **not** the source of the released CLIP tensor. Truncating the last
    three rows — the intuitive fix — would feed unknown text into the low-power
    branch and low-power text into the high-power branch, misaligning both class
    branches with no error raised.

    The original file remains unchanged as an audit-only upstream artifact.
    RCC configs instead select
    `rcc_text_prompt_features_clip_rn50_verified.pt`, an exact ordered
    re-encoding of all 27 upstream CSV rows with the official CLIP RN50
    checkpoint. The derived payload embeds the prompts, source/checkpoint
    hashes, feature-space ID, and half-open row-role spans. Compiler, doctor,
    and runtime checks reject a bare or reordered tensor.

The CSV—not CLIP—is the canonical bank. The local model width now follows
`feature_dim`, and `scripts/build_cod_mil_prompt_features.py` accepts the three
families for which upstream released RCC prompt artifacts: `clip-rn50`, `plip`,
and `quiltnet`. Upstream's model code itself hardcodes 1024 dimensions, so a
PLIP/QuiltNet run is honestly labelled a width-parameterized implementation
extension (`upstream_fidelity: partial`), not an upstream reproduction. It must
use patch bags and a verified prompt tensor from the same feature space. This
is not a width projection: vectors from unrelated encoders are never compared
merely because their dimensions match. Runtime-cached encoding can consume the
same CSV directly when the matching text checkpoint is available.

Upstream's model slices normal prompts as `text_feature[2C:-1]`, dropping the
final row — an ordinary tissue prompt, not a sentinel. That slice remains
unchanged from upstream; only the formerly hardcoded feature width is
parameterized for the explicitly labelled PLIP/QuiltNet extension.

### WSI-FiVE — the vision tower is not method-owned

The compatibility analysis originally attributed 18 blocked cells to "the method
owns its vision tower (`wsi_five`, `convlm`)". That framing was wrong for both
methods: each consumes offline visual features. For WSI-FiVE specifically,
with the shipped default `IS_IMG_PTH: True`, upstream sets
`self.visual = nn.Identity()` (MedCLIP vision is commented out), hardcodes
`embed_dim = 512`, never parses the CLIP state dict, and reads precomputed
features from CSV. The paper confirms it: *"we employed ResNet following [15] as
image encoder… while pre-trained BioClinicalBERT as text encoder"* — [15] is
DSMIL.

**No MedCLIP weights load on any path.** `MedCLIPTextModel`, the only MedCLIP
class instantiated, calls `AutoModel.from_pretrained('./Bio_ClinicalBERT')`.
Every `torch.load` in `MedCLIPModel.py` belongs to a vision class that is never
constructed, so `checkpoints/wsi_five/medclip_vit.bin` is unreachable and is not
a blocker. The name is code provenance — upstream's README says *"parts of the
codes are borrowed from X-CLIP, MedCLIP"* — which this project's documentation
had turned into a model claim.

#### The vendored orchestrator does not run

`FiVE.py` cannot be imported (it wants `VisionTransformer`; canonical CLIP
defines `VisualTransformer`) and cannot be constructed (`CLIP.__init__` raises
`Trying to create tensor with negative dimension -1` on the `vocab_size=-1`
arguments `build_model` passes). Reusing it was never an option, so the model is
rebuilt from the vendored components that do work.

#### What is reproduced

| Mechanism | Source |
| --- | --- |
| Bag aggregation, self-attention + cross-attention branches, concat fusion | vendored `PatchFusionTransformer` |
| Six clinical questions as cross-attention **queries** | release `PROMPT_LIST` |
| BioClinicalBERT text tower with LoRA (r=8, α=32) | vendored `MedCLIPTextModel` + `LoraWrap` |
| 16 learnable soft prompts injected into BERT embeddings | `encode_prompt_embed` |
| Positions from patch index within slide; `sample_range` padding masks | vendored `get_pos_embed` |
| `logit_scale = 300` | hardcoded upstream, not learned |

Verified: BioClinicalBERT weights load identically to the original checkpoint,
LoRA leaves 1.34M of 117.9M parameters trainable with the base frozen, a forward
pass over a 137-patch bag returns finite logits, and gradients reach both the
soft prompts and the fusion transformer.

#### Native text supervision and evaluation

FiVE does not train against class names and a slide's answers do not condition
its fusion transformer. The official release assigns three separate roles to
text:

| Text asset | Role | Local provenance |
| --- | --- | --- |
| Six clinical questions | Cross-attention queries that condition patch aggregation | Copied from the upstream lung configuration |
| Six GPT-derived answers per case | Training answer candidates and targets | Copied from upstream `gpt_preprocess/*.xlsx` into `nsclc_report_answers.csv` |
| LUAD/LUSC diagnostic descriptions | Validation/test comparison bank | Copied from upstream `LUAD_LUSC_labels_{train,val}_reid.csv` into `nsclc_evaluation_prompts.json` |

The NSCLC answer asset covers 912 of 946 metadata cases, with all six fields
present. For each benchmark fold, the adapter reads answers only for training
rows, constructs and hashes a unique candidate bank from that fold, randomly
drops zero to five aligned question/answer fields, removes `Unknown` answers,
and shuffles retained answer segments. This restores the released
`aug_question` and label-hashing semantics without admitting validation/test
answers into the bank. The saved fold trace records the candidate count, asset
sources, and a SHA-256 digest, but deliberately does not duplicate patient text.

At validation and test time the dataset does not require or return an answer
for inference. The slide is aggregated from patch features, the six questions,
and the learned soft prompts, then compared only with the two released
diagnostic descriptions in label-index order. This closes the former
privileged-text path and makes missing answers on held-out cases harmless.

Native mode is currently declared only for TCGA-NSCLC. The RCC and UBC example
configs explicitly use generated, task-specific six-question banks with
`simplified_classnames`; WSI-FiVE published neither native answer banks nor
evaluation descriptions for those tasks. Their generated question provenance
is recorded, and a generated answer/evaluation bank would be a separate
experimental condition that must not be reported as upstream.

!!! warning "Remaining deviations"
    - **Rebuilt orchestration.** The released `FiVE.py` cannot be imported or
      constructed, so the lifecycle is rebuilt around its usable vendored
      components and remains marked partial.
    - **Fold scope.** The release publishes a fixed full training candidate
      bank. PGVL-Gym rebuilds it from each benchmark training fold to prevent
      answer leakage across few-shot folds.
    - **`NUM_FRAMES`.** The local default is 2048 rather than 16384 upstream.
    - **Feature provenance.** The paper uses DSMIL SimCLR-ResNet18 512-d
      features. A run using a different 512-d feature space is dimensionally
      valid but not the paper's encoder condition.

### ConVLM — offline visual features, locally reconstructed attributes

ConVLM does not train a vision foundation model inside the benchmark. Upstream
extracts patch and ROI embeddings with UNI before training, and
`AttributeConVLM` consumes those bags through its configured `feature_dim`.
Accordingly, its contract is `PRECOMPUTED/PATCH_BAG`, the protocol compiler
binds a `bag` feature source, and the unified bag loader reads either the
manifest's `feature_path_column` or `data_folder_s/<slide_id>.pt`.

The attribute banks built for all five cohorts still substitute for the
paper's unpublished Quilt-LLaVA bank. That text-side provenance is a real
reproducibility limitation and must be reported, but it is separate from the
offline visual-feature boundary. No active benchmark protocol currently
enables ConVLM by default.

### MUSE — no upstream prompts for RCC or UBC-OCEAN

Upstream covers `camelyon_all`, `tcga_brca`, `tcga_nsclc` — all imported. Its
`kidney` folder is **IgA nephropathy grading**, a different disease and task from
renal cell carcinoma subtyping; "carcinoma" appears zero times across its six
files. It is not a substitute for RCC.

RCC's MUSE banks are compiled from the same GPT descriptions MSCPT uses and
report `prompt_provenance: generated`. A MUSE result on RCC is not paper-faithful
and must not be tabled beside BRCA/NSCLC MUSE, which read the authors' own CSVs.

### PathPT — recipe locked across backbones

Optimizer, LR, epochs, prompt length and loss weights are held constant across
PLIP/CONCH/KEEP/MUSK so a backbone's number reflects the encoder rather than
training-recipe variance. The adapter overrides `build_optimizer` and
`build_scheduler` to enforce this regardless of the YAML.

Generated configs use `training_mode: upstream_patch_ssl`. After the fold's
model and training loader are built, PathPT encodes all combinations of its 22
templates and task synonyms, scores 200 sampled prompt classifiers on training slides only,
and averages the best 100. That frozen classifier creates Normal, known subtype,
and negative candidate patch labels. Training calls the vendored `PatchSSLoss`;
WSI inference counts patch votes, removes the synthetic Normal class, and uses
the released all-Normal/tie fallback. The selected classifier and full score
trace are persisted per fold. The old mean-patch-probability slide CE path is
retained as the explicit `simplified_slide_ce` compatibility mode.

Prompt provenance is cohort-specific:

| Cohort | Bank | Fidelity note |
| --- | --- | --- |
| TCGA-BRCA | upstream PathPT `brca_names` | native subtype pipeline |
| UBC-OCEAN | upstream PathPT `ubc_names` | native subtype pipeline |
| TCGA-NSCLC | generated here | native algorithm, local task bank |
| TCGA-RCC | generated here | native algorithm, local task bank |
| CAMELYON16 | upstream PathPT `camelyon_names` | binary WSI adaptation; implementation fidelity remains partial |

The CAMELYON upstream list has a missing comma that concatenates
`non-cancerous tissue` and `normal breast tissue`. The upstream condition is
kept byte-for-byte and the defect is recorded in `text_prompts/PROVENANCE.json`.
Correcting it would be a separate derived prompt condition.

---

## 8. Decisions still open

| Item | Question |
| --- | --- |
| ConVLM | Release or reproducibly regenerate the data-specific QuiltNet attribute bank? |
