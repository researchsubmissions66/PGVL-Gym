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
produced it.** Three provenance fields travel from the protocol into the
generated config, the run matrix, and `aggregate_results.csv`.

| Field | Values | Answers |
| --- | --- | --- |
| `encoder_provenance` | `native`, `adapted` | Did the method's own code support this encoder, or does a trainable projection bridge it? |
| `prompt_provenance` | `upstream`, `generated`, `explicit`, `classname_template`, `<compiled spec>`, `upstream_chain_with_generated_normal_tissue` | Where did the embedded text come from? |
| `prompt_source` | per-method, e.g. `cod_chain_runtime_clip_rn50` | Which asset and encoding path? |

All three are **derived from the declared contract and the files on disk**, not
set by hand, so they cannot drift from what the run actually does.

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

## 2. Implementation provenance — the known gap

The README states the framework "preserves each paper's model-specific
architecture." That is verifiable for eight of twelve methods. For four it is
not, and **no field currently records the difference.**

| Method | Upstream LOC used | Local model LOC | Status |
| --- | ---: | ---: | --- |
| `cod_mil`, `maple`, `mscpt`, `pathpt`, `top` | 1,195–6,301 | 0 | fully vendored |
| `focus`, `vila_mil` | 245–409 | 3 | vendored (trivial shim) |
| `slip` | 4,411 | 114 | mixed, mostly vendored |
| `wsi_five` | ~1,000 used of 4,071 | 175 | rebuilt from vendored components; `FiVE.py` orchestrator is unusable |
| `muse` | 0 | 105 | reimplemented (self-declared: "feature-space portion") |
| `convlm` | 0 | 124 | reimplemented (self-declared) |
| `sldpc` | 0 | 246 | reimplemented (CPI/DHNO/SICL written here) |

`wsi_five` was the most misleading case and has since been rebuilt: the adapter
now drives the vendored `PatchFusionTransformer`, `MedCLIPTextModel` and
`LoraWrap`. What remains unused is `FiVE.py` (500 lines) and
`_datasets/pipeline.py` (1,950 lines) — and `FiVE.py` is not merely unused but
**unrunnable**, failing both import and construction (see §7). Its supervision
signal, not its architecture, is now the limiting factor.

**Open decision.** An `implementation_provenance` field (`vendored` /
`reimplemented` / `partial`), declared per adapter, would close this the same
way `encoder_provenance` closed the encoder question.

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
3. **`train.py`** enforces a batch-failure ceiling (`max_batch_failure_rate`,
   default 0.1). Individual batches may fail — one unreadable slide should not
   end a five-fold run — but past that fraction the run fails rather than
   reporting a number computed on a partial split. Counts land in
   `metrics.json` as `batch_failures`.

### Trade-off accepted

The 0.1 ceiling is a judgement, not a measurement. A cohort with a known-bad
slide rate above that should set the threshold in its protocol rather than
having runs fail.

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

    So the CSV is **not** the source of the CLIP tensor. Truncating the last
    three rows — the intuitive fix — would feed unknown text into the low-power
    branch and low-power text into the high-power branch, misaligning both class
    branches with no error raised. Use the PLIP or QuiltNet tensor as the
    reference bank, or re-encode the 27 CSV prompts with CLIP RN50 directly.

Upstream's model slices normal prompts as `text_feature[2C:-1]`, dropping the
final row — an ordinary tissue prompt, not a sentinel. This is reproduced,
because `model.py` is vendored verbatim.

### WSI-FiVE — the vision tower is not method-owned

The compatibility analysis originally attributed 18 blocked cells to "the method
owns its vision tower (`wsi_five`, `convlm`)". **For `wsi_five` that was wrong.**
With the shipped default `IS_IMG_PTH: True`, upstream sets
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

#### The supervision signal is the real constraint

FiVE does not classify against class names. It compares the slide vector against
encoded **answers to its six clinical questions**, GPT-generated per slide and
shipped in `gpt_preprocess/*.xlsx`. Those answers are extracted to
`text_prompts/wsi_five/nsclc_report_answers.csv` (912 cases, six fields each).

| Cohort | Free-text report | Six-question answers |
| --- | ---: | ---: |
| TCGA-NSCLC | 880/946 | **864/946** |
| TCGA-BRCA | 846/900 | none |
| TCGA-RCC | 873/897 | none |
| UBC-OCEAN, CAMELYON16 | none | none |

**WSI-FiVE is therefore paper-faithful only on NSCLC.** On BRCA and RCC the
free-text reports exist but the structured answers do not; on the two non-TCGA
cohorts there is no report text at all. This is a data contract, not a code
limitation, and it is why the method cannot simply be pointed at an arbitrary
cohort.

!!! warning "Remaining deviations"
    - **Classification target.** The comparison text is still the class names;
      upstream builds class candidates from the answer text with label hashing
      inside a data pipeline that is not recoverable from the release.
    - **`aug_question`** training-time text augmentation is not reproduced; it
      depends on the release's BERT token scheme (`STA 101, END 102, SEP 132`).
    - **`NUM_FRAMES`** defaults to 2048 here against 16384 upstream.
    - **Feature provenance.** The paper uses DSMIL SimCLR-ResNet18 512-d
      features; this benchmark would supply CONCH 512-d — same width, different
      space.

### ConVLM — no pretrained vision weights

`methods/convlm/model.py` contains **no weight loading of any kind**.
`AttributeConVLM` is a 12-layer, 768-wide, 448 px ViT built from architecture
arguments and trained from scratch. On 4–16 shots it cannot produce a meaningful
number. Eleven of twelve adapters call `load_encoder`/`build_encoder`; ConVLM is
the only one that does not.

The attribute banks built for all five cohorts substitute for the paper's
unpublished Quilt-LLaVA bank. They are **deliberately unwired**: they address
the text side while the vision side remains untrained.

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

---

## 8. Decisions still open

| Item | Question |
| --- | --- |
| `implementation_provenance` | Should reimplemented methods be labelled in the results table? |
| ConVLM | Source pretrained weights, or exclude the method? |
| MSCPT `Lung.json` | 9 of 40 LUSC descriptors assert adenocarcinoma features, mean-pooled into the class centroid. Byte-identical to upstream. Correct and report, or keep and disclose? |
| MSCPT BRCA | 10 `big_mag` prompts against 30 elsewhere — cohorts are not comparable |
| PathPT `Normal` | Upstream prepends a Normal class (3-way on BRCA); this benchmark runs 2-way |
| TOP prompts | Wired from the NeurIPS supplementary; the released code uses shorter phrasing and produced the paper's numbers |
