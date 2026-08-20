# Which method and encoder combinations are possible

The benchmark aims to run every method against every registered vision-language
encoder. This page records how far that reaches, and — more usefully — exactly
why the remainder does not, so an absent cell is a documented boundary rather
than an unexplained gap.

Regenerate the underlying data at any time with:

```bash
python scripts/list_backbone_compatibility.py
python scripts/list_backbone_compatibility.py --method mscpt --json
```

## Summary

Across 13 methods and 10 registered encoders there are 130 combinations:

| Outcome | Count | Meaning |
| --- | ---: | --- |
| `native` | 48 | The method's encoder boundary supports this encoder without a learned compatibility projection. |
| `adapt` | 34 | Capabilities are satisfied; only the feature width differs, so a declared projection can bridge it. |
| `blocked` | 48 | Cannot run without changing what the method is. |

Every encoder listed is a dual-tower vision-language model. Vision-only
pathology foundation models — UNI, GigaPath, Virchow, Phikon, CTransPath,
H-optimus, ResNet50 — are deliberately absent: a method that learns or injects
text prompts has nothing to attach to without a text tower.

## The matrix

```
              biomedcl clip-rn5 clip-vit    conch hf-clip-     keep     musk     plip quiltnet    titan
focus            adapt    adapt    adapt   native    adapt    adapt    adapt    adapt    adapt    adapt
vila_mil         adapt   native    adapt    adapt    adapt    adapt    adapt    adapt    adapt        -
cod_mil              -   native        -        -        -        -        -   native   native        -
maple            adapt    adapt    adapt    adapt   native    adapt    adapt   native    adapt        -
mscpt                -        -        -   native   native        -        -   native        -        -
pathpt           adapt    adapt    adapt   native    adapt   native   native   native    adapt        -
top                  -   native        -        -        -        -        -        -        -        -
slip            native   native   native    adapt    adapt    adapt    adapt   native    adapt        -
wsi_five             -        -        -        -        -        -        -        -        -        -
muse            native   native   native   native   native   native   native   native   native   native
convlm               -        -        -        -        -        -        -        -        -        -
sldpc           native   native   native   native   native   native   native   native   native   native
composite       native   native   native   native   native   native   native   native   native        -
```

## Why the blocked cells are blocked

The 45 blocked combinations fall into four remaining causes; a fifth has since
been resolved. Only that fifth was ever a limitation of this benchmark rather
than of the methods or encoders themselves.

### 1. The method fixes its visual representation boundary (20 cells)

`wsi_five` and the local `convlm` reconstruction are pinned to
`wsi-five-vit` and `convlm-vit`.

For **ConVLM**, `PRECOMPUTED` describes PGVL's local adaptation, not the
released training path. The release has a separate UNI extraction utility, but
its `train.py` constructs a raw-image ViT and does not consume those bags. The
local fixed boundary remains meaningful: patch bags must declare their feature
space, and encoded attributes must bind class order, source prompts, and text
encoder identity. Changing either side creates a different reconstructed
condition; none is presented as the upstream protocol.

For **WSI-FiVE**, the pin likewise describes a fixed precomputed feature
boundary, not an owned image encoder. WSI-FiVE has no vision tower at all.
With its shipped default `IS_IMG_PTH: True` the upstream model sets
`self.visual = nn.Identity()`,
hardcodes `embed_dim = 512`, never parses the CLIP state dict, and reads
precomputed patch features from disk. The paper is explicit: *"we employed
ResNet following [15] as image encoder to extract image features, while
pre-trained BioClinicalBERT from [30] as text encoder"* — reference [15] is
DSMIL. No MedCLIP weights are loaded on any code path: `MedCLIPTextModel`, the
only MedCLIP class instantiated, loads `Bio_ClinicalBERT` from Hugging Face, and
every checkpoint-loading path in `MedCLIPModel.py` belongs to a vision class
that is never constructed. X-CLIP contributes `PatchFusionTransformer`, an
aggregation module, not an encoder.

What is genuinely method-owned is the **text** side: a BioClinicalBERT tower
fine-tuned with LoRA, learnable prompt vectors, and a patch-fusion transformer
that cross-attends patch features to encoded clinical questions.

**Resolution: `wsi_five` is a patch-bag consumer**, not an encoder-owning
method, and its nine cells are blocked by the current contract rather than by
the architecture. See [Design decisions](design-decisions.md) for the full
analysis, restored native text objective, and remaining fidelity gaps.

### 2. The method hardcodes one encoder's geometry (16 cells)

`top` is `fixed` to CLIP-RN50 because its instance and bag prompt attention is
written against RN50 width. `cod_mil` is `precomputed`: it consumes text
embeddings that were produced in a specific feature space, and comparing them to
patches from a different encoder compares vectors that do not share a geometry.
Its width is parameterized for the CLIP-RN50, PLIP, and QuiltNet families
released upstream, but each run must use a verified prompt tensor and dual-scale
patch bags from the same family.

**Resolution: regenerate the complete paired artifact set in a supported
space.** A width adapter does not help, because the mismatch is semantic rather
than dimensional.

### 3. The encoder has no deep-prompt hooks (7 cells)

`mscpt` requires `deep_text_prompt` and `deep_vision_prompt`: it injects learned
prompts into *intermediate layers* of both towers, not just at the input. Only
`conch`, `hf-clip-vitb` and `plip` expose that access.

**Resolution: per-encoder implementation, not adaptation.** No projection can
create layer-wise injection points in an encoder that does not expose them.

### 4. TITAN is a slide encoder (5 cells)

`titan` declares `paired_slide_text`, not `paired_tile_text`. It emits one vector
per slide, so patch-bag and dual-scale methods — `vila_mil`, `maple`, `pathpt`,
`mscpt`, `composite` — have no patch sequence to aggregate.

**Resolution: none, and none is wanted.** This is a feature-level mismatch the
registry models correctly. `sldpc` and `muse` accept TITAN because they consume
slide embeddings.

### 5. KEEP and MUSK declared no text tower — resolved

`keep` and `musk` carried the capability bundle `{soft_prompt, paired_tile_text}`
and so failed the `text_encode` requirement of `muse`, `sldpc`, `composite` and
`slip`. The bundle they shared was named `_NATIVE_PATHPT_TILE`, which describes
what PathPT needs of them rather than what they can do.

Both are in fact dual-tower models, and both now declare `TEXT_ENCODE`. Verified
through `bundle.encode_text()` — the call the dependent methods make — rather
than by inspecting attributes:

| | width | matches `shared_dim` | finite | cos(IDC,ILC) | cos(IDC,normal) | discriminative |
| --- | ---: | --- | --- | ---: | ---: | --- |
| KEEP | 768 | yes | yes | 0.6553 | 0.4972 | yes |
| MUSK | 1024 | yes | yes | 0.8946 | 0.7947 | yes |

MUSK's width is the load-bearing number. Its text tokens are 768 wide and its
shared space is 1024; only `with_head=True` applies `language_head` and projects
between them. A 768-wide result would have meant the towers did not share a
space and every similarity was computed across mismatched geometries — a silent
numerical error rather than a failure.

Reaching that point took four fixes, each hidden behind the previous one:

1. the specs did not declare `TEXT_ENCODE`;
2. `_NativeText` could not call KEEP's `encode_text(text_inputs)`, which takes
   the token mapping as one positional argument, and could not drive MUSK at
   all, whose API is a unified `forward(...) -> (vision_cls, language_cls)`;
3. the MUSK loader looked for `tokenizer.spm` only inside the installed package,
   which does not ship it, though the published snapshot does;
4. `sentencepiece`, required by MUSK's tokenizer, was missing from the
   `pathpt-musk` extra.

Every intermediate state would have passed `validate_config` and failed at
runtime. That is the reason a capability is only declared once the operation has
been exercised end to end: a declaration that validates and then breaks is worse
than an honest refusal.

## Native versus adapted results

A `native` and an `adapt` run are different claims and must not share a results
table.

* **native** — the method's published code supports the encoder. `pathpt_keep`
  and `pathpt_musk` are native: PathPT ships `PathPT_model_KEEP.py` and
  `PathPT_model_MUSK.py`.
* **adapted** — a learned projection bridges the feature width. What is measured
  is the method *and* the adapter, not the encoder alone. `muse_musk` is adapted:
  MUSK supplies 1024-d patches while CONCH remains the text tower, with a
  trainable visual adapter between them.

Equal widths never establish compatibility on their own. Two 512-d encoders
occupy different semantic spaces, and no adapter is inserted implicitly — it must
be declared, exactly as `slide_projection_mode` declares SLDPC's `native`,
`linear` and `mlp` variants.

Neither outcome establishes paper fidelity. That is recorded separately as
`implementation_provenance` and `upstream_fidelity`; for example, PathPT can
have a native KEEP encoder boundary while its current unified objective is
still labelled partial.

## Adding an encoder or opening a method

See [Backbone interfaces and swap boundaries](BACKBONE_INTERFACES.md) for the
registration checklist. In short: declare only capabilities the bundle really
implements, keep the original experiment alongside any variant, and record which
of the two a run used.
