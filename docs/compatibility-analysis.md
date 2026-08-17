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

Across 13 methods and 9 registered encoders there are 117 combinations:

| Outcome | Count | Meaning |
| --- | ---: | --- |
| `native` | 37 | The method's own code supports this encoder. Paper-faithful. |
| `adapt` | 27 | Capabilities are satisfied; only the feature width differs, so a declared projection can bridge it. |
| `blocked` | 53 | Cannot run without changing what the method is. |

Every encoder listed is a dual-tower vision-language model. Vision-only
pathology foundation models — UNI, GigaPath, Virchow, Phikon, CTransPath,
H-optimus, ResNet50 — are deliberately absent: a method that learns or injects
text prompts has nothing to attach to without a text tower.

## The matrix

```
              biomedcl clip-rn5 clip-vit    conch hf-clip-     keep     musk     plip    titan
focus            adapt    adapt    adapt   native    adapt    adapt    adapt    adapt    adapt
vila_mil         adapt   native    adapt    adapt    adapt    adapt    adapt    adapt        -
cod_mil              -   native        -        -        -        -        -        -        -
maple            adapt    adapt    adapt    adapt   native    adapt    adapt   native        -
mscpt                -        -        -   native   native        -        -   native        -
pathpt           adapt    adapt    adapt   native    adapt   native   native   native        -
top                  -   native        -        -        -        -        -        -        -
slip            native   native   native    adapt    adapt        -        -   native        -
wsi_five             -        -        -        -        -        -        -        -        -
muse            native   native   native   native   native        -        -   native   native
convlm               -        -        -        -        -        -        -        -        -
sldpc           native   native   native   native   native        -        -   native   native
composite       native   native   native   native   native        -        -   native        -
```

## Why the blocked cells are blocked

The 53 blocked combinations fall into five causes. Only one of them is a
limitation of this benchmark rather than of the methods or encoders themselves.

### 1. The method owns its vision tower (18 cells)

`wsi_five` and `convlm` are pinned to `wsi-five-vit` and `convlm-vit`. These are
not interchangeable encoders sitting behind an interface; the tower is part of
the published architecture. WSI-FiVE pairs a MedCLIP/X-CLIP vision tower with a
ClinicalBERT report tower, and ConVLM classifies raw tiles against a QuiltNet
attribute bank. Substituting the encoder does not produce "WSI-FiVE with a
different backbone" — it produces a different method.

**Resolution: none.** These are correctly `fixed`.

### 2. The method hardcodes one encoder's geometry (16 cells)

`top` is `fixed` to CLIP-RN50 because its instance and bag prompt attention is
written against RN50 width. `cod_mil` is `precomputed`: it consumes text
embeddings that were produced in a specific feature space, and comparing them to
patches from a different encoder compares vectors that do not share a geometry.

**Resolution: only by rewriting the method.** A width adapter does not help,
because the mismatch is semantic rather than dimensional.

### 3. The encoder has no deep-prompt hooks (6 cells)

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

### 5. KEEP and MUSK declare no text tower (8 cells)

`keep` and `musk` carry the capability bundle `{soft_prompt, paired_tile_text}`
and therefore fail the `text_encode` requirement of `muse`, `sldpc`, `composite`
and `slip`.

This is the one cause that may be a limitation of the registry rather than the
models: both are dual-tower VLMs, and both loaders return a working tokenizer.
The bundle they share is named `_NATIVE_PATHPT_TILE`, which describes what PathPT
needs from them rather than what they can do.

**Resolution: verify, then implement.** Declaring `TEXT_ENCODE` without the
bundle actually exposing `encode_text` would convert a clean upfront rejection
into a late runtime failure, which is strictly worse. The capability should be
added only once the operation works end to end; doing so would move 8 cells from
`blocked` to `native`.

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

## Adding an encoder or opening a method

See [Backbone interfaces and swap boundaries](BACKBONE_INTERFACES.md) for the
registration checklist. In short: declare only capabilities the bundle really
implements, keep the original experiment alongside any variant, and record which
of the two a run used.
