# Backbone interfaces and swap boundaries

The unified loader separates two questions that are easy to conflate:

1. Can the repository load an encoder through a common API?
2. Can a particular paper architecture use that encoder without being
   redesigned?

The first is answered by an `EncoderBundle`. The second is answered by the
method's `MethodBackboneContract`. A matching embedding width alone is never
treated as evidence that two feature spaces are aligned.

## Runtime interface

New integrations should call `build_encoder`; `build_backbone` is preserved for
vendored code that still expects `(model, tokenizer, info)`.

```python
from common.backbones import BackboneCapability as Cap, build_encoder

encoder = build_encoder("conch", device="cuda")
encoder.require(Cap.TEXT_ENCODE, consumer="my method")
text_features = encoder.encode_text(["lung adenocarcinoma"], normalize=True)
```

The main types live in `common/backbones/interfaces.py`:

- `BackboneSpec` records the canonical name, family, dimensions, context/image
  sizes, feature-space identifier, revision, aliases, and capabilities.
- `EncoderBundle` carries the `BackboneSpec`, native objects, preprocessing,
  and narrow text/tile/slide wrappers. `raw_model` and `raw_tokenizer` let a
  validated adapter pass the original objects to unchanged paper code.
- `TokenBatch` normalizes token IDs, attention masks, end-of-text positions,
  and vendor-specific tensor fields.
- `TextEncoder`, `PromptableTextEncoder`, `TileEncoder`, and `SlideProjector`
  specify the operations consumed by adapters.
- `MethodBackboneContract` validates the selected name, input feature level,
  required capabilities, known feature widths, and swap policy before model
  construction.

`BaseMethod.load_encoder()` is the normal adapter entry point. It calls the
registry and validates the returned bundle against the adapter's contract.
Methods that traverse vendor-specific transformer blocks do so only after this
check; the common layer does not replace those blocks.

## Capabilities

Capabilities describe behavior, not approximate architecture labels.

| Capability | Guarantee |
| --- | --- |
| `text_encode` | Encode ordinary text into the model's native shared space. |
| `soft_prompt` | The native text tower supports differentiable embedded context tokens. |
| `deep_text_prompt` | The native text transformer exposes the layerwise hooks used for deep prompts. |
| `tile_encode` | Encode raw image tiles. |
| `deep_vision_prompt` | The vision transformer exposes the layerwise hooks used for deep prompts. |
| `slide_project` | Project raw slide embeddings through the paired model's native slide projection. |
| `paired_tile_text` | Tile and text outputs belong to the same trained comparison space. |
| `paired_slide_text` | Projected slide and text outputs belong to the same trained comparison space. |

Other capability values reserve explicit boundaries for patch projection and
specialized text-supervision methods. Declaring a capability does not add an adapter or
projection to a model. Capability-policy methods require the corresponding
bundle wrapper operation. Allowlisted methods may instead consume the
validated native object through an existing family-specific implementation.

## Swap policies

- `capability`: another registered bundle is accepted if it satisfies all
  required capabilities. Data dimensions and feature provenance still have to
  agree with that bundle.
- `allowlist`: the paper code contains a native implementation branch for each
  listed family. A new family needs a new architecture branch even when it
  advertises similar capabilities.
- `fixed`: the encoder/tower is structurally part of the method. It is exposed
  in the contract for inspection, not advertised as swappable.
- `precomputed`: the runtime does not load an encoder. Patch and text artifacts
  must already have been produced by the declared, aligned feature space.

Inspect the effective declarations without allocating a model:

```bash
python scripts/list_backbone_compatibility.py
python scripts/list_backbone_compatibility.py --json
python scripts/list_backbone_compatibility.py --method muse
```

## Method matrix

| Method | Policy | Validated boundary | Why the boundary stops there |
| --- | --- | --- | --- |
| Composite | capability | Paired tile/text bundle with black-box text encoding | Individual enabled prompt modules can impose stricter native soft-prompt requirements; patch and text dimensions must match because selectors compare them directly. |
| FOCUS | allowlist: CONCH | One high-resolution bag plus CONCH soft prompting | Its prompt learner accesses the CONCH text tower; the upstream model accepts a low-scale argument but does not consume it. |
| ViLa-MIL | allowlist: CLIP RN50 | Paired 1024-wide RN50 patch/text space and soft prompting | Prompted text and patch vectors are compared directly. |
| CoD-MIL | precomputed: CLIP RN50, PLIP, or QuiltNet | Aligned dual-scale bags, metadata-bound prompt tensors, and cross-scale maps in one feature space | The CSV bank is encoder-independent, but every encoded prompt/patch artifact must use the same tower; runtime-cached CSV encoding is also supported with that tower's checkpoint. |
| MAPLE | allowlist: PLIP or Hugging Face CLIP ViT-B | Paired 512-wide bags and native soft-prompt text layers | MAPLE traverses Hugging Face CLIP/PLIP internals. |
| MSCPT | allowlist: PLIP, Hugging Face CLIP ViT-B, or CONCH | Paired 512-wide bags with deep text and vision prompt hooks | It injects prompts at multiple layers in both towers. |
| PathPT | allowlist: PLIP, CONCH, KEEP, or MUSK | Backbone-specific paired feature width and soft-prompt implementation | The release contains a distinct native prompt class for each family. |
| TOP | fixed: CLIP RN50 | 1024-wide RN50 bags and two-level native prompts | RN50 dimensions and instance/bag prompt attention are structural assumptions. |
| SLIP | allowlist: CLIP ViT-B, CLIP RN50, PLIP, or BiomedCLIP | Paired patch/text features and the matching native prompt branch | Each supported family uses its own tokenizer and prompt implementation. |
| WSI-FiVE | precomputed: 512-wide patch bag | Offline patch features; native training also requires a fold-local six-answer bank | Its patch-fusion transformer and ClinicalBERT question/prompt tower are method-owned; per-slide answers are targets, never inference inputs. |
| MUSE | capability | Any registered static patch source plus any registered black-box `text_encode` prompt bundle | `patch_encoder` provenance is independent from `prompt_feature_space_id`; a learned visual adapter maps patch `feature_dim` to the prompt encoder's `embed_dim`. CONCH/CONCH is the native encoder condition, while implementation fidelity is separately marked partial. |
| ConVLM | local precomputed patch-bag reconstruction | A declared patch-bag space plus metadata-bound attribute vectors from any declared text encoder | This is PGVL's adapter boundary, not the released training boundary: upstream `train.py` feeds RGB images to its ViT and loads an absent `att_splits.mat`. Prompt artifacts bind class order, source-bank digest, encoder checkpoint hash, and feature space. |
| SLDPC | capability | Promptable text tower and native slide projector from one paired slide-text model | A slide-only tower, unrelated text encoder, or width-only match cannot reproduce the trained comparison space. TITAN is the default. |

This matrix deliberately distinguishes architectural compatibility from an
experimental model substitution. A capability-accepted replacement can run
through the same architecture, but it does not inherit the original paper's
reported comparability or pretrained alignment quality.

## Registering another encoder

Registration is process-local. Import the module containing the registration
before constructing the method (for example, in a project-specific launcher).
A builder receives `weights_path`, `device`, and any loader options, and returns
an `EncoderBundle` whose spec name matches the registered name.

```python
from common.backbones import (
    BackboneCapability as Cap,
    BackboneSpec,
    EncoderBundle,
    register_backbone,
)

spec = BackboneSpec(
    name="my-text-model",
    family="my_family",
    feature_space_id="my-org/my-text-model@revision",
    capabilities=frozenset({Cap.TEXT_ENCODE}),
    shared_dim=768,
)

def build_my_text(*, weights_path=None, device="cuda", **options):
    model, tokenizer = load_native_model(weights_path, device=device, **options)
    text = MyTextEncoder(model, tokenizer)  # implements the TextEncoder protocol
    return EncoderBundle(
        raw_model=model,
        raw_tokenizer=tokenizer,
        spec=spec,
        text=text,
    )

register_backbone(spec, build_my_text)
```

That minimal bundle can satisfy MUSE when `embed_dim: 768`. It cannot serve as
SLDPC's prompt backbone because it lacks promptable embedded-text operations.

### Slide-embedding and SLDPC replacement checklist

All adapters declaring `FeatureLevel.SLIDE_EMBEDDING` use the shared
`common.datasets.slide_embeddings` loader. A registered offline source must
record its feature key, width, feature-space/checkpoint identity, resolution,
storage layout, and path template. It does not need a runtime backbone loader;
set `runtime_encoder: false` when only its cached embeddings are consumed.

SLDPC additionally needs a runtime prompt backbone with a
`PromptableTextEncoder` in `bundle.text`, including tokenization,
`token_width`, token embedding, and embedded-prompt encoding. Two alignment
modes are supported:

1. `native`: the prompt bundle must also expose `SlideProjector`,
   `slide_project`, and `paired_slide_text`. Its `slide_input_dim` and exact
   feature space must match the offline files.
2. `linear` or `mlp`: the offline slide encoder may be unrelated and have any
   declared width. A trainable framework-level adapter aligns it to the prompt
   backbone's declared `shared_dim` during Stage 1.

Set `backbone`/`backbone_weights` for the prompt bundle, and record the offline
source separately under `slide_encoder`. `prompt_feature_space_id` validates
the runtime prompt bundle, while `feature_space_id` validates the cached slide
vectors. The projection mode is mandatory in generated benchmark configs.

The unified SLDPC adapter also retains the released two-stage checkpoint
semantics in memory: it restores the best validation prompt before CPI and
again before final evaluation. Consequently, `epochs` must equal
`stage1_epochs + stage2_epochs`, and the outer unified-loop
`early_stopping` must remain disabled.

Its learned prompt input is `prompt_classnames`, an ordered sequence of fixed
class-code tokens. Do not substitute the SLDPC synonym YAML here: upstream uses
that YAML with a 23-template ensemble only for a separate, untrained TITAN
zero-shot baseline. The doctor rejects the old ambiguous
`prompt_reference_yaml` field and verifies active-token and optional
zero-shot-reference digests independently.

There is no automatic projection fallback. Selecting `linear` or `mlp` is an
explicit method variant because it introduces a learned alignment model;
results must report the projection mode separately from native SLDPC.

## Data provenance when changing a backbone

For patch-bag methods, change the cached patch features together with the
backbone. For precomputed prompt methods, regenerate the text tensors too. For
slide-embedding methods, register the replacement slide-vector store and keep
its provenance separate from the runtime prompt/model space. Record exact
feature-space and checkpoint revisions; two files with the same last dimension
can still represent incompatible spaces.
