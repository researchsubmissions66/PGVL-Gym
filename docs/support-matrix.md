# Method support matrix

Every adapter declares its input level and encoder swap policy. This table
describes architectural compatibility; it does not imply that every required
feature or checkpoint is currently present.

| Method | Input contract | Swap policy | Supported encoder boundary |
| --- | --- | --- | --- |
| Composite | Composite patch/text pipeline | Capability | Paired tile-text and text encoding capabilities |
| FOCUS | Single high-resolution patch bag | Allowlist | CONCH |
| ViLa-MIL | Dual-scale patch bags | Allowlist | CLIP-RN50 |
| CoD-MIL | Dual-scale patch bags plus map | Precomputed | CLIP-RN50, PLIP, QuiltNet; prompt and patch banks must share one verified feature space |
| MAPLE | Dual-scale patch bags | Allowlist | PLIP, HF CLIP ViT-B |
| MSCPT | Dual-scale patch bags | Allowlist | PLIP, HF CLIP ViT-B, CONCH |
| PathPT | Patch bag | Allowlist | PLIP, CONCH, KEEP, MUSK |
| TOP | Patch bag | Fixed | CLIP-RN50 |
| SLIP | Patch bag | Allowlist | CLIP ViT-B, CLIP-RN50, PLIP, BiomedCLIP |
| WSI-FiVE | Patch bag; native train-time answer bank | Precomputed | Offline 512-wide features plus WSI-FiVE aggregation/text towers |
| MUSE | Patch bag | Capability | Any declared patch space with a compatible text encoder and learned adapter |
| ConVLM | Local attribute-conditioned patch-bag reconstruction | Precomputed | Any declared patch bags plus metadata-bound attribute vectors; not the released raw-image training path |
| SLDPC | Slide embedding | Capability | Soft-prompt and text-encoding prompt tower; slide source declared separately |

Query the live contracts at any time:

```bash
python scripts/list_backbone_compatibility.py
python scripts/list_backbone_compatibility.py --json
```

## Protocol experiment variants

The standard protocols expand 13 method families into 19 declared variants:

| Family | Registered variants |
| --- | --- |
| PathPT | CONCH 10x, MUSK 10x, KEEP 20x |
| MUSE | CONCH 10x, MUSK 10x, KEEP 20x patch bags |
| FOCUS | CONCH 20x; the accepted low-scale argument is unused upstream and is not a separate variant |
| MSCPT | CONCH 10x/20x and 5x/20x |
| MAPLE | PLIP 10x/20x and 5x/20x |
| ViLa-MIL | CLIP-RN50 5x/20x |
| CoD-MIL | CLIP-RN50 5x/20x plus cross-scale map in the standard protocol; PLIP/QuiltNet are supported when matching dual-scale bags, maps, and verified prompt tensors are supplied |
| TOP | CLIP-RN50 10x |
| SLIP | CLIP-RN50 10x |
| WSI-FiVE | 512-d patch bag; six fixed questions, train-only answer candidates, and a fixed evaluation description bank (BioClinicalBERT tower) |
| SLDPC | TITAN 20x slide embeddings and prompt tower |
| ConVLM | Local patch-bag reconstruction; audited generated prompts or metadata-bound encoded attributes (no upstream bank is released) |
| Composite | CLIP-RN50 10x classname baseline |

FOCUS prompt fidelity is upstream for TCGA-NSCLC, UBC-OCEAN, and CAMELYON16,
whose native positional CSVs are byte-exact copies. TCGA-BRCA and TCGA-RCC use
generated native-format extensions because the release contains no matching
banks. UBC's released file order is explicitly rebound to benchmark order.

ViLa-MIL prompt fidelity is task-specific: TCGA-NSCLC and TCGA-RCC select
byte-exact released native CSVs, while TCGA-BRCA, UBC-OCEAN, and CAMELYON16 use
generated native-format task extensions. The RCC copy preserves upstream's
`CRCC` text while explicitly binding that positional slot to `CHRCC`.

WSI-FiVE native NSCLC uses three hash-bound roles: six upstream questions in a
derived container, a complete 939-case answer bank containing 912 upstream
answers plus 27 disclosed generated completions, and two upstream evaluation
descriptions in a derived container. RCC and UBC use generated questions plus
classname comparison and are explicitly simplified conditions, not upstream
prompt banks.

Architectural compatibility is not implementation fidelity. Generated configs
and result tables separately carry `implementation_provenance` and
`upstream_fidelity`; partial integrations must not be presented as upstream
reproductions.

PathPT generated configs run the native patch-supervision mode: training-fold
prompt selection, synthetic Normal for subtype tasks, `PatchSSLoss`, and
Normal-excluding patch voting. Its older mean-patch slide-CE integration is
available as `simplified_slide_ce` but reports partial fidelity. CAMELYON is
also partial because its normal-vs-tumour slide labels require a disclosed
binary adaptation; prompt-bank provenance is an independent field.

## What “supported” means

- **Capability:** any registered encoder bundle with the required operations
  may be used.
- **Allowlist:** only explicitly named architecture branches are implemented.
- **Fixed:** the paper implementation owns a specific representation boundary.
- **Precomputed:** runtime behavior assumes cached tensors from a declared
  feature space and may require auxiliary files.

Feature dimensions never establish compatibility on their own. A 512-wide
CONCH tensor is not interchangeable with a 512-wide PLIP tensor, and an
ImageNet ResNet representation is not an OpenAI CLIP-RN50 representation.

## Readiness layers

Use three separate questions:

1. Is the method/encoder pairing architecturally supported?
2. Is the generated config valid under that contract?
3. Are all metadata, features, prompts, weights, and auxiliaries available?

The compatibility command answers the first. `config_audit.csv` answers the
second. `run_matrix.csv` and feature coverage answer the third.
