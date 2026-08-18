# Method support matrix

Every adapter declares its input level and encoder swap policy. This table
describes architectural compatibility; it does not imply that every required
feature or checkpoint is currently present.

| Method | Input contract | Swap policy | Supported encoder boundary |
| --- | --- | --- | --- |
| Composite | Composite patch/text pipeline | Capability | Paired tile-text and text encoding capabilities |
| FOCUS | Dual-scale patch bags | Allowlist | CONCH |
| ViLa-MIL | Dual-scale patch bags | Allowlist | CLIP-RN50 |
| CoD-MIL | Dual-scale patch bags plus map | Precomputed | CLIP-RN50 |
| MAPLE | Dual-scale patch bags | Allowlist | PLIP, HF CLIP ViT-B |
| MSCPT | Dual-scale patch bags | Allowlist | PLIP, HF CLIP ViT-B, CONCH |
| PathPT | Patch bag | Allowlist | PLIP, CONCH, KEEP, MUSK |
| TOP | Patch bag | Fixed | CLIP-RN50 |
| SLIP | Patch bag | Allowlist | CLIP ViT-B, CLIP-RN50, PLIP, BiomedCLIP |
| WSI-FiVE | Patch sequence plus report | Fixed | WSI-FiVE ViT |
| MUSE | Patch bag | Capability | Any declared patch space with a compatible text encoder and learned adapter |
| ConVLM | Attribute-conditioned raw tiles | Fixed | ConVLM ViT |
| SLDPC | Slide embedding | Capability | Soft-prompt and text-encoding prompt tower; slide source declared separately |

Query the live contracts at any time:

```bash
python scripts/list_backbone_compatibility.py
python scripts/list_backbone_compatibility.py --json
```

## Protocol experiment variants

The standard protocols expand 13 method families into 20 comparable variants:

| Family | Registered variants |
| --- | --- |
| PathPT | CONCH 10x, MUSK 10x, KEEP 20x |
| MUSE | CONCH 10x, MUSK 10x, KEEP 20x patch bags |
| FOCUS | CONCH 10x/20x and 5x/20x |
| MSCPT | CONCH 10x/20x and 5x/20x |
| MAPLE | PLIP 10x/20x and 5x/20x |
| ViLa-MIL | CLIP-RN50 5x/20x |
| CoD-MIL | CLIP-RN50 5x/20x plus cross-scale map |
| TOP | CLIP-RN50 10x |
| SLIP | CLIP-RN50 10x |
| WSI-FiVE | 512-d patch bag plus per-slide report text (BioClinicalBERT tower) |
| SLDPC | TITAN 20x slide embeddings and prompt tower |
| ConVLM | 20x raw tile directory |
| Composite | CLIP-RN50 10x classname baseline |

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
