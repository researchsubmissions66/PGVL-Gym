# Composite WSI/VLM Model

The composite system fuses **all nine papers' contributions** into one
configurable model. Pick which contributions to enable in YAML.

## Anatomy of a composite config

```
selectors:    [list of patch-selector stages, applied sequentially]
prompts:      {prompt_module: {enabled, params}, ..., fusion: {mode}}
aggregators:  {aggregator: {enabled, params}, ..., fusion: ...}
recipe:       {type, lr, epochs, loss: {weights}}
```

Each block has an **on/off toggle per component**:

| Block | Components | Toggle |
|---|---|---|
| `selectors` | `focus_three_stage`, `cod_chain_mask`, `mscpt_topk`, `identity` | applied in YAML order; omit / set `enabled: false` to skip |
| `prompts` | `coop_flat`, `top_two_level`, `maple_graph`, `cod_chain`, `slip_tissue` | each enabled module contributes a (C, D) text tensor; fused via `average` / `weighted_sum` / `concat` / `first` |
| `aggregators` | `attn_pool`, `vila_prototypes`, `slip_routing`, `pathpt_conv1d`, `focus_mha`, `transmil` | each enabled aggregator runs in parallel; fused via `logit_ensemble` (level 1) or `vector_fusion` (level 2) |
| `recipe` | `focus`, `pathpt`, `top`, `slip`, `wsi_five` | pick **one**; recipes are mutually exclusive |

## Fusion modes

### Aggregator fusion
- **`logit_ensemble`** (Level 1) — each aggregator has its own classifier head
  and emits `(C,)` logits. Logits combine via `mean`, `weighted_sum`,
  `max`, or `voting`. Fully decoupled; safest if any aggregator is buggy.
- **`vector_fusion`** (Level 2) — each aggregator emits a `(D,)` slide
  vector. Vectors are combined via `concat`, `mean`, `weighted_sum`,
  `gated`, or `cross_attention`, then a single classifier produces logits.
  Tighter coupling, often higher capacity.

### Prompt fusion
- `average` — mean of (C, D) tensors across modules.
- `weighted_sum` — softmax-normalized learnable weights over modules.
- `concat` — channel-wise concat then `Linear(D·k → D)` projection.
- `first` — use the first enabled module only (degenerate).

## Example configurations

| File | What it does |
|---|---|
| `vanilla_ubc.yaml` | Plain CoOp + attention pool. Reference baseline. |
| `kitchen_sink_ubc.yaml` | All nine contributions enabled, logit-ensemble fusion. Maximum capacity. |
| `vector_fusion_ubc.yaml` | All on, vector-level fusion with gated combination. |
| `ablate_focus_ubc.yaml` | Kitchen sink minus FOCUS selector. Template for ablations. |

## How to ablate

To remove any single contribution, set `enabled: false`:

```yaml
selectors:
  - {type: focus_three_stage, enabled: false}      # drop FOCUS compression
  - {type: cod_chain_mask,    enabled: true}
  ...
prompts:
  slip_tissue:   {enabled: false}                   # drop SLIP tissue routing
  ...
aggregators:
  vila_prototypes: {enabled: false}                 # drop ViLa prototypes
  ...
```

## Honest constraints (read these)

1. **Selectors stack** but **aggregators ensemble**. You cannot chain
   aggregators because each collapses (N, D) → (C,) or (D,).
2. **Prompt modules emit different things** (CoOp = (C, D); CoD-MIL =
   hierarchy; MAPLE = entity attributes). The fusion module unifies
   them by averaging the (C, D) parts and merging the rest into `aux`.
3. **Recipes are mutually exclusive**. TOP wants lr=0.02 / 8000 epochs;
   PathPT wants lr=1e-4 / 20 epochs. Pick one.
4. **`slip_routing`** depends on `slip_tissue` prompts. If you enable
   `slip_routing` without `slip_tissue`, it gracefully falls back to
   plain text-cosine pooling.
5. **`maple_graph` aux-loss** depends on `aux_attribute_weight > 0`
   in the recipe. Otherwise the aux features are ignored.
6. **`vector_fusion` requires identical `out_dim`** across aggregators
   for `mean`/`weighted_sum`/`gated`/`cross_attention` modes. `concat`
   handles mismatched dims by Linear-projecting the concatenation.

## Dispatch

```bash
python train.py --method composite --config configs/composite/kitchen_sink_ubc.yaml
```

The unified `train.py` dispatches `composite` to
`methods/composite/adapter.py::CompositeMethod`, which builds a
`CompositeModel` from the YAML.
