# CoD-MIL cross-magnification maps — BRCA, 10x -> 20x

`(N_low, 4)` int64 tensors, one per slide, named by manifest slide id. Row `i`
holds the indices of 20x patches whose centres lie inside 10x patch `i`, padded
with `-1`. Consumed by `methods/cod_mil/model.py`, which selects the 16 most
diagnostic low-power patches and expands them through this table into at most
64 high-power instances.

## Provenance

Reconstructed, not upstream. The CoD-MIL release references
`map_10x_20x_files/<slide>.pt` but ships neither the maps nor a generator, so
these were built by `scripts/generate_cross_magnification_maps.py` from the
`coords` datasets of the CLIP RN50 feature files themselves (not the patch
store), which guarantees the indices address the exact bags the model loads.

Correspondence rule: a 20x patch belongs to the 10x patch whose rectangle
contains its centre, in the level-0 coordinate frame. The paper introduces an
alignment matrix but does not state the geometric rule, so this is an inference;
it is robust to tissue filtering in a way index arithmetic is not.

Patch sizes come from each slide's own `patch_size_level0` attribute rather than
a global constant. This cohort spans three scanner base magnifications and the
level-0 extents differ accordingly:

| base | 10x  | 20x | slides |
| ---- | ---- | --- | -----: |
| 40x  |  896 | 448 |    913 |
| 30x  | 1344 | 672 |      8 |
| 20x  |  448 | 224 |     39 |

A single global patch size would have bound roughly four times too many 20x
patches per row for the 47 non-40x slides, silently.

## Not comparable to published numbers

Upstream `create_patches_fp.py` defaults to 256 px / stride 256; this store uses
224 px. The grids differ, so instance counts do not match the paper's Table I.
