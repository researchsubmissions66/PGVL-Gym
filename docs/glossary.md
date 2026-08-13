# Glossary

**Adapter**
: The method-specific class that connects a paper implementation to common
  model construction, training, evaluation, and optimizer hooks.

**Auxiliary asset**
: A non-feature input required by a method, such as a cross-scale map, clinical
  report, attribute tensor, or prompt embedding.

**Backbone**
: A registered runtime encoder implementation. This is distinct from the
  offline model that produced a cached feature store.

**Capability**
: An operation exposed by an encoder bundle, such as text encoding, soft
  prompting, or paired tile-text projection.

**Cohort**
: One annotation-defined collection of slides/cases with a fixed task and
  label order.

**Config validity**
: Whether a generated run is internally consistent and satisfies the adapter's
  declared contract. Validity does not require files to exist.

**Experiment variant**
: A reportable method configuration with a fixed feature source, resolution
  role binding, prompt mode, and recipe. Encoder or projection changes receive
  separate variant names.

**Feature level**
: The semantic input type consumed by a method: patch bag, dual-scale bags,
  patch sequence, slide embedding, raw tile directory, or composite input.

**Feature role**
: A method-facing name such as `bag`, `low`, `high`, or `tiles`. Roles are
  mapped to independently registered feature sources.

**Feature space**
: The representation produced by one exact encoder/checkpoint. It is tracked
  by `feature_space_id`; tensor width alone does not define it.

**Fold**
: One outer patient/case-disjoint train, validation, and test partition.

**Input kind**
: The physical/semantic storage contract, including `patch_bag`,
  `slide_embedding`, `patch_sequence`, or `raw_tile_directory`.

**Native projection**
: A paired projector released with a model that maps its own visual and text
  representations into a shared space. It cannot be borrowed by an unrelated
  slide encoder.

**Prompt encoder**
: The runtime text tower used to encode class descriptions or learn soft
  context. It may differ from the offline visual feature producer when the
  method explicitly defines an alignment.

**Prompt profile**
: The canonical, task-level description source compiled into method-native
  prompt assets.

**Protocol**
: The source-of-truth YAML defining cohorts, features, experiments, folds,
  shots, prompts, and output conventions.

**Readiness**
: Whether every metadata, split, feature, weight, prompt, and auxiliary file
  needed for a valid run currently exists.

**Resolution**
: A declared semantic magnification such as 5x, 10x, or 20x. `low` and `high`
  are roles and are not hard-coded magnifications.

**Run config**
: One generated YAML passed to an adapter invocation. It records resolved paths
  and provenance for one experiment, cohort, and shot level.

**Shot**
: The number of labeled training patients/cases per class, with the same count
  used for validation under the standard protocols.

**Smoke test**
: Isolated model construction plus a method-appropriate dummy forward pass. It
  validates wiring and finite output shape, not scientific performance.

**Upstream reproduction**
: A run that follows the released task, architecture, features, prompts, and
  recipe closely enough to claim reproduction.

**Framework extension**
: A valid adapter run on a new task, prompt source, encoder, projection, or
  recipe that was not released upstream.
