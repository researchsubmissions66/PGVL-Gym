# Extending the framework

## Add a dataset

1. Add a task block with ordered `labels`, `classnames`, and a canonical
   `prompt_spec` to a protocol.
2. Define metadata columns for slide ID, case/patient ID, and label.
3. Register the feature sources the task will use.
4. Include the task in the protocol's cohort list.
5. Run `scripts/tcga_benchmark.py all --protocol <path>`.
6. Resolve validation errors and inspect feature readiness.
7. Run the dummy-feature matrix before launching real-data jobs.

The prompt compiler emits FOCUS/ViLa two-scale CSVs, MUSE description CSVs,
MSCPT description JSON, MAPLE entity JSON, CoD chains, SLIP tissues, SLDPC
aliases, and ConVLM attributes from that single canonical profile.

## Add a feature source or resolution

Register a new source under `feature_registry`; do not add a new loader solely
because the resolution is new. Set `resolution` to the intended semantic value
and map that source to an experiment role such as `bag`, `low`, or `high`.

When changing the producer checkpoint, use a new `feature_space_id` even if
the tensor width is unchanged.

## Add an encoder

1. Create a `BackboneSpec` with exact dimensions, provenance, aliases, and
   capabilities.
2. Implement only the narrow wrappers the encoder actually supports.
3. Register a builder returning an `EncoderBundle`.
4. Add contract tests for valid and invalid consumers.
5. Document whether cached feature extraction must be repeated.

See [Backbone interfaces](BACKBONE_INTERFACES.md) for code examples and SLDPC's
paired slide/text requirements.

## Add a method

1. Create `methods/<name>/adapter.py` and subclass `BaseMethod`.
2. Declare a `MethodBackboneContract`.
3. Implement `build_model`, `train_step`, and `eval_step`.
4. Keep paper-specific implementation files in the method directory.
5. Reuse tensor-level loaders from `common.datasets` where possible.
6. Register aliases in `methods.get_method` and add the canonical name to
   `methods.list_methods`.
7. Add the experiment to protocol registries and regenerate configs.
8. Add unit tests and run the dummy-feature smoke matrix.

Public classes and functions need Google-style docstrings because the API
website is generated directly from source. Use the documentation quality check
described in [Writing documentation](contributing-docs.md).
