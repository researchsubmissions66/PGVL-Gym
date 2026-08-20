import json
from pathlib import Path
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts.tcga_benchmark import (
    _experiment_supports_task,
    _feature_path,
    _feature_root,
    _load_task_metadata,
    _manifest_path,
    _prompt_provenance,
    _resolve_feature_bindings,
    _validate_feature_roles,
    _validate_mscpt_prompt,
    _validate_muse_prompts,
    _validate_protocol_registry,
    _validate_protocol_schedule,
    _validate_slide_embedding_source_config,
    _validate_sldpc_encoder_config,
    aggregate_results,
    build_generated_prompt_assets,
    build_manifests,
    build_splits,
    validate,
)
from common.prompts import (
    compile_task_prompt_assets,
    load_convlm_prompt_bank,
    load_focus_prompt_bank,
    load_vila_prompt_bank,
)
from common.run_state import run_identity


def test_task_metadata_adapter_combines_sources_and_normalizes_columns(
    tmp_path: Path,
):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame([
        {"image": "normal_001.svs", "patient": "p1", "diagnosis": "Normal"},
    ]).to_csv(train, index=False)
    pd.DataFrame([
        {"image": "tumor_001.svs", "patient": "p2", "diagnosis": "TUMOR"},
    ]).to_csv(test, index=False)
    task_cfg = {
        "metadata_csvs": [
            {"path": str(train), "partition": "official_train"},
            {"path": str(test), "partition": "official_test"},
        ],
        "slide_id_column": "image",
        "case_id_column": "patient",
        "label_column": "diagnosis",
        "label_transform": "lower",
    }

    frame = _load_task_metadata("camelyon", task_cfg)

    assert frame is not None
    assert frame["slide_id"].tolist() == ["normal_001", "tumor_001"]
    assert frame["case_id"].tolist() == ["p1", "p2"]
    assert frame["label"].tolist() == ["normal", "tumor"]
    assert frame["source_partition"].tolist() == [
        "official_train", "official_test"]


def _write_aggregate_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    results = tmp_path / "results"
    results.mkdir()
    cfg = {
        "method": "focus", "results_dir": str(results),
        "k_start": 0, "k_end": 3, "evaluate_test": True,
    }
    config = tmp_path / "run.yaml"
    config.write_text(json.dumps(cfg))
    (results / "config.json").write_text(json.dumps(cfg))
    folds = [
        {"fold": 2, "best_val_loss": 0.7, "test_acc": 0.8},
        {"fold": 0, "best_val_loss": 0.9, "test_acc": 0.2},
    ]
    (results / "metrics.json").write_text(json.dumps({
        "method": "focus",
        "run_identity": run_identity("focus", cfg),
        "folds": folds,
    }))
    pd.DataFrame([{
        "experiment": "focus", "method": "focus",
        "feature_signature": "conch@20x",
        "resolution_signature": "20x", "backbone": "conch",
        "encoder_provenance": "native", "cohort": "toy", "shots": 4,
        "config": str(config),
    }]).to_csv(tmp_path / "run_matrix.csv", index=False)
    return results, config, cfg


def test_aggregate_validates_identity_and_uses_population_std(tmp_path: Path):
    _write_aggregate_fixture(tmp_path)

    aggregate = aggregate_results(tmp_path)

    folds = pd.read_csv(tmp_path / "fold_results.csv")
    assert folds["fold"].tolist() == [2, 0]
    test_accuracy = aggregate.loc[aggregate["metric"] == "test_acc"].iloc[0]
    assert test_accuracy["mean"] == pytest.approx(0.5)
    assert test_accuracy["std"] == pytest.approx(0.3)
    assert test_accuracy["folds"] == 2


def test_aggregate_rejects_metrics_from_another_config(tmp_path: Path):
    _results, config, cfg = _write_aggregate_fixture(tmp_path)
    changed = {**cfg, "seed": 9}
    config.write_text(json.dumps(changed))

    with pytest.raises(RuntimeError, match="different method or configuration"):
        aggregate_results(tmp_path)


def test_task_metadata_rejects_blank_required_identifiers(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame([{
        "image": "slide-a.svs", "patient": None, "diagnosis": "A",
    }]).to_csv(metadata, index=False)

    with pytest.raises(ValueError, match="blank values"):
        _load_task_metadata("toy", {
            "metadata_csv": str(metadata),
            "slide_id_column": "image",
            "case_id_column": "patient",
            "label_column": "diagnosis",
        })


def test_protocol_schedule_rejects_duplicate_or_zero_shots():
    with pytest.raises(ValueError, match="positive integers"):
        _validate_protocol_schedule({"shots": [0], "folds": 5, "seed": 1})
    with pytest.raises(ValueError, match="duplicates"):
        _validate_protocol_schedule({"shots": [4, 4], "folds": 5, "seed": 1})


def test_shared_pickle_slide_source_resolves_to_the_file(tmp_path: Path):
    store = tmp_path / "toy_embeddings.pkl"
    store.write_bytes(b"payload")
    source = {
        "input_kind": "slide_embedding",
        "storage": "pkl",
        "path_template": str(store),
    }

    assert _feature_path(source, "slide-a", "toy") == store
    assert _feature_root(source, "toy") == str(store)


def test_portable_manifest_paths_are_expanded_before_readiness_checks(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("PGVL_STORAGE_ROOT", str(tmp_path))

    assert _manifest_path("${PGVL_STORAGE_ROOT}/slide.h5") == \
        tmp_path / "slide.h5"


def test_shared_pickle_slide_source_rejects_per_slide_template(tmp_path: Path):
    source = {
        "input_kind": "slide_embedding",
        "storage": "pkl",
        "path_template": str(tmp_path / "{slide_id}.pkl"),
    }

    with pytest.raises(ValueError, match="dataset-level path"):
        _feature_path(source, "slide-a", "toy")


def test_wide_split_table_joins_complete_manifest_rows(tmp_path: Path):
    from common.datasets.split_tables import load_phase_table

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"slide_id": "train-a", "label": "A", "feature": "/a.pt"},
        {"slide_id": "val-a", "label": "B", "feature": "/b.pt"},
        {"slide_id": "test-a", "label": "A", "feature": "/c.pt"},
    ]).to_csv(manifest, index=False)
    pd.DataFrame({
        "train": ["train-a"], "val": ["val-a"], "test": ["test-a"],
    }).to_csv(tmp_path / "splits_0.csv", index=False)
    cfg = {
        "split_dir": str(tmp_path), "dataset_csv": str(manifest),
        "label_dict": {"A": 0, "B": 1}, "k": 1,
    }

    table = load_phase_table(cfg, "val", 0)

    assert table.to_dict("records") == [{
        "slide_id": "val-a", "label": "B", "feature": "/b.pt",
    }]


def test_upstream_fold_table_uses_phase_specific_labels(tmp_path: Path):
    from common.datasets.split_tables import load_phase_table

    fold = tmp_path / "fold0.csv"
    pd.DataFrame({
        "train": ["train-a"], "train_label": [0],
        "val": ["val-a"], "val_label": [1],
        "test": ["test-a"], "test_label": [0],
    }).to_csv(fold, index=False)
    cfg = {"split_dir": str(tmp_path), "dataset_csv": str(fold), "k": 1}

    table = load_phase_table(cfg, "test", 0)

    assert table["slide_id"].tolist() == ["test-a"]
    assert table["label"].tolist() == [0]


def test_unscoped_phase_file_is_not_reused_across_folds(tmp_path: Path):
    from common.datasets.split_tables import load_phase_table

    pd.DataFrame([{"slide_id": "slide-a", "label": "A"}]).to_csv(
        tmp_path / "train.csv", index=False)
    cfg = {"split_dir": str(tmp_path), "k": 2}

    with pytest.raises(ValueError, match="unscoped phase file"):
        load_phase_table(cfg, "train", 0)


def test_future_metadata_and_task_specific_experiments(tmp_path: Path):
    task_cfg = {
        "metadata_csv": str(tmp_path / "not-yet-downloaded.csv"),
        "metadata_availability": "future",
        "label_column": "label",
    }

    assert _load_task_metadata("ubc_ocean", task_cfg) is None
    experiment = {"tasks": ["ubc_ocean"]}
    assert _experiment_supports_task(experiment, "ubc_ocean")
    assert not _experiment_supports_task(experiment, "camelyon16")
    assert _experiment_supports_task({}, "camelyon16")


def _write_bag(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.ones((3, 512), dtype=np.float32))
        handle.create_dataset("coords", data=np.zeros((3, 2), dtype=np.int32))


def test_tcga_protocol_builds_nested_patient_disjoint_splits(tmp_path: Path):
    rows = []
    feature_root = tmp_path / "features"
    for label in ("A", "B"):
        for index in range(8):
            slide_id = f"{label}-slide-{index}"
            case_id = f"{label}-case-{index}"
            rows.append(
                {"slide_id": slide_id, "case_id": case_id, "OncoTreeCode": label}
            )
            _write_bag(feature_root / "low" / f"{slide_id}.h5")
            _write_bag(feature_root / "high" / f"{slide_id}.h5")
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    protocol = {
        "version": 3,
        "seed": 1,
        "folds": 2,
        "shots": [1, 2],
        "feature_sources": {
            "toy_low": {
                "resolution": "5x",
                "backbone": "conch",
                "feature_key": "features",
                "feature_dim": 512,
                "feature_space_id": "toy:conch",
                "path_template": str(feature_root / "low/{slide_id}.h5"),
            },
            "toy_high": {
                "resolution": "20x",
                "backbone": "conch",
                "feature_key": "features",
                "feature_dim": 512,
                "feature_space_id": "toy:conch",
                "path_template": str(feature_root / "high/{slide_id}.h5"),
            },
        },
        "cohorts": {
            "toy": {
                "metadata_csv": str(metadata),
                "label_column": "OncoTreeCode",
                "labels": ["A", "B"],
            }
        },
    }

    coverage = build_manifests(protocol, tmp_path / "benchmark")
    build_splits(protocol, tmp_path / "benchmark")
    report = validate(protocol, tmp_path / "benchmark")

    assert coverage["available_slides"].eq(16).all()
    assert report["valid"] is True
    assert report["missing_features"] == []
    one_shot = pd.read_csv(
        tmp_path / "benchmark/splits/toy/1shot/fold0/train.csv"
    )
    two_shot = pd.read_csv(
        tmp_path / "benchmark/splits/toy/2shot/fold0/train.csv"
    )
    assert set(one_shot["slide_id"]).issubset(set(two_shot["slide_id"]))


def test_annotation_universe_keeps_rows_without_features(tmp_path: Path):
    metadata = tmp_path / "future.csv"
    pd.DataFrame(
        [
            {"slide_id": "slide-a", "case_id": "case-a", "label": "A"},
            {"slide_id": "slide-b", "case_id": "case-b", "label": "B"},
        ]
    ).to_csv(metadata, index=False)
    protocol = {
        "version": 3,
        "feature_sources": {
            "future": {
                "resolution": "10x",
                "backbone": "conch",
                "feature_key": "features",
                "feature_dim": 512,
                "feature_space_id": "future:encoder",
                "path_template": str(
                    tmp_path / "future-features/{slide_id}.h5"
                ),
            }
        },
        "cohorts": {
            "future": {
                "metadata_csv": str(metadata),
                "label_column": "label",
                "labels": ["A", "B"],
            }
        },
    }

    coverage = build_manifests(protocol, tmp_path / "benchmark")
    manifest = pd.read_csv(tmp_path / "benchmark/data/future/manifest.csv")

    assert len(manifest) == 2
    assert coverage["available_slides"].tolist() == [0]


def test_rcc_prompt_json_converts_to_muse_csv_banks(tmp_path: Path):
    prompt_json = tmp_path / "descriptions.json"
    prompt_json.write_text(
        '{"A": {"small_mag": ["low A"], "big_mag": ["high A"]},'
        ' "B": {"small_mag": ["low B"], "big_mag": ["high B"]}}'
    )
    protocol = {
        "cohorts": {
            "future": {
                "labels": ["A", "B"],
                "muse_prompt_json": str(prompt_json),
            }
        }
    }

    generated = build_generated_prompt_assets(protocol, tmp_path / "benchmark")

    assert len(generated["future"]) == 2
    assert len(pd.read_csv(generated["future"][0])) == 2


def test_canonical_prompt_profile_compiles_every_method_schema(tmp_path: Path):
    descriptions = {
        label: {
            "classname": classname,
            "low_res": [f"{label} low {index}" for index in range(10)],
            "high_res": [f"{label} high {index}" for index in range(10)],
            "aliases": [classname, label],
        }
        for label, classname in (("A", "class alpha"), ("B", "class beta"))
    }
    task_cfg = {
        "labels": ["A", "B"],
        "classnames": ["class alpha", "class beta"],
        "prompt_spec": {
            "version": 1,
            "context": "A label-agnostic diagnostic context.",
            "classes": descriptions,
        },
    }

    assets = compile_task_prompt_assets(
        "toy", task_cfg, tmp_path, repo_root=tmp_path)

    assert set(assets).issuperset({
        "focus", "vila_mil", "muse", "mscpt", "maple", "cod_mil",
        "slip", "sldpc_zero_shot", "convlm",
    })
    assert "wsi_five_default_report" not in assets
    assert assets["provenance"] == "generated"
    assert assets["source_profile_provenance"] == "user_defined"
    focus = load_focus_prompt_bank(
        assets["focus"], class_names=["A", "B"],
        file_class_names=["A", "B"],
        record={"provenance": "generated"})
    assert focus.prompts == ("A low 0", "B low 0", "A high 0", "B high 0")
    assert assets["vila_mil"] != assets["focus"]
    vila = load_vila_prompt_bank(
        assets["vila_mil"], class_names=["A", "B"],
        file_class_names=["A", "B"],
        record={"provenance": "generated"})
    assert vila.prompts == ("A low 0", "B low 0", "A high 0", "B high 0")
    assert list(json.loads(Path(assets["cod_mil"]).read_text())) == [
        "class alpha", "class beta"]
    mscpt = json.loads(Path(assets["mscpt"]).read_text())
    assert len(mscpt["A"]["small_mag"]) == 10
    assert len(mscpt["B"]["big_mag"]) == 10
    maple = json.loads(Path(assets["maple"]).read_text())
    assert maple["_provenance"] == "generated"
    assert maple["_metadata"]["classnames"] == [
        "class alpha", "class beta"]
    convlm = load_convlm_prompt_bank(
        assets["convlm"], classnames=["class alpha", "class beta"])
    assert convlm.provenance == "generated"
    convlm_payload = json.loads(Path(assets["convlm"]).read_text())
    assert convlm.prompt_counts == tuple(
        convlm_payload["_metadata"]["prompt_counts_per_class"].values())


def test_sldpc_prompt_provenance_tracks_active_tokens_not_reference_yaml():
    cohort = {
        "sldpc_prompt_classnames": ["IDC", "ILC"],
        "sldpc_prompt_provenance": "derived",
        "sldpc_zero_shot_prompt_yaml": "text_prompts/sldpc/tcga_brca.yaml",
        "prompt_provenance": "upstream",
    }

    assert _prompt_provenance(cohort, "sldpc") == "derived"


def test_benchmark_generator_uses_shared_muse_csv_validator(tmp_path: Path):
    prompt = tmp_path / "class_a.csv"
    prompt.write_text(",0\n0,a diagnostic description\n")

    _validate_muse_prompts({
        "classnames": ["class a"],
        "prompt_csvs": {"class a": str(prompt)},
    })


def test_dual_feature_roles_select_resolutions_independently():
    protocol = {
        "feature_sources": {
            "encoder_5x": {
                "resolution": "5x",
                "backbone": "conch",
                "feature_key": "features",
                "feature_dim": 512,
                "feature_space_id": "toy:conch",
            },
            "encoder_20x": {
                "resolution": "20x",
                "backbone": "conch",
                "feature_key": "features",
                "feature_dim": 512,
                "feature_space_id": "toy:conch",
            },
        }
    }
    experiment = {"features": {"low": "encoder_5x", "high": "encoder_20x"}}

    bindings = _resolve_feature_bindings(protocol, "dual_5x20x", experiment)
    _validate_feature_roles("mscpt", "dual_5x20x", bindings)

    assert bindings["low"]["config"]["resolution"] == "5x"
    assert bindings["high"]["config"]["resolution"] == "20x"


def test_dual_feature_roles_reject_incompatible_dimensions():
    protocol = {
        "feature_sources": {
            "encoder_5x": {
                "resolution": "5x",
                "backbone": "conch",
                "feature_key": "features",
                "feature_dim": 512,
                "feature_space_id": "toy:conch",
            },
            "encoder_20x": {
                "resolution": "20x",
                "backbone": "conch",
                "feature_key": "features",
                "feature_dim": 768,
                "feature_space_id": "toy:conch",
            },
        }
    }
    experiment = {"features": {"low": "encoder_5x", "high": "encoder_20x"}}
    bindings = _resolve_feature_bindings(protocol, "bad_dual", experiment)

    with pytest.raises(ValueError, match="incompatible feature_dim"):
        _validate_feature_roles("mscpt", "bad_dual", bindings)


def test_dual_feature_roles_reject_malformed_resolutions():
    bindings = {
        "low": {"config": {
            "resolution": "overview", "backbone": "conch",
            "encoder_weights": "weights", "feature_dim": 512,
            "feature_space_id": "space", "feature_key": "features",
        }},
        "high": {"config": {
            "resolution": "20x", "backbone": "conch",
            "encoder_weights": "weights", "feature_dim": 512,
            "feature_space_id": "space", "feature_key": "features",
        }},
    }

    with pytest.raises(ValueError, match="resolution"):
        _validate_feature_roles("mscpt", "bad_resolution", bindings)


def test_focus_requires_only_the_high_resolution_role():
    protocol = {
        "feature_sources": {
            "encoder_20x": {
                "resolution": "20x", "backbone": "conch",
                "feature_key": "features", "feature_dim": 512,
                "feature_space_id": "toy:conch",
            },
        },
    }
    bindings = _resolve_feature_bindings(
        protocol, "focus", {"features": {"high": "encoder_20x"}})

    _validate_feature_roles("focus", "focus", bindings)

    assert set(bindings) == {"high"}


def test_convlm_requires_a_precomputed_patch_bag_role():
    protocol = {
        "feature_sources": {
            "uni_bag": {
                "input_kind": "patch_bag",
                "resolution": "20x",
                "backbone": "uni",
                "feature_key": "features",
                "feature_dim": 1024,
                "feature_space_id": "hf:MahmoodLab/UNI",
            },
            "raw_tiles": {
                "input_kind": "raw_tile_directory",
                "resolution": "20x",
                "backbone": "none",
                "feature_key": "images",
                "feature_dim": 3,
                "feature_space_id": "rgb",
            },
        }
    }
    bag = _resolve_feature_bindings(
        protocol, "convlm_bag", {"features": {"bag": "uni_bag"}})
    raw = _resolve_feature_bindings(
        protocol, "convlm_raw", {"features": {"tiles": "raw_tiles"}})

    _validate_feature_roles("convlm", "convlm_bag", bag)
    with pytest.raises(ValueError, match="requires inputs.*bag"):
        _validate_feature_roles("convlm", "convlm_raw", raw)


def test_mscpt_prompt_labels_must_match_the_subtyping_task(tmp_path: Path):
    prompt_dir = tmp_path / "gpt" / "description"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / "BRCA.json"
    prompt_path.write_text(
        '{"High": {"small_mag": ["high"], "big_mag": ["high"]},'
        ' "Low": {"small_mag": ["low"], "big_mag": ["low"]}}'
    )
    cfg = {
        "description_prompt_path": str(prompt_path),
        "gpt_dir": str(tmp_path / "gpt"),
        "dataset_name": "BRCA",
        "label_dict": {"IDC": 0, "ILC": 1},
        "n_high": 1,
    }

    with pytest.raises(ValueError, match="do not match task labels"):
        _validate_mscpt_prompt(cfg)


def test_sldpc_reads_registered_h5_slide_embedding(tmp_path: Path):
    from methods.sldpc.dataset import SlideEmbeddingDataset

    embedding_path = tmp_path / "slide-a.h5"
    with h5py.File(embedding_path, "w") as handle:
        handle.create_dataset("features", data=np.ones(768, dtype=np.float32))
    split = tmp_path / "train.csv"
    pd.DataFrame([{
        "slide_id": "slide-a", "label": "A",
        "feature__titan": str(embedding_path),
    }]).to_csv(split, index=False)

    dataset = SlideEmbeddingDataset(
        "per_slide_h5", tmp_path, split, {"A": 0},
        feature_path_column="feature__titan")

    assert dataset[0]["feat"].shape == (768,)
    assert dataset[0]["label"] == 0


def test_sldpc_reads_arbitrary_torch_slide_embeddings_and_feature_keys(
        tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset

    embedding_path = tmp_path / "slide-b.pt"
    torch.save({"custom_slide_vector": torch.ones(13)}, embedding_path)
    split = tmp_path / "train.csv"
    pd.DataFrame([{
        "slide_id": "slide-b", "label": "B",
        "feature__custom": str(embedding_path),
    }]).to_csv(split, index=False)

    dataset = SlideEmbeddingDataset(
        "per_slide_torch", tmp_path, split, {"B": 0},
        feature_path_column="feature__custom",
        feature_key="custom_slide_vector", feature_dim=13)

    assert dataset[0]["feat"].shape == (13,)
    assert dataset[0]["label"] == 0


def test_slide_embedding_loader_rejects_any_missing_split_member(tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset

    torch.save({"features": torch.ones(4)}, tmp_path / "present.pt")
    split = tmp_path / "train.csv"
    pd.DataFrame([
        {"slide_id": "present", "case_id": "case-a", "label": "A"},
        {"slide_id": "missing", "case_id": "case-b", "label": "A"},
    ]).to_csv(split, index=False)

    with pytest.raises(FileNotFoundError, match="missing"):
        SlideEmbeddingDataset(
            "per_slide_torch", tmp_path, split, {"A": 0}, feature_dim=4)


def test_slide_embedding_loader_rejects_duplicate_normalized_sources(
        tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset

    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    torch.save(torch.ones(4), first / "slide-a.pt")
    torch.save(torch.ones(4), second / "slide-a.pth")
    split = tmp_path / "train.csv"
    pd.DataFrame([
        {"slide_id": "slide-a", "label": "A"},
    ]).to_csv(split, index=False)

    with pytest.raises(ValueError, match="duplicate normalized slide IDs"):
        SlideEmbeddingDataset(
            "per_slide_torch", tmp_path, split, {"A": 0}, feature_dim=4)


def test_slide_embedding_loader_rejects_scalar_pickle_id_field(tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset

    store = tmp_path / "slides.pkl"
    with store.open("wb") as handle:
        pickle.dump({
            "features": np.ones((1, 4), dtype=np.float32),
            "filenames": "slide-a",
        }, handle)
    split = tmp_path / "train.csv"
    pd.DataFrame([{"slide_id": "slide-a", "label": "A"}]).to_csv(
        split, index=False)

    with pytest.raises(TypeError, match="must be a sequence of IDs"):
        SlideEmbeddingDataset(
            "pkl", store, split, {"A": 0}, feature_dim=4)


def test_slide_embedding_loader_decodes_pickle_byte_ids(tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset

    store = tmp_path / "slides.pkl"
    with store.open("wb") as handle:
        pickle.dump({
            "features": np.ones((1, 4), dtype=np.float32),
            "filenames": np.asarray([b"slide-a.svs"]),
        }, handle)
    split = tmp_path / "train.csv"
    pd.DataFrame([{"slide_id": "slide-a", "label": "A"}]).to_csv(
        split, index=False)

    dataset = SlideEmbeddingDataset(
        "pkl", store, split, {"A": 0}, feature_dim=4)

    assert dataset[0]["slide_id"] == "slide-a"


def test_slide_embedding_loader_accepts_integral_wide_table_labels(
        tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset

    torch.save(torch.ones(4), tmp_path / "slide-a.pt")
    table = pd.DataFrame([{"slide_id": "slide-a", "label": 0.0}])

    dataset = SlideEmbeddingDataset(
        "per_slide_torch", tmp_path, table, {"A": 0}, feature_dim=4)

    assert dataset[0]["label"] == 0


def test_slide_embedding_loader_requires_configured_feature_column(
        tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset

    torch.save(torch.ones(4), tmp_path / "slide-a.pt")
    split = tmp_path / "train.csv"
    pd.DataFrame([{
        "slide_id": "slide-a", "label": "A",
    }]).to_csv(split, index=False)

    with pytest.raises(ValueError, match="no configured feature column"):
        SlideEmbeddingDataset(
            "per_slide_torch", tmp_path, split, {"A": 0},
            feature_path_column="feature__registered", feature_dim=4)


def test_slide_embedding_loader_rejects_blank_configured_feature_path(
        tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset

    torch.save(torch.ones(4), tmp_path / "slide-a.pt")
    split = tmp_path / "train.csv"
    split.write_text(
        "slide_id,label,feature__registered\n"
        "slide-a,A,\n")

    with pytest.raises(ValueError, match="blank configured feature path"):
        SlideEmbeddingDataset(
            "per_slide_torch", tmp_path, split, {"A": 0},
            feature_path_column="feature__registered", feature_dim=4)


def test_sldpc_learned_projection_keeps_slide_and_prompt_provenance_separate():
    cfg = {
        "backbone": "titan",
        "prompt_feature_space_id": "hf:MahmoodLab/TITAN@revision",
        "feature_space_id": "hf:another/slide-encoder@revision",
        "feature_dim": 1024,
        "feature_key": "slide_vector",
        "source_type": "per_slide_torch",
        "slide_features": "/features/another",
        "feature_path_column": "feature__another",
        "slide_encoder": {
            "input_kind": "slide_embedding",
            "name": "another-slide-encoder",
            "weights": "/models/another-slide-encoder",
            "feature_space_id": "hf:another/slide-encoder@revision",
            "feature_dim": 1024,
            "feature_key": "slide_vector",
            "resolution": "20x",
            "path_template": "/features/another/{slide_id}.pt",
            "runtime_encoder": False,
        },
        "slide_projection": {
            "mode": "linear",
            "input_dim": 1024,
            "output_dim": 768,
            "trainable": True,
        },
    }

    _validate_slide_embedding_source_config(cfg)
    _validate_sldpc_encoder_config(cfg)


def test_unified_trainer_dispatches_slide_vectors_by_feature_level(
        tmp_path: Path):
    from common.datasets.slide_embeddings import SlideEmbeddingDataset
    from train import build_loaders

    embedding_path = tmp_path / "slide-a.h5"
    with h5py.File(embedding_path, "w") as handle:
        handle.create_dataset(
            "custom", data=np.ones(5, dtype=np.float32))
    split_root = tmp_path / "splits" / "fold0"
    split_root.mkdir(parents=True)
    split = pd.DataFrame([{"slide_id": "slide-a", "label": "A"}])
    for name in ("train", "val", "test"):
        split.to_csv(split_root / f"{name}.csv", index=False)
    cfg = {
        "source_type": "per_slide_h5",
        "slide_features": str(tmp_path),
        "feature_key": "custom",
        "feature_dim": 5,
        "label_dict": {"A": 0},
        "split_dir": str(tmp_path / "splits"),
        "batch_size": 1,
        "num_workers": 0,
    }

    loaders = build_loaders("sldpc", cfg, 0)

    assert all(isinstance(loader.dataset, SlideEmbeddingDataset)
               for loader in loaders)
    assert next(iter(loaders[0]))["feat"].shape == (1, 5)


def test_protocol_accepts_an_unregistered_offline_slide_encoder(tmp_path: Path):
    slide_weights = tmp_path / "slide-encoder.ckpt"
    prompt_weights = tmp_path / "prompt-encoder.ckpt"
    slide_weights.touch()
    prompt_weights.touch()
    protocol = {
        "feature_sources": {
            "custom_slide": {
                "input_kind": "slide_embedding",
                "runtime_encoder": False,
                "resolution": "5x",
                "backbone": "my-private-slide-encoder",
                "encoder_weights": str(slide_weights),
                "feature_key": "slide_vector",
                "feature_dim": 1536,
                "feature_space_id": "private:slide-encoder@checkpoint-7",
                "path_template": str(tmp_path / "{slide_id}.pt"),
            },
        },
        "cohorts": {
            "toy": {
                "labels": ["A", "B"],
                "classnames": ["class alpha", "class beta"],
            },
        },
        "experiments": {
            "sldpc_custom": {
                "method": "sldpc",
                "features": {"bag": "custom_slide"},
                "prompt_encoder": {
                    "name": "titan",
                    "weights": str(prompt_weights),
                    "feature_space_id": "hf:MahmoodLab/TITAN",
                },
                "slide_projection": {"mode": "mlp"},
            },
        },
    }

    _validate_protocol_registry(protocol)


def test_cod_mil_reads_h5_bags_through_manifest_columns(tmp_path: Path):
    from methods.cod_mil.dataset import CoDMILFeaturesDataset

    low_path = tmp_path / "low.h5"
    high_path = tmp_path / "high.h5"
    for path, patches in ((low_path, 4), (high_path, 8)):
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "features", data=np.ones((patches, 1024), dtype=np.float32))
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    torch.save(torch.zeros((4, 1), dtype=torch.long), map_dir / "slide-a.pt")
    annotations = pd.DataFrame([{
        "slide_id": "slide-a", "label": "A",
        "low_path": str(low_path), "high_path": str(high_path),
    }])

    dataset = CoDMILFeaturesDataset(
        annotations, tmp_path, tmp_path, map_dir, {"A": 0},
        low_feature_column="low_path", high_feature_column="high_path")
    low, high, mapping, label = dataset[0]

    assert low.shape == (4, 1024)
    assert high.shape == (8, 1024)
    assert mapping.shape == (4, 1)
    assert label == 0


def test_cod_mil_collates_requested_prediction_metadata(tmp_path: Path):
    from methods.cod_mil.dataset import CoDMILFeaturesDataset, _collate

    low_path = tmp_path / "low.pt"
    high_path = tmp_path / "high.pt"
    torch.save(torch.ones(2, 4), low_path)
    torch.save(torch.ones(3, 4), high_path)
    maps = tmp_path / "maps"
    maps.mkdir()
    torch.save(torch.tensor([[0], [1]], dtype=torch.long), maps / "slide-a.pt")
    annotations = pd.DataFrame([{
        "slide_id": "slide-a", "case_id": "case-a", "label": "A",
        "low": str(low_path), "high": str(high_path),
    }])
    dataset = CoDMILFeaturesDataset(
        annotations, tmp_path, tmp_path, maps, {"A": 0},
        low_feature_column="low", high_feature_column="high",
        feature_dim=4, include_metadata=True)

    batch = _collate([dataset[0]])

    assert batch[-2] == {"slide_id": ["slide-a"], "case_id": ["case-a"]}
    assert batch[-1].tolist() == [0]


def test_cod_mil_rejects_out_of_range_correspondence_indices(tmp_path: Path):
    from methods.cod_mil.dataset import CoDMILFeaturesDataset

    low_path = tmp_path / "low.pt"
    high_path = tmp_path / "high.pt"
    torch.save(torch.ones(2, 4), low_path)
    torch.save(torch.ones(3, 4), high_path)
    maps = tmp_path / "maps"
    maps.mkdir()
    torch.save(torch.tensor([[0], [3]], dtype=torch.long), maps / "slide-a.pt")
    annotations = pd.DataFrame([{
        "slide_id": "slide-a", "label": "A",
        "low": str(low_path), "high": str(high_path),
    }])
    dataset = CoDMILFeaturesDataset(
        annotations, tmp_path, tmp_path, maps, {"A": 0},
        low_feature_column="low", high_feature_column="high", feature_dim=4)

    with pytest.raises(ValueError, match="outside"):
        dataset[0]


def test_wsi_five_matches_patient_filename_reports_to_tcga_case(tmp_path: Path):
    from methods.wsi_five.dataset import WSI_FiVE_Dataset

    feature_path = tmp_path / "slide-a.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=np.ones((3, 512), dtype=np.float32))
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame([{
        "slide_id": "TCGA-AA-0001-slide", "case_id": "TCGA-AA-0001",
        "label": "A", "feature_path": str(feature_path),
    }]).to_csv(annotations, index=False)
    reports = tmp_path / "reports.csv"
    pd.DataFrame([{
        "patient_filename": "TCGA-AA-0001.some-report-id",
        "text": "diagnostic pathology report",
    }]).to_csv(reports, index=False)
    dataset = WSI_FiVE_Dataset(
        annotations, tmp_path, reports, {"A": 0}, max_patches=8)
    dataset.feature_path_column = "feature_path"

    features, report, patch_info, label = dataset[0]
    assert features.shape == (3, 512)
    # The fusion transformer positions patches by their index within the
    # slide and masks padding per slide, so these travel with the bag.
    assert patch_info["sample_range"] == 3
    assert int(patch_info["patch_pub_cnt"]) == 3
    assert patch_info["patch_inds"].tolist() == [0.0, 1.0, 2.0]
    assert report == "diagnostic pathology report"
    assert label == 0


def test_wsi_five_uses_class_agnostic_context_when_reports_are_unavailable(
        tmp_path: Path):
    from methods.wsi_five.dataset import WSI_FiVE_Dataset

    feature_path = tmp_path / "slide-a.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset(
            "features", data=np.ones((3, 512), dtype=np.float32))
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame([{
        "slide_id": "slide-a", "case_id": "case-a", "label": "A",
        "feature_path": str(feature_path),
    }]).to_csv(annotations, index=False)
    context = "Label-agnostic lymph-node metastasis assessment."
    dataset = WSI_FiVE_Dataset(
        annotations, tmp_path, None, {"A": 0}, max_patches=8,
        default_report=context)
    dataset.feature_path_column = "feature_path"
    dataset.require_report = False

    _, report, _, _ = dataset[0]
    assert report == context


def test_wsi_five_adapter_does_not_pass_report_text_into_the_model():
    from methods.wsi_five.adapter import WSIFiVEMethod

    cfg = yaml.safe_load((
        Path(__file__).resolve().parents[1]
        / "configs" / "wsi_five" / "rcc.yaml").read_text())
    cfg.update({
        "feature_dim": 512,
        "n_classes": 2, "training_mode": "simplified_classnames",
        "classnames": ["class A", "class B"],
        "label_dict": {"A": 0, "B": 1},
    })
    method = WSIFiVEMethod(cfg, device="cpu")

    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.comparison_texts = None

        def forward(self, features, patch_info, comparison_texts):
            self.comparison_texts = comparison_texts
            return torch.tensor([[1.0, 0.0]])

    model = Capture()
    batch = (
        torch.ones((1, 3, 512)), ["specific diagnostic report"],
        {
            "patch_inds": torch.arange(3).reshape(1, 3),
            "patch_pub_cnt": torch.tensor([3.0]),
            "sample_range": [3],
        },
        torch.tensor([0]),
    )

    method.eval_step(batch, model)

    assert model.comparison_texts == ("class A", "class B")


def test_cod_mil_runtime_encodes_compiled_chain_once(tmp_path: Path):
    from methods.cod_mil.adapter import CoDMILMethod

    chain = {
        "class alpha": {"broad": ["alpha low"], "specific": ["alpha high"]},
        "class beta": {"broad": ["beta low"], "specific": ["beta high"]},
    }
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(chain))
    method = CoDMILMethod({
        "backbone": "clip-rn50",
        "feature_dim": 1024,
        "feature_space_id": "openai/clip-rn50@official",
        "text_feature_space_id": "openai/clip-rn50@official",
        "n_classes": 2,
        "classnames": ["class alpha", "class beta"],
        "text_prompt_path": str(path),
        "text_prompt_features": None,
        "prompt_encoding": "runtime_cached",
    }, device="cpu")

    class DummyBundle:
        def __init__(self):
            self.calls = 0
            self.prompts = []

        def freeze(self):
            return self

        def encode_text(self, prompts, normalize=True):
            self.calls += 1
            self.prompts = list(prompts)
            return torch.ones((len(prompts), 1024))

    bundle = DummyBundle()
    method.load_encoder = lambda **_: bundle

    first = method._prepare_text_features()
    second = method._prepare_text_features()

    # 2*C class prompts followed by the normal-tissue bank the auxiliary
    # contrastive branch contrasts against. The released kidney bank is
    # 3 low + 3 high + 21 normal; here it is 2 + 2 + the 15 organ-independent
    # rows reused verbatim from it.
    n_classes = 2
    assert first.shape == (n_classes * 2 + 15, 1024)
    assert first is second
    assert bundle.calls == 1
    assert bundle.prompts[:4] == [
        "alpha low", "beta low", "alpha high", "beta high"]
    background = bundle.prompts[n_classes * 2:]
    assert len(background) >= 15
    # Real tissue phenotypes, not templates naming the tumour class.
    assert all("alpha" not in p and "beta" not in p for p in background)
