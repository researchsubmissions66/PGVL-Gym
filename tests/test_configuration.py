from pathlib import Path

import pytest
import yaml

from common.configuration import (
    expand_environment, expand_path, load_dotenv, load_yaml_config,
)


def test_dotenv_loads_without_overriding_process_environment(
    tmp_path: Path, monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text("LOCAL_ROOT=/from/file\nQUOTED='a value'\n")
    monkeypatch.setenv("LOCAL_ROOT", "/from/process")

    load_dotenv(env_file)

    assert expand_environment("${LOCAL_ROOT}/data") == "/from/process/data"
    assert expand_environment("${QUOTED}") == "a value"


def test_yaml_loader_recursively_expands_environment(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("TEST_STORAGE", "/portable/storage")
    config = tmp_path / "config.yaml"
    config.write_text(
        "root: ${TEST_STORAGE}/features\n"
        "nested:\n"
        "  - ${TEST_STORAGE}/prompts.csv\n")

    loaded = load_yaml_config(config)

    assert loaded["root"] == "/portable/storage/features"
    assert loaded["nested"] == ["/portable/storage/prompts.csv"]


def test_yaml_loader_recursively_expands_home_relative_paths(
    tmp_path: Path, monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config = tmp_path / "config.yaml"
    config.write_text(
        "split_dir: ~/splits\n"
        "encoder:\n"
        "  weights: ~/weights/model.pt\n"
        "prompt: '~ is used here as prose'\n")

    loaded = load_yaml_config(config)

    assert loaded["split_dir"] == str(home / "splits")
    assert loaded["encoder"]["weights"] == str(home / "weights/model.pt")
    assert loaded["prompt"] == "~ is used here as prose"


def test_undefined_environment_reference_fails_clearly(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNDEFINED_PGVL_VAR", raising=False)

    with pytest.raises(ValueError, match="DEFINITELY_UNDEFINED_PGVL_VAR"):
        expand_environment("${DEFINITELY_UNDEFINED_PGVL_VAR}/features")


def test_dotenv_rejects_invalid_shell_variable_names(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("1INVALID=value\n")

    with pytest.raises(ValueError, match="invalid .env entry"):
        load_dotenv(env_file)


def test_expand_path_expands_user_home(monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/pgvl-home")

    assert expand_path("~/features") == "/tmp/pgvl-home/features"


@pytest.mark.parametrize("source, expected", [
    ("revision: 2025-01-02\n", "quote dates"),
    ("label_dict:\n  1: 0\n", "non-string mapping key"),
    ("lr: .nan\n", "non-finite number"),
    ("classes: !!set {A: null}\n", "unsupported YAML value"),
    ("loop: &loop [*loop]\n", "cyclic YAML alias"),
])
def test_yaml_loader_rejects_values_that_cannot_roundtrip_through_json(
    tmp_path: Path, source: str, expected: str,
):
    config = tmp_path / "invalid.yaml"
    config.write_text(source)

    with pytest.raises(ValueError, match=expected):
        load_yaml_config(config)


def test_yaml_loader_rejects_duplicate_keys_at_any_depth(tmp_path: Path):
    config = tmp_path / "duplicate.yaml"
    config.write_text(
        "seed: 1\n"
        "nested:\n"
        "  learning_rate: 0.1\n"
        "  learning_rate: 0.2\n")

    with pytest.raises(yaml.YAMLError, match="duplicate key 'learning_rate'"):
        load_yaml_config(config)
