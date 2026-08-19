"""Portable configuration loading and environment-variable expansion."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which refuses ambiguous duplicate mapping keys."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _value_node in node.value:
            # SafeLoader gives merge keys defined override behavior. Reject
            # only keys repeated directly within the same authored mapping.
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                duplicate = False
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r}", key_node.start_mark)
            try:
                seen.add(key)
            except TypeError:
                # SafeLoader will emit its standard unhashable-key error.
                pass
        return super().construct_mapping(node, deep=deep)


def load_dotenv(path: str | Path | None = None, *,
                override: bool = False) -> Path | None:
    """Load simple ``KEY=VALUE`` entries from the repository's ignored .env.

    Existing process variables win unless ``override`` is true. This keeps
    scheduler- or shell-provided values authoritative and avoids a dependency
    on python-dotenv.
    """
    env_path = Path(path) if path is not None else REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    for line_number, raw_line in enumerate(
            env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or _VARIABLE_NAME.fullmatch(key) is None:
            raise ValueError(f"invalid .env entry at {env_path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def expand_environment(value: Any, *,
                       environ: Mapping[str, str] | None = None) -> Any:
    """Recursively expand ``${NAME}`` references in configuration data.

    Raises:
        ValueError: If a referenced variable is not defined.
    """
    variables = os.environ if environ is None else environ
    if isinstance(value, str):
        missing = sorted({name for name in _VARIABLE.findall(value)
                          if name not in variables})
        if missing:
            raise ValueError(
                "undefined environment variable(s): " + ", ".join(missing))
        return _VARIABLE.sub(lambda match: variables[match.group(1)], value)
    if isinstance(value, dict):
        return {key: expand_environment(item, environ=variables)
                for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item, environ=variables) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment(item, environ=variables)
                     for item in value)
    return value


def expand_user_paths(value: Any) -> Any:
    """Recursively expand explicit home-relative path strings.

    Only ``~`` and ``~/...`` are interpreted. This avoids treating arbitrary
    prompt prose containing a tilde as a filesystem path while ensuring every
    runtime consumer sees the same paths that preflight checks.
    """
    if isinstance(value, str):
        if value == "~" or value.startswith("~/"):
            return os.path.expanduser(value)
        return value
    if isinstance(value, dict):
        return {key: expand_user_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_user_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_user_paths(item) for item in value)
    return value


def expand_path(value: str | Path) -> str:
    """Load local environment settings and expand one filesystem path."""
    load_dotenv()
    return os.path.expanduser(str(expand_environment(str(value))))


def collapse_environment(value: Any) -> Any:
    """Recursively replace configured machine roots with portable variables."""
    load_dotenv()
    roots = [
        (os.environ[name].rstrip("/"), "${" + name + "}")
        for name in ("PGVL_REPO_ROOT", "PGVL_STORAGE_ROOT", "PGVL_USER_ROOT")
        if os.environ.get(name)
    ]
    roots.sort(key=lambda item: len(item[0]), reverse=True)
    if isinstance(value, str):
        for root, variable in roots:
            if value == root or value.startswith(root + "/"):
                return variable + value[len(root):]
        return value
    if isinstance(value, dict):
        return {key: collapse_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [collapse_environment(item) for item in value]
    if isinstance(value, tuple):
        return tuple(collapse_environment(item) for item in value)
    return value


def load_yaml_file(path: str | Path) -> Any:
    """Safely load one YAML document while rejecting duplicate keys."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_UniqueKeyLoader)


def _validate_config_tree(
    value: Any, location: str = "config", ancestors: set[int] | None = None,
) -> None:
    """Require the stable JSON value model used by snapshots and hashes."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{location} contains a non-finite number {value!r}")
        return
    ancestors = set() if ancestors is None else ancestors
    if isinstance(value, (list, dict)) and id(value) in ancestors:
        raise ValueError(f"{location} contains a cyclic YAML alias")
    if isinstance(value, list):
        ancestors.add(id(value))
        try:
            for index, item in enumerate(value):
                _validate_config_tree(
                    item, f"{location}[{index}]", ancestors)
        finally:
            ancestors.remove(id(value))
        return
    if isinstance(value, dict):
        ancestors.add(id(value))
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"{location} contains non-string mapping key {key!r}; "
                        "quote YAML keys that look numeric")
                _validate_config_tree(item, f"{location}.{key}", ancestors)
        finally:
            ancestors.remove(id(value))
        return
    raise ValueError(
        f"{location} contains unsupported YAML value {value!r} "
        f"({type(value).__name__}); quote dates and use only JSON-compatible "
        "configuration values")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping after applying repository-local environment data."""
    load_dotenv()
    config_path = Path(expand_path(path))
    loaded = load_yaml_file(config_path)
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    resolved = expand_user_paths(expand_environment(loaded))
    _validate_config_tree(resolved)
    return resolved
