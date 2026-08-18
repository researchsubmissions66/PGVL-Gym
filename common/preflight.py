"""Preconditions a benchmark configuration must meet before it can run.

A benchmark is a large matrix of configurations, and at any moment some of them
reference assets that have not been produced yet -- a prompt file that was never
imported, an encoder whose weights are not downloaded, features that have not
been extracted for a cohort. Those configurations must *skip*, cleanly and
visibly, rather than abort the campaign or (worse) train on nothing and report a
number.

:func:`preflight` inspects one resolved run config and answers a single
question: can this configuration produce a meaningful result? It reads only the
filesystem -- no model is constructed and no feature file is opened -- so it is
cheap enough to run at the top of every job.

Fatal findings become ``problems`` and the run is skipped. Recoverable ones
become ``warnings``: the run proceeds on the slides that do exist, and the
reduced count is recorded so the shortfall is never silent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Config keys whose value is a path to an asset the run reads. Absent keys are
# fine -- only a key that is present and points nowhere is a problem.
ASSET_KEYS = (
    "dataset_csv",
    "backbone_weights",
    "conch_ckpt",
    "text_prompt_path",
    "description_prompt_path",
    "prompt_reference_yaml",
    "gpt_dir",
    "feat_data_dir",
    "selected_5x_dir",
    "feature_root",
    "data_folder_s",
    "data_folder_l",
    "slide_features",
    "instance_prompt_path",
)

# Encoder blocks each name their own checkpoint, and those are separate assets
# from ``backbone_weights``: SLDPC declares an offline slide encoder and a
# runtime prompt encoder independently, and either can be absent on its own.
# Only ``weights`` is checked here -- sibling keys such as ``path_template``
# hold a ``{slide_id}`` pattern rather than a path, and are resolved per slide.
ASSET_BLOCK_KEYS = (
    "encoder",
    "patch_encoder",
    "slide_encoder",
    "prompt_encoder",
    "attribute_encoder",
)
ASSET_BLOCK_FIELDS = ("weights",)

# Keys holding a collection of asset paths. The shape varies by method: a plain
# list of paths, a list of {path, partition} records, or -- for MUSE -- a
# mapping of classname to that class's description CSV.
ASSET_LIST_KEYS = ("prompt_csvs", "muse_prompt_csvs", "metadata_csvs")

# (manifest, column) -> (available, expected); see _check_features.
_COVERAGE_CACHE: dict[tuple[str, str], tuple[int, int]] = {}


@dataclass
class PreflightReport:
    """Outcome of inspecting one run configuration."""

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    slides_expected: int = 0
    slides_available: int = 0

    @property
    def ok(self) -> bool:
        """Return True when the configuration can produce a real result."""
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "problems": self.problems,
            "warnings": self.warnings,
            "coverage": self.coverage,
            "slides_expected": self.slides_expected,
            "slides_available": self.slides_available,
        }

    def summary(self) -> str:
        return "; ".join(self.problems) if self.problems else "ready"


def _exists(value: Any) -> bool:
    """Return True when the path exists and is not an empty placeholder.

    An interrupted download leaves a zero-byte file behind, which satisfies a
    plain existence check and then fails at model build with an opaque
    ``EOFError: Ran out of input``. Treating an empty file as absent turns that
    into a skip naming the actual asset.
    """
    if not value:
        return False
    path = str(value)
    if not os.path.exists(path):
        return False
    if os.path.isfile(path) and os.path.getsize(path) == 0:
        return False
    if os.path.isdir(path) and not os.listdir(path):
        return False
    return True


def _asset_paths(value: Any) -> list[str]:
    """Extract asset paths from any of the shapes listed on ASSET_LIST_KEYS."""
    if not value:
        return []
    if isinstance(value, dict):
        # classname -> path, e.g. MUSE's per-class description CSVs
        entries: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple)):
        entries = value
    else:
        entries = [value]

    paths: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            entry = entry.get("path")
        if isinstance(entry, str) and entry:
            paths.append(entry)
    return paths


def _iter_split_files(split_dir: Path, k_start: int, k_end: int) -> Iterable[Path]:
    """Yield the split files for each fold, in either supported layout."""
    for fold in range(k_start, k_end):
        flat = split_dir / f"splits_{fold}.csv"
        nested = split_dir / f"fold{fold}"
        if flat.exists():
            yield flat
        elif nested.is_dir():
            yield nested
        else:
            yield flat  # report the conventional name when neither exists


def _feature_columns(cfg: dict[str, Any]) -> list[str]:
    return [
        str(value) for key, value in cfg.items()
        if key.startswith("feature_path_column") and value
    ]


def _check_features(cfg: dict[str, Any], report: PreflightReport) -> None:
    """Measure how many of the manifest's referenced feature files exist."""
    manifest = cfg.get("dataset_csv")
    if not _exists(manifest):
        return                                   # already reported as a problem

    columns = _feature_columns(cfg)
    if not columns:
        return              # method resolves features by root + slide id instead

    try:
        import csv
        with open(str(manifest), newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        report.problems.append(f"cannot read manifest {manifest}: {error}")
        return

    if not rows:
        report.problems.append(f"manifest has no rows: {manifest}")
        return

    for column in columns:
        if column not in rows[0]:
            report.problems.append(
                f"manifest {manifest} has no column {column!r}")
            continue
        # Coverage is a property of (manifest, column), and a campaign plan asks
        # about the same pair once per experiment and shot count. Cache it so a
        # whole-matrix plan does not restat the same thousands of paths.
        cache_key = (str(manifest), column)
        if cache_key in _COVERAGE_CACHE:
            available, expected = _COVERAGE_CACHE[cache_key]
        else:
            paths = [row[column] for row in rows if row.get(column)]
            available = sum(1 for path in paths if os.path.exists(path))
            expected = len(rows)
            _COVERAGE_CACHE[cache_key] = (available, expected)
        fraction = available / expected if expected else 0.0
        report.coverage[column] = round(fraction, 4)
        report.slides_expected = max(report.slides_expected, expected)
        report.slides_available = (
            available if not report.slides_available
            else min(report.slides_available, available))

        if available == 0:
            report.problems.append(
                f"no feature files exist for {column} "
                f"(0 of {expected} slides)")
        elif available < expected:
            report.warnings.append(
                f"{expected - available} of {expected} slides have no "
                f"{column} features; the run continues on the remaining "
                f"{available}")


def preflight(cfg: dict[str, Any], *,
              check_features: bool = True) -> PreflightReport:
    """Check whether one resolved run configuration can produce a result.

    Args:
        cfg: The loaded run configuration.
        check_features: When True, stat every feature file the manifest names.
            That is the authoritative check and belongs at the start of a job.
            A campaign planner should pass False and rely on the run matrix's
            recorded coverage instead: the scan costs thousands of network
            filesystem stats per cohort, which is worth paying once per job but
            not once per row of a whole-matrix plan.

    Returns:
        A :class:`PreflightReport`. ``report.ok`` is False when the run must be
        skipped, and ``report.problems`` explains why in terms of the specific
        asset that is absent.
    """
    report = PreflightReport()

    for key in ASSET_KEYS:
        if key in cfg and cfg[key] is not None and not _exists(cfg[key]):
            report.problems.append(f"{key} does not exist: {cfg[key]}")

    for key in ASSET_LIST_KEYS:
        for path in _asset_paths(cfg.get(key)):
            if not _exists(path):
                report.problems.append(f"{key} entry does not exist: {path}")

    for block in ASSET_BLOCK_KEYS:
        section = cfg.get(block)
        if not isinstance(section, dict):
            continue
        for field in ASSET_BLOCK_FIELDS:
            value = section.get(field)
            # A value carrying a brace is a per-slide pattern, not a path.
            if value and "{" not in str(value) and not _exists(value):
                report.problems.append(
                    f"{block}.{field} does not exist: {value}")

    split_dir = cfg.get("split_dir")
    if not split_dir:
        report.problems.append("split_dir is not set")
    elif not os.path.isdir(str(split_dir)):
        report.problems.append(f"split_dir does not exist: {split_dir}")
    else:
        k_start = int(cfg.get("k_start", 0) or 0)
        k_end = int(cfg.get("k_end", cfg.get("k", 5)) or 5)
        missing = [
            str(path) for path in _iter_split_files(Path(str(split_dir)), k_start, k_end)
            if not path.exists()
        ]
        if len(missing) == k_end - k_start:
            report.problems.append(
                f"no fold splits exist under {split_dir}")
        elif missing:
            report.warnings.append(
                f"{len(missing)} of {k_end - k_start} fold splits are missing")

    if check_features:
        _check_features(cfg, report)
    return report
