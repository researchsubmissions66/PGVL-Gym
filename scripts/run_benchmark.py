#!/usr/bin/env python3
"""Resume-aware launcher for the whole PGVL-Gym campaign.

The launcher walks every benchmark's ``run_matrix.csv``, decides what each run
still needs, and submits only that. A run that cannot proceed -- missing
features, missing metadata, an ungenerated split -- is *skipped with a recorded
reason* rather than being submitted to fail on a GPU, and never stops the walk.

State is derived from what is on disk (feature manifests and files,
``metrics.json`` per results directory) plus the live SLURM queue. Before each
plan, feature coverage and feature-derived matrix readiness are refreshed
without rebuilding prompts, splits, or configs. The launcher can therefore be
re-run after an asynchronous feature backfill and will pick up newly complete
runs.

Usage
-----
    # See the plan without touching the queue (always do this first).
    python scripts/run_benchmark.py --dry-run

    # Submit everything that is ready and not already done or queued.
    python scripts/run_benchmark.py

    # Regenerate manifests/splits/configs first, then submit.
    python scripts/run_benchmark.py --regenerate

    # Canary: one cohort, smallest shot count, at most three jobs.
    python scripts/run_benchmark.py --cohort brca --shots 4 --limit 3

Run states
----------
``done``    every fold present in ``metrics.json``; nothing submitted.
``resume``  some folds present; submitted, ``train.py`` continues where it left off.
``queued``  an identically named job is already pending or running.
``submit``  ready and not started.
``skip``    not ready; the reason is recorded and the walk continues.
``error``   the row itself could not be interpreted; recorded, walk continues.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


REPO = Path(__file__).resolve().parent.parent
JOB_SCRIPT = REPO / "scripts" / "pgvl_job.sh"

sys.path.insert(0, str(REPO))
from common.preflight import preflight  # noqa: E402
from common.configuration import expand_path, load_yaml_config  # noqa: E402
from common.readiness import refresh_benchmark_readiness  # noqa: E402
from common.run_state import validate_resume_state  # noqa: E402

# One benchmark directory per cohort. Multi-cohort protocols were split up so a
# cohort whose data is not ready cannot hold back the cohorts that are.
DEFAULT_BENCHMARKS = ("tcga_brca", "tcga_nsclc", "tcga_rcc",
                      "ubc_ocean", "camelyon16")

DONE, RESUME, QUEUED, SUBMIT, SKIP, ERROR = (
    "done", "resume", "queued", "submit", "skip", "error")

READINESS_COLUMNS = (
    ("metadata_ready", "metadata"),
    ("split_ready", "splits"),
    ("encoder_ready", "encoder weights"),
    ("config_valid", "config"),
    ("auxiliary_ready", "prompt assets"),
)
REQUIRED_MATRIX_COLUMNS = frozenset({
    "experiment", "method", "cohort", "shots", "config", "ready",
    "missing_feature_files", "missing_auxiliary_files",
    *(column for column, _label in READINESS_COLUMNS),
})


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _matrix_boolean(value: Any, field_name: str) -> bool:
    """Parse a run-matrix boolean without treating typos as false."""
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(
        f"{field_name} must be true or false, got {value!r}")


def _count(value: Any, field_name: str = "count") -> int:
    """Parse a non-negative matrix count, rejecting corrupt cells."""
    try:
        parsed = int(str(value or "").strip() or 0)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a non-negative integer, got {value!r}") \
            from error
    if parsed < 0:
        raise ValueError(
            f"{field_name} must be a non-negative integer, got {parsed}")
    return parsed


def _validate_ready_row(
    ready: bool, readiness: dict[str, bool], missing_features: int,
    missing_auxiliary: int,
) -> None:
    """Reject a matrix row that claims readiness despite known blockers."""
    if not ready:
        return
    contradictions = [
        column for column, value in readiness.items() if not value]
    if missing_features:
        contradictions.append("missing_feature_files")
    if missing_auxiliary:
        contradictions.append("missing_auxiliary_files")
    if contradictions:
        raise ValueError(
            "ready=true contradicts blocking fields: "
            + ", ".join(contradictions))


def _validate_matrix_header(fields: Sequence[str | None]) -> None:
    """Require the coverage/readiness evidence planning relies upon."""
    if not fields:
        raise ValueError("run matrix has no header")
    blank = [index for index, field in enumerate(fields)
             if field is None or not field.strip()]
    if blank:
        raise ValueError(
            f"run matrix has blank header fields at positions {blank}")
    duplicates = sorted({
        field for field in fields if fields.count(field) > 1})
    if duplicates:
        raise ValueError(
            f"run matrix has duplicate columns: {', '.join(duplicates)}")
    missing = sorted(REQUIRED_MATRIX_COLUMNS - set(fields))
    if missing:
        raise ValueError(
            "run matrix is missing required columns: " + ", ".join(missing))


def _benchmark_directory(benchmark: str) -> Path | None:
    """Resolve only simple benchmark names below the repository registry."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", benchmark) is None:
        return None
    return REPO / "benchmarks" / benchmark


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def _fold_progress(
    cfg: dict[str, Any], method_name: str | None = None,
) -> tuple[int, int, Path | None]:
    """Return (completed folds, total folds, results dir) for one run."""
    results_dir = cfg.get("results_dir")
    raw_start = cfg.get("k_start", 0)
    raw_end = cfg["k_end"] if "k_end" in cfg else cfg.get("k", 5)
    if (isinstance(raw_start, bool) or not isinstance(raw_start, int)
            or isinstance(raw_end, bool) or not isinstance(raw_end, int)):
        raise ValueError("k_start and k_end/k must be integer fold indices")
    k_start = raw_start
    k_end = raw_end
    if k_start < 0 or k_end <= k_start:
        raise ValueError(f"invalid fold range [{k_start}, {k_end})")
    total = k_end - k_start
    if not results_dir:
        return 0, total, None

    path = Path(results_dir)
    metrics = path / "metrics.json"
    if not metrics.exists():
        return 0, total, path
    try:
        with open(metrics) as handle:
            state = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"cannot read resume state {metrics}: {error}") from error
    method = method_name or cfg.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError(
            f"cannot validate resume state {metrics} without a method name")
    try:
        folds = validate_resume_state(
            state, path / "config.json", method, cfg)
    except RuntimeError as error:
        raise ValueError(f"invalid resume state {metrics}: {error}") from error
    indices = [entry["fold"] for entry in folds]
    outside = sorted(
        index for index in indices if index < k_start or index >= k_end)
    if outside:
        raise ValueError(
            f"resume state has folds outside [{k_start}, {k_end}): {outside}")
    # Count the actual set rather than inferring progress from max(index): a
    # preempted or manually repaired run can contain holes such as {0, 2}.
    return len(indices), total, path


def _recorded_skip(results_dir: Path | None) -> str | None:
    """Return the reason a previous attempt skipped this run, if it did."""
    if results_dir is None:
        return None
    marker = results_dir / "skipped.json"
    if not marker.exists():
        return None
    try:
        with open(marker) as handle:
            problems = json.load(handle).get("problems", [])
    except (json.JSONDecodeError, OSError):
        return "recorded in skipped.json"
    return "; ".join(problems[:2]) if problems else "recorded in skipped.json"


def _queued_job_names() -> dict[str, str] | None:
    """Map job name -> job id for this user's pending and running jobs."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    try:
        output = subprocess.run(
            ["squeue", "-h", "-u", user, "--format=%i|%j"],
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"! cannot read the SLURM queue ({error})", file=sys.stderr)
        return None
    if output.returncode != 0:
        print(f"! squeue failed: {output.stderr.strip()}", file=sys.stderr)
        return None

    jobs: dict[str, str] = {}
    for line in output.stdout.splitlines():
        job_id, _, name = line.strip().partition("|")
        if name:
            jobs.setdefault(name, job_id)
    return jobs


def _submission_environment_error() -> str | None:
    """Return why compute jobs cannot enter the project environment."""
    environment = os.environ.get("PGVL_CONDA_ENV", "").strip()
    if environment:
        return None
    return (
        "PGVL_CONDA_ENV is unset; refusing to submit jobs that would fall "
        "back to the incomplete site PyTorch module. Set it to the Conda "
        "environment created from environment.yml. Dry-run planning does "
        "not require the training environment.")


def _skip_reason(row: dict[str, str]) -> str:
    reasons = [
        f"{label} not ready"
        for column, label in READINESS_COLUMNS
        if column in row and not _truthy(row[column])
    ]
    missing_features = _count(row.get("missing_feature_files"))
    missing_auxiliary = _count(row.get("missing_auxiliary_files"))
    if missing_features:
        reasons.append(f"{missing_features} missing feature files")
    if missing_auxiliary:
        reasons.append(f"{missing_auxiliary} missing auxiliary files")
    return "; ".join(reasons) or "marked not ready by the run matrix"


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
@dataclass
class Run:
    """One row of a benchmark run matrix, plus everything decided about it."""

    benchmark: str
    experiment: str
    method: str
    cohort: str
    shots: str
    config: Path
    ready: bool
    missing_features: int
    missing_auxiliary: int
    state: str = SUBMIT
    reason: str = ""
    folds_done: int = 0
    folds_total: int = 0
    results_dir: Path | None = None
    job_id: str = ""

    @property
    def job_name(self) -> str:
        return (f"pgvl_{self.benchmark}_{self.experiment}"
                f"_{self.cohort}_{self.shots}shot")

    @property
    def label(self) -> str:
        return (f"{self.benchmark}/{self.experiment}/"
                f"{self.cohort}/{self.shots}shot")


@dataclass
class Plan:
    runs: list[Run] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def by_state(self, state: str) -> list[Run]:
        return [run for run in self.runs if run.state == state]


# -----------------------------------------------------------------------------
# planning
# -----------------------------------------------------------------------------
def regenerate(benchmark: str, benchmark_dir: Path) -> str | None:
    """Rebuild manifests, splits, configs and the run matrix for one benchmark.

    Returns a note describing the failure, or ``None`` on success. A benchmark
    that cannot be regenerated (for example a cohort whose metadata CSV does not
    exist yet) is reported and skipped; the other benchmarks still proceed.
    """
    protocol = benchmark_dir / "protocol.yaml"
    if not protocol.exists():
        return f"{benchmark}: no protocol.yaml; nothing to regenerate"

    print(f"  regenerating {benchmark} ...", flush=True)
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "tcga_benchmark.py"),
         "all", "--protocol", str(protocol)],
        cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return None

    tail = (result.stderr.strip() or result.stdout.strip()
            or "no output").splitlines()[-1][:200]
    return f"{benchmark}: regeneration failed ({tail})"


def plan_benchmark(benchmark: str, args: argparse.Namespace,
                   queued: dict[str, str]) -> tuple[list[Run], list[str]]:
    """Classify every run of one benchmark. Never raises for a bad row."""
    benchmark_dir = _benchmark_directory(benchmark)
    notes: list[str] = []

    if benchmark_dir is None:
        return [], [
            f"{benchmark}: invalid benchmark name; use a directory name "
            "directly under benchmarks/"]

    if not benchmark_dir.is_dir():
        return [], [f"{benchmark}: directory does not exist"]

    matrix_path = benchmark_dir / "run_matrix.csv"
    if not matrix_path.exists():
        return [], [
            f"{benchmark}: no run_matrix.csv "
            "(run with --regenerate, or generate it with scripts/tcga_benchmark.py)"]

    try:
        with open(matrix_path, newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_matrix_header(list(reader.fieldnames or []))
            rows = list(reader)
    except (OSError, ValueError) as error:
        return [], [f"{benchmark}: cannot read run_matrix.csv ({error})"]

    runs: list[Run] = []
    for row_number, row in enumerate(rows, start=2):
        raw_config = str(row.get("config") or "").strip()
        config_error = None
        try:
            config_path = Path(expand_path(raw_config)) if raw_config else Path()
            if raw_config and not config_path.is_absolute():
                config_path = REPO / config_path
        except (OSError, UnicodeError, ValueError) as error:
            config_path = Path(raw_config) if raw_config else Path()
            config_error = f"cannot resolve config path ({error})"
        row_error = None
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("row has the wrong number of fields")
            ready = _matrix_boolean(row.get("ready"), "ready")
            missing_features = _count(
                row.get("missing_feature_files"), "missing_feature_files")
            missing_auxiliary = _count(
                row.get("missing_auxiliary_files"),
                "missing_auxiliary_files")
            readiness = {}
            for column, _label in READINESS_COLUMNS:
                if column in row:
                    readiness[column] = _matrix_boolean(row[column], column)
            _validate_ready_row(
                ready, readiness, missing_features, missing_auxiliary)
        except ValueError as error:
            ready = False
            missing_features = missing_auxiliary = 0
            row_error = f"run matrix row {row_number}: {error}"
        run = Run(
            benchmark=benchmark,
            experiment=row.get("experiment") or row.get("method") or "?",
            method=row.get("method") or "",
            cohort=row.get("cohort") or "",
            shots=str(row.get("shots") or ""),
            config=config_path,
            ready=ready,
            missing_features=missing_features,
            missing_auxiliary=missing_auxiliary,
        )

        if not _selected(run, args):
            continue

        if row_error:
            run.state, run.reason = ERROR, row_error
            runs.append(run)
            continue

        if not run.method or not raw_config:
            run.state, run.reason = ERROR, "run matrix row has no method or config"
            runs.append(run)
            continue

        if (len(run.job_name) > 128
                or re.fullmatch(r"[A-Za-z0-9_.-]+", run.job_name) is None):
            run.state, run.reason = (
                ERROR, f"invalid SLURM job name derived from row: "
                f"{run.job_name!r}")
            runs.append(run)
            continue

        if config_error:
            run.state, run.reason = ERROR, config_error
            runs.append(run)
            continue

        if not run.config.exists():
            run.state, run.reason = SKIP, f"config file is missing: {run.config}"
            runs.append(run)
            continue

        # --- readiness -----------------------------------------------------
        if not run.ready and not args.force:
            tolerable = (
                args.allow_missing_features > 0
                and run.missing_features <= args.allow_missing_features
                and run.missing_auxiliary == 0
                and all(_truthy(row[column])
                        for column, _ in READINESS_COLUMNS if column in row)
            )
            if not tolerable:
                run.state, run.reason = SKIP, _skip_reason(row)
                runs.append(run)
                continue
            run.reason = (f"proceeding with {run.missing_features} missing "
                          "feature files (--allow-missing-features)")

        # --- resume --------------------------------------------------------
        try:
            cfg = _load_config(run.config)
        except (ValueError, OSError, yaml.YAMLError) as error:
            run.state, run.reason = ERROR, f"cannot parse config ({error})"
            runs.append(run)
            continue

        try:
            if args.rerun:
                done, total, _ = _fold_progress(
                    {**cfg, "results_dir": None}, run.method)
                configured_results = cfg.get("results_dir")
                results_dir = (
                    Path(configured_results) if configured_results else None)
            else:
                done, total, results_dir = _fold_progress(cfg, run.method)
        except ValueError as error:
            run.state, run.reason = ERROR, str(error)
            runs.append(run)
            continue
        run.folds_done, run.folds_total, run.results_dir = done, total, results_dir

        # A previous attempt already determined this configuration cannot run.
        # Honour that instead of resubmitting it every campaign.
        recorded = _recorded_skip(results_dir)
        if recorded and not (args.retry_skipped or args.force):
            run.state, run.reason = SKIP, f"previously skipped: {recorded}"
            runs.append(run)
            continue

        # The run matrix already measured feature coverage, so preflight here
        # only checks the assets a config names directly -- prompt files,
        # encoder weights, the manifest, the splits. A missing prompt then skips
        # at plan time rather than failing on a GPU. The authoritative
        # feature scan runs once inside the job, where its cost is worth paying.
        if not args.force:
            checked = preflight(cfg, check_features=False)
            if not checked.ok:
                run.state, run.reason = SKIP, "; ".join(checked.problems[:2])
                runs.append(run)
                continue
            if checked.warnings:
                run.reason = checked.warnings[0]

        if total and done >= total:
            run.state = DONE
            run.reason = f"{done}/{total} folds already complete"
        elif run.job_name in queued:
            run.state = QUEUED
            run.job_id = queued[run.job_name]
            run.reason = f"job {run.job_id} is already in the queue"
        elif done > 0:
            run.state = RESUME
            run.reason = f"resuming from fold {done} of {total}"
        else:
            run.state = SUBMIT

        runs.append(run)

    return runs, notes


def _mark_plan_collisions(plan: Plan) -> None:
    """Reject rows that could submit duplicate jobs or share output state."""
    collisions: dict[int, list[str]] = {}

    def record(groups: dict[str, list[Run]], description: str) -> None:
        for key, runs in groups.items():
            if len(runs) < 2:
                continue
            for run in runs:
                collisions.setdefault(id(run), []).append(
                    f"duplicate {description} {key!r}")

    job_groups: dict[str, list[Run]] = {}
    result_groups: dict[str, list[Run]] = {}
    for run in plan.runs:
        if run.state == ERROR:
            continue
        job_groups.setdefault(run.job_name, []).append(run)
        if run.results_dir is not None:
            key = str(run.results_dir.expanduser().resolve(strict=False))
            result_groups.setdefault(key, []).append(run)
    record(job_groups, "SLURM job name")
    record(result_groups, "results directory")

    for run in plan.runs:
        reasons = collisions.get(id(run))
        if reasons:
            run.state = ERROR
            run.reason = "; ".join(reasons)


def _selected(run: Run, args: argparse.Namespace) -> bool:
    if args.cohort and run.cohort not in args.cohort:
        return False
    if args.method and run.method not in args.method:
        return False
    if args.experiment and run.experiment not in args.experiment:
        return False
    if args.shots and run.shots not in {str(shot) for shot in args.shots}:
        return False
    return True


# -----------------------------------------------------------------------------
# submission
# -----------------------------------------------------------------------------
def submit(run: Run, args: argparse.Namespace, log_dir: Path) -> tuple[bool, str]:
    """sbatch one run. A submission failure is returned, never raised."""
    command = [
        "sbatch",
        f"--job-name={run.job_name}",
        f"--output={log_dir}/{run.job_name}_%j.out",
        f"--partition={args.partition}",
        f"--gpus={args.gpus}",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.mem}",
        f"--time={args.time}",
    ]
    if args.account:
        command.append(f"--account={args.account}")
    command += [str(JOB_SCRIPT), run.method, str(run.config), args.device]
    if args.rerun:
        command.append("--rerun")

    if args.dry_run:
        return True, "dry-run"

    result = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False, (result.stderr.strip() or "sbatch failed").splitlines()[-1][:200]

    match = re.fullmatch(
        r"Submitted batch job ([0-9]+)", result.stdout.strip())
    if match is None:
        return False, (
            "sbatch returned success but its job ID could not be parsed: "
            f"{result.stdout.strip()[:120]!r}")
    job_id = match.group(1)
    return True, job_id


# -----------------------------------------------------------------------------
# reporting
# -----------------------------------------------------------------------------
def write_report(plan: Plan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temporary, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "benchmark", "experiment", "method", "cohort", "shots",
                "state", "folds_done", "folds_total", "job_id", "reason",
                "config", "results_dir",
            ])
            for run in plan.runs:
                writer.writerow([
                    run.benchmark, run.experiment, run.method, run.cohort,
                    run.shots, run.state, run.folds_done, run.folds_total,
                    run.job_id, run.reason, run.config, run.results_dir or "",
                ])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _report_destination_error(
    plan: Plan, report: Path, benchmarks: Sequence[str] = (),
) -> str | None:
    """Return why a report path would overwrite campaign input or run state."""
    destination = report.expanduser().resolve(strict=False)
    protected: dict[Path, str] = {}
    for benchmark in benchmarks:
        benchmark_dir = _benchmark_directory(benchmark)
        if benchmark_dir is None:
            continue
        for name, label in (("run_matrix.csv", "run matrix"),
                            ("protocol.yaml", "benchmark protocol")):
            path = benchmark_dir / name
            protected[path.resolve(strict=False)] = f"{label} {path}"
    for run in plan.runs:
        protected[run.config.expanduser().resolve(strict=False)] = (
            f"run config {run.config}")
        benchmark_dir = _benchmark_directory(run.benchmark)
        if benchmark_dir is not None:
            matrix = benchmark_dir / "run_matrix.csv"
            protected[matrix.resolve(strict=False)] = f"run matrix {matrix}"
        if run.results_dir is not None:
            for name in ("config.json", "metrics.json", "skipped.json"):
                state = run.results_dir / name
                protected[state.expanduser().resolve(strict=False)] = (
                    f"run state {state}")
    if destination in protected:
        return f"refusing to overwrite {protected[destination]}"
    if report.exists() and not report.is_file():
        return f"report path exists but is not a regular file: {report}"
    return None


def summarize(plan: Plan, report_path: Path, dry_run: bool) -> None:
    order = (SUBMIT, RESUME, QUEUED, DONE, SKIP, ERROR)
    print("\n" + "=" * 74)
    print("  Campaign summary" + ("  (dry run -- nothing submitted)" if dry_run else ""))
    print("=" * 74)
    for state in order:
        runs = plan.by_state(state)
        if runs:
            print(f"  {state:<8} {len(runs):>4}")

    skipped = plan.by_state(SKIP) + plan.by_state(ERROR)
    if skipped:
        grouped: dict[str, list[Run]] = {}
        for run in skipped:
            grouped.setdefault(run.reason, []).append(run)
        print("\n  Skipped, grouped by reason:")
        for reason, runs in sorted(grouped.items(), key=lambda item: -len(item[1])):
            examples = ", ".join(run.label for run in runs[:2])
            more = f", +{len(runs) - 2} more" if len(runs) > 2 else ""
            print(f"    {len(runs):>4}  {reason}")
            print(f"          e.g. {examples}{more}")

    for note in plan.notes:
        print(f"  ! {note}")

    print(f"\n  Full report: {report_path}")
    print("=" * 74)


# -----------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS),
                        help="benchmark directories under benchmarks/ to walk")
    parser.add_argument("--regenerate", action="store_true",
                        help="rebuild manifests, splits, configs and run "
                             "matrices before planning")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without submitting anything")

    selection = parser.add_argument_group("selection")
    selection.add_argument("--cohort", nargs="+", default=None)
    selection.add_argument("--method", nargs="+", default=None)
    selection.add_argument("--experiment", nargs="+", default=None,
                           help="experiment/variant name, e.g. pathpt_keep")
    selection.add_argument(
        "--shots", nargs="+", type=_positive_int, default=None)
    selection.add_argument("--limit", type=_nonnegative_int, default=None,
                           help="submit at most this many jobs this invocation")

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--rerun", action="store_true",
                           help="ignore existing metrics.json and start every "
                                "selected run from fold 0")
    behaviour.add_argument("--retry-skipped", action="store_true",
                           help="reconsider runs a previous attempt recorded as "
                                "skipped, e.g. after extracting the missing "
                                "features or importing the missing prompts")
    behaviour.add_argument("--force", action="store_true",
                           help="submit runs the matrix marks not ready "
                                "(they are expected to fail; for debugging)")
    behaviour.add_argument(
        "--no-refresh-readiness", action="store_true",
        help="plan from the existing feature readiness cells without checking "
             "feature files (normally readiness is refreshed automatically)")
    behaviour.add_argument(
        "--best-effort", action="store_true",
        help="return success even when selected rows or submissions error")
    behaviour.add_argument("--allow-missing-features", type=_nonnegative_int,
                           default=0,
                           metavar="N",
                           help="submit an otherwise-ready run when at most N "
                                "slide feature files are missing; the count is "
                                "recorded in the report")

    slurm = parser.add_argument_group("slurm")
    slurm.add_argument("--account", default=os.environ.get("PGVL_ACCOUNT", "bhwm-delta-gpu"))
    slurm.add_argument("--partition", default=os.environ.get("PGVL_PARTITION", "gpuA100x4"))
    slurm.add_argument("--gpus", type=_positive_int, default=1)
    slurm.add_argument("--cpus", type=_positive_int, default=8)
    slurm.add_argument("--mem", default="64G")
    slurm.add_argument("--time", default="12:00:00")
    slurm.add_argument("--device", default="cuda:0")
    slurm.add_argument("--log-dir", type=Path, default=REPO / "logs" / "slurm")
    slurm.add_argument("--report", type=Path,
                       default=REPO / "benchmarks" / "launch_report.csv")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.log_dir = args.log_dir.expanduser()
    args.report = args.report.expanduser()
    plan = Plan()

    if not JOB_SCRIPT.exists():
        print(f"FATAL: job script not found: {JOB_SCRIPT}", file=sys.stderr)
        return 1
    if not args.dry_run:
        environment_error = _submission_environment_error()
        if environment_error:
            print(f"FATAL: {environment_error}", file=sys.stderr)
            return 1
    if not args.dry_run and not os.access(JOB_SCRIPT, os.X_OK):
        JOB_SCRIPT.chmod(0o755)

    if args.regenerate:
        print("Regenerating benchmark artefacts")
        for benchmark in args.benchmarks:
            benchmark_dir = _benchmark_directory(benchmark)
            note = (
                regenerate(benchmark, benchmark_dir)
                if benchmark_dir is not None else
                f"{benchmark}: invalid benchmark name; use a directory name "
                "directly under benchmarks/")
            if note:
                plan.notes.append(note)
                print(f"  ! {note}")

    refresh_failed: set[str] = set()
    if not args.no_refresh_readiness:
        print("Refreshing feature readiness", flush=True)
        for benchmark in args.benchmarks:
            benchmark_dir = _benchmark_directory(benchmark)
            if benchmark_dir is None:
                continue
            try:
                refreshed = refresh_benchmark_readiness(benchmark_dir)
            except (OSError, ValueError) as error:
                note = f"{benchmark}: readiness refresh failed ({error})"
                plan.notes.append(note)
                refresh_failed.add(benchmark)
                print(f"  ! {note}", flush=True)
            else:
                print(
                    f"  {benchmark}: {refreshed.feature_sources} feature "
                    f"sources, {refreshed.matrix_rows} rows, "
                    f"{refreshed.changed_rows} changed", flush=True)

    queued = _queued_job_names()
    if queued is None:
        if not args.dry_run and not args.force:
            print(
                "FATAL: queue state is unavailable; refusing to submit without "
                "duplicate-job protection (use --force to override)",
                file=sys.stderr)
            return 1
        queued = {}
    print(f"\nPlanning ({len(queued)} of your jobs already in the queue)")
    for benchmark in args.benchmarks:
        if benchmark in refresh_failed:
            continue
        runs, notes = plan_benchmark(benchmark, args, queued)
        plan.runs.extend(runs)
        plan.notes.extend(notes)
        for note in notes:
            print(f"  ! {note}")
        if runs:
            ready = sum(run.state in {SUBMIT, RESUME} for run in runs)
            print(f"  {benchmark}: {len(runs)} runs, {ready} to submit")

    _mark_plan_collisions(plan)

    if not plan.runs and not plan.notes:
        plan.notes.append("no run-matrix rows matched the requested selection")

    if not args.dry_run:
        try:
            args.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(
                f"FATAL: cannot prepare log directory {args.log_dir}: {error}",
                file=sys.stderr)
            return 1
    pending = [run for run in plan.runs if run.state in {SUBMIT, RESUME}]
    if args.limit is not None:
        held = pending[args.limit:]
        pending = pending[:args.limit]
        for run in held:
            run.state = SKIP
            run.reason = f"held back by --limit {args.limit}"

    report_error = _report_destination_error(plan, args.report, args.benchmarks)
    if report_error:
        print(f"FATAL: {report_error}", file=sys.stderr)
        return 1
    if not args.dry_run:
        # Prove the report destination is writable before the first external
        # submission. The final report replaces this provisional plan after
        # job IDs and any submission failures are known.
        try:
            write_report(plan, args.report)
        except OSError as error:
            print(
                f"FATAL: cannot write campaign report {args.report}: {error}",
                file=sys.stderr)
            return 1

    if pending:
        print(f"\nSubmitting {len(pending)} jobs")
    for run in pending:
        ok, detail = submit(run, args, args.log_dir)
        if ok:
            run.job_id = detail
            if not args.dry_run:
                prior_reason = run.reason
                run.state = QUEUED
                run.reason = f"submitted as job {detail}"
                if prior_reason:
                    run.reason += f"; {prior_reason}"
            note = f" ({run.reason})" if run.reason else ""
            print(f"  [{detail}] {run.label}{note}")
        else:
            # One rejected submission must not abort the campaign.
            run.state, run.reason = ERROR, f"sbatch rejected the job: {detail}"
            print(f"  ! {run.label}: {run.reason}", file=sys.stderr)

    try:
        write_report(plan, args.report)
    except OSError as error:
        print(
            f"FATAL: cannot finalize campaign report {args.report}: {error}",
            file=sys.stderr)
        return 1
    summarize(plan, args.report, args.dry_run)
    failed = bool(plan.by_state(ERROR) or plan.notes)
    return 1 if failed and not args.best_effort else 0


if __name__ == "__main__":
    raise SystemExit(main())
