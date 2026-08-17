#!/usr/bin/env python3
"""Resume-aware launcher for the whole PGVL-Gym campaign.

The launcher walks every benchmark's ``run_matrix.csv``, decides what each run
still needs, and submits only that. A run that cannot proceed -- missing
features, missing metadata, an ungenerated split -- is *skipped with a recorded
reason* rather than being submitted to fail on a GPU, and never stops the walk.

State is derived from what is on disk (``metrics.json`` per results directory)
plus the live SLURM queue. There is no separate bookkeeping file to drift out of
sync, so the launcher can be re-run at any time and will submit exactly the
runs that are still outstanding.

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


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _count(value: Any) -> int:
    try:
        return int(str(value).strip() or 0)
    except ValueError:
        return 0


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _fold_progress(cfg: dict[str, Any]) -> tuple[int, int, Path | None]:
    """Return (completed folds, total folds, results dir) for one run."""
    results_dir = cfg.get("results_dir")
    k_start = int(cfg.get("k_start", 0) or 0)
    k_end = int(cfg.get("k_end", cfg.get("k", 5)) or 5)
    total = max(k_end - k_start, 0)
    if not results_dir:
        return 0, total, None

    path = Path(results_dir)
    metrics = path / "metrics.json"
    if not metrics.exists():
        return 0, total, path
    try:
        with open(metrics) as handle:
            folds = json.load(handle).get("folds", [])
    except (json.JSONDecodeError, OSError):
        # A truncated metrics.json (preempted mid-write) means "start over"
        # rather than "crash the launcher".
        return 0, total, path

    indices = {
        entry["fold"] for entry in folds
        if isinstance(entry, dict) and isinstance(entry.get("fold"), int)
    }
    completed = max(indices) + 1 - k_start if indices else len(folds)
    return max(min(completed, total), 0), total, path


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


def _queued_job_names() -> dict[str, str]:
    """Map job name -> job id for this user's pending and running jobs."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    try:
        output = subprocess.run(
            ["squeue", "-h", "-u", user, "--format=%i|%j"],
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"! cannot read the SLURM queue ({error}); "
              "duplicate-submission protection is off", file=sys.stderr)
        return {}
    if output.returncode != 0:
        print(f"! squeue failed: {output.stderr.strip()}", file=sys.stderr)
        return {}

    jobs: dict[str, str] = {}
    for line in output.stdout.splitlines():
        job_id, _, name = line.strip().partition("|")
        if name:
            jobs.setdefault(name, job_id)
    return jobs


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
    benchmark_dir = REPO / "benchmarks" / benchmark
    notes: list[str] = []

    if not benchmark_dir.is_dir():
        return [], [f"{benchmark}: directory does not exist"]

    matrix_path = benchmark_dir / "run_matrix.csv"
    if not matrix_path.exists():
        return [], [
            f"{benchmark}: no run_matrix.csv "
            "(run with --regenerate, or generate it with scripts/tcga_benchmark.py)"]

    try:
        with open(matrix_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        return [], [f"{benchmark}: cannot read run_matrix.csv ({error})"]

    runs: list[Run] = []
    for row in rows:
        run = Run(
            benchmark=benchmark,
            experiment=row.get("experiment") or row.get("method") or "?",
            method=row.get("method") or "",
            cohort=row.get("cohort") or "",
            shots=str(row.get("shots") or ""),
            config=Path(row.get("config") or ""),
            ready=_truthy(row.get("ready")),
            missing_features=_count(row.get("missing_feature_files")),
            missing_auxiliary=_count(row.get("missing_auxiliary_files")),
        )

        if not _selected(run, args):
            continue

        if not run.method or not str(run.config):
            run.state, run.reason = ERROR, "run matrix row has no method or config"
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
        except (yaml.YAMLError, OSError) as error:
            run.state, run.reason = ERROR, f"cannot parse config ({error})"
            runs.append(run)
            continue

        done, total, results_dir = _fold_progress(cfg)
        run.folds_done, run.folds_total, run.results_dir = done, total, results_dir
        if args.rerun:
            run.folds_done = done = 0

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

    if args.dry_run:
        return True, "dry-run"

    result = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False, (result.stderr.strip() or "sbatch failed").splitlines()[-1][:200]

    # "Submitted batch job 12345"
    job_id = result.stdout.strip().rsplit(" ", 1)[-1]
    return True, job_id


# -----------------------------------------------------------------------------
# reporting
# -----------------------------------------------------------------------------
def write_report(plan: Plan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "benchmark", "experiment", "method", "cohort", "shots", "state",
            "folds_done", "folds_total", "job_id", "reason", "config",
            "results_dir",
        ])
        for run in plan.runs:
            writer.writerow([
                run.benchmark, run.experiment, run.method, run.cohort,
                run.shots, run.state, run.folds_done, run.folds_total,
                run.job_id, run.reason, run.config, run.results_dir or "",
            ])


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
    selection.add_argument("--shots", nargs="+", type=int, default=None)
    selection.add_argument("--limit", type=int, default=None,
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
    behaviour.add_argument("--allow-missing-features", type=int, default=0,
                           metavar="N",
                           help="submit an otherwise-ready run when at most N "
                                "slide feature files are missing; the count is "
                                "recorded in the report")

    slurm = parser.add_argument_group("slurm")
    slurm.add_argument("--account", default=os.environ.get("PGVL_ACCOUNT", "bhwm-delta-gpu"))
    slurm.add_argument("--partition", default=os.environ.get("PGVL_PARTITION", "gpuA100x4"))
    slurm.add_argument("--gpus", default="1")
    slurm.add_argument("--cpus", default="8")
    slurm.add_argument("--mem", default="64G")
    slurm.add_argument("--time", default="12:00:00")
    slurm.add_argument("--device", default="cuda:0")
    slurm.add_argument("--log-dir", type=Path, default=REPO / "logs" / "slurm")
    slurm.add_argument("--report", type=Path,
                       default=REPO / "benchmarks" / "launch_report.csv")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = Plan()

    if not JOB_SCRIPT.exists():
        print(f"FATAL: job script not found: {JOB_SCRIPT}", file=sys.stderr)
        return 1
    if not os.access(JOB_SCRIPT, os.X_OK):
        JOB_SCRIPT.chmod(0o755)

    if args.regenerate:
        print("Regenerating benchmark artefacts")
        for benchmark in args.benchmarks:
            note = regenerate(benchmark, REPO / "benchmarks" / benchmark)
            if note:
                plan.notes.append(note)
                print(f"  ! {note}")

    queued = _queued_job_names()
    print(f"\nPlanning ({len(queued)} of your jobs already in the queue)")
    for benchmark in args.benchmarks:
        runs, notes = plan_benchmark(benchmark, args, queued)
        plan.runs.extend(runs)
        plan.notes.extend(notes)
        for note in notes:
            print(f"  ! {note}")
        if runs:
            ready = sum(run.state in {SUBMIT, RESUME} for run in runs)
            print(f"  {benchmark}: {len(runs)} runs, {ready} to submit")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    pending = [run for run in plan.runs if run.state in {SUBMIT, RESUME}]
    if args.limit is not None:
        held = pending[args.limit:]
        pending = pending[:args.limit]
        for run in held:
            run.state = SKIP
            run.reason = f"held back by --limit {args.limit}"

    if pending:
        print(f"\nSubmitting {len(pending)} jobs")
    for run in pending:
        ok, detail = submit(run, args, args.log_dir)
        if ok:
            run.job_id = detail
            note = f" ({run.reason})" if run.reason else ""
            print(f"  [{detail}] {run.label}{note}")
        else:
            # One rejected submission must not abort the campaign.
            run.state, run.reason = ERROR, f"sbatch rejected the job: {detail}"
            print(f"  ! {run.label}: {run.reason}", file=sys.stderr)

    write_report(plan, args.report)
    summarize(plan, args.report, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
