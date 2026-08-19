#!/usr/bin/env python3
"""Diagnose whether PGVL-Gym and its run configurations are ready.

Normal checks avoid importing PyTorch and only inspect filesystem metadata.
Opt-in ``--deep`` validation opens feature tensors, but no mode constructs a
model. The command is useful both as an interactive ``doctor`` and as a
machine-readable campaign gate.
"""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.configuration import load_dotenv, load_yaml_config  # noqa: E402
from common.preflight import PREFLIGHT_CHECKS, preflight  # noqa: E402


EXIT_HEALTHY = 0
EXIT_UNHEALTHY = 1

CORE_MODULES = {
    "torch": ("PyTorch", "torch", ((2, 4), (2, 7))),
    "torchvision": ("torchvision", "torchvision", ((0, 19), (0, 22))),
    "numpy": ("NumPy", "numpy", ((1, 24), (2, 0))),
    "scipy": ("SciPy", "scipy", None),
    "pandas": ("pandas", "pandas", None),
    "yaml": ("PyYAML", "PyYAML", None),
    "sklearn": ("scikit-learn", "scikit-learn", None),
    "tensorboard": ("TensorBoard", "tensorboard", None),
    "tensorboardX": ("tensorboardX", "tensorboardX", None),
    "h5py": ("h5py", "h5py", None),
    "transformers": ("Transformers", "transformers", ((4, 40), (5, 0))),
    "ftfy": ("ftfy", "ftfy", None),
    "regex": ("regex", "regex", None),
    "tqdm": ("tqdm", "tqdm", None),
    "termcolor": ("termcolor", "termcolor", None),
}
ROOT_VARIABLES = ("PGVL_REPO_ROOT", "PGVL_USER_ROOT", "PGVL_STORAGE_ROOT")


class Palette:
    """Small ANSI palette that automatically degrades for redirected output."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def pass_(self, text: str) -> str:
        return self.paint(text, "32")

    def warn(self, text: str) -> str:
        return self.paint(text, "33")

    def fail(self, text: str) -> str:
        return self.paint(text, "31")

    def dim(self, text: str) -> str:
        return self.paint(text, "2")

    def heading(self, text: str) -> str:
        return self.paint(text, "1;36")


def _coverage_fraction(value: str) -> float:
    """Argparse converter for a closed-interval coverage fraction."""
    try:
        fraction = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from error
    if not 0.0 <= fraction <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return fraction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose feature coverage, prompts, splits, encoder weights, "
            "and host setup without loading a model."),
        epilog=(
            "Examples:\n"
            "  %(prog)s run.yaml\n"
            "  %(prog)s run.yaml --quick --strict\n"
            "  %(prog)s --system\n"
            "  %(prog)s configs/focus/*.yaml --json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("configs", nargs="*", metavar="CONFIG",
                        help="run YAML configuration(s) to diagnose")
    selection = parser.add_argument_group(
        "check selection (combine freely; omitted means --all)")
    for name in sorted(PREFLIGHT_CHECKS):
        selection.add_argument(
            f"--{name}", action="store_true",
            help=f"check {name}")
    selection.add_argument("--all", action="store_true",
                           help="run every configuration health check")
    selection.add_argument(
        "--system", action="store_true",
        help="also diagnose Python, core packages, and PGVL root variables")
    parser.add_argument(
        "--min-feature-coverage", type=_coverage_fraction, metavar="FRACTION",
        help="override the config's required feature coverage for this check")
    parser.add_argument(
        "--no-feature-scan", action="store_true",
        help="check feature roots without statting every manifest row")
    parser.add_argument(
        "--quick", action="store_true",
        help="alias for --no-feature-scan")
    parser.add_argument(
        "--deep", action="store_true",
        help="open feature payloads and validate keys, shape, width, and values")
    parser.add_argument(
        "--strict", action="store_true",
        help="treat warnings as an unhealthy diagnosis")
    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true",
                        help="emit stable machine-readable JSON only")
    output.add_argument("--quiet", action="store_true",
                        help="print only failures, warnings, and the summary")
    output.add_argument("--verbose", action="store_true",
                        help="show every configured path, including healthy ones")
    output.add_argument("--no-color", action="store_true",
                        help="disable ANSI colors")
    return parser


def _selected_checks(args: argparse.Namespace) -> set[str]:
    selected = {name for name in PREFLIGHT_CHECKS if getattr(args, name)}
    return set(PREFLIGHT_CHECKS) if args.all or not selected else selected


def _diagnostic(name: str, status: str, message: str,
                fix: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": name,
        "status": status,
        "message": message,
    }
    if fix:
        item["fix"] = fix
    return item


def _major_minor(package_version: str) -> tuple[int, int] | None:
    """Parse the numeric major/minor prefix used by project version bounds."""
    pieces = package_version.split(".", 2)
    if len(pieces) < 2:
        return None
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError:
        return None


def _torchvision_compatible(torch_version: str,
                            torchvision_version: str) -> bool:
    """Check the official Torch 2.x to torchvision 0.x release pairing."""
    torch_pair = _major_minor(torch_version)
    vision_pair = _major_minor(torchvision_version)
    if torch_pair is None or vision_pair is None:
        return False
    return (torch_pair[0] == 2 and vision_pair[0] == 0
            and vision_pair[1] == torch_pair[1] + 15)


def _system_diagnostics() -> dict[str, Any]:
    """Inspect the base host setup without importing heavyweight packages."""
    started = time.monotonic()
    checks: list[dict[str, Any]] = []
    package_versions: dict[str, str] = {}
    supported = (3, 10) <= sys.version_info[:2] < (3, 12)
    checks.append(_diagnostic(
        "python", "pass" if supported else "fail",
        f"Python {platform.python_version()} at {sys.executable}",
        (None if supported else
         "Create the Python 3.10 environment from environment.yml."),
    ))

    repository_ok = (ROOT / "pyproject.toml").is_file()
    checks.append(_diagnostic(
        "repository", "pass" if repository_ok else "fail", str(ROOT),
        None if repository_ok else "Run the command from a complete PGVL-Gym checkout.",
    ))

    dotenv = None
    roots_before_dotenv = {
        variable: os.environ.get(variable) for variable in ROOT_VARIABLES
    }
    try:
        dotenv = load_dotenv()
    except (OSError, ValueError) as error:
        checks.append(_diagnostic(
            "environment.dotenv", "fail", str(error),
            "Correct malformed KEY=VALUE entries in .env and rerun the doctor.",
        ))
    for variable in ROOT_VARIABLES:
        raw = os.environ.get(variable)
        if not raw:
            checks.append(_diagnostic(
                f"environment.{variable}", "fail", "not set",
                "Copy .env.example to .env and set all PGVL root paths.",
            ))
            continue
        expanded = Path(os.path.expanduser(raw))
        exists = expanded.is_dir()
        wrong_repository = False
        if variable == "PGVL_REPO_ROOT" and exists:
            try:
                wrong_repository = expanded.resolve() != ROOT.resolve()
                exists = not wrong_repository
            except OSError:
                exists = False
        if roots_before_dotenv[variable] is not None:
            source = " via process environment"
        elif dotenv:
            source = f" via {dotenv}"
        else:
            source = ""
        checks.append(_diagnostic(
            f"environment.{variable}", "pass" if exists else "fail",
            f"{expanded}{source}",
            (None if exists else
             (f"Set {variable} to this checkout ({ROOT})."
              if wrong_repository else
              f"Create the directory or correct {variable} in .env.")),
        ))

    for module, (label, distribution, bounds) in CORE_MODULES.items():
        try:
            installed = find_spec(module) is not None
        except (ImportError, AttributeError, ValueError):
            installed = False
        package_version = None
        if installed:
            try:
                package_version = version(distribution)
            except PackageNotFoundError:
                pass
        if package_version:
            package_versions[module] = package_version
        version_ok = True
        if package_version and bounds:
            parsed = _major_minor(package_version)
            version_ok = parsed is not None and bounds[0] <= parsed < bounds[1]
        installed = installed and version_ok
        package_message = (
            f"{label} {package_version} is available" if package_version
            else f"{label} is {'available' if installed else 'not importable'}")
        if package_version and not version_ok:
            package_message = (
                f"{label} {package_version} is outside the supported range "
                f">={bounds[0][0]}.{bounds[0][1]},"
                f"<{bounds[1][0]}.{bounds[1][1]}")
        checks.append(_diagnostic(
            f"package.{module}", "pass" if installed else "fail",
            package_message,
            (None if installed else
             "Install the base environment with `pip install -e .`."),
        ))

    torch_version = package_versions.get("torch")
    vision_version = package_versions.get("torchvision")
    if torch_version and vision_version:
        compatible = _torchvision_compatible(torch_version, vision_version)
        checks.append(_diagnostic(
            "package.torchvision_compatibility",
            "pass" if compatible else "fail",
            f"torch {torch_version} with torchvision {vision_version}",
            (None if compatible else
             "Install the matching torchvision release (torch 2.4/2.5/2.6 "
             "pairs with torchvision 0.19/0.20/0.21)."),
        ))

    return {
        "ok": all(item["status"] == "pass" for item in checks),
        "checks": checks,
        "duration_seconds": round(time.monotonic() - started, 4),
    }


def _load_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def _advice_for(message: str) -> str:
    """Translate common diagnoses into a concrete next action."""
    lowered = message.lower()
    if "invalid .env entry" in lowered:
        return "Correct malformed KEY=VALUE entries in .env and rerun the doctor."
    if "undefined environment variable" in lowered:
        return "Copy .env.example to .env and define the named PGVL root variable."
    if "results_dir" in lowered or "writable/searchable" in lowered:
        return (
            "Choose a writable results_dir or correct the parent directory "
            "permissions.")
    if "cannot read manifest" in lowered or "manifest has no" in lowered:
        return (
            "Regenerate the cohort manifest with "
            "`scripts/tcga_benchmark.py prepare`.")
    if "split" in lowered or "fold range" in lowered:
        return (
            "Regenerate patient-disjoint splits with "
            "`scripts/tcga_benchmark.py prepare`.")
    if any(token in lowered for token in ("prompt", "description", "gpt_dir")):
        return (
            "Generate/import the method prompt assets, then update the run "
            "YAML path.")
    if "feature" in lowered:
        return (
            "Extract the missing slide features or correct the manifest "
            "feature paths.")
    if any(token in lowered for token in (
            "encoder", "backbone", "checkpoint", "weights", "conch_ckpt")):
        return "Download the configured encoder checkpoint or correct its weights path."
    if "dataset_csv" in lowered:
        return (
            "Correct dataset_csv or regenerate benchmark data with the "
            "protocol compiler.")
    if "does not exist" in lowered or "empty" in lowered:
        return "Create the asset or correct its path in the run configuration."
    return "Correct the run configuration, then rerun this doctor command."


def _fixes_for(item: dict[str, Any]) -> list[str]:
    fixes: list[str] = []
    for message in item.get("problems", []):
        fix = _advice_for(message)
        if fix not in fixes:
            fixes.append(fix)
    return fixes


def _effective_ok(item: dict[str, Any], strict: bool) -> bool:
    return bool(item["ok"] and (not strict or not item.get("warnings")))


def _status_label(status: str, palette: Palette) -> str:
    if status == "pass":
        return palette.pass_("PASS")
    if status == "warn":
        return palette.warn("WARN")
    return palette.fail("FAIL")


def _format_bytes(value: int | None) -> str:
    if value is None:
        return ""
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return ""


def _render_system(system: dict[str, Any], palette: Palette,
                   quiet: bool) -> None:
    if quiet and system["ok"]:
        return
    print(palette.heading("Host environment"))
    for check in system["checks"]:
        if quiet and check["status"] == "pass":
            continue
        marker = _status_label(check["status"], palette)
        print(f"  {marker} {check['name']}: {check['message']}")
        if check.get("fix"):
            print(f"       fix: {check['fix']}")


def _render_config(item: dict[str, Any], palette: Palette, *,
                   quiet: bool, verbose: bool, strict: bool) -> None:
    effective_ok = _effective_ok(item, strict)
    status = "pass" if effective_ok else "fail"
    duration = palette.dim(f"({item['duration_seconds']:.2f}s)")
    print(f"{_status_label(status, palette)} {item['config']} {duration}")
    if item.get("load_error"):
        print(f"  {_status_label('fail', palette)} configuration: {item['load_error']}")
    elif not quiet:
        print(f"  checks: {', '.join(item['checks'])}")
        if "features" in item["checks"] and not item.get("feature_scan", True):
            print("  feature scan: skipped (root paths only)")

    paths = item.get("checked_paths", {})
    visible_paths = paths.items() if verbose else (
        (name, detail) for name, detail in paths.items()
        if not detail["available"])
    for label, detail in visible_paths:
        state = "pass" if detail["available"] else "fail"
        resolved = detail.get("resolved_path") or detail["path"]
        size = _format_bytes(detail.get("size_bytes"))
        suffix = f" ({size})" if size else ""
        reason = f" — {detail['reason']}" if detail.get("reason") else ""
        print(f"  {_status_label(state, palette)} {label}: {resolved}{suffix}{reason}")

    coverage = item.get("coverage", {})
    if coverage and not quiet:
        print("  feature coverage:")
        for name, fraction in coverage.items():
            print(f"    {name}: {fraction:.1%}")
        print(
            f"    complete rows: {item['slides_available']}/"
            f"{item['slides_expected']}")
    if item.get("deep_features_checked") and not quiet:
        print(
            "  deep feature payloads checked: "
            f"{item['deep_features_checked']}")

    for warning in item.get("warnings", []):
        suffix = " (strict mode)" if strict else ""
        print(f"  {_status_label('warn', palette)} {warning}{suffix}")
    failed_path_prefixes = tuple(
        f"{label}:" for label, detail in paths.items()
        if not detail["available"])
    for problem in item.get("problems", []):
        if (problem != item.get("load_error")
                and not problem.startswith(failed_path_prefixes)):
            print(f"  {_status_label('fail', palette)} {problem}")

    fixes = _fixes_for(item)
    for index, fix in enumerate(fixes, start=1):
        print(f"       fix {index}: {fix}")


def _summary(results: list[dict[str, Any]], system: dict[str, Any] | None,
             strict: bool, elapsed: float) -> dict[str, Any]:
    config_healthy = sum(_effective_ok(item, strict) for item in results)
    warning_count = sum(len(item.get("warnings", [])) for item in results)
    problem_count = sum(len(item.get("problems", [])) for item in results)
    system_failures = 0
    if system is not None:
        system_failures = sum(
            item["status"] == "fail" for item in system["checks"])
    healthy = (
        config_healthy == len(results)
        and system_failures == 0
        and (not strict or warning_count == 0)
    )
    return {
        "healthy": healthy,
        "configs_checked": len(results),
        "configs_healthy": config_healthy,
        "problems": problem_count,
        "warnings": warning_count,
        "system_failures": system_failures,
        "duration_seconds": round(elapsed, 4),
    }


def _render_text(results: list[dict[str, Any]], system: dict[str, Any] | None,
                 summary: dict[str, Any], args: argparse.Namespace) -> None:
    palette = Palette(
        enabled=not args.no_color and sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None)
    if not args.quiet:
        print(palette.heading("PGVL-Gym doctor"))
        print(palette.dim("Read-only diagnostics; no model weights are loaded."))
        print()
    visible_system = system is not None and (not args.quiet or not system["ok"])
    visible_results = [
        item for item in results
        if not args.quiet or not _effective_ok(item, args.strict)
    ]
    if visible_system:
        assert system is not None
        _render_system(system, palette, args.quiet)
        if visible_results:
            print()
    for index, item in enumerate(visible_results):
        if index:
            print()
        _render_config(
            item, palette, quiet=args.quiet, verbose=args.verbose,
            strict=args.strict)

    if visible_results or visible_system:
        print()
    label = _status_label("pass" if summary["healthy"] else "fail", palette)
    parts = []
    if results:
        parts.extend([
            (f"{summary['configs_healthy']}/{summary['configs_checked']} "
             "configs ready"),
            f"{summary['problems']} problems",
            f"{summary['warnings']} warnings",
        ])
    if system is not None:
        parts.append(f"{summary['system_failures']} host failures")
    print(f"{label} Diagnosis: {' · '.join(parts)} "
          f"· {summary['duration_seconds']:.2f}s")
    if summary["healthy"]:
        print("Everything checked is ready.")
    else:
        print("Apply the fixes above and rerun the same command.")


def _diagnose_configs(paths: Iterable[str], args: argparse.Namespace,
                      selected: set[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw_path in paths:
        started = time.monotonic()
        path = Path(raw_path)
        try:
            cfg = _load_config(path)
            if args.min_feature_coverage is not None:
                cfg["min_feature_coverage"] = args.min_feature_coverage
            report = preflight(
                cfg, checks=selected,
                check_features=not (args.no_feature_scan or args.quick),
                deep_features=args.deep)
            result = {"config": str(path), "ok": report.ok,
                      **report.as_dict()}
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
            result = {
                "config": str(path), "ok": False,
                "checks": sorted(selected), "load_error": str(error),
                "problems": [str(error)], "warnings": [], "coverage": {},
                "checked_paths": {}, "slides_expected": 0,
                "slides_available": 0, "deep_features_checked": 0,
            }
        result["feature_scan"] = bool(
            "features" in selected
            and not (args.no_feature_scan or args.quick))
        result["duration_seconds"] = round(time.monotonic() - started, 4)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.configs and not args.system:
        parser.error("provide at least one CONFIG or use --system")
    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose cannot be used together")
    if args.deep and (args.no_feature_scan or args.quick):
        parser.error(
            "--deep cannot be combined with --quick or --no-feature-scan")

    started = time.monotonic()
    selected = _selected_checks(args)
    if args.deep and "features" not in selected:
        parser.error("--deep requires the --features check")
    system = _system_diagnostics() if args.system else None
    results = _diagnose_configs(args.configs, args, selected)
    summary = _summary(
        results, system, args.strict, time.monotonic() - started)

    if args.json:
        payload = {
            "schema_version": 1,
            "healthy": summary["healthy"],
            "strict": args.strict,
            "summary": summary,
            "system": system,
            "results": results,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_text(results, system, summary, args)
    return EXIT_HEALTHY if summary["healthy"] else EXIT_UNHEALTHY


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Unix commands should compose cleanly with `head`, `less`, and pipes.
        raise SystemExit(EXIT_HEALTHY) from None
