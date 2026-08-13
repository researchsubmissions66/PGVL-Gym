#!/usr/bin/env python3
"""Run real model construction and dummy-feature forwards for the framework.

The default matrix mode selects one generated config per experiment variant,
then runs each in an isolated subprocess. Isolation prevents one foundation
model from retaining GPU memory and gives every experiment a hard timeout.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import pandas as pd
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MATRIX = REPO_ROOT / "benchmarks" / "additional_tasks" / "run_matrix.csv"


def _label() -> torch.Tensor:
    return torch.tensor([0], dtype=torch.long)


def _bag(patches: int, width: int, *, batched: bool) -> torch.Tensor:
    shape = (1, patches, width) if batched else (patches, width)
    return torch.randn(shape, dtype=torch.float32)


def synthetic_batch(method: str, cfg: dict[str, Any]) -> Any:
    """Mirror each unified loader's output using small in-memory tensors."""
    width = int(cfg.get("feature_dim", 512))
    label = _label()
    if method in {"focus", "vila_mil", "maple"}:
        return (_bag(32, width, batched=False),
                _bag(64, width, batched=False), label)
    if method == "mscpt":
        # Native MSCPT performs hard-coded top-k(100) and top-k(5).
        return (_bag(100, width, batched=True),
                _bag(8, width, batched=True), label)
    if method == "cod_mil":
        low_count, high_count, children = 32, 128, 4
        mapping = torch.arange(low_count * children).reshape(
            low_count, children).remainder(high_count)
        return (_bag(low_count, width, batched=False),
                _bag(high_count, width, batched=False), mapping.long(), label)
    if method == "pathpt":
        features = _bag(32, width, batched=True)
        coordinates = torch.zeros((1, 32, 2), dtype=torch.float32)
        return features, coordinates, label
    if method in {"muse", "top", "slip", "composite"}:
        return _bag(32, width, batched=True), label
    if method == "wsi_five":
        frames = min(int(cfg.get("num_frames", 8)), 8)
        report = str(cfg.get(
            "default_report", "Class-agnostic pathology assessment."))
        return _bag(frames, width, batched=True), [report], label
    if method == "sldpc":
        return {
            "feat": torch.randn((1, width), dtype=torch.float32),
            "label": label,
            "slide_id": ["dummy-slide"],
            "case_id": ["dummy-case"],
        }
    if method == "convlm":
        image_size = int(cfg.get("image_size", 64))
        images = torch.randn(
            (1, 1, 3, image_size, image_size), dtype=torch.float32)
        return images, label
    raise KeyError(f"No synthetic batch contract for method {method!r}")


def _smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["num_workers"] = 0
    cfg["include_metadata"] = False
    if cfg["method"] == "wsi_five":
        cfg["num_frames"] = min(int(cfg.get("num_frames", 8)), 8)
    elif cfg["method"] == "convlm":
        # Preserve all transformer layers and widths while reducing the dummy
        # spatial token count from 28x28 to 4x4.
        cfg["image_size"] = 64
        cfg["max_tiles_per_slide"] = 1
    return cfg


def smoke_one(config_path: Path, device: str) -> dict[str, Any]:
    started = time.monotonic()
    resolved_device = torch.device(device)
    with config_path.open(encoding="utf-8") as handle:
        cfg = _smoke_overrides(yaml.safe_load(handle))
    method_name = str(cfg["method"])
    result: dict[str, Any] = {
        "experiment": cfg.get("experiment", method_name),
        "method": method_name,
        "config": str(config_path),
        "device": device,
        "status": "failed",
        "stage": "adapter_init",
    }
    torch.manual_seed(7)
    model = method = batch = None
    try:
        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)
        from methods import get_method

        print(f"  [{method_name}] adapter init", flush=True)
        method = get_method(method_name)(cfg, device=device)
        result["stage"] = "model_build"
        print(f"  [{method_name}] model build", flush=True)
        model = method.build_model()
        model.eval()
        result["trainable_parameters"] = int(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad))
        result["stage"] = "dummy_forward"
        batch = synthetic_batch(method_name, cfg)
        print(f"  [{method_name}] dummy forward", flush=True)
        with torch.inference_mode():
            output = method.eval_step(batch, model)
        logits = output.get("logits") if isinstance(output, dict) else None
        if not torch.is_tensor(logits):
            raise TypeError("eval_step did not return tensor logits")
        if logits.ndim != 2 or logits.shape != (1, int(cfg["n_classes"])):
            raise ValueError(
                f"Expected logits [1, {cfg['n_classes']}], got "
                f"{list(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Dummy forward returned non-finite logits")
        result.update({
            "status": "passed",
            "stage": "complete",
            "logits_shape": list(logits.shape),
            "logits_finite": True,
        })
    except Exception as error:
        result.update({
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
    finally:
        result["seconds"] = round(time.monotonic() - started, 3)
        if resolved_device.type == "cuda" and torch.cuda.is_available():
            result["peak_gpu_mib"] = round(
                torch.cuda.max_memory_allocated(resolved_device) / 1024 ** 2, 1)
        del batch, model, method
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _representative_configs(
    matrix_path: Path, cohort: str,
) -> list[tuple[str, Path]]:
    matrix = pd.read_csv(matrix_path)
    subset = matrix[matrix["cohort"] == cohort].copy()
    if subset.empty:
        raise ValueError(f"No {cohort!r} configs in {matrix_path}")
    subset = subset.sort_values(["experiment", "shots"])
    subset = subset.drop_duplicates("experiment", keep="first")
    return [
        (str(row.experiment), Path(str(row.config)).resolve())
        for row in subset.itertuples()
    ]


def smoke_matrix(
    matrix_path: Path, cohort: str, device: str, timeout: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    selected = _representative_configs(matrix_path, cohort)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pgvl-smoke-") as temporary:
        temporary_root = Path(temporary)
        for index, (experiment, config_path) in enumerate(selected, start=1):
            result_path = temporary_root / f"{index}.json"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--config", str(config_path), "--device", device,
                "--child", "--result-json", str(result_path),
            ]
            print(
                f"\n[{index}/{len(selected)}] {experiment}: {config_path.name}",
                flush=True)
            started = time.monotonic()
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            next_heartbeat = 20
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                if elapsed >= next_heartbeat:
                    print(f"  still running ({int(elapsed)}s)", flush=True)
                    next_heartbeat += 20
                time.sleep(1)
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                elapsed = round(time.monotonic() - started, 3)
                result = {
                    "experiment": experiment,
                    "config": str(config_path),
                    "device": device,
                    "status": "timeout" if elapsed >= timeout else "failed",
                    "stage": "subprocess",
                    "seconds": elapsed,
                    "error": (
                        f"No result produced; child exit code {process.returncode}"),
                }
            results.append(result)
            status = result["status"].upper()
            detail = (
                f"logits={result.get('logits_shape')}"
                if result["status"] == "passed"
                else f"{result.get('error_type', '')}: {result.get('error', '')}")
            print(
                f"  {status} stage={result.get('stage')} "
                f"time={result.get('seconds')}s {detail}", flush=True)
            _write_result(output_path, {
                "matrix": str(matrix_path), "cohort": cohort,
                "device": device, "results": results})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--cohort", default="camelyon16")
    parser.add_argument("--device", default=(
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config:
        result = smoke_one(args.config.expanduser().resolve(), args.device)
        if args.result_json:
            _write_result(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["status"] == "passed" else 1
    output = (args.result_json or (
        args.matrix.expanduser().resolve().parent /
        f"smoke_report_{args.cohort}.json")).expanduser().resolve()
    results = smoke_matrix(
        args.matrix.expanduser().resolve(), args.cohort, args.device,
        args.timeout, output)
    failures = [item for item in results if item["status"] != "passed"]
    print(
        f"\nSmoke summary: {len(results) - len(failures)}/{len(results)} passed; "
        f"report={output}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
