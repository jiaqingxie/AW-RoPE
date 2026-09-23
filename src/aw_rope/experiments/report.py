"""Aggregate final AW-RoPE trials and verify the complete run artifact set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev
import subprocess
import sys
from typing import Any

from .datasets import STAGE_DATASETS
from .grid import FINAL_SEEDS


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _bootstrap_ci(values: list[float], *, seed: int = 91_777, draws: int = 10_000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    if len(values) == 1:
        return [values[0], values[0]]
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choices(values, k=len(values))) for _ in range(draws)
    )
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]]


def _final_results(run_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for stage in STAGE_DATASETS:
        for path in (run_root / stage / "final").glob("**/result.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_path"] = str(path.relative_to(run_root))
            results.append(payload)
    return results


def aggregate(run_root: Path) -> dict[str, Any]:
    results = _final_results(run_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        config = result["config"]
        grouped.setdefault((config["dataset"], config["method"]), []).append(result)

    rows: list[dict[str, Any]] = []
    by_seed: dict[tuple[str, str, int], float] = {}
    for (dataset, method), group in sorted(grouped.items()):
        metric = group[0]["metric_name"]
        values = [float(result["test"][metric]) for result in group]
        for result, value in zip(group, values):
            by_seed[(dataset, method, int(result["config"]["seed"]))] = value
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "metric_mode": group[0]["metric_mode"],
                "seeds": len(values),
                "mean": mean(values),
                "standard_error": stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
                "bootstrap_95_low": _bootstrap_ci(values)[0],
                "bootstrap_95_high": _bootstrap_ci(values)[1],
                "mean_parameters": mean(float(result["parameter_count"]) for result in group),
                "mean_epoch_seconds": mean(float(result["mean_epoch_seconds"]) for result in group),
                "peak_gpu_memory_bytes": max(int(result["peak_gpu_memory_bytes"]) for result in group),
            }
        )

    paired: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] == "nope":
            continue
        dataset, method = row["dataset"], row["method"]
        seeds = sorted(
            seed for (candidate, candidate_method, seed) in by_seed
            if candidate == dataset and candidate_method == method and (dataset, "nope", seed) in by_seed
        )
        improvements = []
        for seed in seeds:
            baseline = by_seed[(dataset, "nope", seed)]
            candidate = by_seed[(dataset, method, seed)]
            improvements.append(baseline - candidate if row["metric_mode"] == "min" else candidate - baseline)
        paired.append(
            {
                "dataset": dataset,
                "method": method,
                "paired_seeds": len(improvements),
                "mean_improvement_over_nope": mean(improvements) if improvements else float("nan"),
                "bootstrap_95": _bootstrap_ci(improvements),
            }
        )

    search_selection = {}
    for stage in STAGE_DATASETS:
        path = run_root / stage / "search" / "summary.json"
        if path.exists():
            search_selection[stage] = json.loads(path.read_text(encoding="utf-8"))

    payload = {
        "protocol": "validation-selected grid; test evaluated once per final-fit seed",
        "final_trials": len(results),
        "results": rows,
        "paired_against_nope": paired,
        "search": search_selection,
    }
    aggregate_dir = run_root / "aggregate"
    _atomic_json(aggregate_dir / "summary.json", payload)
    with (aggregate_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# AW-RoPE experiment results",
        "",
        "Hyperparameters were selected using validation only. Test metrics below are from final fits.",
        "",
        "| Dataset | Method | Metric | Seeds | Mean | SE | Bootstrap 95% CI |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['metric']} | {row['seeds']} | "
            f"{row['mean']:.6g} | {row['standard_error']:.3g} | "
            f"[{row['bootstrap_95_low']:.6g}, {row['bootstrap_95_high']:.6g}] |"
        )
    (aggregate_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def verify(run_root: Path, *, require_gpu: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    runtime_path = run_root / "runtime.json"
    if not runtime_path.exists():
        errors.append("missing runtime.json")
        project_root = None
    else:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        project_root = Path(runtime["project_root"])
        if runtime.get("proxy_mode") != "direct-no-proxy":
            errors.append("runtime did not record direct-no-proxy mode")
    if project_root is not None:
        dataset_manifest_path = project_root / "dataset_manifest.json"
        if not dataset_manifest_path.exists():
            errors.append("missing dataset_manifest.json")
        else:
            dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            if dataset_manifest.get("network_mode") != "direct-no-proxy":
                errors.append("dataset manifest network mode is not direct-no-proxy")
            not_ready = [
                name for name, status in dataset_manifest.get("local_status", {}).items()
                if not status.get("ready")
            ]
            if not_ready:
                errors.append(f"datasets not ready: {not_ready}")
    theory_path = run_root / "stage1" / "theory.json"
    if not theory_path.exists():
        errors.append("missing stage1/theory.json")
    else:
        theory = json.loads(theory_path.read_text(encoding="utf-8"))
        if theory.get("path_resolvent_phase_max_error_float64", float("inf")) >= 1e-8:
            errors.append("path graph exact RoPE recovery threshold failed")
        if require_gpu and not theory.get("gpu_name"):
            errors.append("stage1 theory benchmark did not run on GPU")
    expected_search = {stage: 22 * len(datasets) for stage, datasets in STAGE_DATASETS.items()}
    expected_final = {
        stage: sum(4 * FINAL_SEEDS[dataset] for dataset in datasets)
        for stage, datasets in STAGE_DATASETS.items()
    }
    for stage in STAGE_DATASETS:
        marker = run_root / ".state" / f"{stage}.done.json"
        if not marker.exists():
            errors.append(f"missing completion marker: {marker}")
        summary_path = run_root / stage / "search" / "summary.json"
        if not summary_path.exists():
            errors.append(f"missing search summary: {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["completed_trials"] != expected_search[stage]:
            errors.append(f"{stage} search count {summary['completed_trials']} != {expected_search[stage]}")
        final_paths = list((run_root / stage / "final").glob("**/result.json"))
        if len(final_paths) != expected_final[stage]:
            errors.append(f"{stage} final count {len(final_paths)} != {expected_final[stage]}")
        for result_path in final_paths:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("test") is None:
                errors.append(f"missing test metric: {result_path}")
            if require_gpu and not result.get("gpu_name"):
                errors.append(f"trial did not run on GPU: {result_path}")
            checkpoint = result_path.with_name("best.pt")
            if not checkpoint.exists() or checkpoint.stat().st_size == 0:
                errors.append(f"missing checkpoint: {checkpoint}")
    failure_files = [path for path in run_root.glob("**/failures.json") if path.stat().st_size > 3]
    if failure_files:
        errors.extend(f"trial failure file present: {path}" for path in failure_files)
    aggregate_path = run_root / "aggregate" / "summary.json"
    if not aggregate_path.exists():
        errors.append("missing aggregate/summary.json")

    manifest_entries = []
    for path in sorted(run_root.glob("**/*")):
        if not path.is_file() or path.suffix == ".pt":
            continue
        relative = str(path.relative_to(run_root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    payload = {
        "valid": not errors,
        "errors": errors,
        "expected_search_trials": expected_search,
        "expected_final_trials": expected_final,
        "manifest": manifest_entries,
    }
    _atomic_json(run_root / "verification" / "artifact_manifest.json", payload)
    if errors:
        raise RuntimeError("artifact verification failed:\n" + "\n".join(errors[:30]))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("aggregate", "verify", "full-verify"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "aggregate":
        payload = aggregate(args.run_root)
    else:
        if args.command == "full-verify":
            subprocess.run([sys.executable, "-m", "pytest"], check=True)
        payload = verify(args.run_root, require_gpu=not args.allow_cpu)
    print(json.dumps({"command": args.command, "ok": True, "items": len(payload)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
