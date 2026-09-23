"""Resumable exhaustive grid search with one worker per visible GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import torch

from .datasets import STAGE_DATASETS
from .train import TrialConfig


DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "monochromatic-0": dict(hidden_dim=32, num_layers=4, batch_size=16, epochs=250),
    "monochromatic-5": dict(hidden_dim=32, num_layers=4, batch_size=16, epochs=250),
    "monochromatic-10": dict(hidden_dim=32, num_layers=4, batch_size=16, epochs=250),
    "monochromatic-15": dict(hidden_dim=32, num_layers=4, batch_size=16, epochs=250),
    "watts-strogatz-spd": dict(hidden_dim=32, num_layers=4, batch_size=16, epochs=250),
    "mnist": dict(hidden_dim=64, num_layers=3, batch_size=64, epochs=150, learning_rate=1e-3),
    "cifar10": dict(hidden_dim=64, num_layers=3, batch_size=64, epochs=150, learning_rate=1e-3),
    "pattern": dict(hidden_dim=64, num_layers=6, batch_size=32, epochs=100, learning_rate=5e-4),
    "cluster": dict(hidden_dim=48, num_layers=16, batch_size=32, epochs=100, learning_rate=5e-4),
    "peptides-func": dict(hidden_dim=96, num_layers=4, batch_size=128, epochs=150, learning_rate=3e-4),
    "peptides-struct": dict(hidden_dim=96, num_layers=4, batch_size=128, epochs=200, learning_rate=3e-4),
    "pascalvoc-sp": dict(hidden_dim=96, num_layers=4, batch_size=32, epochs=300, learning_rate=5e-4),
    "malnet-tiny": dict(hidden_dim=64, num_layers=6, batch_size=4, epochs=150, learning_rate=5e-4),
    "ogbg-molhiv": dict(hidden_dim=64, num_layers=10, batch_size=32, epochs=100, learning_rate=1e-4),
    "ogbg-molpcba": dict(hidden_dim=384, num_layers=5, batch_size=128, epochs=100, learning_rate=5e-4),
    "ogbg-code2": dict(hidden_dim=256, num_layers=4, batch_size=32, epochs=30, learning_rate=1e-4),
    "modelnet40": dict(hidden_dim=128, num_layers=4, batch_size=4, epochs=100, learning_rate=5e-5, weight_decay=1e-2),
    "shapenet": dict(hidden_dim=128, num_layers=4, batch_size=2, epochs=100, learning_rate=5e-5, weight_decay=1e-2),
}


FINAL_SEEDS: dict[str, int] = {
    **{name: 10 for name in STAGE_DATASETS["stage1"]},
    "mnist": 10,
    "cifar10": 10,
    "pattern": 10,
    "cluster": 10,
    "peptides-func": 4,
    "peptides-struct": 4,
    "pascalvoc-sp": 4,
    "malnet-tiny": 3,
    "ogbg-molhiv": 10,
    "ogbg-molpcba": 10,
    "ogbg-code2": 6,
    "modelnet40": 4,
    "shapenet": 4,
}


def dataset_grid(dataset: str) -> list[TrialConfig]:
    defaults = DATASET_DEFAULTS[dataset]
    base_lr = float(defaults.get("learning_rate", 2e-4))
    learning_rates = (base_lr / 2, base_lr)
    common = {key: value for key, value in defaults.items() if key != "learning_rate"}
    trials: list[TrialConfig] = []
    for learning_rate in learning_rates:
        trials.append(TrialConfig(dataset=dataset, method="nope", learning_rate=learning_rate, **common))
    for method in ("aw", "nb"):
        for learning_rate, num_steps, z in itertools.product(
            learning_rates, (8, 16), (0.6, 0.8)
        ):
            trials.append(
                TrialConfig(
                    dataset=dataset,
                    method=method,
                    learning_rate=learning_rate,
                    num_steps=num_steps,
                    z=z,
                    **common,
                )
            )
    for learning_rate, num_steps in itertools.product(learning_rates, (8, 16)):
        trials.append(
            TrialConfig(
                dataset=dataset,
                method="multiscale-nb",
                learning_rate=learning_rate,
                num_steps=num_steps,
                **common,
            )
        )
    return trials


def stage_grid(stage: str) -> list[TrialConfig]:
    return [trial for dataset in STAGE_DATASETS[stage] for trial in dataset_grid(dataset)]


def synthetic_grid() -> list[TrialConfig]:
    """Compatibility helper used by tests and local callers."""
    return stage_grid("stage1")


def _result_path(root: Path, trial: TrialConfig) -> Path:
    return root / trial.dataset / trial.method / trial.trial_id / "result.json"


def _write_config(root: Path, trial: TrialConfig) -> Path:
    path = root / "_configs" / f"{trial.trial_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(trial), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _gpu_ids(requested_workers: int | None) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
    elif torch.cuda.is_available():
        devices = [str(index) for index in range(torch.cuda.device_count())]
    else:
        devices = [""]
    if requested_workers is not None:
        devices = devices[:requested_workers]
    return devices or [""]


def _summarize(output_root: Path, trials: Iterable[TrialConfig]) -> dict[str, Any]:
    results = []
    for trial in trials:
        path = _result_path(output_root, trial)
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    best: dict[str, dict[str, Any]] = {}
    for result in results:
        config = result["config"]
        key = f"{config['dataset']}::{config['method']}"
        incumbent = best.get(key)
        better = incumbent is None
        if incumbent is not None:
            if result["metric_mode"] == "min":
                better = result["validation_metric"] < incumbent["validation_metric"]
            else:
                better = result["validation_metric"] > incumbent["validation_metric"]
        if better:
            best[key] = result
    summary = {
        "completed_trials": len(results),
        "expected_trials": len(list(trials)) if not isinstance(trials, list) else len(trials),
        "best_by_dataset_method": best,
    }
    path = output_root / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return summary


def run_trials(
    trials: list[TrialConfig],
    *,
    data_root: Path,
    output_root: Path,
    workers: int | None,
    tasks_per_gpu: int = 1,
    max_attempts_per_trial: int = 1,
) -> int:
    if tasks_per_gpu < 1:
        raise ValueError("tasks_per_gpu must be positive")
    if max_attempts_per_trial < 1:
        raise ValueError("max_attempts_per_trial must be positive")
    pending = [trial for trial in trials if not _result_path(output_root, trial).exists()]
    devices = _gpu_ids(workers)
    running: dict[subprocess.Popen[str], tuple[TrialConfig, object, str]] = {}
    failures: list[dict[str, Any]] = []
    attempts: dict[str, int] = {}
    print(json.dumps({
        "total": len(trials),
        "pending": len(pending),
        "gpus": len(devices),
        "tasks_per_gpu": tasks_per_gpu,
        "max_concurrent_trials": len(devices) * tasks_per_gpu,
    }), flush=True)
    while pending or running:
        occupancy = {
            device: sum(entry[2] == device for entry in running.values())
            for device in devices
        }
        available_devices = [
            device
            for device in devices
            for _ in range(tasks_per_gpu - occupancy[device])
        ]
        while pending and available_devices:
            trial = pending.pop(0)
            device = available_devices.pop(0)
            attempts[trial.trial_id] = attempts.get(trial.trial_id, 0) + 1
            config_path = _write_config(output_root, trial)
            log_path = config_path.with_suffix(".log")
            log_handle = log_path.open(
                "w" if attempts[trial.trial_id] == 1 else "a",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            for proxy_name in (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy",
            ):
                environment.pop(proxy_name, None)
            if device:
                environment["CUDA_VISIBLE_DEVICES"] = device
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m", "aw_rope.experiments.train",
                    "--config", str(config_path),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            running[process] = (trial, log_handle, device)
            print(json.dumps({
                "started": trial.trial_id,
                "dataset": trial.dataset,
                "method": trial.method,
                "device": device,
                "attempt": attempts[trial.trial_id],
            }), flush=True)

        finished = [process for process in running if process.poll() is not None]
        for process in finished:
            trial, log_handle, _device = running.pop(process)
            log_handle.close()
            if process.returncode:
                if attempts[trial.trial_id] < max_attempts_per_trial:
                    pending.append(trial)
                    print(json.dumps({
                        "retrying": trial.trial_id,
                        "returncode": process.returncode,
                        "next_attempt": attempts[trial.trial_id] + 1,
                    }), flush=True)
                else:
                    failures.append({"trial_id": trial.trial_id, "returncode": process.returncode})
                    print(json.dumps({"failed": trial.trial_id, "returncode": process.returncode}), flush=True)
            else:
                print(json.dumps({"finished": trial.trial_id}), flush=True)
        _summarize(output_root, trials)
        if running and not finished:
            time.sleep(2)

    failure_path = output_root / "failures.json"
    if failures:
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    elif failure_path.exists():
        failure_path.unlink()
    summary = _summarize(output_root, trials)
    print(json.dumps({"completed": summary["completed_trials"], "failures": len(failures)}), flush=True)
    return 1 if failures else 0


def _final_trials(search_summary: dict[str, Any], *, seed_limit: int | None = None) -> list[TrialConfig]:
    trials: list[TrialConfig] = []
    for key, result in sorted(search_summary["best_by_dataset_method"].items()):
        del key
        selected = TrialConfig(**result["config"])
        seeds = FINAL_SEEDS[selected.dataset]
        if seed_limit is not None:
            seeds = min(seeds, seed_limit)
        epochs = max(1, int(result["best_epoch"]))
        for seed in range(seeds):
            trials.append(
                replace(
                    selected,
                    seed=seed,
                    epochs=epochs,
                    patience=epochs,
                    final_fit=True,
                )
            )
    return trials


def run_stage(
    stage: str,
    *,
    data_root: Path,
    output_root: Path,
    workers: int | None,
    mode: str = "all",
    datasets: set[str] | None = None,
    limit_trials: int | None = None,
    epochs_override: int | None = None,
    seed_limit: int | None = None,
) -> int:
    trials = stage_grid(stage)
    if datasets:
        trials = [trial for trial in trials if trial.dataset in datasets]
    if epochs_override is not None:
        trials = [replace(trial, epochs=epochs_override, patience=max(1, epochs_override)) for trial in trials]
    if limit_trials is not None:
        trials = trials[:limit_trials]
    search_root = output_root / stage / "search"
    if mode in {"search", "all"}:
        code = run_trials(trials, data_root=data_root, output_root=search_root, workers=workers)
        if code:
            return code
    summary = _summarize(search_root, trials)
    if summary["completed_trials"] != len(trials):
        raise RuntimeError(f"{stage} search incomplete: {summary['completed_trials']}/{len(trials)}")
    if mode in {"finalize", "all"}:
        final_trials = _final_trials(summary, seed_limit=seed_limit)
        return run_trials(
            final_trials,
            data_root=data_root,
            output_root=output_root / stage / "final",
            workers=workers,
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_DATASETS), required=True)
    parser.add_argument("--mode", choices=("search", "finalize", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/grid"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--limit-trials", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_stage(
        args.stage,
        data_root=args.data_root,
        output_root=args.output_root,
        workers=args.workers,
        mode=args.mode,
        datasets=set(args.datasets) if args.datasets else None,
        limit_trials=args.limit_trials,
        epochs_override=args.epochs,
        seed_limit=args.seed_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
