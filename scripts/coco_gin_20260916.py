#!/usr/bin/env python3
"""Bounded COCO-SP GIN NoPE/AW cohort, preflight and all-seed reporting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from torch_geometric.loader import DataLoader
from aw_rope.experiments.datasets import load_dataset
from aw_rope.experiments.models import GraphPredictionModel
from aw_rope.experiments.train import TrialConfig, _autocast, _loss, _set_seed, _target


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def trials(manifest):
    assert manifest["protocol"] == "coco-gin-nope-aw-two-seed-20260916-v1"
    assert manifest["dataset"] == "coco-sp"
    assert manifest["methods"] == ["nope", "aw"] and manifest["seeds"] == [0, 1]
    result = [TrialConfig(dataset="coco-sp", method=method, seed=seed, **manifest["config"])
              for seed in manifest["seeds"] for method in manifest["methods"]]
    assert all(t.epochs == 250 and t.patience > t.epochs and t.selection_split == "validation"
               and t.evaluate_test_each_epoch and not t.final_fit and t.backbone == "gin" for t in result)
    return result


def preflight(manifest, data_root, output, device):
    configs = trials(manifest)
    bundle = load_dataset("coco-sp", data_root)
    split_info = {}
    for name, dataset, expected in (("train", bundle.train, 113286),
                                     ("val", bundle.validation, 5000), ("test", bundle.test, 5000)):
        assert len(dataset) == expected, (name, len(dataset), expected)
        labels = dataset._data.y
        assert labels.min().item() >= 0 and labels.max().item() < 81
        sample = dataset[0]
        assert sample.x.shape[1] == 14 and torch.isfinite(sample.x).all()
        path = data_root / "lrgb" / "coco-sp" / "processed" / f"{name}.pt"
        split_info[name] = dict(graphs=len(dataset), nodes=labels.numel(),
                                labels_min=int(labels.min()), labels_max=int(labels.max()),
                                path=str(path.resolve()), size=path.stat().st_size,
                                mtime_ns=path.stat().st_mtime_ns)
    batch = next(iter(DataLoader(bundle.train, batch_size=16, shuffle=False, num_workers=0))).to(device)
    probe_rows = []
    initial_states = {}
    for method in manifest["methods"]:
        config = next(t for t in configs if t.method == method and t.seed == 0)
        _set_seed(0)
        model = GraphPredictionModel(bundle.input_dim, bundle.output_dim,
            hidden_dim=config.hidden_dim, num_layers=config.num_layers, dropout=config.dropout,
            method=method, num_steps=config.num_steps, z=config.z,
            learnable_frequencies=config.learnable_frequencies, task=bundle.task,
            encoder=bundle.encoder, gin_aw_variant=config.gin_aw_variant).to(device)
        initial_states[method] = {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                 for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with _autocast(device):
            output_tensor = model(batch)
            loss = _loss(output_tensor, _target(batch, bundle), bundle)
        assert output_tensor.shape == (batch.num_nodes, 81) and torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        probe_rows.append(dict(method=method, loss=float(loss.detach()), seconds=time.monotonic()-started,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0))
        del model, optimizer, output_tensor, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    shared_keys = initial_states["nope"].keys() & initial_states["aw"].keys()
    mismatches = sorted(k for k in shared_keys if initial_states["nope"][k] != initial_states["aw"][k])
    audit = dict(protocol=manifest["protocol"], device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        torch=torch.__version__, python=sys.executable, splits=split_info, probes=probe_rows,
        initialization=dict(shared_tensor_count=len(shared_keys), unequal_shared_tensors=mismatches,
            note="Historical legacy GIN initialization is retained; AW branch draws can change head initialization. Same seeds do not imply all tensors are identical."))
    write_json(output, audit)
    print(json.dumps(audit), flush=True)


def summarize(manifest, run_root, require_complete=False):
    rows = []
    progress = []
    for trial in trials(manifest):
        directory = run_root / trial.dataset / trial.method / trial.trial_id
        path = directory / "result.json"
        metrics = directory / "epoch_metrics.jsonl"
        history = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
        progress.append(dict(method=trial.method, seed=trial.seed, trial_id=trial.trial_id,
                             epochs=len(history), complete=path.exists()))
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        assert result["config"] == asdict(trial) and result["trial_id"] == trial.trial_id
        assert len(result["history"]) == trial.epochs and len(history) == trial.epochs
        assert [row["epoch"] for row in history] == list(range(1, trial.epochs + 1))
        assert result["selection_split"] == "validation" and not result["test_selection_bias"]
        assert result["test_selection"] == "paired-with-best-validation-epoch"
        best = max(result["history"], key=lambda row: row["validation"]["macro-f1"])
        assert result["best_epoch"] == best["epoch"] and result["test"] == best["test"]
        assert math.isfinite(result["test"]["macro-f1"])
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        assert checkpoint["config"] == asdict(trial) and checkpoint["best_epoch"] == best["epoch"]
        rows.append(dict(method=trial.method, seed=trial.seed, trial_id=trial.trial_id,
                         best_epoch=best["epoch"], validation=best["validation"]["macro-f1"],
                         test=best["test"]["macro-f1"], result=str(path)))
    summary = dict(protocol=manifest["protocol"], expected=4, completed=len(rows),
                   progress=progress, rows=rows, methods={})
    for method in manifest["methods"]:
        values = [row["test"] for row in rows if row["method"] == method]
        summary["methods"][method] = dict(n=len(values), mean=statistics.mean(values) if values else None,
            sample_std=statistics.stdev(values) if len(values) > 1 else None)
    write_json(run_root / "cohort-summary.json", summary)
    print(json.dumps(summary), flush=True)
    if require_complete:
        assert len(rows) == 4, "cohort incomplete"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "summarize"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/coco_gin_20260916.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    if args.action == "preflight":
        preflight(manifest, args.data_root, args.run_root / f"preflight-{args.device.replace(':', '-')}.json", torch.device(args.device))
    elif args.action == "summarize":
        summarize(manifest, args.run_root)
    else:
        import fcntl
        from aw_rope.experiments.grid import run_trials
        assert torch.cuda.is_available() and torch.cuda.device_count() == manifest["resources"]["gpus"]
        args.run_root.mkdir(parents=True, exist_ok=True)
        with (args.run_root / "cohort.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            configs = trials(manifest)
            write_json(args.run_root / "plan.json", dict(manifest=manifest,
                trials=[dict(trial_id=t.trial_id, config=asdict(t)) for t in configs],
                python=sys.executable, source=str(ROOT), pid=os.getpid()))
            # Refuse malformed prior results before the generic dispatcher can skip them.
            summarize(manifest, args.run_root)
            code = run_trials(configs, data_root=args.data_root, output_root=args.run_root,
                workers=2, tasks_per_gpu=2, max_attempts_per_trial=1)
            summarize(manifest, args.run_root, require_complete=True)
            if code:
                raise RuntimeError("one or more cohort trials failed")
            write_json(args.run_root / "completed.json", dict(protocol=manifest["protocol"], time=time.time()))


if __name__ == "__main__":
    main()
