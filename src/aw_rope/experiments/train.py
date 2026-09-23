"""Single-trial training entry point used by local and Inspire grid jobs."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader

from aw_rope import AWRoPE, resolvent_tail_bound

from .datasets import DatasetBundle, load_dataset
from .models import (
    GraphPredictionModel,
    Method,
    OfficialGraphTransformerPredictionModel,
    OfficialPointCloudTransformerPredictionModel,
)


_RESUME_PROTOCOL = "aw-rope-single-trial-resume-v1"


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    if torch.cuda.is_available():
        cuda_states = [state.cpu() for state in payload.get("torch_cuda", [])]
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "resume checkpoint CUDA RNG state does not match visible devices"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _rewrite_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _scheduler_t_max(config: "TrialConfig") -> int:
    raw = os.environ.get("AW_ROPE_SCHEDULER_T_MAX")
    value = config.epochs if raw is None else int(raw)
    if value < 1:
        raise ValueError("scheduler T_max must be positive")
    return value


@dataclass(frozen=True)
class TrialConfig:
    dataset: str
    method: Method
    seed: int = 0
    hidden_dim: int = 32
    num_layers: int = 4
    dropout: float = 0.2
    num_steps: int = 8
    z: float = 0.8
    learnable_frequencies: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    scheduler_step_unit: Literal["epoch", "optimizer-step"] = "epoch"
    epochs: int = 40
    patience: int = 15
    num_workers: int = 4
    final_fit: bool = False
    backbone: str = "gin"
    heads: int = 4
    field_type: str = "local-antisymmetric"
    evaluate_test_each_epoch: bool = False
    selection_split: str = "validation"
    gin_aw_variant: str = "legacy"
    diffusion_time: float = 2.0
    exact_heat_kernel_method: str = "precomputed"

    @property
    def trial_id(self) -> str:
        fields = asdict(self)
        # Evaluation tracing is opt-in and was added after existing runs.
        # Preserve every historical trial ID when it remains disabled.
        if not self.evaluate_test_each_epoch:
            fields.pop("evaluate_test_each_epoch")
        if self.selection_split == "validation":
            fields.pop("selection_split")
        if self.gradient_accumulation_steps == 1:
            fields.pop("gradient_accumulation_steps")
        if self.warmup_steps == 0:
            fields.pop("warmup_steps")
        if self.scheduler_step_unit == "epoch":
            fields.pop("scheduler_step_unit")
        if self.gin_aw_variant == "legacy":
            fields.pop("gin_aw_variant")
        if self.method != "exact-holonomy":
            fields.pop("diffusion_time")
            fields.pop("exact_heat_kernel_method")
        # Preserve IDs of the already-running GIN experiment round.  The
        # official-GT-only fields were added later and must not move or orphan
        # its existing result/checkpoint paths when they retain defaults.
        if self.backbone == "gin":
            fields.pop("backbone")
            fields.pop("heads")
            fields.pop("field_type")
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")


def _loader(dataset: object, config: TrialConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        # Scenic drops the training tail before sharding.  The optimizer-step
        # protocol additionally drops an incomplete accumulation window below.
        drop_last=shuffle and config.scheduler_step_unit == "optimizer-step",
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def _optimizer_schedule_shape(
    config: TrialConfig, micro_batches: int
) -> tuple[int, int]:
    """Return optimizer steps and usable micro-batches for one epoch."""
    if config.scheduler_step_unit == "optimizer-step":
        steps = micro_batches // config.gradient_accumulation_steps
        usable = steps * config.gradient_accumulation_steps
    else:
        steps = math.ceil(micro_batches / config.gradient_accumulation_steps)
        usable = micro_batches
    if steps < 1 or usable < 1:
        raise ValueError(
            "one epoch does not contain a complete optimizer accumulation window"
        )
    return steps, usable


def _target(batch: object, bundle: DatasetBundle) -> Tensor:
    if bundle.task in {"graph-classification", "node-classification"}:
        return batch.y.long().view(-1)
    if bundle.task == "sequence-prediction":
        return batch.y.long().view(-1, int(bundle.metadata.get("sequence_length", 5)))
    return batch.y.float().reshape(-1, bundle.output_dim)


def _loss(prediction: Tensor, target: Tensor, bundle: DatasetBundle) -> Tensor:
    if bundle.task in {"graph-classification", "node-classification"}:
        return torch.nn.functional.cross_entropy(prediction.float(), target)
    if bundle.task == "sequence-prediction":
        return torch.nn.functional.cross_entropy(
            prediction.float().flatten(0, 1), target.flatten()
        )
    valid = torch.isfinite(target)
    if bundle.task == "graph-multilabel":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float()[valid], target[valid]
        )
    return torch.nn.functional.mse_loss(prediction.float()[valid], target[valid])


def _autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class _MetricAccumulator:
    def __init__(self, bundle: DatasetBundle) -> None:
        self.bundle = bundle
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.count = 0
        self.correct = 0
        self.confusion = torch.zeros(bundle.output_dim, bundle.output_dim, dtype=torch.long)
        self.predictions: list[Tensor] = []
        self.targets: list[Tensor] = []
        self.sequence_f1 = 0.0
        self.sequences = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = prediction.detach().float().cpu()
        target = target.detach().cpu()
        metric = self.bundle.metric
        if metric in {"normalized-rmse", "mae"}:
            valid = torch.isfinite(target)
            error = prediction[valid] - target[valid]
            self.squared_error += float(error.square().sum())
            self.absolute_error += float(error.abs().sum())
            self.count += error.numel()
        elif metric == "accuracy":
            label = prediction.argmax(dim=-1)
            self.correct += int((label == target).sum())
            self.count += target.numel()
        elif metric == "macro-f1":
            label = prediction.argmax(dim=-1).view(-1)
            truth = target.view(-1)
            flat = truth * self.bundle.output_dim + label
            self.confusion += torch.bincount(
                flat, minlength=self.bundle.output_dim**2
            ).reshape(self.bundle.output_dim, self.bundle.output_dim)
        elif metric in {"average-precision", "rocauc"}:
            self.predictions.append(prediction)
            self.targets.append(target.float())
        elif metric == "sequence-f1":
            predicted_ids = prediction.argmax(dim=-1)
            eos = self.bundle.output_dim - 1
            for predicted, truth in zip(predicted_ids, target):
                predicted_list = predicted.tolist()
                truth_list = truth.tolist()
                predicted_list = predicted_list[: predicted_list.index(eos)] if eos in predicted_list else predicted_list
                truth_list = truth_list[: truth_list.index(eos)] if eos in truth_list else truth_list
                predicted_counter, truth_counter = Counter(predicted_list), Counter(truth_list)
                common = sum((predicted_counter & truth_counter).values())
                precision = common / len(predicted_list) if predicted_list else 0.0
                recall = common / len(truth_list) if truth_list else 0.0
                self.sequence_f1 += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                self.sequences += 1
        else:
            raise KeyError(metric)

    def compute(self) -> dict[str, float]:
        metric = self.bundle.metric
        if metric in {"normalized-rmse", "mae"}:
            rmse = math.sqrt(self.squared_error / max(self.count, 1))
            return {
                "rmse": rmse,
                "normalized-rmse": rmse / self.bundle.target_scale,
                "mae": self.absolute_error / max(self.count, 1),
            }
        if metric == "accuracy":
            return {"accuracy": self.correct / max(self.count, 1)}
        if metric == "macro-f1":
            true_positive = self.confusion.diag().float()
            predicted = self.confusion.sum(0).float()
            actual = self.confusion.sum(1).float()
            denominator = predicted + actual
            present = denominator > 0
            f1 = torch.where(present, 2 * true_positive / denominator.clamp_min(1), 0)
            return {"macro-f1": float(f1[present].mean()) if present.any() else 0.0}
        if metric in {"average-precision", "rocauc"}:
            from sklearn.metrics import average_precision_score, roc_auc_score

            predictions = torch.cat(self.predictions).numpy()
            targets = torch.cat(self.targets).numpy()
            values = []
            for column in range(targets.shape[1]):
                valid = np.isfinite(targets[:, column])
                labels = targets[valid, column]
                if valid.sum() == 0 or np.unique(labels).size < 2:
                    continue
                scores = predictions[valid, column]
                function = average_precision_score if metric == "average-precision" else roc_auc_score
                values.append(float(function(labels, scores)))
            return {metric: float(np.mean(values)) if values else float("nan")}
        if metric == "sequence-f1":
            return {metric: self.sequence_f1 / max(self.sequences, 1)}
        raise KeyError(metric)


def _evaluate(
    model: GraphPredictionModel,
    loader: DataLoader,
    bundle: DatasetBundle,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulator = _MetricAccumulator(bundle)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with _autocast(device):
                prediction = model(batch)
            accumulator.update(prediction, _target(batch, bundle))
    return accumulator.compute()


def _mean_within_graph_cosine(features: Tensor, node_batch: Tensor) -> float:
    """Mean cosine over distinct node pairs, without a quadratic matrix."""
    normalized = torch.nn.functional.normalize(features.detach().float(), dim=-1)
    graph_count = int(node_batch.max()) + 1 if node_batch.numel() else 1
    sums = torch.zeros(
        graph_count, normalized.shape[-1], dtype=normalized.dtype, device=normalized.device
    )
    sums.index_add_(0, node_batch, normalized)
    counts = torch.bincount(node_batch, minlength=graph_count).to(normalized.dtype)
    pair_sums = sums.square().sum(-1) - counts
    pair_counts = counts * (counts - 1)
    valid = pair_counts > 0
    if not valid.any():
        return float("nan")
    return float((pair_sums[valid].sum() / pair_counts[valid].sum()).cpu())


def _gin_model_diagnostics(
    model: GraphPredictionModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Capture collapse and AW-scale diagnostics on one deterministic batch."""
    batch = next(iter(loader)).to(device, non_blocking=True)
    node_batch = batch.batch
    layer_cosines: list[float] = []
    hooks = [
        normalization.register_forward_hook(
            lambda _module, _inputs, output: layer_cosines.append(
                _mean_within_graph_cosine(output, node_batch)
            )
        )
        for normalization in model.normalizations
    ]
    model.eval()
    try:
        with torch.no_grad(), _autocast(device):
            logits = model(batch).detach().float()
    finally:
        for hook in hooks:
            hook.remove()
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    branches: list[dict[str, Any]] = []
    for branch in model.analytic_branches:
        rope = getattr(branch, "rope", None)
        z = float(rope.z.detach().cpu()) if isinstance(rope, AWRoPE) else None
        branches.append(
            {
                "gate": float(
                    (branch.gate if branch.zero_init_gate else torch.sigmoid(branch.gate))
                    .detach()
                    .cpu()
                ),
                "z": z,
                "displacement_rms": float(
                    getattr(branch, "last_displacement_rms", torch.tensor(float("nan")))
                    .detach()
                    .cpu()
                ),
                "phase_temperature": (
                    float(branch.log_phase_temperature.detach().exp().cpu())
                    if branch.log_phase_temperature is not None
                    else None
                ),
            }
        )
    return {
        "layer_mean_within_graph_cosine": layer_cosines,
        "logit_std": float(logits.std(unbiased=False).cpu()),
        "mean_predictive_entropy": float(entropy.mean().cpu()),
        # Graph classification produces ``[batch, classes]`` logits, while
        # sequence tasks such as Code2 produce ``[batch, steps, classes]``.
        # ``torch.bincount`` accepts only a one-dimensional tensor, so count
        # predictions over every non-class axis after flattening them.
        "predicted_class_counts": torch.bincount(
            logits.argmax(dim=-1).reshape(-1), minlength=model.output_dim
        ).cpu().tolist(),
        "analytic_branches": branches,
    }


def run_trial(
    config: TrialConfig,
    data_root: Path,
    output_root: Path,
    *,
    attention_type: str = "Full",
) -> dict[str, Any]:
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.scheduler_step_unit not in {"epoch", "optimizer-step"}:
        raise ValueError(f"unknown scheduler_step_unit: {config.scheduler_step_unit}")
    if config.selection_split not in {"validation", "test"}:
        raise ValueError(f"unknown selection split: {config.selection_split}")
    if config.selection_split == "test" and not config.evaluate_test_each_epoch:
        raise ValueError("test selection requires evaluate_test_each_epoch=True")
    if attention_type not in {"Full", "Linear"}:
        raise ValueError(f"unknown attention type: {attention_type}")
    if config.backbone != "official-gt" and attention_type != "Full":
        raise ValueError("Linear attention is only available for the official GT backbone")
    if config.evaluate_test_each_epoch and config.final_fit:
        raise ValueError(
            "per-epoch validation/test requires final_fit=False so validation stays independent"
        )
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_started = time.monotonic()
    exact_num_frequencies: int | None = None
    if config.method == "exact-holonomy":
        if config.exact_heat_kernel_method != "precomputed":
            raise ValueError("canonical exact Holonomy training requires precomputed sidecars")
        rotary_width = (
            config.hidden_dim
            if config.backbone == "gin"
            else config.hidden_dim // config.heads
        )
        exact_num_frequencies = (rotary_width - rotary_width % 2) // 2
    bundle = load_dataset(
        config.dataset,
        data_root,
        final_fit=config.final_fit,
        pointcloud_wire_protocol=(
            config.dataset in {"modelnet40", "shapenet"}
            and config.scheduler_step_unit == "optimizer-step"
        ),
        exact_holonomy_num_frequencies=exact_num_frequencies,
        exact_holonomy_diffusion_time=config.diffusion_time,
    )
    data_preparation_seconds = time.monotonic() - load_started
    if config.backbone == "official-gt":
        official_method = {
            "nope": "none",
            "aw": "aw",
            "nb": "aw-nb",
            "multiscale-nb": "aw-nb-ms",
            "none": "none",
            "wire": "wire",
            "aw-nb": "aw-nb",
            "aw-nb-ms": "aw-nb-ms",
            "lr-aw": "lr-aw",
            "exact-holonomy": "exact-holonomy",
        }.get(config.method)
        if official_method is None:
            raise ValueError(f"unsupported official GT method: {config.method}")
        model_type = (
            OfficialPointCloudTransformerPredictionModel
            if (
                config.dataset in {"modelnet40", "shapenet"}
                and config.scheduler_step_unit == "optimizer-step"
            )
            else OfficialGraphTransformerPredictionModel
        )
        model = model_type(
            bundle.input_dim,
            bundle.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            method=official_method,
            num_steps=config.num_steps,
            z=config.z,
            field_type=config.field_type,
            position_dim=bundle.position_dim,
            learnable_frequencies=config.learnable_frequencies,
            task=bundle.task,
            attention_type=attention_type,
            diffusion_time=config.diffusion_time,
            exact_heat_kernel_method=config.exact_heat_kernel_method,
        ).to(device)
    elif config.backbone == "gin":
        model = GraphPredictionModel(
            bundle.input_dim,
            bundle.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            method=config.method,
            num_steps=config.num_steps,
            z=config.z,
            learnable_frequencies=config.learnable_frequencies,
            task=bundle.task,
            encoder=bundle.encoder,
            position_dim=bundle.position_dim,
            sequence_length=int(bundle.metadata.get("sequence_length", 5)),
            gin_aw_variant=config.gin_aw_variant,
            diffusion_time=config.diffusion_time,
            exact_heat_kernel_method=config.exact_heat_kernel_method,
        ).to(device)
    else:
        raise ValueError(f"unknown backbone: {config.backbone}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = _loader(bundle.train, config, shuffle=True)
    validation_loader = _loader(bundle.validation, config, shuffle=False) if bundle.validation is not None else None
    test_loader = _loader(bundle.test, config, shuffle=False)
    optimizer_steps_per_epoch, usable_train_batches = _optimizer_schedule_shape(
        config, len(train_loader)
    )
    dropped_micro_batches_per_epoch = len(train_loader) - usable_train_batches
    if config.scheduler_step_unit == "optimizer-step":
        scheduler_t_max = config.epochs * optimizer_steps_per_epoch
        warmup_steps = min(config.warmup_steps, max(0, scheduler_t_max - 1))

        def lr_factor(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1, step + 1) / warmup_steps
            decay_steps = max(1, scheduler_t_max - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        )
    else:
        scheduler_t_max = _scheduler_t_max(config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_t_max
        )

    trial_dir = output_root / config.dataset / config.method / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    result_path = trial_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    epoch_metrics_path = trial_dir / "epoch_metrics.jsonl"
    last_checkpoint_path = trial_dir / "last.pt"

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.monotonic()
    elapsed_before_resume = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    minimize = bundle.metric_mode == "min"
    best_metric = float("inf") if minimize else -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int | dict[str, float]]] = []
    best_state: dict[str, Tensor] | None = None
    best_validation_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    progress_history: list[dict[str, Any]] = []
    start_epoch = 1

    if last_checkpoint_path.is_file():
        resume = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if resume.get("protocol") != _RESUME_PROTOCOL:
            raise RuntimeError(f"invalid resume protocol: {last_checkpoint_path}")
        if resume.get("trial_id") != config.trial_id:
            raise RuntimeError(f"resume trial identity mismatch: {last_checkpoint_path}")
        if resume.get("config") != asdict(config):
            raise RuntimeError(f"resume config mismatch: {last_checkpoint_path}")
        if resume.get("attention_type") != attention_type:
            raise RuntimeError(f"resume attention type mismatch: {last_checkpoint_path}")
        completed_epoch = int(resume.get("epoch", 0))
        history = list(resume.get("history", []))
        progress_history = list(resume.get("progress_history", []))
        if (
            completed_epoch < 1
            or len(history) != completed_epoch
            or len(progress_history) != completed_epoch
            or int(history[-1].get("epoch", -1)) != completed_epoch
            or int(progress_history[-1].get("epoch", -1)) != completed_epoch
        ):
            raise RuntimeError(f"incomplete resume history: {last_checkpoint_path}")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(resume["scheduler"])
        best_metric = float(resume["best_metric"])
        best_epoch = int(resume["best_epoch"])
        stale_epochs = int(resume["stale_epochs"])
        best_state = resume["best_state"]
        best_validation_metrics = resume.get("best_validation_metrics")
        best_test_metrics = resume.get("best_test_metrics")
        elapsed_before_resume = float(resume.get("elapsed_seconds", 0.0))
        _restore_rng_state(resume["rng_state"])
        _rewrite_epoch_metrics(epoch_metrics_path, progress_history)
        start_epoch = completed_epoch + 1
        print(
            json.dumps(
                {
                    "event": "resume",
                    "trial_id": config.trial_id,
                    "completed_epoch": completed_epoch,
                    "next_epoch": start_epoch,
                    "checkpoint": str(last_checkpoint_path),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    else:
        # A pre-resume run without result.json/last.pt cannot reconstruct the
        # optimizer or RNG state.  Start it cleanly and never mix attempts in
        # one live metric stream.
        epoch_metrics_path.unlink(missing_ok=True)

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        loss_sum = 0.0
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        total_batches = usable_train_batches
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= total_batches:
                break
            batch = batch.to(device, non_blocking=True)
            with _autocast(device):
                prediction = model(batch)
                loss = _loss(prediction, _target(batch, bundle), bundle)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in trial {config.trial_id}")
            window_start = (
                batch_index // config.gradient_accumulation_steps
            ) * config.gradient_accumulation_steps
            window_size = min(
                config.gradient_accumulation_steps,
                total_batches - window_start,
            )
            (loss / window_size).backward()
            end_of_window = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == total_batches
            )
            if end_of_window:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if config.scheduler_step_unit == "optimizer-step":
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            batches += 1
        if config.scheduler_step_unit == "epoch":
            scheduler.step()

        train_loss = loss_sum / max(batches, 1)
        if validation_loader is not None:
            validation = _evaluate(model, validation_loader, bundle, device)
        else:
            validation = {"train-loss": train_loss}
        epoch_test = (
            _evaluate(model, test_loader, bundle, device)
            if config.evaluate_test_each_epoch
            else None
        )
        if config.selection_split == "test":
            assert epoch_test is not None
            selection_metric = epoch_test[bundle.metric]
        elif validation_loader is not None:
            selection_metric = validation[bundle.metric]
        else:
            selection_metric = train_loss
        epoch_seconds = time.monotonic() - epoch_started
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
            "selection_split": config.selection_split,
            "selection_metric": selection_metric,
            "epoch_seconds": epoch_seconds,
        }
        if epoch_test is not None:
            epoch_record["test"] = epoch_test
        history.append(epoch_record)
        # Final fits keep the last epoch. Other protocols select on the split
        # explicitly declared in the config (validation by default, test only
        # when the caller deliberately requests the biased test-selection
        # protocol).
        improved = config.final_fit or (
            math.isfinite(selection_metric)
            and ((selection_metric < best_metric) if minimize else (selection_metric > best_metric))
        )
        if improved:
            best_metric = selection_metric
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_validation_metrics = dict(validation)
            best_test_metrics = dict(epoch_test) if epoch_test is not None else None
            if config.evaluate_test_each_epoch:
                checkpoint_path = trial_dir / "best.pt"
                temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "model": best_state,
                        "config": asdict(config),
                        "attention_type": attention_type,
                        "best_epoch": best_epoch,
                        "selection_split": config.selection_split,
                        "selection_metric": best_metric,
                        "validation": best_validation_metrics,
                        "test": best_test_metrics,
                        "metric_name": bundle.metric,
                        "metric_mode": bundle.metric_mode,
                        "test_selection_bias": config.selection_split == "test",
                    },
                    temporary_checkpoint,
                )
                temporary_checkpoint.replace(checkpoint_path)
        else:
            stale_epochs += 1
        progress: dict[str, Any] = {
            "trial_id": config.trial_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
            "selection_split": config.selection_split,
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "best_selection_metric": best_metric,
            "epoch_seconds": epoch_seconds,
        }
        if epoch_test is not None:
            progress["test"] = epoch_test
        progress_history.append(progress)
        with epoch_metrics_path.open("a", encoding="utf-8") as metrics_log:
            metrics_log.write(json.dumps(progress, separators=(",", ":")) + "\n")
            metrics_log.flush()
            os.fsync(metrics_log.fileno())
        print(
            json.dumps(progress, separators=(",", ":")),
            flush=True,
        )
        _atomic_torch_save(
            {
                "protocol": _RESUME_PROTOCOL,
                "trial_id": config.trial_id,
                "config": asdict(config),
                "attention_type": attention_type,
                "epoch": epoch,
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
                "progress_history": progress_history,
                "best_state": best_state,
                "best_validation_metrics": best_validation_metrics,
                "best_test_metrics": best_test_metrics,
                "rng_state": _capture_rng_state(),
                "elapsed_seconds": elapsed_before_resume
                + time.monotonic()
                - started,
            },
            last_checkpoint_path,
        )
        if validation_loader is not None and stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model_diagnostics = (
        _gin_model_diagnostics(
            model,
            validation_loader if validation_loader is not None else test_loader,
            device,
        )
        if isinstance(model, GraphPredictionModel)
        else None
    )
    checkpoint_path = trial_dir / "best.pt"
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": best_state,
            "config": asdict(config),
            "attention_type": attention_type,
            "best_epoch": best_epoch,
            "selection_split": config.selection_split,
            "selection_metric": best_metric,
            "validation": best_validation_metrics,
            "test": best_test_metrics,
            "metric_name": bundle.metric,
            "metric_mode": bundle.metric_mode,
            "test_selection_bias": config.selection_split == "test",
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint_path)
    if config.evaluate_test_each_epoch:
        if best_test_metrics is None:
            raise RuntimeError("per-epoch testing produced no best-epoch test metrics")
        test_metrics = best_test_metrics
    else:
        test_metrics = _evaluate(model, test_loader, bundle, device) if config.final_fit else None
    validation_metric = (
        best_validation_metrics[bundle.metric]
        if best_validation_metrics is not None and bundle.metric in best_validation_metrics
        else best_metric
    )
    result: dict[str, Any] = {
        "trial_id": config.trial_id,
        "config": asdict(config),
        "attention_type": attention_type,
        "method_label": (
            "Exact/Full Holonomy-RoPE"
            if config.method == "exact-holonomy"
            else "LR-CRF-AW-RoPE Performer"
            if config.method == "lr-aw" and attention_type == "Linear"
            else "LR-CRF-AW-RoPE full"
            if config.method == "lr-aw"
            else "AW-RoPE Performer"
            if config.method == "aw" and attention_type == "Linear"
            else "AW-RoPE full"
            if config.method == "aw"
            else str(config.method)
        ),
        "gin_aw_variant": config.gin_aw_variant,
        "task": bundle.task,
        "metric_name": bundle.metric,
        "metric_mode": bundle.metric_mode,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": parameter_count,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "scheduler_t_max": scheduler_t_max,
        "scheduler_step_unit": config.scheduler_step_unit,
        "warmup_steps": config.warmup_steps,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "dropped_micro_batches_per_epoch": dropped_micro_batches_per_epoch,
        "data_preparation_seconds": data_preparation_seconds,
        "best_epoch": best_epoch,
        "selection_split": config.selection_split,
        "selection_metric": best_metric,
        "test_selection_bias": config.selection_split == "test",
        "validation_metric": validation_metric,
        "validation": best_validation_metrics,
        "test": test_metrics,
        "epoch_metrics_file": str(epoch_metrics_path),
        "test_selection": (
            "best-test-epoch-biased"
            if config.selection_split == "test"
            else "paired-with-best-validation-epoch"
            if config.evaluate_test_each_epoch
            else "final-fit-last-epoch" if config.final_fit else None
        ),
        "elapsed_seconds": elapsed_before_resume + time.monotonic() - started,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "model_diagnostics": model_diagnostics,
        "history": history,
    }
    if config.method in {"aw", "nb", "lr-aw"}:
        result["truncation_tail_bound"] = float(
            resolvent_tail_bound(config.z, config.num_steps)
        )
        result["approximation_label"] = (
            "resolvent-approximation"
            if result["truncation_tail_bound"] <= 1e-3
            else "truncated-walk-polynomial"
        )
    elif config.method == "multiscale-nb":
        result["truncation_tail_bounds"] = {
            str(scale): float(resolvent_tail_bound(scale, config.num_steps))
            for scale in (0.2, 0.4, 0.6, 0.8)
        }
        result["approximation_label"] = (
            "resolvent-approximation"
            if max(result["truncation_tail_bounds"].values()) <= 1e-3
            else "truncated-walk-polynomial"
        )
    if bundle.metric == "normalized-rmse":
        result["validation_normalized_rmse"] = validation_metric
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/grid"))
    parser.add_argument(
        "--attention-type",
        choices=("Full", "Linear"),
        default="Full",
        help="Official GraphRoPE global attention implementation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = TrialConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    result = run_trial(
        config,
        args.data_root,
        args.output_root,
        attention_type=args.attention_type,
    )
    print(json.dumps({"completed": result["trial_id"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
