"""Correctness and sparse-scaling measurements for AW-RoPE stage 1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from aw_rope import (
    edge_displacement_from_positions,
    standard_frequencies,
    truncated_walk_resolvent,
)


def _path_edges(nodes: int, device: torch.device) -> torch.Tensor:
    forward = torch.arange(nodes - 1, device=device)
    return torch.stack(
        (torch.cat((forward, forward + 1)), torch.cat((forward + 1, forward)))
    )


def _measure(nodes: int, *, dim: int, steps: int, device: torch.device) -> dict[str, Any]:
    edge_index = _path_edges(nodes, device)
    position = torch.arange(nodes, dtype=torch.float32, device=device)
    displacement = edge_displacement_from_positions(position, edge_index)
    features = torch.randn(nodes, dim, device=device)
    frequencies = standard_frequencies(dim, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = truncated_walk_resolvent(
        features, edge_index, displacement, frequencies, z=0.6, num_steps=steps
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "nodes": nodes,
        "directed_edges": edge_index.shape[1],
        "dimension": dim,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "finite": bool(torch.isfinite(output).all()),
    }


def _path_phase_error(device: torch.device) -> float:
    nodes = 12
    edge_index = _path_edges(nodes, device)
    position = torch.arange(nodes, dtype=torch.float64, device=device)
    displacement = edge_displacement_from_positions(position, edge_index)
    source, target = edge_index
    degree = torch.bincount(source, minlength=nodes).double()
    transition = torch.zeros(nodes, nodes, dtype=torch.complex128, device=device)
    omega = 0.73
    transition[source, target] = torch.exp(1j * omega * displacement) / degree[source]
    resolvent = torch.linalg.inv(torch.eye(nodes, dtype=torch.complex128, device=device) - 0.6 * transition)
    reference = torch.exp(
        1j * omega * (position[None, :] - position[:, None])
    )
    nonzero = resolvent.abs() > 1e-12
    phase = resolvent[nonzero] / resolvent[nonzero].abs()
    return float((phase - reference[nonzero]).abs().max().cpu())


def run(output: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measurements = [_measure(nodes, dim=64, steps=16, device=device) for nodes in (256, 512, 1_024, 2_048, 4_096)]
    log_edges = torch.tensor([math.log(row["directed_edges"]) for row in measurements])
    log_time = torch.tensor([math.log(max(row["elapsed_seconds"], 1e-9)) for row in measurements])
    centered = log_edges - log_edges.mean()
    slope = float((centered * (log_time - log_time.mean())).sum() / centered.square().sum())
    payload = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "path_resolvent_phase_max_error_float64": _path_phase_error(device),
        "scaling_log_time_vs_log_edges_slope": slope,
        "measurements": measurements,
        "complexity_claim": "O(K|E|d)",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    if result["path_resolvent_phase_max_error_float64"] >= 1e-8:
        raise RuntimeError("path phase recovery exceeded the 1e-8 acceptance threshold")
    if not all(row["finite"] for row in result["measurements"]):
        raise RuntimeError("non-finite value in scaling benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
