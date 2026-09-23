"""Paths and dataset adapters for offline Exact Holonomy transports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .static_holonomy import STATIC_FIELD_PROTOCOL


SIDECAR_PROTOCOL = "precomputed-exact-holonomy-transport-v1"


def exact_holonomy_sidecar_path(
    data_root: str | Path,
    *,
    consumer: str,
    dataset: str,
    split: str | None,
    num_frequencies: int,
    frequency_base: float,
    diffusion_time: float,
) -> Path:
    """Return the canonical path for a static pairwise transport sidecar."""
    if consumer not in {"graphgps", "gin"}:
        raise ValueError(f"unsupported exact Holonomy consumer: {consumer}")
    if num_frequencies < 1:
        raise ValueError("num_frequencies must be positive")
    if consumer == "gin" and split not in {"train", "validation", "test"}:
        raise ValueError("GIN sidecars require train/validation/test split")
    if consumer == "graphgps" and split is not None:
        raise ValueError("GraphGPS uses one joined sidecar and no split name")
    leaf = Path(split) if split is not None else Path("joined")
    base_text = format(float(frequency_base), ".12g")
    time_text = format(float(diffusion_time), ".12g")
    return (
        Path(data_root)
        / ".exact_holonomy_cache"
        / STATIC_FIELD_PROTOCOL
        / consumer
        / dataset
        / leaf
        / f"f{num_frequencies}-base{base_text}-t{time_text}.pt"
    )


def exact_holonomy_infeasible_path(sidecar: str | Path) -> Path:
    return Path(str(sidecar) + ".infeasible.json")


class PrecomputedExactHolonomyDataset(Dataset):
    """Attach one immutable transport slice to each item of a PyG dataset."""

    def __init__(
        self,
        dataset: Dataset,
        sidecar: str | Path,
        *,
        expected_frequencies: int,
        expected_diffusion_time: float,
    ) -> None:
        self.dataset = dataset
        self.sidecar = Path(sidecar)
        marker = Path(str(self.sidecar) + ".done.json")
        if not self.sidecar.is_file() or not marker.is_file():
            raise FileNotFoundError(
                f"static exact Holonomy sidecar is missing: {self.sidecar}"
            )
        payload: dict[str, Any] = torch.load(
            self.sidecar, map_location="cpu", mmap=True, weights_only=True
        )
        if payload.get("protocol") != SIDECAR_PROTOCOL:
            raise RuntimeError(f"invalid exact Holonomy protocol: {self.sidecar}")
        if payload.get("field_protocol") != STATIC_FIELD_PROTOCOL:
            raise RuntimeError(f"invalid exact Holonomy field: {self.sidecar}")
        if int(payload.get("num_frequencies", -1)) != expected_frequencies:
            raise RuntimeError(f"exact Holonomy frequency mismatch: {self.sidecar}")
        if abs(
            float(payload.get("diffusion_time", -1.0))
            - float(expected_diffusion_time)
        ) > 1e-12:
            raise RuntimeError(f"exact Holonomy diffusion-time mismatch: {self.sidecar}")
        node_counts = payload.get("node_counts")
        transport_slices = payload.get("transport_slices")
        transport = payload.get("transport")
        if not isinstance(node_counts, torch.Tensor) or len(node_counts) != len(dataset):
            raise RuntimeError(f"exact Holonomy graph-count mismatch: {self.sidecar}")
        if (
            not isinstance(transport_slices, torch.Tensor)
            or tuple(transport_slices.shape) != (len(dataset) + 1,)
        ):
            raise RuntimeError(f"invalid exact Holonomy slices: {self.sidecar}")
        if (
            not isinstance(transport, torch.Tensor)
            or not torch.is_complex(transport)
            or transport.ndim != 2
            or transport.shape[1] != expected_frequencies
            or int(transport_slices[-1]) != transport.shape[0]
        ):
            raise RuntimeError(f"invalid exact Holonomy transport: {self.sidecar}")
        self.node_counts = node_counts.to(torch.int64)
        self.transport_slices = transport_slices.to(torch.int64)
        self.transport = transport

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Data:
        data = self.dataset[index]
        if not isinstance(data, Data):
            raise TypeError(f"expected PyG Data, got {type(data)!r}")
        data = data.clone()
        if int(data.num_nodes) != int(self.node_counts[index]):
            raise RuntimeError(
                f"graph {index} changed after Exact Holonomy precomputation"
            )
        start = int(self.transport_slices[index])
        stop = int(self.transport_slices[index + 1])
        data.exact_holonomy_transport = self.transport[start:stop]
        return data

