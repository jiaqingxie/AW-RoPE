"""Offline dataset adapters for the staged AW-RoPE experiments.

Every loader points at an already prepared local asset.  It never downloads a
dataset.  Official validation/test splits are preserved; datasets without an
official validation split receive a deterministic split of the training set.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset
from torch_geometric.data import Data, InMemoryDataset

from aw_rope.exact_holonomy_cache import (
    PrecomputedExactHolonomyDataset,
    exact_holonomy_sidecar_path,
)


TaskType = Literal[
    "graph-regression",
    "graph-classification",
    "graph-multilabel",
    "node-classification",
    "sequence-prediction",
]
EncoderType = Literal["continuous", "constant", "ogb-atom", "ast"]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task: TaskType
    train: Dataset
    validation: Dataset | None
    test: Dataset
    input_dim: int
    output_dim: int
    metric: str
    metric_mode: Literal["min", "max"]
    target_scale: float = 1.0
    encoder: EncoderType = "continuous"
    position_dim: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class _CollatedDataset(InMemoryDataset):
    def __init__(self, data: object, slices: object) -> None:
        super().__init__(root=None)
        self._data = data
        self.slices = slices


class _TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable[[Data, int], Data]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Data:
        data = self.dataset[index]
        if not isinstance(data, Data):
            raise TypeError(f"expected a PyG Data object, got {type(data)!r}")
        return self.transform(data.clone(), index)


def _require(paths: list[Path], dataset: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{dataset} is not prepared locally; missing {missing}. "
            "Run scripts/prepare_datasets.py outside the GPU job."
        )


def _search_split(
    dataset: Dataset,
    train_indices: list[int] | None = None,
    *,
    validation_size: int,
    split_seed: int = 17_291,
) -> tuple[Subset, Subset]:
    indices = list(range(len(dataset))) if train_indices is None else train_indices
    if not 0 < validation_size < len(indices):
        raise ValueError("validation_size must be between zero and the dataset size")
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(split_seed))
    validation_positions = order[:validation_size].tolist()
    training_positions = order[validation_size:].tolist()
    return (
        Subset(dataset, [indices[index] for index in training_positions]),
        Subset(dataset, [indices[index] for index in validation_positions]),
    )


def _stratified_search_split(
    dataset: Dataset,
    *,
    validation_size: int,
    split_seed: int = 17_291,
) -> tuple[Subset, Subset]:
    """Make an exact-size deterministic split while preserving class ratios."""
    if not 0 < validation_size < len(dataset):
        raise ValueError("validation_size must be between zero and the dataset size")
    groups: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        target = getattr(dataset[index], "y", None)
        if not isinstance(target, Tensor) or target.numel() != 1:
            raise ValueError("stratified split requires one scalar class target per item")
        groups.setdefault(int(target.item()), []).append(index)
    if validation_size < len(groups):
        raise ValueError("validation_size must include at least one item per class")
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("stratified split requires at least two items per class")

    expected = {
        label: validation_size * len(indices) / len(dataset)
        for label, indices in groups.items()
    }
    allocations = {
        label: min(len(groups[label]) - 1, max(1, int(expected[label])))
        for label in groups
    }
    while sum(allocations.values()) < validation_size:
        eligible = [
            label for label in groups
            if allocations[label] < len(groups[label]) - 1
        ]
        if not eligible:
            raise RuntimeError("cannot fill the requested stratified validation split")
        label = max(
            eligible,
            key=lambda item: (expected[item] - allocations[item], -item),
        )
        allocations[label] += 1
    while sum(allocations.values()) > validation_size:
        eligible = [label for label in groups if allocations[label] > 1]
        if not eligible:
            raise RuntimeError("cannot shrink the requested stratified validation split")
        label = max(
            eligible,
            key=lambda item: (allocations[item] - expected[item], -item),
        )
        allocations[label] -= 1

    generator = torch.Generator().manual_seed(split_seed)
    training_indices: list[int] = []
    validation_indices: list[int] = []
    for label in sorted(groups):
        indices = groups[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        count = allocations[label]
        validation_indices.extend(indices[position] for position in order[:count])
        training_indices.extend(indices[position] for position in order[count:])
    if len(validation_indices) != validation_size:
        raise RuntimeError("stratified validation split has the wrong size")
    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def _train_and_validation(
    train: Dataset,
    validation: Dataset,
    final_fit: bool,
) -> tuple[Dataset, Dataset | None]:
    if final_fit:
        return ConcatDataset((train, validation)), None
    return train, validation


def _load_synthetic_payload(path: Path) -> tuple[_CollatedDataset, dict[str, object]]:
    (data, slices), metadata = torch.load(path, map_location="cpu", weights_only=False)
    return _CollatedDataset(data, slices), metadata


def _load_synthetic(name: str, root: Path, final_fit: bool) -> DatasetBundle:
    if name.startswith("monochromatic-"):
        deleted_edges = int(name.rsplit("-", 1)[1])
        if deleted_edges not in {0, 5, 10, 15}:
            raise ValueError("monochromatic deletion setting must be one of 0, 5, 10, 15")
        path = root / "synthetic" / "monochromatic_subgraphs" / f"deleted_{deleted_edges}.pt"
        input_dim, target_scale = 1, 25.0
    elif name == "watts-strogatz-spd":
        path = root / "synthetic" / "watts_strogatz_spd" / "data.pt"
        input_dim, target_scale = 2, 10.0
    else:
        raise KeyError(name)

    _require([path], name)
    dataset, metadata = _load_synthetic_payload(path)
    train_indices = [int(index) for index in metadata["train"]]
    test_indices = [int(index) for index in metadata["test"]]
    if final_fit:
        train: Dataset = Subset(dataset, train_indices)
        validation = None
    else:
        train, validation = _search_split(
            dataset, train_indices, validation_size=min(1_000, len(train_indices) // 10)
        )
    return DatasetBundle(
        name=name,
        task="graph-regression",
        train=train,
        validation=validation,
        test=Subset(dataset, test_indices),
        input_dim=input_dim,
        output_dim=1,
        metric="normalized-rmse",
        metric_mode="min",
        target_scale=target_scale,
    )


def _load_gnn_benchmark(name: str, root: Path, final_fit: bool) -> DatasetBundle:
    from torch_geometric.datasets import GNNBenchmarkDataset

    canonical = {"mnist": "MNIST", "cifar10": "CIFAR10", "pattern": "PATTERN", "cluster": "CLUSTER"}[name]
    base = root / "gnn_benchmark"
    _require([base / canonical / "processed" / "train_data.pt"], name)
    train_raw = GNNBenchmarkDataset(str(base), canonical, split="train")
    validation_raw = GNNBenchmarkDataset(str(base), canonical, split="val")
    test = GNNBenchmarkDataset(str(base), canonical, split="test")
    train, validation = _train_and_validation(train_raw, validation_raw, final_fit)
    node_task = name in {"pattern", "cluster"}
    return DatasetBundle(
        name=name,
        task="node-classification" if node_task else "graph-classification",
        train=train,
        validation=validation,
        test=test,
        input_dim=train_raw.num_features,
        output_dim=train_raw.num_classes,
        metric="accuracy",
        metric_mode="max",
    )


def _load_lrgb(name: str, root: Path, final_fit: bool) -> DatasetBundle:
    base = root / "lrgb"
    _require([base / name / "processed" / f"{split}.pt" for split in ("train", "val", "test")], name)
    # LRGBDataset.__init__ checks raw files before processed files and may
    # download even when every processed split is present. Load the same
    # PyG stores directly so prepared-only assets remain strictly offline.
    def prepared_split(split: str) -> _CollatedDataset:
        # These trusted local stores include PyG Data objects, not just
        # weights. Be explicit for PyTorch >= 2.6 as well as older images.
        payload = torch.load(
            base / name / "processed" / f"{split}.pt",
            map_location="cpu", weights_only=False,
        )
        if not isinstance(payload, tuple) or len(payload) not in (2, 3):
            raise ValueError(f"invalid prepared {name}/{split} PyG store")
        data, slices = payload[:2]
        data_cls = payload[2] if len(payload) == 3 else Data
        if isinstance(data, dict):
            data = data_cls.from_dict(data)
        return _CollatedDataset(data, slices)

    train_raw = prepared_split("train")
    validation_raw = prepared_split("val")
    test = prepared_split("test")
    train, validation = _train_and_validation(train_raw, validation_raw, final_fit)
    if name == "peptides-func":
        task, output_dim, metric, mode, encoder = "graph-multilabel", 10, "average-precision", "max", "ogb-atom"
    elif name == "peptides-struct":
        task, output_dim, metric, mode, encoder = "graph-regression", 11, "mae", "min", "ogb-atom"
    else:
        task, output_dim, metric, mode, encoder = "node-classification", (81 if name == "coco-sp" else 21), "macro-f1", "max", "continuous"
    return DatasetBundle(
        name=name,
        task=task,  # type: ignore[arg-type]
        train=train,
        validation=validation,
        test=test,
        input_dim=train_raw.num_features,
        output_dim=output_dim,
        metric=metric,
        metric_mode=mode,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
    )


def _degree_features(data: Data, _index: int) -> Data:
    degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
    data.x = torch.log1p(degree)[:, None]
    return data


def _load_malnet(root: Path, final_fit: bool) -> DatasetBundle:
    from torch_geometric.datasets import MalNetTiny

    base = root / "malnet_tiny"
    _require([base / "processed" / "data.pt"], "malnet-tiny")
    train_raw = MalNetTiny(str(base), split="train")
    validation_raw = MalNetTiny(str(base), split="val")
    test_raw = MalNetTiny(str(base), split="test")
    train, validation = _train_and_validation(train_raw, validation_raw, final_fit)
    wrap = lambda dataset: _TransformDataset(dataset, _degree_features)
    return DatasetBundle(
        name="malnet-tiny",
        task="graph-classification",
        train=wrap(train),
        validation=wrap(validation) if validation is not None else None,
        test=wrap(test_raw),
        input_dim=1,
        output_dim=5,
        metric="accuracy",
        metric_mode="max",
    )


def _augment_code2(data: Data, _index: int, vocab: dict[str, int]) -> Data:
    edge_index = data.edge_index
    inverse = edge_index.flip(0)
    attributed = torch.where(data.node_is_attributed.view(-1) == 1)[0]
    if attributed.numel() > 1:
        token_edge = torch.stack((attributed[:-1], attributed[1:]))
        edges = (edge_index, inverse, token_edge, token_edge.flip(0))
    else:
        edges = (edge_index, inverse)
    data.edge_index = torch.cat(edges, dim=1)
    sequence = list(data.y)
    eos = vocab["__EOS__"]
    unk = vocab["__UNK__"]
    encoded = [vocab.get(token, unk) for token in sequence[:5]]
    encoded.extend([eos] * (5 - len(encoded)))
    data.y = torch.tensor(encoded, dtype=torch.long)
    data.x = data.x.long()
    data.node_depth = data.node_depth.view(-1).long().clamp(max=20)
    return data


def _code2_vocabulary(dataset: Any, train_indices: list[int], cache_path: Path) -> dict[str, int]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return {str(key): int(value) for key, value in payload["vocab"].items()}
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for index in train_indices:
        for token in dataset._data.y[index]:
            if token not in first_seen:
                first_seen[token] = len(first_seen)
            counts[token] += 1
    ordered = sorted(counts, key=lambda token: (-counts[token], first_seen[token]))[:5_000]
    vocab = {token: index for index, token in enumerate(ordered)}
    vocab["__UNK__"] = 5_000
    vocab["__EOS__"] = 5_001
    cache_path.write_text(
        json.dumps({"vocab": vocab, "train_graphs": len(train_indices)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return vocab


def _load_ogb(name: str, root: Path, final_fit: bool) -> DatasetBundle:
    from ogb.graphproppred import PygGraphPropPredDataset

    canonical = f"ogbg-{name.removeprefix('ogbg-')}"
    local = root / "ogb" / canonical.replace("-", "_")
    _require([local / "processed" / "geometric_data_processed.pt"], canonical)
    dataset = PygGraphPropPredDataset(name=canonical, root=str(root / "ogb"))
    split = {key: value.view(-1).tolist() for key, value in dataset.get_idx_split().items()}
    train_raw: Dataset = Subset(dataset, split["train"])
    validation_raw: Dataset = Subset(dataset, split["valid"])
    test_raw: Dataset = Subset(dataset, split["test"])

    if canonical == "ogbg-code2":
        vocab = _code2_vocabulary(dataset, split["train"], local / "aw_rope_vocab_5000.json")
        transform = lambda data, index: _augment_code2(data, index, vocab)
        train_raw = _TransformDataset(train_raw, transform)
        validation_raw = _TransformDataset(validation_raw, transform)
        test_raw = _TransformDataset(test_raw, transform)
    train, validation = _train_and_validation(train_raw, validation_raw, final_fit)
    specs: dict[str, tuple[TaskType, int, str, Literal["min", "max"], EncoderType, int]] = {
        "ogbg-molhiv": ("graph-multilabel", 1, "rocauc", "max", "ogb-atom", 9),
        "ogbg-molpcba": ("graph-multilabel", 128, "average-precision", "max", "ogb-atom", 9),
        "ogbg-code2": ("sequence-prediction", 5_002, "sequence-f1", "max", "ast", 2),
    }
    task, output_dim, metric, mode, encoder, input_dim = specs[canonical]
    return DatasetBundle(
        name=canonical,
        task=task,
        train=train,
        validation=validation,
        test=test_raw,
        input_dim=input_dim,
        output_dim=output_dim,
        metric=metric,
        metric_mode=mode,
        encoder=encoder,
        metadata={"sequence_length": 5} if task == "sequence-prediction" else {},
    )


class _PointGraphDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        mesh: bool,
        training: bool = False,
        xyz_only: bool = False,
        points: int = 2_048,
        k: int = 20,
    ) -> None:
        self.dataset = dataset
        self.mesh = mesh
        self.training = training
        self.xyz_only = xyz_only
        self.points = points
        self.k = k

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Data:
        data = self.dataset[index].clone()
        # WIRE uses Scenic's point-cloud input pipeline: training examples get
        # fresh point sampling/noise, whereas validation and test are stable
        # across worker counts and resumptions.
        sampling_context = (
            nullcontext()
            if self.training
            else torch.random.fork_rng(devices=[])
        )
        with sampling_context:
            if not self.training:
                torch.manual_seed(91_733 + index)
            if self.mesh:
                from torch_geometric.transforms import SamplePoints

                data = SamplePoints(self.points, include_normals=True)(data)
            else:
                count = data.pos.shape[0]
                choice = torch.randperm(count)[: self.points] if count >= self.points else torch.randint(count, (self.points,))
                data.pos = data.pos[choice]
                if data.x is not None:
                    data.x = data.x[choice]
                if data.y.numel() == count:
                    data.y = data.y[choice]
        positions = data.pos.float().contiguous()
        if self.training:
            # Scenic clips N(0, 0.01) coordinate noise to [-0.02, 0.02].
            positions = positions + torch.randn_like(positions).clamp(-2.0, 2.0) * 0.01
        data.pos = positions
        # A CPU KD-tree avoids materialising a 2048 x 2048 distance matrix in
        # every DataLoader worker.  Make the kNN graph explicitly undirected so
        # non-backtracking reverse-edge lookup has an exact counterpart.
        from scipy.spatial import cKDTree
        from torch_geometric.utils import to_undirected

        neighbors = cKDTree(positions.numpy()).query(
            positions.numpy(), k=min(self.k + 1, positions.shape[0])
        )[1][:, 1:]
        neighbors = torch.from_numpy(neighbors).long()
        source = torch.arange(positions.shape[0]).repeat_interleave(neighbors.shape[1])
        data.edge_index = to_undirected(
            torch.stack((source, neighbors.reshape(-1))),
            num_nodes=positions.shape[0],
        )
        if self.xyz_only:
            # The WIRE/Scenic PCT consumes xyz only; normals are deliberately
            # not added as extra features so the comparison keeps its inputs.
            data.x = data.pos.float()
        else:
            feature_parts = [data.pos.float()]
            normal = getattr(data, "normal", None)
            if normal is not None:
                feature_parts.append(normal.float())
            elif data.x is not None:
                feature_parts.append(data.x.float())
            data.x = torch.cat(feature_parts, dim=-1)
        data.face = None
        return data


def _load_pointcloud(
    name: str,
    root: Path,
    final_fit: bool,
    *,
    wire_protocol: bool = False,
) -> DatasetBundle:
    from torch_geometric.datasets import ModelNet, ShapeNet

    if name == "modelnet40":
        base = root / "pointcloud" / "ModelNet40"
        _require([base / "processed" / "training.pt", base / "processed" / "test.pt"], name)
        full_train = ModelNet(str(base), name="40", train=True)
        test_raw = ModelNet(str(base), name="40", train=False)
        if final_fit:
            train_raw, validation_raw = full_train, None
        else:
            # Hold out 624 examples for validation while retaining exactly
            # nine complete effective batches (9 * 1024 = 9216) per epoch.
            # Scenic's 2048-point H5 variant has 9840 training examples and
            # likewise performs nine optimizer steps after dropping its tail.
            train_raw, validation_raw = _stratified_search_split(
                full_train, validation_size=624
            )
        return DatasetBundle(
            name=name,
            task="graph-classification",
            train=_PointGraphDataset(
                train_raw,
                mesh=True,
                training=wire_protocol,
                xyz_only=wire_protocol,
            ),
            validation=(
                _PointGraphDataset(validation_raw, mesh=True, xyz_only=wire_protocol)
                if validation_raw is not None else None
            ),
            test=_PointGraphDataset(test_raw, mesh=True, xyz_only=wire_protocol),
            input_dim=3 if wire_protocol else 6,
            output_dim=40,
            metric="accuracy",
            metric_mode="max",
            position_dim=3,
        )

    base = root / "pointcloud" / "ShapeNet"
    processed = base / "processed"
    if not list(processed.glob("*_trainval.pt")) or not list(processed.glob("*_test.pt")):
        _require([processed / "*_trainval.pt", processed / "*_test.pt"], name)
    test_raw = ShapeNet(str(base), split="test", include_normals=True)
    if wire_protocol:
        train_split = ShapeNet(str(base), split="train", include_normals=True)
        validation_split = ShapeNet(str(base), split="val", include_normals=True)
        if final_fit:
            # Preserve the official train+validation population for final fits.
            train_raw, validation_raw = ConcatDataset((train_split, validation_split)), None
        else:
            train_raw, validation_raw = train_split, validation_split
    else:
        full_train = ShapeNet(str(base), split="trainval", include_normals=True)
        if final_fit:
            train_raw, validation_raw = full_train, None
        else:
            train_raw, validation_raw = _search_split(full_train, validation_size=1_400)
    return DatasetBundle(
        name="shapenet",
        task="node-classification",
        train=_PointGraphDataset(
            train_raw,
            mesh=False,
            training=wire_protocol,
            xyz_only=wire_protocol,
        ),
        validation=(
            _PointGraphDataset(validation_raw, mesh=False, xyz_only=wire_protocol)
            if validation_raw is not None else None
        ),
        test=_PointGraphDataset(test_raw, mesh=False, xyz_only=wire_protocol),
        input_dim=3 if wire_protocol else 6,
        output_dim=50,
        # WIRE Table 4 reports ShapeNet point accuracy.  Use the same metric
        # so the official-GT point-cloud row is directly comparable.
        metric="accuracy",
        metric_mode="max",
        position_dim=3,
    )


STAGE_DATASETS: dict[str, tuple[str, ...]] = {
    "stage1": (
        "monochromatic-0", "monochromatic-5", "monochromatic-10",
        "monochromatic-15", "watts-strogatz-spd",
    ),
    "stage2": (
        "mnist", "cifar10", "pattern", "cluster", "peptides-func",
        "peptides-struct", "pascalvoc-sp", "malnet-tiny",
    ),
    "stage3": (
        "ogbg-molhiv", "ogbg-molpcba", "ogbg-code2",
        "modelnet40", "shapenet",
    ),
}


def load_dataset(
    name: str,
    root: str | Path = "data",
    *,
    final_fit: bool = False,
    pointcloud_wire_protocol: bool = False,
    exact_holonomy_num_frequencies: int | None = None,
    exact_holonomy_diffusion_time: float = 2.0,
    exact_holonomy_frequency_base: float = 10_000.0,
) -> DatasetBundle:
    """Load one experiment dataset strictly from prepared local files."""
    root = Path(root)
    name = name.lower()
    if name.startswith("monochromatic-") or name == "watts-strogatz-spd":
        bundle = _load_synthetic(name, root, final_fit)
    elif name in {"mnist", "cifar10", "pattern", "cluster"}:
        bundle = _load_gnn_benchmark(name, root, final_fit)
    elif name in {"peptides-func", "peptides-struct", "pascalvoc-sp", "coco-sp"}:
        bundle = _load_lrgb(name, root, final_fit)
    elif name == "malnet-tiny":
        bundle = _load_malnet(root, final_fit)
    elif name.startswith("ogbg-"):
        bundle = _load_ogb(name, root, final_fit)
    elif name in {"modelnet40", "shapenet"}:
        bundle = _load_pointcloud(
            name,
            root,
            final_fit,
            wire_protocol=pointcloud_wire_protocol,
        )
    else:
        raise KeyError(f"unknown dataset {name!r}")

    if exact_holonomy_num_frequencies is None:
        return bundle
    if final_fit:
        raise ValueError("static Exact Holonomy sidecars require fixed train/validation splits")

    def attach(split_name: str, dataset: Dataset | None) -> Dataset | None:
        if dataset is None:
            return None
        path = exact_holonomy_sidecar_path(
            root,
            consumer="gin",
            dataset=name,
            split=split_name,
            num_frequencies=exact_holonomy_num_frequencies,
            frequency_base=exact_holonomy_frequency_base,
            diffusion_time=exact_holonomy_diffusion_time,
        )
        return PrecomputedExactHolonomyDataset(
            dataset,
            path,
            expected_frequencies=exact_holonomy_num_frequencies,
            expected_diffusion_time=exact_holonomy_diffusion_time,
        )

    return DatasetBundle(
        **{
            **bundle.__dict__,
            "train": attach("train", bundle.train),
            "validation": attach("validation", bundle.validation),
            "test": attach("test", bundle.test),
        }
    )
