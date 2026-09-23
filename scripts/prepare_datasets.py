#!/usr/bin/env python3
"""Download and prepare every dataset used by the WIRE paper.

Network access is deliberately direct: common proxy environment variables are
removed and urllib is configured with an empty proxy map before any dataset
loader is imported.  Use ``--group core`` for the moderate-size graph suite or
``--group all`` for the complete WIRE suite, including large OGB and point-cloud
datasets.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable


PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


CATALOG: dict[str, dict[str, Any]] = {
    "monochromatic-subgraphs": {
        "family": "synthetic",
        "task": "graph regression: largest monochromatic connected component",
        "source": "generated as specified in WIRE Section 4.1",
        "paper_count": "10,000 train + 1,000 test per deletion setting",
    },
    "watts-strogatz-spd": {
        "family": "synthetic",
        "task": "graph regression: source-target shortest path distance",
        "source": "generated as specified in WIRE Section 4.1",
        "paper_count": "10,000 train + 1,000 test",
    },
    "MNIST": {
        "family": "gnn-benchmark",
        "task": "graph classification",
        "source": "https://data.pyg.org/datasets/benchmarking-gnns/MNIST_v2.zip",
        "paper_count": 70_000,
    },
    "CIFAR10": {
        "family": "gnn-benchmark",
        "task": "graph classification",
        "source": "https://data.pyg.org/datasets/benchmarking-gnns/CIFAR10_v2.zip",
        "paper_count": 60_000,
    },
    "PATTERN": {
        "family": "gnn-benchmark",
        "task": "inductive node classification",
        "source": "https://data.pyg.org/datasets/benchmarking-gnns/PATTERN_v2.zip",
        "paper_count": 14_000,
    },
    "CLUSTER": {
        "family": "gnn-benchmark",
        "task": "inductive node classification",
        "source": "https://data.pyg.org/datasets/benchmarking-gnns/CLUSTER_v2.zip",
        "paper_count": 12_000,
    },
    "ogbg-molhiv": {
        "family": "ogb",
        "task": "binary graph classification",
        "source": "https://ogb.stanford.edu/docs/graphprop/#ogbg-mol",
        "paper_count": 41_127,
    },
    "ogbg-molpcba": {
        "family": "ogb",
        "task": "128-task graph classification",
        "source": "https://ogb.stanford.edu/docs/graphprop/#ogbg-mol",
        "paper_count": 437_929,
    },
    "ogbg-code2": {
        "family": "ogb",
        "task": "graph-to-subtoken sequence prediction",
        "source": "https://ogb.stanford.edu/docs/graphprop/#ogbg-code2",
        "paper_count": 452_741,
    },
    "Peptides-func": {
        "family": "lrgb",
        "task": "10-task graph classification",
        "source": "https://zenodo.org/records/6975830/files/peptide_multi_class_dataset.csv.gz?download=1",
        "paper_count": 15_535,
    },
    "Peptides-struct": {
        "family": "lrgb",
        "task": "11-task graph regression",
        "source": "https://zenodo.org/records/6975830/files/peptide_structure_normalized_dataset.csv.gz?download=1",
        "paper_count": 15_535,
    },
    "PascalVOC-SP": {
        "family": "lrgb",
        "task": "inductive node classification",
        "source": "https://zenodo.org/records/6975830/files/voc_superpixels_edge_wt_region_boundary.zip?download=1",
        "paper_count": 11_355,
    },
    "MalNet-Tiny": {
        "family": "malnet",
        "task": "graph classification",
        "source": "http://malnet.cc.gatech.edu/graph-data/malnet-graphs-tiny.tar.gz",
        "paper_count": 5_000,
    },
    "ModelNet40": {
        "family": "pointcloud",
        "task": "point-cloud classification",
        "source": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
        "paper_count": 12_311,
    },
    "ShapeNet": {
        "family": "pointcloud",
        "task": "point-cloud part segmentation",
        # The Stanford endpoint used by PyG is no longer reliably reachable.
        # This is a byte-identical public mirror of the original archive.
        "source": "https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip?download=true",
        "original_source": "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
        "sha256": "0e26411700bae2da38ee8ecc719ba4db2e6e0133486e258665952ad5dfced0fe",
        "paper_count": 16_881,
    },
}

GROUPS = {
    "quick": ["monochromatic-subgraphs", "watts-strogatz-spd", "MNIST"],
    "core": [
        "monochromatic-subgraphs",
        "watts-strogatz-spd",
        "MNIST",
        "CIFAR10",
        "PATTERN",
        "CLUSTER",
        "Peptides-func",
        "Peptides-struct",
        "PascalVOC-SP",
        "MalNet-Tiny",
    ],
    "ogb": ["ogbg-molhiv", "ogbg-molpcba", "ogbg-code2"],
    "pointcloud": ["ModelNet40", "ShapeNet"],
}
GROUPS["all"] = list(CATALOG)


def disable_proxies() -> dict[str, str]:
    removed = {name: os.environ.pop(name) for name in PROXY_VARIABLES if name in os.environ}
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    return removed


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_download(
    url: str,
    target: Path,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Download one file with urllib's explicitly proxy-free global opener."""
    checksum_ok = (
        (expected_md5 is None or md5sum(target) == expected_md5)
        and (expected_sha256 is None or sha256sum(target) == expected_sha256)
    ) if target.exists() else False
    if checksum_ok:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "aw-rope-dataset-preparer/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=2**20)
    if expected_md5 is not None and md5sum(partial) != expected_md5:
        raise ValueError(f"checksum mismatch for {url}")
    if expected_sha256 is not None and sha256sum(partial) != expected_sha256:
        raise ValueError(f"SHA-256 checksum mismatch for {url}")
    partial.replace(target)
    return target


def save_manifest(root: Path, status: dict[str, Any], direct_mode: bool = True) -> None:
    manifest = {
        "wire_paper": "https://arxiv.org/abs/2509.22259",
        "wire_code": "https://github.com/cederikHoefs/Graph-RoPE",
        "prepared_at_unix": int(time.time()),
        "network_mode": "direct-no-proxy" if direct_mode else "unknown",
        "catalog": CATALOG,
        "local_status": status,
    }
    target = root.parent / "dataset_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def directed_edges(graph: Any) -> Any:
    import torch

    pairs = []
    for u, v in graph.edges():
        pairs.extend(((u, v), (v, u)))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def largest_monochromatic_component(graph: Any, colors: Any) -> int:
    import networkx as nx

    largest = 0
    for color in range(int(colors.max()) + 1):
        nodes = [node for node in graph.nodes if int(colors[node]) == color]
        subgraph = graph.subgraph(nodes)
        component_sizes = [len(component) for component in nx.connected_components(subgraph)]
        largest = max([largest, *component_sizes])
    return largest


def generate_monochromatic(root: Path) -> dict[str, Any]:
    import networkx as nx
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    target_dir = root / "synthetic" / "monochromatic_subgraphs"
    target_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for deleted_edges in (0, 5, 10, 15):
        target = target_dir / f"deleted_{deleted_edges}.pt"
        if target.exists():
            payload = torch.load(target, weights_only=False)
            counts[str(deleted_edges)] = int(payload[1]["num_graphs"])
            continue
        data_list = []
        for index in range(11_000):
            rng = random.Random(17_000 * deleted_edges + index)
            graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5))
            if deleted_edges:
                removed = rng.sample(list(graph.edges), deleted_edges)
                graph.remove_edges_from(removed)
            generator = torch.Generator().manual_seed(31_337 * (deleted_edges + 1) + index)
            colors = torch.randint(0, 3, (25,), generator=generator)
            label = largest_monochromatic_component(graph, colors)
            data_list.append(
                Data(
                    x=colors[:, None],
                    edge_index=directed_edges(graph),
                    y=torch.tensor([float(label)]),
                    num_nodes=25,
                )
            )
        data, slices = InMemoryDataset.collate(data_list)
        metadata = {
            "num_graphs": len(data_list),
            "train": list(range(10_000)),
            "test": list(range(10_000, 11_000)),
            "deleted_edges": deleted_edges,
            "num_colors": 3,
            "seed_scheme": "17000*deleted_edges+index / 31337*(deleted_edges+1)+index",
        }
        torch.save(((data, slices), metadata), target)
        counts[str(deleted_edges)] = len(data_list)
    return {"num_graphs_by_deleted_edges": counts, "path": str(target_dir)}


def generate_shortest_path(root: Path) -> dict[str, Any]:
    import networkx as nx
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    target_dir = root / "synthetic" / "watts_strogatz_spd"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "data.pt"
    if target.exists():
        payload = torch.load(target, weights_only=False)
        return {"num_graphs": int(payload[1]["num_graphs"]), "path": str(target)}
    data_list = []
    for index in range(11_000):
        graph_seed = 70_000 + index
        graph = nx.watts_strogatz_graph(10, 2, 0.6, seed=graph_seed)
        while not nx.is_connected(graph):
            graph_seed += 11_000
            graph = nx.watts_strogatz_graph(10, 2, 0.6, seed=graph_seed)
        rng = random.Random(90_000 + index)
        source, target_node = rng.sample(range(10), 2)
        marker = torch.zeros((10, 2), dtype=torch.float32)
        marker[source, 0] = 1.0
        marker[target_node, 1] = 1.0
        distance = nx.shortest_path_length(graph, source, target_node)
        data_list.append(
            Data(
                x=marker,
                edge_index=directed_edges(graph),
                y=torch.tensor([float(distance)]),
                source=source,
                target=target_node,
                num_nodes=10,
            )
        )
    data, slices = InMemoryDataset.collate(data_list)
    metadata = {
        "num_graphs": len(data_list),
        "train": list(range(10_000)),
        "test": list(range(10_000, 11_000)),
        "n": 10,
        "k": 2,
        "rewiring_probability": 0.6,
    }
    torch.save(((data, slices), metadata), target)
    return {"num_graphs": len(data_list), "path": str(target)}


def prepare_gnn_benchmark(name: str, root: Path) -> dict[str, Any]:
    from torch_geometric.datasets import GNNBenchmarkDataset

    counts = {}
    for split in ("train", "val", "test"):
        dataset = GNNBenchmarkDataset(str(root / "gnn_benchmark"), name=name, split=split)
        counts[split] = len(dataset)
    return {"num_graphs": sum(counts.values()), "splits": counts}


def prepare_peptides(name: str, root: Path) -> dict[str, Any]:
    """Build standard PyG LRGB processed files from the official Zenodo release."""
    import pandas as pd
    import torch
    from ogb.utils import smiles2graph
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.datasets import LRGBDataset

    dataset_name = name.lower()
    base = root / "lrgb" / dataset_name
    raw = base / "raw"
    processed = base / "processed"
    raw_files = [raw / f"{split}.pt" for split in ("train", "val", "test")]
    processed_files = [processed / f"{split}.pt" for split in ("train", "val", "test")]
    if all(path.exists() for path in processed_files):
        # PyG checks raw files before processed files.  Materialize the standard
        # tuple representation so a later LRGBDataset load remains fully
        # offline instead of falling back to its historical Dropbox URL.
        raw.mkdir(parents=True, exist_ok=True)
        for split, processed_path, raw_path in zip(
            ("train", "val", "test"), processed_files, raw_files
        ):
            if raw_path.exists():
                continue
            dataset = InMemoryDataset()
            dataset.load(processed_path)
            graphs = [
                (
                    graph.x.clone(),
                    graph.edge_attr.clone(),
                    graph.edge_index.clone(),
                    graph.y.clone(),
                )
                for graph in dataset
            ]
            torch.save(graphs, raw_path)
        counts = {
            split: len(LRGBDataset(str(root / "lrgb"), name=name, split=split))
            for split in ("train", "val", "test")
        }
        return {"num_graphs": sum(counts.values()), "splits": counts, "mirror": "Zenodo 6975830"}

    source_dir = root / "lrgb" / "_source"
    split_path = direct_download(
        "https://zenodo.org/records/6975830/files/splits_random_stratified_peptide.pickle?download=1",
        source_dir / "splits_random_stratified_peptide.pickle",
        "5a0114bdadc80b94fc7ae974f13ef061",
    )
    if name == "Peptides-func":
        csv_path = direct_download(
            CATALOG[name]["source"],
            source_dir / "peptide_multi_class_dataset.csv.gz",
            "701eb743e899f4d793f0e13c8fa5a1b4",
        )
    else:
        csv_path = direct_download(
            CATALOG[name]["source"],
            source_dir / "peptide_structure_normalized_dataset.csv.gz",
            "c240c1c15466b5c907c63e180fa8aa89",
        )

    frame = pd.read_csv(csv_path)
    target_names = [
        "Inertia_mass_a",
        "Inertia_mass_b",
        "Inertia_mass_c",
        "Inertia_valence_a",
        "Inertia_valence_b",
        "Inertia_valence_c",
        "length_a",
        "length_b",
        "length_c",
        "Spherocity",
        "Plane_best_fit",
    ]
    data_list = []
    print(f"Converting {len(frame)} peptide SMILES strings to graphs ...", flush=True)
    for row_index, row in frame.iterrows():
        graph = smiles2graph(row["smiles"])
        if name == "Peptides-func":
            label = torch.tensor([ast.literal_eval(row["labels"])], dtype=torch.float32)
        else:
            label = torch.tensor([[float(row[column]) for column in target_names]], dtype=torch.float32)
        data_list.append(
            Data(
                x=torch.from_numpy(graph["node_feat"]).long(),
                edge_index=torch.from_numpy(graph["edge_index"]).long(),
                edge_attr=torch.from_numpy(graph["edge_feat"]).long(),
                y=label,
                num_nodes=int(graph["num_nodes"]),
            )
        )
        if (row_index + 1) % 2_000 == 0:
            print(f"  converted {row_index + 1}/{len(frame)}", flush=True)

    with split_path.open("rb") as handle:
        split_indices = pickle.load(handle)
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        indices = split_indices[split]
        split_data = [data_list[int(index)] for index in indices]
        torch.save(
            [(graph.x, graph.edge_attr, graph.edge_index, graph.y) for graph in split_data],
            raw / f"{split}.pt",
        )
        InMemoryDataset.save(split_data, processed / f"{split}.pt")
        counts[split] = len(indices)
    return {"num_graphs": sum(counts.values()), "splits": counts, "mirror": "Zenodo 6975830"}


def prepare_pascalvoc(root: Path) -> dict[str, Any]:
    from torch_geometric.datasets import LRGBDataset

    lrgb_root = root / "lrgb"
    raw_dir = lrgb_root / "pascalvoc-sp" / "raw"
    processed_dir = lrgb_root / "pascalvoc-sp" / "processed"
    if not all((processed_dir / f"{split}.pt").exists() for split in ("train", "val", "test")):
        archive = direct_download(
            CATALOG["PascalVOC-SP"]["source"],
            lrgb_root / "_source" / "voc_superpixels_edge_wt_region_boundary.zip",
            "9a535cb17ab5c6ca94d0b93f0b5293e7",
        )
        extraction = lrgb_root / "_source" / "pascalvoc_extracted"
        extraction.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(extraction)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            matches = list(extraction.rglob(f"{split}.pickle"))
            if len(matches) != 1:
                raise RuntimeError(f"expected exactly one {split}.pickle in PascalVOC archive")
            shutil.copy2(matches[0], raw_dir / f"{split}.pickle")

    counts = {}
    for split in ("train", "val", "test"):
        dataset = LRGBDataset(str(lrgb_root), name="PascalVOC-SP", split=split)
        counts[split] = len(dataset)
    return {"num_graphs": sum(counts.values()), "splits": counts, "mirror": "Zenodo 6975830"}


def prepare_lrgb(name: str, root: Path) -> dict[str, Any]:
    if name.startswith("Peptides-"):
        return prepare_peptides(name, root)
    if name == "PascalVOC-SP":
        return prepare_pascalvoc(root)
    raise KeyError(name)


def prepare_malnet(root: Path) -> dict[str, Any]:
    from torch_geometric.datasets import MalNetTiny

    dataset = MalNetTiny(str(root / "malnet_tiny"))
    return {"num_graphs": len(dataset)}


def prepare_ogb(name: str, root: Path) -> dict[str, Any]:
    from ogb.graphproppred import PygGraphPropPredDataset

    dataset = PygGraphPropPredDataset(name=name, root=str(root / "ogb"))
    split = dataset.get_idx_split()
    return {
        "num_graphs": len(dataset),
        "splits": {key: int(value.numel()) for key, value in split.items()},
    }


def prepare_modelnet(root: Path) -> dict[str, Any]:
    from torch_geometric.datasets import ModelNet

    train = ModelNet(str(root / "pointcloud" / "ModelNet40"), name="40", train=True)
    test = ModelNet(str(root / "pointcloud" / "ModelNet40"), name="40", train=False)
    return {"num_graphs": len(train) + len(test), "splits": {"train": len(train), "test": len(test)}}


def prepare_shapenet(root: Path) -> dict[str, Any]:
    from torch_geometric.datasets import ShapeNet

    dataset_root = root / "pointcloud" / "ShapeNet"
    raw_dir = dataset_root / "raw"
    expected_raw_entries = list(ShapeNet.category_ids.values()) + ["train_test_split"]
    if not all((raw_dir / entry).exists() for entry in expected_raw_entries):
        source_dir = dataset_root / "_source"
        archive = direct_download(
            CATALOG["ShapeNet"]["source"],
            source_dir / "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
            expected_sha256=CATALOG["ShapeNet"]["sha256"],
        )
        extraction = source_dir / "extracted"
        extracted_dataset = extraction / "shapenetcore_partanno_segmentation_benchmark_v0_normal"
        if not extracted_dataset.exists():
            extraction.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zip_file:
                zip_file.extractall(extraction)
        if raw_dir.exists():
            raw_dir.rmdir()  # Only the empty directory left by an interrupted download is safe here.
        shutil.move(str(extracted_dataset), raw_dir)

    trainval = ShapeNet(str(dataset_root), split="trainval")
    test = ShapeNet(str(dataset_root), split="test")
    return {
        "num_graphs": len(trainval) + len(test),
        "splits": {"trainval": len(trainval), "test": len(test)},
        "archive_sha256": CATALOG["ShapeNet"]["sha256"],
        "mirror": "Hugging Face cminst/ShapeNet",
    }


def prepare_one(name: str, root: Path) -> dict[str, Any]:
    family = CATALOG[name]["family"]
    if name == "monochromatic-subgraphs":
        details = generate_monochromatic(root)
        local_root = root / "synthetic" / "monochromatic_subgraphs"
    elif name == "watts-strogatz-spd":
        details = generate_shortest_path(root)
        local_root = root / "synthetic" / "watts_strogatz_spd"
    elif family == "gnn-benchmark":
        details = prepare_gnn_benchmark(name, root)
        local_root = root / "gnn_benchmark" / name
    elif family == "lrgb":
        details = prepare_lrgb(name, root)
        local_root = root / "lrgb" / name.lower()
    elif family == "malnet":
        details = prepare_malnet(root)
        local_root = root / "malnet_tiny"
    elif family == "ogb":
        details = prepare_ogb(name, root)
        local_root = root / "ogb" / name.replace("-", "_")
    elif name == "ModelNet40":
        details = prepare_modelnet(root)
        local_root = root / "pointcloud" / "ModelNet40"
    elif name == "ShapeNet":
        details = prepare_shapenet(root)
        local_root = root / "pointcloud" / "ShapeNet"
    else:
        raise KeyError(name)
    return {"ready": True, "bytes": directory_size(local_root), "local_root": str(local_root), **details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"), help="dataset root")
    parser.add_argument("--group", choices=sorted(GROUPS), default="core")
    parser.add_argument("--only", nargs="+", choices=sorted(CATALOG), help="explicit dataset names")
    parser.add_argument("--list", action="store_true", help="print the catalog and exit")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed dataset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print(json.dumps(CATALOG, indent=2, ensure_ascii=False))
        return 0

    removed = disable_proxies()
    print(f"Direct network mode enabled; disabled {len(removed)} proxy environment variable(s).")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = args.only or GROUPS[args.group]
    status: dict[str, Any] = {}
    manifest_path = root.parent / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            status.update(json.loads(manifest_path.read_text(encoding="utf-8")).get("local_status", {}))
        except (OSError, json.JSONDecodeError):
            pass

    failures = 0
    for position, name in enumerate(selected, 1):
        print(f"[{position}/{len(selected)}] Preparing {name} ...", flush=True)
        started = time.monotonic()
        try:
            details = prepare_one(name, root)
            details["elapsed_seconds"] = round(time.monotonic() - started, 3)
            status[name] = details
            print(
                f"  ready: {details.get('num_graphs', details.get('num_graphs_by_deleted_edges'))}; "
                f"{details['bytes'] / 2**20:.1f} MiB",
                flush=True,
            )
        except Exception as error:  # keep the expensive batch resumable
            failures += 1
            status[name] = {
                "ready": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            if args.fail_fast:
                save_manifest(root, status)
                raise
        save_manifest(root, status)

    print(f"Manifest: {manifest_path}")
    print(f"Prepared {len(selected) - failures}/{len(selected)} selected datasets.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
