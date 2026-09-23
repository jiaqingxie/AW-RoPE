"""Regression for subset-index leakage in the prepared MalNet loader."""
import ast
from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset


def join_function():
    source = Path(__file__).resolve().parents[1] / 'external/Graph-RoPE/graphgps/loader/master_loader.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'join_dataset_splits')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[node.name]


def dataset():
    d = InMemoryDataset()
    d.data, d.slices = d.collate([Data(x=torch.tensor([[float(i)]]), y=torch.tensor([i])) for i in range(9)])
    return d


def test_subset_indices_are_respected_without_running_transform():
    d = dataset()
    d.transform = lambda _: (_ for _ in ()).throw(AssertionError('runtime transform executed'))
    splits = [d[[5, 1, 8]], d[[4, 2]], d[[7, 0, 3, 6]]]
    joined = join_function()(splits)
    assert [joined.get(i).y.item() for i in range(9)] == [5, 1, 8, 4, 2, 7, 0, 3, 6]
    assert joined.split_idxs == [[0, 1, 2], [3, 4], [5, 6, 7, 8]]


def test_independent_stores_keep_original_order():
    splits = [dataset(), dataset(), dataset()]
    joined = join_function()(splits)
    assert [joined.get(i).y.item() for i in range(27)] == list(range(9)) * 3
