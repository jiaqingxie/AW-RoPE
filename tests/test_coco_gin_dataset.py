"""Check COCO labels/splits and refuse incomplete offline assets."""
import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from aw_rope.experiments.datasets import load_dataset
from aw_rope.experiments.models import GraphPredictionModel


def save_split(root, split, label, legacy=False):
    path = root / "lrgb" / "coco-sp" / "processed" / f"{split}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = Data(x=torch.ones(3, 14), y=torch.tensor([0, label, 80]),
                 edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
    if legacy:
        torch.save(InMemoryDataset.collate([graph, graph.clone()]), path)
    else:
        InMemoryDataset.save([graph, graph.clone()], path)


@pytest.mark.parametrize("legacy", [False, True])
def test_coco_official_splits_and_81_class_models(tmp_path, monkeypatch, legacy):
    monkeypatch.setattr("torch_geometric.datasets.LRGBDataset.download",
                        lambda self: pytest.fail("processed-only assets must stay offline"))
    for split, label in (("train", 3), ("val", 4), ("test", 5)):
        save_split(tmp_path, split, label, legacy=legacy)
    torch_load = torch.load
    def require_explicit_dataset_loading(*args, **kwargs):
        assert kwargs.get("weights_only") is False
        return torch_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", require_explicit_dataset_loading)
    bundle = load_dataset("coco-sp", tmp_path)
    assert bundle.output_dim == 81 and bundle.metric == "macro-f1"
    assert [split[0].y[1].item() for split in (bundle.train, bundle.validation, bundle.test)] == [3, 4, 5]
    for method in ("nope", "aw"):
        model = GraphPredictionModel(bundle.input_dim, bundle.output_dim,
                                     method=method, task=bundle.task, num_steps=2)
        output = model(bundle.train[0])
        assert output.shape == (3, 81)
        loss = torch.nn.functional.cross_entropy(output, bundle.train[0].y)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_missing_validation_is_rejected_before_pyg_download(tmp_path, monkeypatch):
    save_split(tmp_path, "train", 3)
    monkeypatch.setattr("torch_geometric.datasets.LRGBDataset.download",
                        lambda self: pytest.fail("offline loader attempted download"))
    with pytest.raises(FileNotFoundError, match="val.pt"):
        load_dataset("coco-sp", tmp_path)
