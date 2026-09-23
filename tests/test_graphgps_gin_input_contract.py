from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym.config import cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROPE_ROOT = PROJECT_ROOT / "external" / "Graph-RoPE"
if str(GRAPH_ROPE_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPH_ROPE_ROOT))


def test_pattern_keeps_official_wire_preformat_then_casts_at_shared_encoder(
    monkeypatch,
) -> None:
    from graphgps.loader import master_loader
    from graphgps.config.posenc_config import set_cfg_posenc
    from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder

    class TinySplit(InMemoryDataset):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__(root=None)
            graph = Data(
                x=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                y=torch.tensor([0, 1]),
            )
            self.data, self.slices = self.collate([graph])

    monkeypatch.setattr(master_loader, "GNNBenchmarkDataset", TinySplit)
    dataset = master_loader.preformat_GNNBenchmarkDataset("unused", "PATTERN")

    # The serialized/preformatted representation remains byte-compatible with
    # the official Graph-RoPE/WIRE loader.
    assert dataset[0].x.dtype == torch.long

    cfg.defrost()
    if not hasattr(cfg, "posenc_LapPE"):
        set_cfg_posenc(cfg)
    cfg.share.dim_in = 3
    cfg.posenc_LapPE.dim_pe = 2
    cfg.posenc_LapPE.model = "DeepSet"
    cfg.posenc_LapPE.layers = 2
    cfg.posenc_LapPE.n_heads = 1
    cfg.posenc_LapPE.post_layers = 0
    cfg.posenc_LapPE.eigen.max_freqs = 2
    cfg.posenc_LapPE.raw_norm_type = "none"
    cfg.posenc_LapPE.pass_as_var = False
    encoder = LapPENodeEncoder(emb_dim=6).eval()
    batch = dataset[0]
    batch.EigVecs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    batch.EigVals = torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])

    encoded = encoder(batch)

    assert encoded.x.dtype == encoder.linear_x.weight.dtype
    assert encoded.x.shape == (2, 6)


def test_linear_edge_accepts_aw_prepared_lowercase_names_and_casts_float() -> None:
    from graphgps.encoder.linear_edge_encoder import LinearEdgeEncoder

    cfg.defrost()
    for name in ("mnist", "cifar10"):
        cfg.dataset.name = name
        encoder = LinearEdgeEncoder(8)
        batch = Data(edge_attr=torch.tensor([1, 2, 3], dtype=torch.long))

        encoded = encoder(batch)

        assert encoded.edge_attr.shape == (3, 8)
        assert encoded.edge_attr.dtype == encoder.encoder.weight.dtype


def test_voc_edge_encoder_maps_aw_prepared_name_to_two_continuous_features() -> None:
    from graphgps.encoder.voc_superpixels_encoder import VOCEdgeEncoder

    cfg.defrost()
    cfg.dataset.name = "pascalvoc-sp"
    encoder = VOCEdgeEncoder(8)
    batch = Data(edge_attr=torch.tensor([[1, 2], [3, 4]], dtype=torch.long))

    encoded = encoder(batch)

    assert encoder.encoder.in_features == 2
    assert encoded.edge_attr.shape == (2, 8)
    assert encoded.edge_attr.dtype == encoder.encoder.weight.dtype
