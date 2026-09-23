from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "Graph-RoPE"))

from graphgps.layer.gps_layer import GPSLayer
from graphgps.layer.official_performer_aw import OfficialPerformerAWRoPE


def _aw_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_steps=2,
        z=0.4,
        z_values=[0.2, 0.4, 0.6, 0.8],
        field_hidden_dim=8,
        max_displacement=3.141592653589793,
        frequency_base=10000.0,
        learnable_frequencies=True,
        learnable_z=True,
        normalize_resolvent=False,
        residual_mix=1.0,
        preserve_input_norm=False,
        norm_group_size=0,
        field_type="local-antisymmetric",
        position_dim=3,
    )


def _graphrope(method: str) -> SimpleNamespace:
    return SimpleNamespace(
        enable=method != "none",
        method=method,
        aw=_aw_cfg(),
        t_dim=10,
        init_omega="uniform",
        attn_type="Linear",
        double_omega=False,
        freeze_omega=False,
    )


def _batch() -> Batch:
    return Batch(
        x=torch.randn(7, 256),
        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
             [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]],
            dtype=torch.long,
        ),
    )


def test_official_performer_aw_preserves_paper_attention_shell() -> None:
    torch.manual_seed(0)
    module = OfficialPerformerAWRoPE(
        dim=256,
        heads=8,
        dropout=0.5,
        aw_cfg=_aw_cfg(),
        method="aw",
    )
    assert isinstance(module, SelfAttention)
    assert module.paper_contract == {
        "attention_class": "SelfAttention+AW",
        "model_dim": 256,
        "heads": 8,
        "dim_head": 64,
        "inner_dim": 512,
        "qkv_bias": False,
        "attention_dropout": 0.5,
        "performer_kernel": "softmax-favor+",
        "aw_method": "aw",
    }
    output = module(_batch())
    assert output.shape == (7, 256)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert module.aw_rope.frequencies.grad is not None


def test_gps_nope_stays_the_unmodified_official_self_attention() -> None:
    layer = GPSLayer(
        dim_h=256,
        local_gnn_type="CustomGatedGCN",
        global_model_type="Performer",
        num_heads=8,
        dropout=0.1,
        attn_dropout=0.5,
        batch_norm=True,
        graphrope_cfg=_graphrope("none"),
        graphrope_method_override="none",
    )
    assert type(layer.self_attn) is SelfAttention
    assert layer.self_attn.to_q.out_features == 512
    assert layer.self_attn.dropout.p == 0.5
    assert layer.official_performer_aw is False


def test_gps_aw_uses_official_width_performer_shell() -> None:
    layer = GPSLayer(
        dim_h=256,
        local_gnn_type="CustomGatedGCN",
        global_model_type="Performer",
        num_heads=8,
        dropout=0.1,
        attn_dropout=0.5,
        batch_norm=True,
        graphrope_cfg=_graphrope("aw"),
        graphrope_method_override="aw",
    )
    assert isinstance(layer.self_attn, OfficialPerformerAWRoPE)
    assert layer.self_attn.to_q.out_features == 512
    assert layer.self_attn.dropout.p == 0.5
    assert layer.official_performer_aw is True
