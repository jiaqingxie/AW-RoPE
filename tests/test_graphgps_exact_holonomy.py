from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "external" / "Graph-RoPE"
sys.path.insert(0, str(OFFICIAL_ROOT))

from graphgps.layer.graphrope import GraphRoPE, resolve_positional_method  # noqa: E402
from graphgps.layer.gps_layer import GPSLayer  # noqa: E402


def _aw_config() -> SimpleNamespace:
    return SimpleNamespace(
        field_hidden_dim=4,
        max_displacement=3.141592653589793,
        frequency_base=10_000.0,
        field_type="local-antisymmetric",
        position_dim=3,
        exact=SimpleNamespace(
            diffusion_time=1.5,
            eps=1e-12,
            soft_phase_normalization=False,
            heat_kernel_method="matrix_exp",
            learnable_frequencies=False,
            precomputed=True,
            field_protocol="topology-rwdiag-skew-v1",
        ),
    )


def _learnable_aw_config() -> SimpleNamespace:
    config = _aw_config()
    config.exact.precomputed = False
    config.exact.learnable_frequencies = True
    return config


def _bidirectional(edges: list[tuple[int, int]]) -> torch.Tensor:
    forward = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat((forward, forward.flip(0)), dim=1)


def test_exact_holonomy_aliases() -> None:
    assert resolve_positional_method("exact", True) == "exact-holonomy"
    assert resolve_positional_method("holonomy-rope", True) == "exact-holonomy"


def test_graphrope_exact_handles_variable_pyg_graphs_and_backward() -> None:
    torch.manual_seed(9)
    graph_a = Data(
        x=torch.randn(3, 8),
        edge_index=_bidirectional([(0, 1), (1, 2), (2, 0)]),
        exact_holonomy_transport=torch.ones(3 * 3, 2, dtype=torch.complex64),
    )
    graph_b = Data(
        x=torch.randn(4, 8),
        edge_index=_bidirectional([(0, 1), (1, 2), (2, 3)]),
        exact_holonomy_transport=torch.ones(4 * 4, 2, dtype=torch.complex64),
    )
    batch = Batch.from_data_list((graph_a, graph_b))
    batch.x.requires_grad_(True)
    layer = GraphRoPE(
        k=0,
        d=8,
        num_heads=2,
        dropout=0.0,
        positional_method="exact-holonomy",
        attn_type="Full",
        aw_cfg=_aw_config(),
    )

    output = layer(batch)
    assert output.shape == (7, 8)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert batch.x.grad is not None and torch.isfinite(batch.x.grad).all()
    assert list(layer.exact_holonomy.parameters()) == []


def test_exact_graphrope_rejects_missing_offline_sidecar() -> None:
    graph = Data(
        x=torch.randn(3, 8),
        edge_index=_bidirectional([(0, 1), (1, 2)]),
    )
    batch = Batch.from_data_list((graph,))
    layer = GraphRoPE(
        k=0,
        d=8,
        num_heads=2,
        dropout=0.0,
        positional_method="exact-holonomy",
        attn_type="Full",
        aw_cfg=_aw_config(),
    )
    try:
        layer(batch)
    except RuntimeError as error:
        assert "offline" in str(error)
    else:
        raise AssertionError("exact Holonomy training must require a sidecar")


def test_learnable_exact_holonomy_backpropagates_to_edge_field_and_frequencies() -> None:
    torch.manual_seed(12)
    graph_a = Data(
        x=torch.randn(3, 8),
        edge_index=_bidirectional([(0, 1), (1, 2), (2, 0)]),
    )
    graph_b = Data(
        x=torch.randn(4, 8),
        edge_index=_bidirectional([(0, 1), (1, 2), (2, 3), (3, 0)]),
    )
    batch = Batch.from_data_list((graph_a, graph_b))
    layer = GraphRoPE(
        k=0,
        d=8,
        num_heads=2,
        dropout=0.0,
        positional_method="exact-holonomy",
        attn_type="Full",
        aw_cfg=_learnable_aw_config(),
    )
    output = layer(batch)
    assert output.shape == (7, 8)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    exact = layer.exact_holonomy
    assert exact.precomputed is False
    assert exact.frequencies.grad is not None
    assert torch.isfinite(exact.frequencies.grad).all()
    field_grads = [parameter.grad for parameter in exact.edge_field.parameters()]
    assert field_grads and all(gradient is not None for gradient in field_grads)
    assert all(torch.isfinite(gradient).all() for gradient in field_grads)


def test_learnable_exact_holonomy_uses_one_canonical_copy_per_edge() -> None:
    config = _learnable_aw_config()
    layer = GraphRoPE(
        k=0,
        d=8,
        num_heads=2,
        dropout=0.0,
        positional_method="exact-holonomy",
        attn_type="Full",
        aw_cfg=config,
    )
    directed = _bidirectional([(0, 1), (1, 2), (2, 0)])
    canonical, _ = layer.exact_holonomy._canonical_edges(directed, 3)
    assert canonical.shape == (2, 3)
    assert {tuple(edge) for edge in canonical.t().tolist()} == {(0, 1), (0, 2), (1, 2)}


def test_batched_learnable_heat_kernel_matches_individual_graphs() -> None:
    torch.manual_seed(31)
    graph_a = Data(x=torch.randn(3, 8), edge_index=_bidirectional([(0, 1), (1, 2), (2, 0)]))
    graph_b = Data(x=torch.randn(3, 8), edge_index=_bidirectional([(0, 1), (1, 2)]))
    batch = Batch.from_data_list((graph_a, graph_b))
    layer = GraphRoPE(
        k=0, d=8, num_heads=2, dropout=0.0,
        positional_method="exact-holonomy", attn_type="Full",
        aw_cfg=_learnable_aw_config(),
    )
    exact = layer.exact_holonomy
    batched, mask = exact.pairwise_transport(batch.x, batch.edge_index, batch.batch)
    individual_a = exact._graph_transport(graph_a.x, graph_a.edge_index, None)
    individual_b = exact._graph_transport(graph_b.x, graph_b.edge_index, None)
    assert mask.all()
    torch.testing.assert_close(batched[0], individual_a, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(batched[1], individual_b, rtol=2e-5, atol=2e-5)


def test_gps_layer_preserves_exact_transport_in_graphrope_temporary_batch() -> None:
    graph_cfg = SimpleNamespace(
        t_dim=0,
        enable=True,
        init_omega="uniform",
        attn_type="Full",
        double_omega=False,
        freeze_omega=False,
        method="exact-holonomy",
        aw=_aw_config(),
    )
    layer = GPSLayer(
        dim_h=8,
        local_gnn_type="None",
        global_model_type="GraphRoPE",
        num_heads=2,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=False,
        graphrope_cfg=graph_cfg,
        graphrope_method_override="exact-holonomy",
    )
    graph = Data(
        x=torch.randn(3, 8),
        edge_index=_bidirectional([(0, 1), (1, 2)]),
        exact_holonomy_transport=torch.ones(3 * 3, 2, dtype=torch.complex64),
    )
    batch = Batch.from_data_list((graph,))
    batch._exact_holonomy_geometry_cache = {}
    output = layer(batch)
    assert output.x.shape == (3, 8)
    assert torch.isfinite(output.x).all()
