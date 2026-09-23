"""Contract tests for AW-RoPE integrated into the official WIRE layer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch_geometric.data import Batch, Data


def _load_official_layers():
    """Load only the two dependency-light official layer modules.

    Graph-RoPE's package initializer imports the entire GraphGPS training
    stack.  Unit tests for full attention intentionally avoid those optional
    runtime dependencies.
    """
    layer_root = Path(__file__).parents[1] / "external" / "Graph-RoPE" / "graphgps" / "layer"
    graphgps = types.ModuleType("official_graphgps")
    graphgps.__path__ = [str(layer_root.parent)]
    layer = types.ModuleType("official_graphgps.layer")
    layer.__path__ = [str(layer_root)]
    sys.modules.setdefault("official_graphgps", graphgps)
    sys.modules.setdefault("official_graphgps.layer", layer)
    modules = []
    for name in ("aw_rope", "graphrope"):
        qualified = f"official_graphgps.layer.{name}"
        if qualified not in sys.modules:
            spec = importlib.util.spec_from_file_location(qualified, layer_root / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
        modules.append(sys.modules[qualified])
    return modules


AW, WIRE = _load_official_layers()


def _cycle(num_nodes: int) -> torch.Tensor:
    source = torch.arange(num_nodes)
    target = source.roll(-1)
    directed = torch.stack((source, target))
    return torch.cat((directed, directed.flip(0)), dim=1).long()


def _data(num_nodes: int, dim: int = 8, position_dim: int = 4) -> Data:
    return Data(
        x=torch.randn(num_nodes, dim),
        edge_index=_cycle(num_nodes),
        t=torch.randn(num_nodes, position_dim),
    )


def test_wire_rotate_is_pairwise_complex_rotation() -> None:
    x = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
    angle = torch.tensor([[torch.pi / 2, -torch.pi]])
    actual = WIRE.rotate(x, torch.sin(angle), torch.cos(angle))
    expected = torch.tensor([[1.0, 2.0, -0.5, -3.0]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_legacy_enable_flag_still_selects_wire_and_nope() -> None:
    assert WIRE.resolve_positional_method(None, True) == "wire"
    assert WIRE.resolve_positional_method("", False) == "none"
    assert WIRE.resolve_positional_method("multiscale-nb", True) == "aw-nb-ms"
    assert WIRE.resolve_positional_method("lr-crf-aw-rope", True) == "lr-aw"
    with pytest.raises(ValueError, match="unknown positional method"):
        WIRE.resolve_positional_method("laplacian-magic", True)


def test_zero_initialized_wire_exactly_matches_same_backbone_nope() -> None:
    torch.manual_seed(7)
    nope = WIRE.GraphRoPE(4, 8, 2, dropout=0.0, positional_method="none")
    wire = WIRE.GraphRoPE(
        4, 8, 2, dropout=0.0, positional_method="wire", init_omega="zero"
    )
    wire.WQKV.load_state_dict(nope.WQKV.state_dict())
    wire.WO.load_state_dict(nope.WO.state_dict())
    batch = Batch.from_data_list((_data(3), _data(5)))
    nope.eval()
    wire.eval()
    torch.testing.assert_close(wire(batch), nope(batch), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "method", ["none", "wire", "aw", "aw-nb", "aw-nb-ms", "lr-aw"]
)
def test_attention_has_no_cross_graph_leakage(method: str) -> None:
    torch.manual_seed(11)
    first, second = _data(3), _data(6)
    model = WIRE.GraphRoPE(
        4,
        8,
        2,
        dropout=0.0,
        positional_method=method,
        aw_cfg=types.SimpleNamespace(num_steps=2),
    ).eval()
    alone = model(Batch.from_data_list((first.clone(),)))
    together = model(Batch.from_data_list((first.clone(), second)))[: first.num_nodes]
    torch.testing.assert_close(together, alone, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "method", ["none", "wire", "aw", "aw-nb", "aw-nb-ms", "lr-aw"]
)
def test_official_performer_path_is_finite(method: str) -> None:
    pytest.importorskip("performer_pytorch")
    torch.manual_seed(13)
    batch = Batch.from_data_list((_data(3), _data(5)))
    model = WIRE.GraphRoPE(
        4,
        8,
        2,
        dropout=0.0,
        attn_type="Linear",
        positional_method=method,
        aw_cfg=types.SimpleNamespace(num_steps=2),
    )
    output = model(batch)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_graph_performer_kernel_is_explicit_and_defaults_to_official_favor() -> None:
    pytest.importorskip("performer_pytorch")
    official = WIRE.GraphRoPE(
        0, 8, 2, dropout=0.0, attn_type="Linear",
        positional_method="aw", aw_cfg=types.SimpleNamespace(num_steps=2),
    )
    relu_variant = WIRE.GraphRoPE(
        0, 8, 2, dropout=0.0, attn_type="Linear",
        positional_method="aw",
        aw_cfg=types.SimpleNamespace(num_steps=2, performer_kernel="relu"),
    )
    assert official.performer_kernel == "softmax-favor+"
    assert official.attention.generalized_attention is False
    assert relu_variant.performer_kernel == "generalized-relu"
    assert relu_variant.attention.generalized_attention is True


def test_b_rezero_aw_performer_starts_as_exact_paired_nope() -> None:
    pytest.importorskip("performer_pytorch")
    batch = Batch.from_data_list((_data(4), _data(5)))
    torch.manual_seed(137)
    nope = WIRE.GraphRoPE(
        0, 8, 2, dropout=0.0, attn_type="Linear",
        positional_method="none",
    )
    torch.manual_seed(137)
    b_rezero = WIRE.GraphRoPE(
        0, 8, 2, dropout=0.0, attn_type="Linear",
        positional_method="aw",
        aw_cfg=types.SimpleNamespace(num_steps=2, z=0.8, rezero=True),
    )
    assert b_rezero.aw_rezero is True
    assert b_rezero.aw_rezero_gate.item() == 0.0
    shared_nope = nope.state_dict()
    shared_b = b_rezero.state_dict()
    for name, value in shared_nope.items():
        assert name in shared_b
        torch.testing.assert_close(value, shared_b[name], atol=0.0, rtol=0.0)
    nope.eval()
    b_rezero.eval()
    expected = nope(batch)
    actual = b_rezero(batch)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    actual.square().mean().backward()
    assert b_rezero.aw_rezero_gate.grad is not None
    assert torch.isfinite(b_rezero.aw_rezero_gate.grad)
    assert b_rezero.aw_rezero_gate.grad.abs().item() > 0


def test_b_rezero_aw_initialization_preserves_paired_backbone_rng() -> None:
    pytest.importorskip("performer_pytorch")
    torch.manual_seed(139)
    WIRE.GraphRoPE(
        0, 8, 2, dropout=0.0, attn_type="Linear",
        positional_method="none",
    )
    nope_tail = torch.nn.Linear(8, 8)
    torch.manual_seed(139)
    WIRE.GraphRoPE(
        0, 8, 2, dropout=0.0, attn_type="Linear",
        positional_method="aw",
        aw_cfg=types.SimpleNamespace(num_steps=2, z=0.8, rezero=True),
    )
    b_tail = torch.nn.Linear(8, 8)
    torch.testing.assert_close(nope_tail.weight, b_tail.weight, atol=0.0, rtol=0.0)
    torch.testing.assert_close(nope_tail.bias, b_tail.bias, atol=0.0, rtol=0.0)


def test_low_rank_aw_performer_uses_anchor_lifted_projection() -> None:
    pytest.importorskip("performer_pytorch")
    model = WIRE.GraphRoPE(
        0,
        8,
        2,
        dropout=0.0,
        attn_type="Linear",
        positional_method="lr-aw",
        aw_cfg=types.SimpleNamespace(
            low_rank=types.SimpleNamespace(rank=3, num_steps=2)
        ),
    )
    assert model.attention.dim_heads == 12
    assert model.attention.projection_matrix.shape[1] == 12
    assert model.attention.nb_features == int(4 * torch.log(torch.tensor(4.0)))


def test_pointcloud_official_gt_supports_aw_rope_performer(monkeypatch) -> None:
    pytest.importorskip("performer_pytorch")
    graph_rope_root = Path(__file__).parents[1] / "external" / "Graph-RoPE"
    monkeypatch.syspath_prepend(str(graph_rope_root))
    from aw_rope.experiments.models import OfficialGraphTransformerPredictionModel

    torch.manual_seed(17)
    batch = Batch.from_data_list((
        Data(x=torch.randn(4, 6), pos=torch.randn(4, 3), edge_index=_cycle(4), y=torch.tensor([0])),
        Data(x=torch.randn(5, 6), pos=torch.randn(5, 3), edge_index=_cycle(5), y=torch.tensor([1])),
    ))
    model = OfficialGraphTransformerPredictionModel(
        6,
        3,
        hidden_dim=16,
        num_layers=2,
        heads=2,
        dropout=0.1,
        method="aw",
        num_steps=2,
        field_type="coordinate",
        attention_type="Linear",
    )
    output = model(batch)
    output.square().mean().backward()
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_wire_pyg_pct_port_supports_shapenet_segmentation(monkeypatch) -> None:
    pytest.importorskip("performer_pytorch")
    graph_rope_root = Path(__file__).parents[1] / "external" / "Graph-RoPE"
    monkeypatch.syspath_prepend(str(graph_rope_root))
    from aw_rope.experiments.models import OfficialPointCloudTransformerPredictionModel

    torch.manual_seed(29)
    batch = Batch.from_data_list((
        Data(x=torch.randn(4, 3), pos=torch.randn(4, 3), edge_index=_cycle(4), y=torch.arange(4), category=torch.tensor([2])),
        Data(x=torch.randn(5, 3), pos=torch.randn(5, 3), edge_index=_cycle(5), y=torch.arange(5), category=torch.tensor([7])),
    ))
    model = OfficialPointCloudTransformerPredictionModel(
        3,
        7,
        hidden_dim=16,
        num_layers=2,
        heads=1,
        dropout=0.5,
        method="aw",
        num_steps=2,
        field_type="coordinate",
        task="node-classification",
        attention_type="Linear",
    )
    output = model(batch)
    output.square().mean().backward()
    assert output.shape == (9, 7)
    assert torch.isfinite(output).all()


def test_wire_pyg_pct_port_supports_aw_rope_performer(monkeypatch) -> None:
    pytest.importorskip("performer_pytorch")
    graph_rope_root = Path(__file__).parents[1] / "external" / "Graph-RoPE"
    monkeypatch.syspath_prepend(str(graph_rope_root))
    from aw_rope.experiments.models import OfficialPointCloudTransformerPredictionModel

    torch.manual_seed(23)
    batch = Batch.from_data_list((
        Data(x=torch.randn(4, 3), pos=torch.randn(4, 3), edge_index=_cycle(4), y=torch.tensor([0])),
        Data(x=torch.randn(5, 3), pos=torch.randn(5, 3), edge_index=_cycle(5), y=torch.tensor([1])),
    ))
    model = OfficialPointCloudTransformerPredictionModel(
        3,
        3,
        hidden_dim=16,
        num_layers=2,
        heads=1,
        dropout=0.5,
        method="aw",
        num_steps=2,
        field_type="coordinate",
        attention_type="Linear",
    )
    output = model(batch)
    output.square().mean().backward()
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert all(layer.pointcloud_style for layer in model.layers)
    assert all(not hasattr(layer, "feed_forward") for layer in model.layers)
    assert all(layer.attention.attention.nb_features == 256 for layer in model.layers)
    assert all(
        layer.attention.attention.generalized_attention
        for layer in model.layers
    )
    assert all(
        isinstance(layer.attention.attention.kernel_fn, torch.nn.ReLU)
        for layer in model.layers
    )
    assert all(not layer.attention.attention.no_projection for layer in model.layers)
    assert all(not layer.attention.attention.causal for layer in model.layers)
    assert all(
        layer.attention.attention.projection_matrix.shape == (256, 16)
        for layer in model.layers
    )


@pytest.mark.parametrize("attention_type", ["Full", "Linear"])
def test_official_gt_supports_low_rank_aw_and_refreshes_layer_cache(
    monkeypatch, attention_type: str
) -> None:
    pytest.importorskip("yacs")
    graph_rope_root = Path(__file__).parents[1] / "external" / "Graph-RoPE"
    monkeypatch.syspath_prepend(str(graph_rope_root))
    from aw_rope.experiments.models import OfficialGraphTransformerPredictionModel

    torch.manual_seed(181)
    batch = Batch.from_data_list((
        Data(
            x=torch.randn(4, 6),
            pos=torch.randn(4, 3),
            edge_index=_cycle(4),
            y=torch.tensor([0]),
        ),
        Data(
            x=torch.randn(5, 6),
            pos=torch.randn(5, 3),
            edge_index=_cycle(5),
            y=torch.tensor([1]),
        ),
    ))
    model = OfficialGraphTransformerPredictionModel(
        6,
        3,
        hidden_dim=10,
        num_layers=2,
        heads=2,
        dropout=0.0,
        method="lr-aw",
        num_steps=2,
        field_type="coordinate",
        attention_type=attention_type,
    )
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        output = model(batch)
        output.square().mean().backward()
        assert output.shape == (2, 3)
        assert len(batch._lr_aw_geometry_cache) == 1
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_aw_zero_walk_steps_matches_same_backbone_nope() -> None:
    torch.manual_seed(19)
    nope = WIRE.GraphRoPE(4, 8, 2, dropout=0.0, positional_method="none")
    aw = WIRE.GraphRoPE(
        4,
        8,
        2,
        dropout=0.0,
        positional_method="aw",
        aw_cfg=types.SimpleNamespace(num_steps=0),
    )
    aw.WQKV.load_state_dict(nope.WQKV.state_dict())
    aw.WO.load_state_dict(nope.WO.state_dict())
    batch = Batch.from_data_list((_data(4), _data(5)))
    nope.eval()
    aw.eval()
    torch.testing.assert_close(aw(batch), nope(batch), atol=1e-6, rtol=1e-6)


def test_low_rank_zero_walk_steps_matches_nope_without_extra_parameters() -> None:
    torch.manual_seed(191)
    nope = WIRE.GraphRoPE(0, 10, 2, dropout=0.0, positional_method="none")
    low_rank = WIRE.GraphRoPE(
        0,
        10,
        2,
        dropout=0.0,
        positional_method="lr-aw",
        aw_cfg=types.SimpleNamespace(
            low_rank=types.SimpleNamespace(num_steps=0, num_bands=2)
        ),
    )
    low_rank.WQKV.load_state_dict(nope.WQKV.state_dict())
    low_rank.WO.load_state_dict(nope.WO.state_dict())
    assert sum(parameter.numel() for parameter in low_rank.parameters()) == sum(
        parameter.numel() for parameter in nope.parameters()
    )
    batch = Batch.from_data_list((_data(4, dim=10), _data(5, dim=10)))
    low_rank.eval()
    nope.eval()
    actual = low_rank(batch)
    torch.testing.assert_close(actual, nope(batch), atol=2e-6, rtol=2e-6)
    actual.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in low_rank.parameters()
    )


def test_inactive_aw_keeps_parameters_but_matches_nope() -> None:
    torch.manual_seed(20)
    nope = WIRE.GraphRoPE(4, 8, 2, dropout=0.0, positional_method="none")
    inactive = WIRE.GraphRoPE(
        4,
        8,
        2,
        dropout=0.0,
        positional_method="aw-nb",
        apply_positional=False,
        aw_cfg=types.SimpleNamespace(num_steps=2),
    )
    inactive.WQKV.load_state_dict(nope.WQKV.state_dict())
    inactive.WO.load_state_dict(nope.WO.state_dict())
    assert hasattr(inactive, "aw_rope")
    assert sum(parameter.numel() for parameter in inactive.parameters()) > sum(
        parameter.numel() for parameter in nope.parameters()
    )
    batch = Batch.from_data_list((_data(4), _data(5)))
    nope.eval()
    inactive.eval()
    torch.testing.assert_close(inactive(batch), nope(batch), atol=1e-6, rtol=1e-6)


def test_fused_attention_matches_official_manual_logits_path() -> None:
    torch.manual_seed(21)
    fused = WIRE.GraphRoPE(4, 8, 2, dropout=0.0, positional_method="wire")
    manual = WIRE.GraphRoPE(
        4, 8, 2, dropout=0.0, positional_method="wire", return_logits=True
    )
    manual.load_state_dict(fused.state_dict())
    batch = Batch.from_data_list((_data(2), _data(5)))
    fused.eval()
    manual.eval()
    expected, scores = manual(batch)
    actual = fused(batch)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert scores.shape == (2, 2, 5, 5)


def test_aw_field_is_antisymmetric_and_training_is_finite() -> None:
    torch.manual_seed(23)
    edge_index = _cycle(5)
    x = torch.randn(5, 8, requires_grad=True)
    module = AW.AnalyticWalkRoPE(8, method="aw-nb-ms", num_steps=3)
    displacement = module.edge_field(x, edge_index)
    edge_lookup = {
        (int(source), int(target)): index
        for index, (source, target) in enumerate(edge_index.t())
    }
    for index, (source, target) in enumerate(edge_index.t()):
        reverse = edge_lookup[(int(target), int(source))]
        torch.testing.assert_close(displacement[index], -displacement[reverse])

    query, key = module(x, x.roll(1, 0), x, edge_index)
    loss = query.square().mean() + key.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_aw_zero_residual_mix_returns_unmodified_query_and_key() -> None:
    torch.manual_seed(24)
    edge_index = _cycle(5)
    query = torch.randn(5, 8)
    key = torch.randn(5, 8)
    node_features = torch.randn(5, 8)
    module = AW.AnalyticWalkRoPE(8, num_steps=2, residual_mix=0.0)
    actual_query, actual_key = module(query, key, node_features, edge_index)
    torch.testing.assert_close(actual_query, query, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_key, key, atol=0.0, rtol=0.0)


def test_aw_norm_preservation_matches_each_input_vector_norm() -> None:
    torch.manual_seed(25)
    edge_index = _cycle(5)
    query = torch.randn(5, 8)
    key = torch.randn(5, 8)
    node_features = torch.randn(5, 8)
    module = AW.AnalyticWalkRoPE(
        8,
        num_steps=2,
        residual_mix=0.5,
        preserve_input_norm=True,
    )
    actual_query, actual_key = module(query, key, node_features, edge_index)
    torch.testing.assert_close(
        actual_query.norm(dim=-1), query.norm(dim=-1), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        actual_key.norm(dim=-1), key.norm(dim=-1), atol=1e-6, rtol=1e-6
    )


def test_aw_grouped_norm_preservation_matches_each_rope_pair() -> None:
    torch.manual_seed(26)
    edge_index = _cycle(5)
    query = torch.randn(5, 8)
    key = torch.randn(5, 8)
    node_features = torch.randn(5, 8)
    module = AW.AnalyticWalkRoPE(
        8,
        num_steps=2,
        preserve_input_norm=True,
        norm_group_size=2,
    )
    actual_query, actual_key = module(query, key, node_features, edge_index)
    torch.testing.assert_close(
        actual_query.reshape(5, 4, 2).norm(dim=-1),
        query.reshape(5, 4, 2).norm(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        actual_key.reshape(5, 4, 2).norm(dim=-1),
        key.reshape(5, 4, 2).norm(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_aw_rejects_invalid_residual_mix() -> None:
    with pytest.raises(ValueError, match="residual_mix"):
        AW.AnalyticWalkRoPE(8, residual_mix=1.01)


def test_aw_rejects_norm_group_size_that_does_not_divide_dim() -> None:
    with pytest.raises(ValueError, match="norm_group_size"):
        AW.AnalyticWalkRoPE(8, norm_group_size=3)


def test_low_rank_field_is_parameter_free_and_antisymmetric() -> None:
    torch.manual_seed(231)
    edge_index = _cycle(5)
    values = torch.randn(5, 7)
    field = AW.ParameterFreeEdgeField("features", torch.pi)
    displacement = field(values, edge_index)
    assert sum(parameter.numel() for parameter in field.parameters()) == 0
    edge_lookup = {
        (int(source), int(target)): index
        for index, (source, target) in enumerate(edge_index.t())
    }
    for index, (source, target) in enumerate(edge_index.t()):
        reverse = edge_lookup[(int(target), int(source))]
        torch.testing.assert_close(displacement[index], -displacement[reverse])


def test_low_rank_sparse_propagation_width_is_rank_not_embedding_width() -> None:
    torch.manual_seed(233)
    edge_index = _cycle(6)
    rank = 3
    z0, real, imaginary, anchors = AW.low_rank_complex_walk_features(
        edge_index,
        torch.randn(edge_index.shape[1]),
        torch.zeros(6, dtype=torch.long),
        rank=rank,
        z=0.6,
        num_steps=4,
        carrier_frequencies=torch.tensor([0.1, 0.2]),
    )
    assert z0.shape == anchors.shape == (6, rank)
    assert real.shape == imaginary.shape == (2, 6, rank)
    assert torch.isfinite(z0).all()
    assert torch.isfinite(real).all()
    assert torch.isfinite(imaginary).all()


def test_low_rank_anchors_do_not_depend_on_shuffled_batch_position() -> None:
    rank = 4
    alone = AW.sample_uniform_anchor_probes(
        torch.zeros(5, dtype=torch.long), rank, seed=17
    )
    prefixed = AW.sample_uniform_anchor_probes(
        torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]), rank, seed=17
    )
    torch.testing.assert_close(prefixed[3:], alone)


def test_low_rank_path_recovers_exact_carrier_displacement_at_finite_rank() -> None:
    node_count = 7
    source = torch.arange(node_count - 1)
    target = source + 1
    edge_index = torch.cat(
        (torch.stack((source, target)), torch.stack((target, source))), dim=1
    )
    position = torch.arange(node_count, dtype=torch.float32)
    edge_displacement = position[edge_index[1]] - position[edge_index[0]]
    actual, confidence, real_nodes = AW.low_rank_pairwise_geometry(
        edge_index,
        edge_displacement,
        torch.zeros(node_count, dtype=torch.long),
        rank=3,
        z=0.6,
        num_steps=node_count - 1,
        # The second carrier wraps at the largest separation, exercising the
        # coarse-to-fine branch selection rather than only principal phases.
        carrier_frequencies=torch.tensor([0.1, 1.3]),
    )
    expected = position[None, :] - position[:, None]
    assert real_nodes.all()
    assert torch.all(confidence > 0)
    torch.testing.assert_close(actual[0], expected, atol=2e-5, rtol=2e-5)


def test_low_rank_banded_scores_reduce_to_dot_product_at_zero_phase() -> None:
    torch.manual_seed(239)
    module = AW.LowRankComplexRandomFeatureAWRoPE(
        head_dim=5, num_steps=2, num_bands=2
    )
    query = torch.randn(2, 3, 4, 5)
    key = torch.randn_like(query)
    actual = module.attention_scores(
        query,
        key,
        torch.zeros(2, 4, 4),
        torch.ones(2, 4, 4),
    )
    expected = torch.matmul(query, key.transpose(-2, -1)) / (5 ** 0.5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_low_rank_banded_score_uses_bar_query_times_key_phase_sign() -> None:
    module = AW.LowRankComplexRandomFeatureAWRoPE(head_dim=2, num_bands=1)
    query = torch.tensor([[[[0.0, 1.0]]]])  # q = i
    key = torch.tensor([[[[1.0, 0.0]]]])  # k = 1
    angle = torch.tensor([[[0.7]]])
    actual = module.attention_scores(query, key, angle, torch.ones_like(angle))
    # Re(conj(i) * 1 * exp(i theta)) = sin(theta).
    expected = torch.sin(angle)[:, None] / (2 ** 0.5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_low_rank_performer_lift_matches_anchor_average_banded_score() -> None:
    torch.manual_seed(240)
    module = AW.LowRankComplexRandomFeatureAWRoPE(
        head_dim=5, rank=3, num_bands=2
    )
    query = torch.randn(1, 2, 4, 5)
    key = torch.randn_like(query)
    anchor_displacement = torch.randn(1, 4, 3)
    anchor_valid = torch.ones_like(anchor_displacement, dtype=torch.bool)
    lifted_query, lifted_key = module.performer_lift(
        query, key, anchor_displacement, anchor_valid
    )
    actual = torch.matmul(lifted_query, lifted_key.transpose(-2, -1))
    actual = actual / (module.rank * module.head_dim) ** 0.5

    expected = torch.zeros_like(actual)
    for anchor in range(module.rank):
        pair_displacement = (
            anchor_displacement[:, :, anchor, None]
            - anchor_displacement[:, None, :, anchor]
        )
        expected += module.attention_scores(
            query,
            key,
            pair_displacement,
            torch.ones_like(pair_displacement),
        )
    expected /= module.rank
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_low_rank_anchor_geometry_recovers_path_phase_without_pairwise_matrix() -> None:
    node_count = 7
    source = torch.arange(node_count - 1)
    target = source + 1
    edge_index = torch.cat(
        (torch.stack((source, target)), torch.stack((target, source))), dim=1
    )
    position = torch.arange(node_count, dtype=torch.float32)
    edge_displacement = position[edge_index[1]] - position[edge_index[0]]
    displacement, valid, real_nodes = AW.low_rank_anchor_geometry(
        edge_index,
        edge_displacement,
        torch.zeros(node_count, dtype=torch.long),
        rank=3,
        z=0.6,
        num_steps=node_count - 1,
        carrier_frequencies=torch.tensor([0.1, 1.3]),
    )
    assert displacement.shape == valid.shape == (1, node_count, 3)
    assert real_nodes.shape == (1, node_count)
    # For every anchor supporting both nodes, d(u,a)-d(v,a)=v-u.
    for first in range(node_count):
        for second in range(node_count):
            shared = valid[0, first] & valid[0, second]
            if shared.any():
                actual = (
                    displacement[0, first, shared]
                    - displacement[0, second, shared]
                )
                expected = torch.full_like(actual, float(second - first))
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_low_rank_training_gradient_is_finite_with_padded_graphs() -> None:
    torch.manual_seed(241)
    batch = Batch.from_data_list((_data(3, dim=10), _data(5, dim=10)))
    batch.x.requires_grad_()
    model = WIRE.GraphRoPE(
        0,
        10,
        2,
        dropout=0.0,
        positional_method="lr-aw",
        aw_cfg=types.SimpleNamespace(
            low_rank=types.SimpleNamespace(num_steps=2, num_bands=2)
        ),
    )
    output = model(batch)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert batch.x.grad is not None and torch.isfinite(batch.x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("field_type", ["potential", "coordinate"])
def test_aw_ablation_fields_are_antisymmetric(field_type: str) -> None:
    torch.manual_seed(27)
    edge_index = _cycle(5)
    module = AW.AnalyticWalkRoPE(
        8,
        method="aw",
        num_steps=1,
        field_type=field_type,
        position_dim=3,
    )
    node_features = torch.randn(5, 8)
    positions = torch.randn(5, 3)
    field_input = positions if field_type == "coordinate" else node_features
    displacement = module.edge_field(field_input, edge_index)
    edge_lookup = {
        (int(source), int(target)): index
        for index, (source, target) in enumerate(edge_index.t())
    }
    for index, (source, target) in enumerate(edge_index.t()):
        reverse = edge_lookup[(int(target), int(source))]
        torch.testing.assert_close(displacement[index], -displacement[reverse])


def test_reverse_edge_map_pairs_parallel_edges_and_missing_reverse() -> None:
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 0, 1]])
    reverse = AW.build_reverse_edge_index(edge_index, num_nodes=3)
    assert reverse.tolist() == [2, 3, 0, 1, -1]


def test_shared_multiscale_recurrence_matches_explicit_resolvent_mixture() -> None:
    torch.manual_seed(29)
    edge_index = _cycle(5)
    x = torch.randn(5, 8)
    displacement = torch.randn(edge_index.shape[1])
    frequencies = AW.standard_frequencies(8)
    z_values = torch.tensor([0.2, 0.7])
    mixture = torch.softmax(torch.tensor([-0.4, 0.9]), dim=0)
    reverse = AW.build_reverse_edge_index(edge_index, 5)
    actual = AW.truncated_multiscale_walk_resolvent(
        x,
        edge_index,
        displacement,
        frequencies,
        z_values=z_values,
        mixture=mixture,
        num_steps=4,
        reverse_edge=reverse,
    )
    expected = sum(
        beta
        * AW.truncated_walk_resolvent(
            x,
            edge_index,
            displacement,
            frequencies,
            z=z,
            num_steps=4,
            non_backtracking=True,
            reverse_edge=reverse,
        )
        for beta, z in zip(mixture, z_values)
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
