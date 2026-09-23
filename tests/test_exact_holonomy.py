import torch

from aw_rope import (
    ExactHolonomyTransport,
    HolonomyRoPEAttention,
    build_connection_laplacians,
    build_rope_frequencies,
    connection_heat_kernel,
)


def undirected_path(num_nodes: int) -> torch.Tensor:
    source = torch.arange(num_nodes - 1)
    return torch.stack((source, source + 1)).long()


def test_connection_laplacian_is_hermitian_positive_semidefinite() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    displacement = torch.tensor([0.4, 0.7, 0.5], dtype=torch.float64)
    frequencies = build_rope_frequencies(8, dtype=torch.float64)
    laplacian, _ = build_connection_laplacians(
        3, edge_index, displacement, frequencies
    )

    torch.testing.assert_close(
        laplacian,
        laplacian.conj().transpose(-1, -2),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.linalg.eigvalsh(laplacian).min() >= -1e-11


def test_path_heat_phase_recovers_standard_rope() -> None:
    num_nodes = 8
    positions = torch.arange(num_nodes, dtype=torch.float64)
    edge_index = undirected_path(num_nodes)
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    module = ExactHolonomyTransport(
        rotary_dim=8,
        diffusion_time=2.0,
        eps=1e-14,
    ).double()
    transport = module(num_nodes, edge_index, displacement)
    assert isinstance(transport, torch.Tensor)

    expected = torch.exp(
        1j
        * module.frequencies[:, None, None]
        * (positions[None, None, :] - positions[None, :, None])
    )
    torch.testing.assert_close(transport, expected, atol=2e-10, rtol=2e-10)


def test_triangle_has_prescribed_nontrivial_edge_holonomy() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    displacement = torch.tensor([0.4, 0.7, 0.5], dtype=torch.float64)
    frequencies = build_rope_frequencies(8, dtype=torch.float64)
    _, edge_transport = build_connection_laplacians(
        3, edge_index, displacement, frequencies
    )

    actual = edge_transport.prod(dim=-1)
    expected = torch.exp(1j * frequencies * displacement.sum())
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.max((actual - 1).abs()) > 0.5


def test_eigh_and_matrix_exp_compute_same_heat_kernel() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    displacement = torch.tensor([0.2, -0.5, 0.9], dtype=torch.float64)
    frequencies = build_rope_frequencies(6, dtype=torch.float64)
    laplacian, _ = build_connection_laplacians(
        3, edge_index, displacement, frequencies
    )
    heat_eigh, _, _ = connection_heat_kernel(laplacian, 0.7, method="eigh")
    heat_exp, _, _ = connection_heat_kernel(
        laplacian,
        0.7,
        method="matrix_exp",
        return_eigensystem=False,
    )
    torch.testing.assert_close(heat_eigh, heat_exp, atol=2e-11, rtol=2e-11)


def test_attention_forward_precompute_and_gradients() -> None:
    torch.manual_seed(42)
    num_nodes = 5
    edge_index = undirected_path(num_nodes)
    displacement = torch.ones(num_nodes - 1, requires_grad=True)
    x = torch.randn(2, num_nodes, 24, requires_grad=True)
    attention = HolonomyRoPEAttention(
        embed_dim=24,
        num_heads=3,
        rotary_dim=8,
        diffusion_time=1.3,
        dropout=0.0,
        soft_phase_normalization=True,
        heat_kernel_method="matrix_exp",
    )

    output, weights, transport = attention(
        x,
        edge_index,
        displacement,
        return_attention=True,
        return_transport=True,
    )
    assert output.shape == (2, num_nodes, 24)
    assert weights.shape == (2, 3, num_nodes, num_nodes)
    assert transport.shape == (4, num_nodes, num_nodes)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones_like(weights[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    )

    precomputed = attention(
        x,
        edge_index,
        displacement,
        precomputed_transport=transport,
    )
    torch.testing.assert_close(output, precomputed, atol=1e-6, rtol=1e-6)

    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert displacement.grad is not None and torch.isfinite(displacement.grad).all()


def test_rejects_bidirectional_storage_for_exact_input_contract() -> None:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    displacement = torch.tensor([1.0, -1.0])
    frequencies = build_rope_frequencies(4)
    try:
        build_connection_laplacians(2, edge_index, displacement, frequencies)
    except ValueError as error:
        assert "exactly once" in str(error)
    else:
        raise AssertionError("bidirectional duplicate should have been rejected")
