import math

import pytest
import torch

from aw_rope import (
    AWRoPE,
    MultiScaleAWRoPE,
    build_reverse_edge_index,
    edge_displacement_from_positions,
    non_backtracking_walk_resolvent,
    minimum_resolvent_steps,
    rotate_pairs,
    rotary_edge_scores,
    resolvent_tail_bound,
    truncated_complex_walk_resolvent,
    truncated_walk_resolvent,
    walk_step,
)


def bidirectional_path(num_nodes: int) -> torch.Tensor:
    forward = torch.stack((torch.arange(num_nodes - 1), torch.arange(1, num_nodes)))
    return torch.cat((forward, forward.flip(0)), dim=1).long()


def dense_transport(
    edge_index: torch.Tensor,
    displacement: torch.Tensor,
    omega: float,
    num_nodes: int,
) -> torch.Tensor:
    source, target = edge_index
    degree = torch.bincount(source, minlength=num_nodes).to(torch.float64)
    matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.complex128)
    matrix[source, target] = torch.exp(1j * omega * displacement) / degree[source]
    return matrix


def test_rotate_pairs_is_complex_multiplication() -> None:
    x = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
    angles = torch.tensor([[math.pi / 2, -math.pi]])
    actual = rotate_pairs(x, angles)
    expected = torch.tensor([[1.0, 2.0, -0.5, -3.0]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_walk_matches_dense_transport() -> None:
    edge_index = bidirectional_path(5)
    positions = torch.arange(5, dtype=torch.float64)
    displacement = edge_displacement_from_positions(positions, edge_index)
    omega = 0.73
    x_complex = torch.randn(5, 2, dtype=torch.complex128)
    x_real = torch.view_as_real(x_complex).flatten(1)
    frequencies = torch.full((2,), omega, dtype=torch.float64)

    actual = walk_step(x_real, edge_index, displacement, frequencies)
    transport = dense_transport(edge_index, displacement, omega, 5)
    expected_complex = transport @ x_complex
    expected = torch.view_as_real(expected_complex).flatten(1)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_truncated_resolvent_matches_dense_polynomial() -> None:
    edge_index = bidirectional_path(4)
    positions = torch.arange(4, dtype=torch.float64)
    displacement = edge_displacement_from_positions(positions, edge_index)
    omega, z, steps = 0.4, 0.72, 7
    x = torch.randn(4, 3, dtype=torch.complex128)
    actual = truncated_complex_walk_resolvent(
        x, edge_index, displacement, omega, z=z, num_steps=steps
    )

    transport = dense_transport(edge_index, displacement, omega, 4)
    power = torch.eye(4, dtype=torch.complex128)
    kernel = power.clone()
    for hop in range(1, steps + 1):
        power = power @ transport
        kernel = kernel + z**hop * power
    torch.testing.assert_close(actual, kernel @ x, atol=1e-11, rtol=1e-11)


def test_path_resolvent_recovers_exact_rope_phase() -> None:
    num_nodes = 7
    edge_index = bidirectional_path(num_nodes)
    positions = torch.arange(num_nodes, dtype=torch.float64)
    displacement = edge_displacement_from_positions(positions, edge_index)
    omega, z = 0.61, 0.55
    transport = dense_transport(edge_index, displacement, omega, num_nodes)
    kernel = torch.linalg.inv(torch.eye(num_nodes, dtype=torch.complex128) - z * transport)

    expected_phase = torch.exp(
        1j * omega * (positions[None, :] - positions[:, None])
    )
    normalized = kernel / kernel.abs().clamp_min(1e-14)
    torch.testing.assert_close(normalized, expected_phase, atol=1e-10, rtol=1e-10)


def test_non_backtracking_removes_immediate_return() -> None:
    edge_index = bidirectional_path(2)
    positions = torch.arange(2, dtype=torch.float64)
    displacement = edge_displacement_from_positions(positions, edge_index)
    x = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    frequencies = torch.tensor([0.8], dtype=torch.float64)

    ordinary = truncated_walk_resolvent(
        x, edge_index, displacement, frequencies, z=0.5, num_steps=2
    )
    non_backtracking = non_backtracking_walk_resolvent(
        x, edge_index, displacement, frequencies, z=0.5, num_steps=2
    )
    torch.testing.assert_close(non_backtracking[0], x[0])
    assert ordinary[0, 0] > non_backtracking[0, 0]


def test_non_backtracking_matches_explicit_path_enumeration() -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 1, 3, 2, 3], [1, 0, 2, 1, 3, 1, 3, 2]],
        dtype=torch.long,
    )
    positions = torch.tensor([0.0, 0.7, 2.0, -0.5], dtype=torch.float64)
    displacement = edge_displacement_from_positions(positions, edge_index)
    omega, z, steps = 0.63, 0.71, 4
    x_complex = torch.randn(4, dtype=torch.complex128)
    x_real = torch.view_as_real(x_complex)
    actual = non_backtracking_walk_resolvent(
        x_real,
        edge_index,
        displacement,
        torch.tensor([omega], dtype=torch.float64),
        z=z,
        num_steps=steps,
    )

    source, target = edge_index
    degree = torch.bincount(source, minlength=4).to(torch.float64)
    outgoing = [[index for index in range(source.numel()) if source[index] == node] for node in range(4)]
    expected = x_complex.clone()
    for origin in range(4):
        paths = [(origin, None, torch.ones((), dtype=torch.complex128))]
        for hop in range(1, steps + 1):
            next_paths = []
            for current, previous, amplitude in paths:
                for edge in outgoing[current]:
                    destination = int(target[edge])
                    if destination == previous:
                        continue
                    phase = torch.exp(1j * omega * displacement[edge])
                    next_amplitude = amplitude * phase / degree[current]
                    next_paths.append((destination, current, next_amplitude))
                    expected[origin] += z**hop * next_amplitude * x_complex[destination]
            paths = next_paths

    torch.testing.assert_close(
        torch.view_as_complex(actual.contiguous()), expected, atol=1e-11, rtol=1e-11
    )


def test_reverse_edges_handles_parallel_edges_and_missing_reverse() -> None:
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 0, 1]])
    reverse = build_reverse_edge_index(edge_index)
    assert reverse.tolist() == [2, 3, 0, 1, -1]


def test_permutation_equivariance() -> None:
    torch.manual_seed(1)
    edge_index = bidirectional_path(6)
    positions = torch.arange(6, dtype=torch.float64)
    displacement = edge_displacement_from_positions(positions, edge_index)
    x = torch.randn(6, 4, dtype=torch.float64)
    frequencies = torch.tensor([0.2, 0.9], dtype=torch.float64)
    expected = truncated_walk_resolvent(
        x, edge_index, displacement, frequencies, z=0.8, num_steps=5
    )

    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(6)
    permuted_x = x[permutation]
    permuted_positions = positions[permutation]
    permuted_edges = inverse[edge_index]
    permuted_displacement = edge_displacement_from_positions(permuted_positions, permuted_edges)
    actual = truncated_walk_resolvent(
        permuted_x, permuted_edges, permuted_displacement, frequencies, z=0.8, num_steps=5
    )
    torch.testing.assert_close(actual, expected[permutation], atol=1e-11, rtol=1e-11)


def test_rotary_edge_scores_match_absolute_rope() -> None:
    torch.manual_seed(2)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
    positions = torch.tensor([0.0, 2.0, 5.0])
    displacement = edge_displacement_from_positions(positions, edge_index)
    frequencies = torch.tensor([0.2, 0.8])
    query, key = torch.randn(3, 4), torch.randn(3, 4)

    actual = rotary_edge_scores(
        query, key, edge_index, displacement, frequencies, scale=False
    )
    absolute_angles = positions[:, None] * frequencies[None, :]
    rotated_query = rotate_pairs(query, absolute_angles)
    rotated_key = rotate_pairs(key, absolute_angles)
    source, target = edge_index
    expected = (rotated_query[source] * rotated_key[target]).sum(-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_modules_are_differentiable_and_stable() -> None:
    edge_index = bidirectional_path(5)
    displacement = edge_displacement_from_positions(torch.arange(5.0), edge_index)
    x = torch.randn(5, 6, requires_grad=True)
    model = AWRoPE(6, num_steps=4, learnable_frequencies=True)
    multiscale = MultiScaleAWRoPE(6, num_steps=4)
    loss = model(x, edge_index, displacement).square().mean()
    loss = loss + multiscale(x, edge_index, displacement).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert 0.0 < float(model.z) < 1.0
    assert torch.all((multiscale.z_values > 0) & (multiscale.z_values < 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_training_path_is_finite(dtype: torch.dtype) -> None:
    edge_index = bidirectional_path(5)
    displacement = edge_displacement_from_positions(
        torch.arange(5, dtype=dtype), edge_index
    )
    x = torch.randn(5, 8, dtype=dtype, requires_grad=True)
    model = AWRoPE(8, num_steps=3, non_backtracking=True).to(dtype=dtype)
    output = model(x, edge_index, displacement)
    output.float().square().mean().backward()
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_tail_bound_and_step_selection() -> None:
    for z in (0.3, 0.6, 0.9, 0.99):
        steps = minimum_resolvent_steps(z, 1e-3)
        assert float(resolvent_tail_bound(z, steps)) <= 1e-3
        if steps:
            assert float(resolvent_tail_bound(z, steps - 1)) > 1e-3


def test_multiscale_non_backtracking_matches_explicit_mixture() -> None:
    edge_index = bidirectional_path(4)
    positions = torch.arange(4, dtype=torch.float64)
    displacement = edge_displacement_from_positions(positions, edge_index)
    x = torch.randn(4, 4, dtype=torch.float64)
    model = MultiScaleAWRoPE(
        4,
        z_values=(0.2, 0.7),
        num_steps=4,
        normalized_mixture=True,
        non_backtracking=True,
    ).double()
    with torch.no_grad():
        model.mixture_logits.copy_(torch.tensor([-0.4, 0.9], dtype=torch.float64))
    actual = model(x, edge_index, displacement)
    expected = sum(
        beta
        * non_backtracking_walk_resolvent(
            x,
            edge_index,
            displacement,
            model.frequencies,
            z=z,
            num_steps=4,
        )
        for beta, z in zip(model.mixture, model.z_values)
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("z", [-0.1, 1.0, 1.1])
def test_invalid_resolvent_radius_is_rejected(z: float) -> None:
    edge_index = bidirectional_path(2)
    displacement = torch.tensor([1.0, -1.0])
    with pytest.raises(ValueError, match="z must satisfy"):
        truncated_walk_resolvent(
            torch.ones(2, 2), edge_index, displacement, torch.ones(1), z=z
        )
