import torch

from aw_rope.static_holonomy import (
    static_topology_edge_displacement,
    topology_return_descriptor,
)


def _bidirectional(edges: list[tuple[int, int]]) -> torch.Tensor:
    forward = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat((forward, forward.flip(0)), dim=1)


def test_static_field_is_permutation_equivariant_and_antisymmetric() -> None:
    edge_index = _bidirectional(
        [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 0), (1, 4)]
    )
    edges, displacement, _ = static_topology_edge_displacement(5, edge_index)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted_index = permutation[edge_index]
    permuted_edges, permuted_displacement, _ = static_topology_edge_displacement(
        5, permuted_index
    )
    original_lookup = {}
    for (source, target), value in zip(edges.t().tolist(), displacement):
        original_lookup[(source, target)] = value
        original_lookup[(target, source)] = -value
    for (source, target), value in zip(permuted_edges.t().tolist(), permuted_displacement):
        old_source = int((permutation == source).nonzero()[0])
        old_target = int((permutation == target).nonzero()[0])
        torch.testing.assert_close(value, original_lookup[(old_source, old_target)])


def test_static_field_is_not_forced_to_be_a_gradient() -> None:
    # This asymmetric graph has non-identical structural return signatures.
    edge_index = _bidirectional(
        [(0, 1), (0, 2), (0, 4), (0, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)]
    )
    edges, displacement, _ = static_topology_edge_displacement(6, edge_index)
    incidence = torch.zeros(edges.shape[1], 6)
    rows = torch.arange(edges.shape[1])
    incidence[rows, edges[0]] = -1
    incidence[rows, edges[1]] = 1
    potential = torch.linalg.lstsq(incidence[:, 1:], displacement[:, None]).solution
    residual = (incidence[:, 1:] @ potential).squeeze(1) - displacement
    assert residual.norm() > 1e-4


def test_symmetric_nodes_do_not_receive_artificial_ids() -> None:
    cycle = _bidirectional([(0, 1), (1, 2), (2, 3), (3, 0)])
    descriptor = topology_return_descriptor(4, cycle)
    torch.testing.assert_close(descriptor, torch.zeros_like(descriptor))
    _, displacement, _ = static_topology_edge_displacement(4, cycle)
    torch.testing.assert_close(displacement, torch.zeros_like(displacement))
