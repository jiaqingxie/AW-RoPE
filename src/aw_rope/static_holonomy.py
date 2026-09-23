"""Static, topology-only connection field for precomputed Holonomy-RoPE.

The exact heat-kernel transport can be cached for an entire training run only
when its connection is fixed.  This module therefore deliberately contains no
``nn.Module`` and no learnable parameter.  It turns an undirected graph into a
permutation-equivariant antisymmetric scalar field before any model is built.

For node ``u`` let

    r_u = [log(1 + d_u), (P^2)_uu, ..., (P^8)_uu],  P = D^{-1} A.

After graph-wise channel standardization, a fixed block-skew matrix ``J`` gives

    raw_a_uv = r_u^T J r_v,
    a_uv = max_displacement * tanh(raw_a_uv / scale_G).

``J^T = -J`` makes ``a_vu = -a_uv`` exactly.  Unlike a potential difference
``s_v - s_u``, the bilinear field is generally non-flat and can have non-zero
cycle flux.  The descriptor uses closed-walk return probabilities only to
choose the *fixed connection field*; the subsequent Holonomy-RoPE kernel is
still the full analytic heat kernel, not an AW/random-walk approximation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


STATIC_FIELD_PROTOCOL = "topology-rwdiag-skew-v1"


def canonical_undirected_edges(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Return every non-self-loop undirected edge once as ``min -> max``."""
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise ValueError("edge_index contains an out-of-range node")
    source, target = edge_index
    keep = source != target
    lower = torch.minimum(source[keep], target[keep])
    upper = torch.maximum(source[keep], target[keep])
    if lower.numel() == 0:
        return edge_index.new_empty((2, 0))
    key = lower * num_nodes + upper
    order = torch.argsort(key, stable=True)
    key = key[order]
    first = torch.ones_like(key, dtype=torch.bool)
    first[1:] = key[1:] != key[:-1]
    selected = order[first]
    return torch.stack((lower[selected], upper[selected]))


def topology_return_descriptor(
    num_nodes: int,
    edge_index: Tensor,
    *,
    max_power: int = 8,
    eps: float = 1e-8,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build the fixed node descriptor ``[log(1+d), diag(P^2),...,diag(P^K)]``.

    Channels are standardized within each graph.  Constant channels become
    zero, which respects graph symmetries instead of inventing node identities.
    """
    if max_power < 2 or max_power % 2:
        raise ValueError("max_power must be an even integer >= 2")
    if eps <= 0:
        raise ValueError("eps must be positive")
    edges = canonical_undirected_edges(edge_index, num_nodes)
    device = edge_index.device
    adjacency = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    if edges.numel():
        source, target = edges
        adjacency[source, target] = 1
        adjacency[target, source] = 1
    degree = adjacency.sum(dim=1)
    transition = adjacency / degree.clamp_min(1).unsqueeze(1)
    power = transition
    channels = [torch.log1p(degree)]
    for exponent in range(2, max_power + 1):
        power = power @ transition
        channels.append(power.diagonal())
    descriptor = torch.stack(channels, dim=1)
    mean = descriptor.mean(dim=0, keepdim=True)
    std = descriptor.std(dim=0, unbiased=False, keepdim=True)
    descriptor = torch.where(std > eps, (descriptor - mean) / std.clamp_min(eps), 0)
    return descriptor


def static_topology_edge_displacement(
    num_nodes: int,
    edge_index: Tensor,
    *,
    max_power: int = 8,
    max_displacement: float = math.pi,
    eps: float = 1e-8,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return canonical edges, fixed ``a_uv``, and the node descriptor.

    The returned edge convention is one undirected edge per column.  Reversing
    an edge negates the formula exactly; callers construct the reverse complex
    transport by conjugation.
    """
    if not math.isfinite(max_displacement) or max_displacement <= 0:
        raise ValueError("max_displacement must be finite and positive")
    descriptor = topology_return_descriptor(
        num_nodes,
        edge_index,
        max_power=max_power,
        eps=eps,
        dtype=dtype,
    )
    edges = canonical_undirected_edges(edge_index, num_nodes)
    if edges.numel() == 0:
        return edges, descriptor.new_empty((0,)), descriptor
    source, target = edges
    # The adjacent 2-D blocks implement r_u^T J r_v for a fixed J^T = -J.
    source_even = descriptor[source, 0::2]
    source_odd = descriptor[source, 1::2]
    target_even = descriptor[target, 0::2]
    target_odd = descriptor[target, 1::2]
    raw = (source_even * target_odd - source_odd * target_even).sum(dim=1)
    absolute = raw.abs()
    scale = absolute.median()
    if not bool(torch.isfinite(scale)) or bool(scale <= eps):
        scale = absolute.square().mean().sqrt()
    if not bool(torch.isfinite(scale)) or bool(scale <= eps):
        scale = raw.new_tensor(1.0)
    displacement = max_displacement * torch.tanh(raw / (scale + eps))
    return edges, displacement, descriptor


def cycle_flux(edge_index: Tensor, edge_displacement: Tensor, cycle: Tensor) -> Tensor:
    """Return oriented displacement sum for a closed node sequence.

    ``cycle`` contains nodes ``[v0, ..., vk]`` with ``vk == v0``.  This small
    audit helper is intentionally not used by training.
    """
    if cycle.ndim != 1 or cycle.numel() < 4 or int(cycle[0]) != int(cycle[-1]):
        raise ValueError("cycle must be a closed 1-D node sequence")
    if edge_displacement.shape != (edge_index.shape[1],):
        raise ValueError("edge_displacement must have shape (E,)")
    lookup: dict[tuple[int, int], Tensor] = {}
    for index, (source, target) in enumerate(edge_index.detach().cpu().t().tolist()):
        value = edge_displacement[index]
        lookup[(source, target)] = value
        lookup[(target, source)] = -value
    total = edge_displacement.new_zeros(())
    nodes = cycle.detach().cpu().tolist()
    for source, target in zip(nodes[:-1], nodes[1:]):
        try:
            total = total + lookup[(source, target)]
        except KeyError as error:
            raise ValueError(f"cycle step {(source, target)} is not an edge") from error
    return total
