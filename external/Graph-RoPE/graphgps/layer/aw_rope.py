"""Sparse Analytic-Walk rotary transport for GraphRoPE attention.

This module deliberately has no GraphGym dependency.  It operates on the
sparse node ordering used by PyG and can therefore be tested independently of
the training stack.  For directed edges ``u -> v`` it constructs

    T_omega[u, v] = P[u, v] exp(i * omega * a[u, v])

and applies a truncated analytic resolvent to query and key features:

    sum_{k=0}^K z^k T_omega^k X.

The edge displacement is antisymmetric by construction and is learned from
the current node representations.  Equivalently, with the phase-lifted
random-walk Laplacian ``L_omega = I - T_omega``, the resolvent is a polynomial
in ``L_omega`` applied to the current features.  This operator use of a
Laplacian is different from LapPE/LapRoPE: no Laplacian eigenvectors or dense
positional matrix are constructed here.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn


def rotate_pairs(x: Tensor, angles: Tensor) -> Tensor:
    """Rotate adjacent real feature pairs by ``angles``."""
    if x.ndim != 2 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError(f"x must have shape (N, positive even d), got {tuple(x.shape)}")
    real, imaginary = x[:, 0::2], x[:, 1::2]
    cosine, sine = torch.cos(angles), torch.sin(angles)
    return torch.stack(
        (real * cosine - imaginary * sine,
         real * sine + imaginary * cosine),
        dim=-1,
    ).flatten(-2)


def standard_frequencies(dim: int, base: float = 10_000.0) -> Tensor:
    """Return the standard RoPE frequencies ``base ** (-2i / dim)``."""
    if dim <= 0 or dim % 2:
        raise ValueError(f"dim must be a positive even integer, got {dim}")
    pair = torch.arange(dim // 2, dtype=torch.get_default_dtype())
    return base ** (-2.0 * pair / dim)


def _validate_edges(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be a long tensor with shape (2, E)")
    # Avoid a device synchronization in every attention layer.  CPU inputs
    # receive the friendly bounds error; CUDA gather/scatter operations still
    # fail loudly if an invalid edge somehow passes the data loader.
    if (not edge_index.is_cuda and edge_index.numel()
            and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes)):
        raise ValueError("edge_index contains a node outside the input node range")


def _transition_weights(
    edge_index: Tensor,
    num_nodes: int,
    reference: Tensor,
    edge_weight: Tensor | None,
) -> Tensor:
    source = edge_index[0]
    if edge_weight is None:
        weight = torch.ones(source.numel(), dtype=reference.dtype, device=reference.device)
    else:
        if edge_weight.ndim != 1 or edge_weight.numel() != source.numel():
            raise ValueError("edge_weight must have shape (E,)")
        if not edge_weight.is_cuda and bool(torch.any(edge_weight < 0)):
            raise ValueError("edge_weight must be non-negative")
        weight = edge_weight.to(device=reference.device, dtype=reference.dtype)
    degree = torch.zeros(num_nodes, dtype=reference.dtype, device=reference.device)
    degree.index_add_(0, source, weight)
    return weight / degree[source].clamp_min(torch.finfo(reference.dtype).tiny)


def build_reverse_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Map each edge to its reverse using tensor operations only.

    Parallel edges are paired by their stable input order.  Missing reverse
    edges map to ``-1`` and self-loops map to their corresponding self-loop.
    """
    _validate_edges(edge_index, num_nodes)
    edge_count = edge_index.shape[1]
    if edge_count == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    source, target = edge_index
    pair_key = source * num_nodes + target
    reverse_pair_key = target * num_nodes + source
    order = torch.argsort(pair_key, stable=True)
    sorted_key = pair_key[order]

    new_group = torch.ones(edge_count, dtype=torch.bool, device=edge_index.device)
    new_group[1:] = sorted_key[1:] != sorted_key[:-1]
    group_start = torch.where(new_group, torch.arange(edge_count, device=edge_index.device), 0)
    group_start = torch.cummax(group_start, dim=0).values
    rank_sorted = torch.arange(edge_count, device=edge_index.device) - group_start
    rank = torch.empty_like(rank_sorted)
    rank[order] = rank_sorted

    stride = edge_count + 1
    compound = pair_key * stride + rank
    reverse_compound = reverse_pair_key * stride + rank
    compound_order = torch.argsort(compound)
    sorted_compound = compound[compound_order]
    positions = torch.searchsorted(sorted_compound, reverse_compound)
    in_range = positions < edge_count
    safe_positions = positions.clamp_max(edge_count - 1)
    matched = in_range & (sorted_compound[safe_positions] == reverse_compound)
    reverse = torch.full((edge_count,), -1, dtype=torch.long, device=edge_index.device)
    reverse[matched] = compound_order[safe_positions[matched]]
    return reverse


class AntisymmetricEdgeField(nn.Module):
    """Learn ``a_uv = psi(h_u, h_v) - psi(h_v, h_u)`` locally."""

    def __init__(self, node_dim: int, hidden_dim: int, max_displacement: float | None) -> None:
        super().__init__()
        if node_dim <= 0 or hidden_dim <= 0:
            raise ValueError("node_dim and hidden_dim must be positive")
        if max_displacement is not None and max_displacement <= 0:
            raise ValueError("max_displacement must be positive when provided")
        self.max_displacement = max_displacement
        self.scorer = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        _validate_edges(edge_index, node_features.shape[0])
        source, target = edge_index
        forward = torch.cat((node_features[source], node_features[target]), dim=-1)
        reverse = torch.cat((node_features[target], node_features[source]), dim=-1)
        displacement = self.scorer(forward).squeeze(-1) - self.scorer(reverse).squeeze(-1)
        if self.max_displacement is not None:
            scale = self.max_displacement
            displacement = scale * torch.tanh(displacement / scale)
        return displacement


class MatchedPotentialEdgeField(AntisymmetricEdgeField):
    """Parameter-matched exact gradient; bound node potentials, not edges.

    Uses the same scorer shape/initialization as the unrestricted field.
    Unlike edgewise tanh of a difference, this preserves zero circulation.
    """

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        _validate_edges(edge_index, node_features.shape[0])
        potential = self.scorer(torch.cat((node_features, node_features), dim=-1)).squeeze(-1)
        if self.max_displacement is not None:
            scale = self.max_displacement / 2.0
            potential = scale * torch.tanh(potential / scale)
        source, target = edge_index
        return potential[target] - potential[source]


class ZeroEdgeField(AntisymmetricEdgeField):
    """Zero connection, retaining AW propagation and scorer RNG consumption.

    Scorer parameters are intentionally inactive, not evidence of equal
    effective capacity. They are retained to match other layers' initialization.
    """

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        _validate_edges(edge_index, node_features.shape[0])
        return node_features.new_zeros(edge_index.shape[1])


class PotentialEdgeField(nn.Module):
    """Exact-gradient ablation ``a_uv = g(h_v) - g(h_u)``."""

    def __init__(self, node_dim: int, hidden_dim: int, max_displacement: float | None) -> None:
        super().__init__()
        self.max_displacement = max_displacement
        self.potential = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        _validate_edges(edge_index, node_features.shape[0])
        source, target = edge_index
        potential = self.potential(node_features).squeeze(-1)
        displacement = potential[target] - potential[source]
        if self.max_displacement is not None:
            scale = self.max_displacement
            displacement = scale * torch.tanh(displacement / scale)
        return displacement


class CoordinateEdgeField(nn.Module):
    """Projected geometric displacement for path/grid/point-cloud graphs."""

    def __init__(self, position_dim: int, max_displacement: float | None) -> None:
        super().__init__()
        if position_dim <= 0:
            raise ValueError("position_dim must be positive")
        self.max_displacement = max_displacement
        self.projection = nn.Linear(position_dim, 1, bias=False)

    def forward(self, positions: Tensor, edge_index: Tensor) -> Tensor:
        _validate_edges(edge_index, positions.shape[0])
        source, target = edge_index
        displacement = self.projection(positions[target] - positions[source]).squeeze(-1)
        if self.max_displacement is not None:
            scale = self.max_displacement
            displacement = scale * torch.tanh(displacement / scale)
        return displacement


def _walk_step(
    x: Tensor,
    source: Tensor,
    target: Tensor,
    angles: Tensor,
    transition_weight: Tensor,
) -> Tensor:
    message = rotate_pairs(x[target], angles) * transition_weight[:, None]
    output = torch.zeros_like(x)
    output.index_add_(0, source, message)
    return output


def sample_uniform_anchor_probes(
    batch_index: Tensor,
    rank: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample the scaled coordinate probes from the LR-CRF construction.

    Every graph in a PyG mini-batch receives ``rank`` independent anchors.
    Sampling is with replacement, so ``E[Omega Omega^T] = I`` also when a
    graph contains fewer nodes than the requested rank.  The fixed seed makes
    all GT layers use the same probes without storing trainable embeddings.
    """
    if rank <= 0:
        raise ValueError("anchor rank must be positive")
    if batch_index.dtype != torch.long or batch_index.ndim != 1:
        raise ValueError("batch_index must be a long tensor with shape (N,)")
    if batch_index.numel() == 0:
        return torch.empty((0, rank), device=batch_index.device, dtype=dtype)
    if not batch_index.is_cuda and (
        int(batch_index.min()) < 0
        or not bool(torch.all(batch_index[1:] >= batch_index[:-1]))
    ):
        raise ValueError("batch_index must be non-negative and grouped by graph")

    generator = torch.Generator(device="cpu")
    graph_count = int(batch_index[-1].item()) + 1
    counts = torch.bincount(batch_index, minlength=graph_count).cpu().tolist()
    if any(node_count == 0 for node_count in counts):
        raise ValueError("batch_index graph identifiers must be contiguous")

    selected_by_graph: list[Tensor] = []
    scales: list[float] = []
    offset = 0
    for node_count in counts:
        # Depend only on graph-local size, not its shuffled position in a
        # mini-batch.  Thus a dataset graph keeps the same local anchors in
        # every epoch; correlations between equally-sized disconnected
        # graphs do not affect their separate kernel blocks.
        generator.manual_seed(int(seed) + 97 * node_count)
        local = torch.randint(node_count, (rank,), generator=generator)
        selected_by_graph.append(local + offset)
        scales.extend([math.sqrt(node_count / rank)] * rank)
        offset += node_count

    selected = torch.cat(selected_by_graph).to(device=batch_index.device)
    columns = torch.arange(rank, device=batch_index.device).repeat(graph_count)
    probes = torch.zeros(
        (batch_index.numel(), rank), device=batch_index.device, dtype=dtype
    )
    probes[selected, columns] = torch.tensor(
        scales, device=batch_index.device, dtype=dtype
    )
    return probes


def low_rank_complex_walk_features(
    edge_index: Tensor,
    displacement: Tensor,
    batch_index: Tensor,
    *,
    rank: int,
    z: Tensor | float,
    num_steps: int,
    carrier_frequencies: Tensor,
    anchor_seed: int = 0,
    edge_weight: Tensor | None = None,
    anchors: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute ``Z_0`` and complex ``Z_nu`` in ``O((J+1) K E r)``.

    Complex states are represented by two real tensors.  No tensor in the
    sparse recurrence has a feature width related to the GT hidden dimension.
    Returned carrier tensors have shape ``(J, N, r)``.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if carrier_frequencies.ndim != 1 or carrier_frequencies.numel() == 0:
        raise ValueError("carrier_frequencies must be a non-empty vector")
    if not bool(torch.all(carrier_frequencies > 0)):
        raise ValueError("carrier frequencies must be positive")
    if carrier_frequencies.numel() > 1 and not bool(
        torch.all(carrier_frequencies[1:] > carrier_frequencies[:-1])
    ):
        raise ValueError("carrier frequencies must be strictly increasing")
    _validate_edges(edge_index, batch_index.numel())
    if displacement.shape != (edge_index.shape[1],):
        raise ValueError("displacement must have shape (E,)")

    device = displacement.device
    edge_index = edge_index.to(device)
    batch_index = batch_index.to(device)
    work_dtype = (
        torch.float32
        if displacement.dtype in {torch.float16, torch.bfloat16}
        else displacement.dtype
    )
    if anchors is None:
        anchors = sample_uniform_anchor_probes(
            batch_index, rank, seed=anchor_seed, dtype=work_dtype
        )
    elif anchors.shape != (batch_index.numel(), rank):
        raise ValueError(f"anchors must have shape ({batch_index.numel()}, {rank})")
    else:
        anchors = anchors.to(device=device, dtype=work_dtype)

    source, target = edge_index
    weights = _transition_weights(edge_index, batch_index.numel(), anchors, edge_weight)
    carriers = carrier_frequencies.to(device=device, dtype=work_dtype)
    angles = carriers[:, None] * displacement.to(dtype=work_dtype)[None, :]
    cosine = torch.cos(angles)[..., None]
    sine = torch.sin(angles)[..., None]
    alpha = torch.as_tensor(z, device=device, dtype=work_dtype)
    if alpha.ndim != 0 or not bool((alpha >= 0) & (alpha < 1)):
        raise ValueError("z must be a scalar in [0, 1)")

    zero_state = anchors
    zero_features = anchors.clone()
    real_state = anchors.unsqueeze(0).expand(carriers.numel(), -1, -1).clone()
    imag_state = torch.zeros_like(real_state)
    real_features = real_state.clone()
    imag_features = imag_state.clone()
    coefficient = torch.ones((), device=device, dtype=work_dtype)

    for _ in range(num_steps):
        zero_message = zero_state[target] * weights[:, None]
        next_zero = torch.zeros_like(zero_state)
        next_zero.index_add_(0, source, zero_message)

        target_real = real_state[:, target, :]
        target_imag = imag_state[:, target, :]
        weighted = weights[None, :, None]
        real_message = (target_real * cosine - target_imag * sine) * weighted
        imag_message = (target_real * sine + target_imag * cosine) * weighted
        next_real = torch.zeros_like(real_state)
        next_imag = torch.zeros_like(imag_state)
        next_real.index_add_(1, source, real_message)
        next_imag.index_add_(1, source, imag_message)

        coefficient = coefficient * alpha
        zero_features = zero_features + coefficient * next_zero
        real_features = real_features + coefficient * next_real
        imag_features = imag_features + coefficient * next_imag
        zero_state, real_state, imag_state = next_zero, next_real, next_imag

    return zero_features, real_features, imag_features, anchors


def _dense_by_graph(x: Tensor, batch_index: Tensor) -> tuple[Tensor, Tensor]:
    """Dependency-light equivalent of ``to_dense_batch`` for grouped nodes."""
    if x.ndim < 2 or x.shape[0] != batch_index.numel():
        raise ValueError("x must start with the same node dimension as batch_index")
    if batch_index.numel() == 0:
        return x.new_empty((0, 0, *x.shape[1:])), torch.empty(
            (0, 0), device=x.device, dtype=torch.bool
        )
    graph_count = int(batch_index[-1].item()) + 1
    counts = torch.bincount(batch_index, minlength=graph_count)
    max_nodes = int(counts.max().item())
    dense = x.new_zeros((graph_count, max_nodes, *x.shape[1:]))
    mask = torch.zeros((graph_count, max_nodes), device=x.device, dtype=torch.bool)
    starts = torch.cumsum(counts, dim=0) - counts
    local_index = torch.arange(batch_index.numel(), device=x.device) - starts[batch_index]
    dense[batch_index, local_index] = x
    mask[batch_index, local_index] = True
    return dense, mask


def low_rank_pairwise_geometry(
    edge_index: Tensor,
    displacement: Tensor,
    batch_index: Tensor,
    *,
    rank: int,
    z: Tensor | float,
    num_steps: int,
    carrier_frequencies: Tensor,
    anchor_seed: int = 0,
    denominator_eps: float = 1e-8,
    edge_weight: Tensor | None = None,
    anchors: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return dense effective displacement, confidence, and real-node mask."""
    if denominator_eps <= 0:
        raise ValueError("denominator_eps must be positive")
    z0, carrier_real, carrier_imag, _ = low_rank_complex_walk_features(
        edge_index,
        displacement,
        batch_index,
        rank=rank,
        z=z,
        num_steps=num_steps,
        carrier_frequencies=carrier_frequencies,
        anchor_seed=anchor_seed,
        edge_weight=edge_weight,
        anchors=anchors,
    )
    dense_zero, real_nodes = _dense_by_graph(z0, batch_index)
    c0 = torch.matmul(dense_zero, dense_zero.transpose(-1, -2))
    pair_mask = real_nodes[:, :, None] & real_nodes[:, None, :]
    valid = pair_mask & (c0 > denominator_eps)

    phases: list[Tensor] = []
    confidences: list[Tensor] = []
    carrier_validities: list[Tensor] = []
    for carrier in range(carrier_real.shape[0]):
        real, _ = _dense_by_graph(carrier_real[carrier], batch_index)
        imag, _ = _dense_by_graph(carrier_imag[carrier], batch_index)
        # Z(u) Z(v)^*: (a+ib)(c-id) = ac+bd + i(bc-ad).
        kernel_real = (
            torch.matmul(real, real.transpose(-1, -2))
            + torch.matmul(imag, imag.transpose(-1, -2))
        )
        kernel_imag = (
            torch.matmul(imag, real.transpose(-1, -2))
            - torch.matmul(real, imag.transpose(-1, -2))
        )
        magnitude_squared = kernel_real.square() + kernel_imag.square()
        carrier_valid = valid & (magnitude_squared > denominator_eps ** 2)
        # atan2(0, 0) and sqrt(0) have undefined derivatives.  Masking their
        # outputs afterwards is insufficient because 0 * NaN can still
        # poison gradients.  Substitute a benign unit complex number before
        # either operation, then mark that node pair as unsupported.
        safe_real = torch.where(
            carrier_valid, kernel_real, torch.ones_like(kernel_real)
        )
        safe_imag = torch.where(
            carrier_valid, kernel_imag, torch.zeros_like(kernel_imag)
        )
        phases.append(torch.atan2(safe_imag, safe_real))
        magnitude = torch.sqrt(
            torch.where(
                carrier_valid, magnitude_squared, torch.ones_like(magnitude_squared)
            )
        )
        confidences.append(
            torch.where(
                carrier_valid,
                magnitude / c0.clamp_min(denominator_eps),
                torch.zeros_like(magnitude),
            )
        )
        carrier_validities.append(carrier_valid)

    valid = torch.stack(carrier_validities).all(dim=0)

    carriers = carrier_frequencies.to(device=c0.device, dtype=c0.dtype)
    displacement_dense = phases[0] / carriers[0]
    # Optional two/multi-carrier phase unwrapping.  The low carrier supplies
    # the coarse branch and every higher carrier refines it.
    for carrier in range(1, len(phases)):
        winding = torch.round(
            (carriers[carrier] * displacement_dense - phases[carrier])
            / (2.0 * math.pi)
        )
        displacement_dense = (
            phases[carrier] + 2.0 * math.pi * winding
        ) / carriers[carrier]

    displacement_dense = torch.where(
        valid, displacement_dense, torch.zeros_like(displacement_dense)
    )
    confidence = torch.where(
        valid,
        confidences[-1].clamp(min=0.0, max=1.0),
        torch.zeros_like(confidences[-1]),
    )
    return displacement_dense, confidence, real_nodes


def low_rank_anchor_geometry(
    edge_index: Tensor,
    displacement: Tensor,
    batch_index: Tensor,
    *,
    rank: int,
    z: Tensor | float,
    num_steps: int,
    carrier_frequencies: Tensor,
    anchor_seed: int = 0,
    denominator_eps: float = 1e-8,
    edge_weight: Tensor | None = None,
    anchors: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return separable node-to-anchor displacement for linear attention.

    The full LR-AW relation normalizes a pairwise sum and cannot in general be
    factorized.  This adapter instead keeps each transported anchor phase as a
    node feature.  On a path, every supported anchor has phase
    ``nu * (anchor - node)``, so subtracting node-anchor phases recovers exact
    relative RoPE at finite rank and propagation depth.  General graphs obtain
    a rank-``r`` separable approximation without constructing an ``N x N``
    geometry matrix.
    """
    if denominator_eps <= 0:
        raise ValueError("denominator_eps must be positive")
    z0, carrier_real, carrier_imag, _ = low_rank_complex_walk_features(
        edge_index,
        displacement,
        batch_index,
        rank=rank,
        z=z,
        num_steps=num_steps,
        carrier_frequencies=carrier_frequencies,
        anchor_seed=anchor_seed,
        edge_weight=edge_weight,
        anchors=anchors,
    )

    phases: list[Tensor] = []
    validities: list[Tensor] = []
    base_valid = z0 > denominator_eps
    for carrier in range(carrier_real.shape[0]):
        real = carrier_real[carrier]
        imag = carrier_imag[carrier]
        magnitude_squared = real.square() + imag.square()
        valid = base_valid & (magnitude_squared > denominator_eps ** 2)
        safe_real = torch.where(valid, real, torch.ones_like(real))
        safe_imag = torch.where(valid, imag, torch.zeros_like(imag))
        phases.append(torch.atan2(safe_imag, safe_real))
        validities.append(valid)

    carriers = carrier_frequencies.to(device=z0.device, dtype=z0.dtype)
    anchor_displacement = phases[0] / carriers[0]
    for carrier in range(1, len(phases)):
        winding = torch.round(
            (carriers[carrier] * anchor_displacement - phases[carrier])
            / (2.0 * math.pi)
        )
        anchor_displacement = (
            phases[carrier] + 2.0 * math.pi * winding
        ) / carriers[carrier]

    valid = torch.stack(validities).all(dim=0)
    anchor_displacement = torch.where(
        valid, anchor_displacement, torch.zeros_like(anchor_displacement)
    )
    dense_displacement, real_nodes = _dense_by_graph(
        anchor_displacement, batch_index
    )
    dense_valid, _ = _dense_by_graph(valid.unsqueeze(-1), batch_index)
    return dense_displacement, dense_valid.squeeze(-1), real_nodes


class ParameterFreeEdgeField(nn.Module):
    """A zero-parameter antisymmetric field for LR-CRF-AW-RoPE V1."""

    def __init__(self, source: str, max_displacement: float | None) -> None:
        super().__init__()
        if source not in {"features", "coordinates"}:
            raise ValueError("source must be features or coordinates")
        if max_displacement is not None and max_displacement <= 0:
            raise ValueError("max_displacement must be positive when provided")
        self.source = source
        self.max_displacement = max_displacement

    def forward(self, values: Tensor, edge_index: Tensor) -> Tensor:
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[-1] == 0:
            raise ValueError("edge-field values must have shape (N, positive d)")
        _validate_edges(edge_index, values.shape[0])
        source, target = edge_index
        # A fixed normalized projection is permutation-equivariant with
        # respect to nodes and exactly antisymmetric under edge reversal.
        displacement = (values[target] - values[source]).sum(dim=-1)
        displacement = displacement / math.sqrt(values.shape[-1])
        if self.max_displacement is not None:
            scale = self.max_displacement
            displacement = scale * torch.tanh(displacement / scale)
        return displacement


class LowRankComplexRandomFeatureAWRoPE(nn.Module):
    """Pairwise LR-CRF-AW-RoPE geometry and B-band attention scores.

    The module itself has zero trainable parameters.  Geometry propagation is
    rank-width rather than model-width, while dense pairwise work is fused
    with the full-attention QK score computation.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        rank: int = 8,
        num_steps: int = 6,
        z: float = 0.6,
        carrier_frequencies: Iterable[float] | None = None,
        num_bands: int = 4,
        frequency_base: float = 10_000.0,
        anchor_seed: int = 0,
        denominator_eps: float = 1e-8,
        confidence_power: float = 0.0,
        field_source: str = "features",
        max_displacement: float | None = math.pi,
        share_geometry: bool = True,
    ) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if rank <= 0 or num_bands <= 0:
            raise ValueError("rank and num_bands must be positive")
        if num_steps < 0 or not 0 <= z < 1:
            raise ValueError("num_steps must be non-negative and z must be in [0, 1)")
        if frequency_base <= 0 or denominator_eps <= 0:
            raise ValueError("frequency_base and denominator_eps must be positive")
        if confidence_power < 0:
            raise ValueError("confidence_power must be non-negative")
        if carrier_frequencies is None:
            carrier_frequencies = (math.pi / (4.0 * max(num_steps, 1)),)
        carriers = tuple(float(value) for value in carrier_frequencies)
        if not carriers or any(value <= 0 for value in carriers):
            raise ValueError("carrier_frequencies must be positive")
        if any(right <= left for left, right in zip(carriers, carriers[1:])):
            raise ValueError("carrier_frequencies must be strictly increasing")

        self.head_dim = head_dim
        self.rotary_dim = head_dim - head_dim % 2
        pair_count = self.rotary_dim // 2
        self.num_bands = min(num_bands, pair_count) if pair_count else 0
        self.rank = rank
        self.num_steps = num_steps
        self.z = float(z)
        self.anchor_seed = int(anchor_seed)
        self.denominator_eps = float(denominator_eps)
        self.confidence_power = float(confidence_power)
        self.share_geometry = bool(share_geometry)
        self.edge_field = ParameterFreeEdgeField(field_source, max_displacement)
        self.field_source = field_source
        self.max_displacement = max_displacement
        self._carrier_values = carriers
        self.register_buffer("carrier_frequencies", torch.tensor(carriers))

        if pair_count:
            frequencies = standard_frequencies(self.rotary_dim, frequency_base)
            groups = torch.tensor_split(torch.arange(pair_count), self.num_bands)
            representatives = torch.stack(
                [torch.exp(torch.log(frequencies[group]).mean()) for group in groups]
            )
            band_index = torch.empty(pair_count, dtype=torch.long)
            for index, group in enumerate(groups):
                band_index[group] = index
        else:
            representatives = torch.empty(0)
            band_index = torch.empty(0, dtype=torch.long)
        self.register_buffer("band_frequencies", representatives)
        self.register_buffer("band_index", band_index)

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            "lr-crf-aw-rope-v1",
            self.rank,
            self.num_steps,
            self.z,
            self._carrier_values,
            self.anchor_seed,
            self.denominator_eps,
            self.field_source,
            self.max_displacement,
        )

    @property
    def performer_cache_key(self) -> tuple[object, ...]:
        return (*self.cache_key, "performer-anchor-geometry")

    def geometry(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        *,
        edge_weight: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        values = positions if self.field_source == "coordinates" else node_features
        if values is None:
            raise ValueError("coordinate LR-AW requires node positions")
        edge_index = edge_index.to(node_features.device)
        displacement = self.edge_field(values, edge_index)
        return low_rank_pairwise_geometry(
            edge_index,
            displacement,
            batch_index.to(node_features.device),
            rank=self.rank,
            z=self.z,
            num_steps=self.num_steps,
            carrier_frequencies=self.carrier_frequencies,
            anchor_seed=self.anchor_seed,
            denominator_eps=self.denominator_eps,
            edge_weight=edge_weight,
        )

    def performer_geometry(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        *,
        edge_weight: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build rank-width anchor geometry without any pairwise dense work."""
        values = positions if self.field_source == "coordinates" else node_features
        if values is None:
            raise ValueError("coordinate LR-AW requires node positions")
        edge_index = edge_index.to(node_features.device)
        displacement = self.edge_field(values, edge_index)
        return low_rank_anchor_geometry(
            edge_index,
            displacement,
            batch_index.to(node_features.device),
            rank=self.rank,
            z=self.z,
            num_steps=self.num_steps,
            carrier_frequencies=self.carrier_frequencies,
            anchor_seed=self.anchor_seed,
            denominator_eps=self.denominator_eps,
            edge_weight=edge_weight,
        )

    def performer_lift(
        self,
        query: Tensor,
        key: Tensor,
        anchor_displacement: Tensor,
        anchor_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Lift Q/K into separable anchor-phase features for Performer.

        Input head dimension ``d`` becomes ``r*d``.  The ``r**(-1/4)``
        scaling exactly preserves the usual ``qk / sqrt(d)`` normalization
        when all anchor phases are zero, because Performer normalizes the
        lifted dot product by ``sqrt(r*d)``.
        """
        if query.shape != key.shape or query.ndim != 4:
            raise ValueError("query and key must share shape (B, H, N, d_head)")
        if query.shape[-1] != self.head_dim:
            raise ValueError(f"expected head dimension {self.head_dim}")
        expected = (query.shape[0], query.shape[2], self.rank)
        if anchor_displacement.shape != expected or anchor_valid.shape != expected:
            raise ValueError(f"anchor geometry must have shape {expected}")

        lifted_shape = (*query.shape[:-1], self.rank, self.head_dim)
        query_lifted = query.unsqueeze(-2).expand(lifted_shape).clone()
        key_lifted = key.unsqueeze(-2).expand(lifted_shape).clone()
        if self.rotary_dim:
            pair_frequencies = self.band_frequencies[self.band_index].to(
                device=query.device, dtype=anchor_displacement.dtype
            )
            # A path anchor phase is nu * (anchor - node).  Rotating both Q
            # and K by its negative makes theta_k - theta_q = omega * (v-u).
            angles = -anchor_displacement[..., None] * pair_frequencies
            cosine = torch.cos(angles).to(dtype=query.dtype)[:, None]
            sine = torch.sin(angles).to(dtype=query.dtype)[:, None]

            query_pairs = query_lifted[..., : self.rotary_dim].reshape(
                *query_lifted.shape[:-1], self.rotary_dim // 2, 2
            )
            key_pairs = key_lifted[..., : self.rotary_dim].reshape(
                *key_lifted.shape[:-1], self.rotary_dim // 2, 2
            )
            for pairs in (query_pairs, key_pairs):
                real = pairs[..., 0].clone()
                imaginary = pairs[..., 1].clone()
                pairs[..., 0] = real * cosine - imaginary * sine
                pairs[..., 1] = real * sine + imaginary * cosine

        validity = anchor_valid.to(dtype=query.dtype)[:, None, :, :, None]
        scale = self.rank ** -0.25
        query_lifted = (query_lifted * validity * scale).flatten(-2)
        key_lifted = (key_lifted * validity * scale).flatten(-2)
        return query_lifted, key_lifted

    def attention_scores(
        self,
        query: Tensor,
        key: Tensor,
        displacement: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        """Compute pairwise B-band rotary QK scores.

        ``query`` and ``key`` have shape ``(batch, heads, nodes, head_dim)``.
        An odd final head channel is deliberately left unrotated.
        """
        if query.shape != key.shape or query.ndim != 4:
            raise ValueError("query and key must share shape (B, H, N, d_head)")
        if query.shape[-1] != self.head_dim:
            raise ValueError(f"expected head dimension {self.head_dim}")
        expected = (query.shape[0], query.shape[2], query.shape[2])
        if displacement.shape != expected or confidence.shape != expected:
            raise ValueError(f"pairwise geometry must have shape {expected}")

        scores = query.new_zeros(
            (query.shape[0], query.shape[1], query.shape[2], query.shape[2])
        )
        if self.rotary_dim:
            query_pairs = query[..., : self.rotary_dim].reshape(
                *query.shape[:-1], self.rotary_dim // 2, 2
            )
            key_pairs = key[..., : self.rotary_dim].reshape(
                *key.shape[:-1], self.rotary_dim // 2, 2
            )
            for band in range(self.num_bands):
                pair_mask = self.band_index == band
                q_real = query_pairs[..., pair_mask, 0]
                q_imag = query_pairs[..., pair_mask, 1]
                k_real = key_pairs[..., pair_mask, 0]
                k_imag = key_pairs[..., pair_mask, 1]
                correlation_real = (
                    torch.einsum("bhup,bhvp->bhuv", q_real, k_real)
                    + torch.einsum("bhup,bhvp->bhuv", q_imag, k_imag)
                )
                correlation_imag = (
                    torch.einsum("bhup,bhvp->bhuv", q_real, k_imag)
                    - torch.einsum("bhup,bhvp->bhuv", q_imag, k_real)
                )
                theta = displacement * self.band_frequencies[band].to(
                    device=displacement.device, dtype=displacement.dtype
                )
                cosine = torch.cos(theta).to(dtype=query.dtype)[:, None]
                sine = torch.sin(theta).to(dtype=query.dtype)[:, None]
                scores = scores + correlation_real * cosine - correlation_imag * sine

        if self.rotary_dim < self.head_dim:
            scores = scores + torch.einsum(
                "bhud,bhvd->bhuv",
                query[..., self.rotary_dim :],
                key[..., self.rotary_dim :],
            )
        if self.confidence_power:
            scores = scores * confidence.to(dtype=scores.dtype)[:, None].pow(
                self.confidence_power
            )
        return scores / math.sqrt(self.head_dim)


def truncated_walk_resolvent(
    x: Tensor,
    edge_index: Tensor,
    displacement: Tensor,
    frequencies: Tensor,
    *,
    z: Tensor,
    num_steps: int,
    edge_weight: Tensor | None = None,
    non_backtracking: bool = False,
    reverse_edge: Tensor | None = None,
) -> Tensor:
    """Apply ``sum_(k=0)^K z^k T^k`` in ``O(K E d)`` time."""
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    _validate_edges(edge_index, x.shape[0])
    if displacement.shape != (edge_index.shape[1],):
        raise ValueError("displacement must have shape (E,)")
    if frequencies.shape != (x.shape[-1] // 2,):
        raise ValueError(f"frequencies must have shape ({x.shape[-1] // 2},)")

    edge_index = edge_index.to(x.device)
    source, target = edge_index
    weights = _transition_weights(edge_index, x.shape[0], x, edge_weight)
    angles = (
        displacement.to(device=x.device, dtype=x.dtype)[:, None]
        * frequencies.to(device=x.device, dtype=x.dtype)[None, :]
    )
    z = z.to(device=x.device, dtype=x.dtype)
    result = x
    coefficient = torch.ones((), dtype=x.dtype, device=x.device)

    if non_backtracking:
        if reverse_edge is None:
            reverse_edge = build_reverse_edge_index(edge_index, x.shape[0])
        else:
            reverse_edge = reverse_edge.to(x.device)
        edge_state = x[target]
        for hop in range(1, num_steps + 1):
            contribution = rotate_pairs(edge_state, angles) * weights[:, None]
            state = torch.zeros_like(x)
            state.index_add_(0, source, contribution)
            coefficient = coefficient * z
            result = result + coefficient * state
            if hop != num_steps:
                edge_state = state[target]
                has_reverse = reverse_edge >= 0
                edge_state = edge_state.clone()
                edge_state[has_reverse] -= contribution[reverse_edge[has_reverse]]
        return result

    state = x
    for _ in range(num_steps):
        state = _walk_step(state, source, target, angles, weights)
        coefficient = coefficient * z
        result = result + coefficient * state
    return result


def truncated_multiscale_walk_resolvent(
    x: Tensor,
    edge_index: Tensor,
    displacement: Tensor,
    frequencies: Tensor,
    *,
    z_values: Tensor,
    mixture: Tensor,
    num_steps: int,
    edge_weight: Tensor | None = None,
    reverse_edge: Tensor | None = None,
) -> Tensor:
    """Apply a rational mixture with one shared non-backtracking recurrence."""
    if z_values.ndim != 1 or mixture.shape != z_values.shape:
        raise ValueError("z_values and mixture must have the same one-dimensional shape")
    _validate_edges(edge_index, x.shape[0])
    if displacement.shape != (edge_index.shape[1],):
        raise ValueError("displacement must have shape (E,)")
    if frequencies.shape != (x.shape[-1] // 2,):
        raise ValueError(f"frequencies must have shape ({x.shape[-1] // 2},)")
    edge_index = edge_index.to(x.device)
    source, target = edge_index
    weights = _transition_weights(edge_index, x.shape[0], x, edge_weight)
    angles = (
        displacement.to(device=x.device, dtype=x.dtype)[:, None]
        * frequencies.to(device=x.device, dtype=x.dtype)[None, :]
    )
    if reverse_edge is None:
        reverse_edge = build_reverse_edge_index(edge_index, x.shape[0])
    else:
        reverse_edge = reverse_edge.to(x.device)

    z_values = z_values.to(device=x.device, dtype=x.dtype)
    mixture = mixture.to(device=x.device, dtype=x.dtype)
    powers = torch.ones_like(z_values)
    result = mixture.sum() * x
    edge_state = x[target]
    for hop in range(1, num_steps + 1):
        contribution = rotate_pairs(edge_state, angles) * weights[:, None]
        state = torch.zeros_like(x)
        state.index_add_(0, source, contribution)
        powers = powers * z_values
        result = result + torch.sum(mixture * powers) * state
        if hop != num_steps:
            edge_state = state[target]
            has_reverse = reverse_edge >= 0
            edge_state = edge_state.clone()
            edge_state[has_reverse] -= contribution[reverse_edge[has_reverse]]
    return result


class AnalyticWalkRoPE(nn.Module):
    """Shared AW-RoPE position module for attention queries and keys."""

    _VALID_METHODS = {"aw", "aw-nb", "aw-nb-ms"}

    def __init__(
        self,
        dim: int,
        *,
        field_node_dim: int | None = None,
        method: str = "aw",
        num_steps: int = 8,
        initial_z: float = 0.8,
        z_values: Iterable[float] = (0.2, 0.4, 0.6, 0.8),
        field_hidden_dim: int = 32,
        max_displacement: float | None = 3.141592653589793,
        frequency_base: float = 10_000.0,
        learnable_frequencies: bool = True,
        learnable_z: bool = True,
        normalize_resolvent: bool = False,
        residual_mix: float = 1.0,
        preserve_input_norm: bool = False,
        norm_group_size: int = 0,
        stability_eps: float = 1e-4,
        field_type: str = "local-antisymmetric",
        position_dim: int = 3,
    ) -> None:
        super().__init__()
        if method not in self._VALID_METHODS:
            raise ValueError(f"unsupported AW-RoPE method {method!r}")
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if not 0 <= initial_z < 1:
            raise ValueError("initial_z must satisfy 0 <= z < 1")
        if not 0 < stability_eps < 1:
            raise ValueError("stability_eps must be in (0, 1)")
        if not 0.0 <= residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0, 1]")
        if norm_group_size < 0 or (
            norm_group_size > 0 and dim % norm_group_size != 0
        ):
            raise ValueError("norm_group_size must be zero or divide dim")
        self.dim = dim
        self.method = method
        self.num_steps = num_steps
        self.normalize_resolvent = normalize_resolvent
        self.residual_mix = float(residual_mix)
        self.preserve_input_norm = bool(preserve_input_norm)
        self.norm_group_size = int(norm_group_size)
        self.stability_eps = stability_eps
        self.field_node_dim = dim if field_node_dim is None else int(field_node_dim)
        if self.field_node_dim <= 0:
            raise ValueError("field_node_dim must be positive")
        if field_type == "local-antisymmetric":
            self.edge_field: nn.Module = AntisymmetricEdgeField(
                self.field_node_dim, field_hidden_dim, max_displacement
            )
        elif field_type == "matched-potential":
            self.edge_field = MatchedPotentialEdgeField(
                self.field_node_dim, field_hidden_dim, max_displacement
            )
        elif field_type == "zero":
            self.edge_field = ZeroEdgeField(
                self.field_node_dim, field_hidden_dim, max_displacement
            )
        elif field_type == "potential":
            self.edge_field = PotentialEdgeField(
                self.field_node_dim, field_hidden_dim, max_displacement
            )
        elif field_type == "coordinate":
            self.edge_field = CoordinateEdgeField(position_dim, max_displacement)
        elif field_type == "precomputed-static":
            self.edge_field = None
        else:
            raise ValueError(
                "field_type must be local-antisymmetric, potential, coordinate, "
                "or precomputed-static"
            )
        self.field_type = field_type

        frequencies = standard_frequencies(dim, frequency_base)
        if learnable_frequencies:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)

        if method == "aw-nb-ms":
            z_values = tuple(float(value) for value in z_values)
            if not z_values or any(not 0 < value < 1 for value in z_values):
                raise ValueError("z_values must be a non-empty list with entries in (0, 1)")
            scaled = torch.tensor(z_values) / (1.0 - stability_eps)
            z_logits = torch.logit(scaled.clamp(1e-7, 1.0 - 1e-7))
            if learnable_z:
                self.z_logits = nn.Parameter(z_logits)
            else:
                self.register_buffer("z_logits", z_logits)
            self.mixture_logits = nn.Parameter(torch.zeros(len(z_values)))
        else:
            scaled_z = min(initial_z / (1.0 - stability_eps), 1.0 - 1e-7)
            z_logit = torch.logit(torch.tensor(max(scaled_z, 1e-7)))
            if learnable_z:
                self.z_logit = nn.Parameter(z_logit)
            else:
                self.register_buffer("z_logit", z_logit)

    @property
    def z(self) -> Tensor:
        if self.method == "aw-nb-ms":
            raise AttributeError("multiscale AW-RoPE has z_values instead of z")
        return torch.sigmoid(self.z_logit) * (1.0 - self.stability_eps)

    @property
    def z_values(self) -> Tensor:
        if self.method != "aw-nb-ms":
            raise AttributeError("single-scale AW-RoPE has z instead of z_values")
        return torch.sigmoid(self.z_logits) * (1.0 - self.stability_eps)

    def _normalizer(self, z: Tensor, reference: Tensor) -> Tensor:
        if not self.normalize_resolvent:
            return torch.ones((), dtype=reference.dtype, device=reference.device)
        z = z.to(dtype=reference.dtype, device=reference.device)
        denominator = sum(z**hop for hop in range(self.num_steps + 1))
        return denominator.reciprocal()

    def _transport_single(
        self,
        x: Tensor,
        edge_index: Tensor,
        displacement: Tensor,
        edge_weight: Tensor | None,
        reverse_edge: Tensor | None,
        frequencies: Tensor | None = None,
    ) -> Tensor:
        output = truncated_walk_resolvent(
            x,
            edge_index,
            displacement,
            self.frequencies if frequencies is None else frequencies,
            z=self.z,
            num_steps=self.num_steps,
            edge_weight=edge_weight,
            non_backtracking=self.method == "aw-nb",
            reverse_edge=reverse_edge,
        )
        return output * self._normalizer(self.z, x)

    def _transport_multiscale(
        self,
        x: Tensor,
        edge_index: Tensor,
        displacement: Tensor,
        edge_weight: Tensor | None,
        reverse_edge: Tensor,
        frequencies: Tensor | None = None,
    ) -> Tensor:
        mixture = torch.softmax(self.mixture_logits, dim=0)
        if self.normalize_resolvent:
            normalizers = torch.stack([self._normalizer(z, x) for z in self.z_values])
            mixture = mixture.to(normalizers.dtype) * normalizers
        return truncated_multiscale_walk_resolvent(
            x,
            edge_index,
            displacement,
            self.frequencies if frequencies is None else frequencies,
            z_values=self.z_values,
            mixture=mixture,
            num_steps=self.num_steps,
            edge_weight=edge_weight,
            reverse_edge=reverse_edge,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        node_features: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        reverse_edge: Tensor | None = None,
        positions: Tensor | None = None,
        precomputed_displacement: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if query.shape != key.shape or query.shape[0] != node_features.shape[0]:
            raise ValueError(
                "query/key must share shape and node_features must share their node axis"
            )
        if query.shape[-1] != self.dim:
            raise ValueError(f"expected feature dimension {self.dim}, got {query.shape[-1]}")
        if node_features.ndim != 2 or node_features.shape[-1] != self.field_node_dim:
            raise ValueError(
                f"expected node feature dimension {self.field_node_dim}, "
                f"got {tuple(node_features.shape)}"
            )
        edge_index = edge_index.to(query.device)
        if self.field_type == "precomputed-static":
            if precomputed_displacement is None:
                raise ValueError(
                    "precomputed-static AW requires aw_static_edge_displacement"
                )
            if precomputed_displacement.shape != (edge_index.shape[1],):
                raise ValueError(
                    "aw_static_edge_displacement must have one value per edge"
                )
            displacement = precomputed_displacement.to(
                device=query.device, dtype=query.dtype
            )
        elif self.field_type == "coordinate":
            if positions is None:
                raise ValueError("coordinate AW-RoPE requires node positions")
            displacement = self.edge_field(positions, edge_index)
        else:
            displacement = self.edge_field(node_features, edge_index)
        if reverse_edge is not None:
            reverse_edge = reverse_edge.to(query.device)
        elif self.method in {"aw-nb", "aw-nb-ms"}:
            reverse_edge = build_reverse_edge_index(edge_index, query.shape[0])

        # Q and K use the same graph transport.  Concatenating them preserves
        # pair boundaries (dim is even) while sharing every sparse recurrence.
        query_key = torch.cat((query, key), dim=-1)
        # The transport helpers expect frequencies for the current input
        # width.  Repeat the shared Q/K frequencies without adding parameters.
        repeated_frequencies = self.frequencies.repeat(2)
        if self.method == "aw-nb-ms":
            assert reverse_edge is not None
            output = self._transport_multiscale(
                query_key,
                edge_index,
                displacement,
                edge_weight,
                reverse_edge,
                repeated_frequencies,
            )
        else:
            output = self._transport_single(
                query_key,
                edge_index,
                displacement,
                edge_weight,
                reverse_edge,
                repeated_frequencies,
            )
        transported_query, transported_key = output.split(self.dim, dim=-1)

        # A standard RoPE is norm preserving, whereas graph transport also
        # mixes neighboring Q/K values.  These parameter-free stabilizers let
        # softmax FAVOR+ retain the input scale while still receiving analytic
        # walk phase information.  residual_mix=1 and preserve_input_norm=False
        # exactly reproduce the original implementation.
        if self.residual_mix == 1.0:
            # Preserve the exact pre-stabilizer numerical path for every
            # existing experiment and checkpoint.
            mixed_query = transported_query
            mixed_key = transported_key
        else:
            mixed_query = torch.lerp(query, transported_query, self.residual_mix)
            mixed_key = torch.lerp(key, transported_key, self.residual_mix)
        if self.preserve_input_norm:
            tiny = torch.finfo(query.dtype).tiny

            def preserve(reference: Tensor, value: Tensor) -> Tensor:
                group_size = self.norm_group_size or self.dim
                reference_groups = reference.reshape(
                    *reference.shape[:-1], self.dim // group_size, group_size
                )
                value_groups = value.reshape(
                    *value.shape[:-1], self.dim // group_size, group_size
                )
                reference_norm = torch.linalg.vector_norm(
                    reference_groups, dim=-1, keepdim=True
                )
                value_norm = torch.linalg.vector_norm(
                    value_groups, dim=-1, keepdim=True
                )
                return (
                    value_groups * (reference_norm / value_norm.clamp_min(tiny))
                ).reshape_as(value)

            mixed_query = preserve(query, mixed_query)
            mixed_key = preserve(key, mixed_key)
        return mixed_query, mixed_key
