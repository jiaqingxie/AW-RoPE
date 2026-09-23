"""Exact Holonomy-RoPE geometry for GraphGPS attention.

The fixed reference consumes an offline ``chi`` sidecar.  The learnable
reference builds the complete connection Laplacian from a learned
antisymmetric edge field and differentiates through the exact matrix
exponential.  These are deliberately dense reference implementations, not an
AW truncation, a low-rank approximation, or a linear-attention surrogate.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .aw_rope import AntisymmetricEdgeField, standard_frequencies


class ExactHolonomyRoPE(nn.Module):
    """Parameter-free adapter for a cached exact pairwise transport."""

    def __init__(
        self,
        *,
        head_dim: int,
        field_protocol: str = "topology-rwdiag-skew-v1",
    ) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.head_dim = head_dim
        self.rotary_dim = head_dim - head_dim % 2
        self.num_frequencies = self.rotary_dim // 2
        self.field_protocol = str(field_protocol)

    def pairwise_transport(
        self,
        flat_transport: Tensor,
        batch_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pad flat sidecar slices to ``(B, F, N_max, N_max)``."""
        if batch_index.ndim != 1:
            raise ValueError("batch_index must be one-dimensional")
        if not torch.is_complex(flat_transport) or flat_transport.ndim != 2:
            raise TypeError("precomputed transport must be a complex (sum N^2, F) tensor")
        if flat_transport.shape[1] != self.num_frequencies:
            raise ValueError(
                f"cached transport has {flat_transport.shape[1]} frequencies; "
                f"attention requires {self.num_frequencies}"
            )
        num_graphs = int(batch_index.max()) + 1 if batch_index.numel() else 0
        if num_graphs <= 0:
            raise ValueError("the PyG batch must contain at least one graph")
        counts = torch.bincount(batch_index, minlength=num_graphs)
        max_nodes = int(counts.max())
        expected_rows = int(counts.square().sum())
        if flat_transport.shape[0] != expected_rows:
            raise ValueError(
                f"cached transport has {flat_transport.shape[0]} rows; "
                f"batch graph sizes require {expected_rows}"
            )
        padded: list[Tensor] = []
        cursor = 0
        for count_tensor in counts:
            count = int(count_tensor)
            rows = count * count
            graph_transport = flat_transport[cursor : cursor + rows]
            graph_transport = graph_transport.view(
                count, count, self.num_frequencies
            ).permute(2, 0, 1)
            padded.append(
                F.pad(graph_transport, (0, max_nodes - count, 0, max_nodes - count))
            )
            cursor += rows
        transport = torch.stack(padded)
        node_ids = torch.arange(max_nodes, device=batch_index.device)
        real_nodes = node_ids.unsqueeze(0) < counts.unsqueeze(1)
        return transport, real_nodes

    def attention_scores(self, query: Tensor, key: Tensor, transport: Tensor) -> Tensor:
        """Compute exact non-factorized Holonomy-RoPE logits before softmax."""
        if query.shape != key.shape or query.ndim != 4:
            raise ValueError("query and key must share shape (B, H, N, D_h)")
        if query.shape[-1] != self.head_dim:
            raise ValueError(f"expected head dimension {self.head_dim}")
        expected = (query.shape[0], self.num_frequencies, query.shape[2], query.shape[2])
        if tuple(transport.shape) != expected:
            raise ValueError(f"transport shape {tuple(transport.shape)} != {expected}")

        rotary_query = query[..., : self.rotary_dim].reshape(
            *query.shape[:-1], self.num_frequencies, 2
        )
        rotary_key = key[..., : self.rotary_dim].reshape(
            *key.shape[:-1], self.num_frequencies, 2
        )
        if query.dtype == torch.float64:
            query_complex = torch.complex(rotary_query[..., 0], rotary_query[..., 1])
            key_complex = torch.complex(rotary_key[..., 0], rotary_key[..., 1])
        else:
            query_complex = torch.complex(
                rotary_query[..., 0].float(), rotary_query[..., 1].float()
            )
            key_complex = torch.complex(
                rotary_key[..., 0].float(), rotary_key[..., 1].float()
            )
        transport = transport.to(query_complex.dtype)
        scores = torch.einsum(
            "bhif,bfij,bhjf->bhij",
            query_complex.conj(),
            transport,
            key_complex,
        ).real
        if self.rotary_dim < self.head_dim:
            scores = scores + torch.einsum(
                "bhid,bhjd->bhij",
                query[..., self.rotary_dim :].float(),
                key[..., self.rotary_dim :].float(),
            )
        return scores / math.sqrt(self.head_dim)


class LearnableExactHolonomyRoPE(ExactHolonomyRoPE):
    """End-to-end learnable Full-HPE for variable-size PyG mini-batches.

    For each graph and rotary frequency this module evaluates

    ``a_uv = psi(h_u, h_v) - psi(h_v, h_u)``,
    ``L_omega = D - W * exp(i omega a)``, and
    ``chi_omega = phase(exp(-t L_omega))``.

    PyG normally stores both directions of every undirected edge.  The
    connection Laplacian convention stores an undirected edge once, so this
    module canonicalizes every pair to ``min(u,v) -> max(u,v)`` before the
    learned field is evaluated.  Self loops do not contribute to the
    connection geometry.  The dense transport is recomputed on every forward
    pass because both the current node representation and the learned field
    change during training.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        field_node_dim: int,
        field_hidden_dim: int = 32,
        max_displacement: float | None = math.pi,
        frequency_base: float = 10_000.0,
        learnable_frequencies: bool = True,
        diffusion_time: float = 2.0,
        eps: float = 1e-12,
        soft_phase_normalization: bool = False,
    ) -> None:
        super().__init__(head_dim=head_dim, field_protocol="learned-local-antisymmetric-v1")
        if field_node_dim <= 0:
            raise ValueError("field_node_dim must be positive")
        if diffusion_time < 0:
            raise ValueError("diffusion_time must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.field_node_dim = int(field_node_dim)
        self.diffusion_time = float(diffusion_time)
        self.eps = float(eps)
        self.soft_phase_normalization = bool(soft_phase_normalization)
        self.edge_field = AntisymmetricEdgeField(
            self.field_node_dim,
            int(field_hidden_dim),
            max_displacement,
        )
        frequencies = standard_frequencies(self.rotary_dim, frequency_base)
        if learnable_frequencies:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.precomputed = False

    @staticmethod
    def _canonical_edges(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
        """Return unique undirected edges and indices of their first copies."""
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a long tensor with shape (2, E)")
        source, target = edge_index
        keep = source != target
        low = torch.minimum(source[keep], target[keep])
        high = torch.maximum(source[keep], target[keep])
        if low.numel() == 0:
            return edge_index.new_empty((2, 0)), edge_index.new_empty((0,))
        key = low * num_nodes + high
        order = torch.argsort(key, stable=True)
        sorted_key = key[order]
        first = torch.ones_like(sorted_key, dtype=torch.bool)
        first[1:] = sorted_key[1:] != sorted_key[:-1]
        selected = order[first]
        original_indices = torch.where(keep)[0][selected]
        return torch.stack((low[selected], high[selected])), original_indices

    def _connection_laplacian(
        self,
        num_nodes: int,
        edge_index: Tensor,
        displacement: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        """Build one graph's ``(F,N,N)`` connection Laplacian."""
        real_dtype = torch.float64 if displacement.dtype == torch.float64 else torch.float32
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        frequencies = self.frequencies.to(device=displacement.device, dtype=real_dtype)
        displacement = displacement.to(dtype=real_dtype)

        if edge_weight is None:
            weight = torch.ones(edge_index.shape[1], device=displacement.device, dtype=real_dtype)
        else:
            weight = edge_weight.to(device=displacement.device, dtype=real_dtype)
            if bool(torch.any(weight < 0)):
                raise ValueError("edge_weight must be non-negative")

        angles = frequencies[:, None] * displacement[None, :]
        transport = torch.complex(torch.cos(angles), torch.sin(angles)).to(complex_dtype)
        values = transport * weight[None, :]
        source, target = edge_index
        adjacency = torch.zeros(
            self.num_frequencies,
            num_nodes * num_nodes,
            device=displacement.device,
            dtype=complex_dtype,
        )
        adjacency.index_add_(1, source * num_nodes + target, values)
        adjacency.index_add_(1, target * num_nodes + source, values.conj())
        adjacency = adjacency.view(self.num_frequencies, num_nodes, num_nodes)
        degree = torch.zeros(num_nodes, device=displacement.device, dtype=real_dtype)
        degree.index_add_(0, source, weight)
        degree.index_add_(0, target, weight)
        return torch.diag(degree).to(complex_dtype).unsqueeze(0) - adjacency

    def _heat_transport(self, laplacian: Tensor) -> Tensor:
        """Apply the exact heat kernel and phase map to a matrix batch."""
        # matrix_exp is the exact finite-matrix analytic function.  Unlike an
        # eigendecomposition, its backward pass remains well-defined when a
        # learned connection produces repeated eigenvalues.
        heat_kernel = torch.matrix_exp(-self.diffusion_time * laplacian)
        magnitude = heat_kernel.abs()
        if self.soft_phase_normalization:
            return heat_kernel / torch.sqrt(magnitude.square() + self.eps**2)
        return torch.where(
            magnitude > self.eps,
            heat_kernel / magnitude.clamp_min(self.eps),
            torch.zeros_like(heat_kernel),
        )

    def _graph_transport(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        """Build ``(F,N,N)`` exact transport for one graph."""
        edges, original_indices = self._canonical_edges(edge_index, node_features.shape[0])
        displacement = self.edge_field(node_features, edges)
        weight = edge_weight[original_indices] if edge_weight is not None else None
        laplacian = self._connection_laplacian(
            node_features.shape[0], edges, displacement, weight
        )
        return self._heat_transport(laplacian)

    def pairwise_transport(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return padded exact transports and the corresponding node mask."""
        if node_features.ndim != 2 or node_features.shape[1] != self.field_node_dim:
            raise ValueError(
                f"node_features must have shape (N, {self.field_node_dim})"
            )
        if batch_index.dtype != torch.long or batch_index.ndim != 1:
            raise ValueError("batch_index must be a long tensor with shape (N,)")
        if batch_index.numel() != node_features.shape[0]:
            raise ValueError("batch_index and node_features must share their node axis")
        if edge_weight is not None and edge_weight.shape != (edge_index.shape[1],):
            raise ValueError("edge_weight must have shape (E,)")
        num_graphs = int(batch_index.max()) + 1 if batch_index.numel() else 0
        if num_graphs <= 0:
            raise ValueError("the PyG batch must contain at least one graph")
        source, target = edge_index
        if bool(torch.any(batch_index[source] != batch_index[target])):
            raise ValueError("edge_index contains an edge between different graphs")
        counts = torch.bincount(batch_index, minlength=num_graphs)
        max_nodes = int(counts.max())
        local_index = torch.empty_like(batch_index)
        for graph_id, count_tensor in enumerate(counts):
            nodes = torch.where(batch_index == graph_id)[0]
            local_index[nodes] = torch.arange(int(count_tensor), device=batch_index.device)

        # Learn all edge displacements in one scorer call, then batch every
        # equal-size graph into one matrix exponential.  This preserves the
        # exact per-graph blocks while avoiding B separate Python launches of
        # the expensive analytic function.  Synthetic datasets have one node
        # count, so a whole minibatch becomes a single batched call.
        canonical_edges, original_indices = self._canonical_edges(
            edge_index, node_features.shape[0]
        )
        canonical_displacement = self.edge_field(node_features, canonical_edges)
        canonical_weight = (
            edge_weight[original_indices] if edge_weight is not None else None
        )
        canonical_source = canonical_edges[0]
        grouped_laplacians: dict[int, list[tuple[int, Tensor]]] = {}
        for graph_id, count_tensor in enumerate(counts):
            nodes = torch.where(batch_index == graph_id)[0]
            count = int(count_tensor)
            edge_mask = batch_index[canonical_source] == graph_id
            local_edges = local_index[canonical_edges[:, edge_mask]]
            local_weight = (
                canonical_weight[edge_mask] if canonical_weight is not None else None
            )
            laplacian = self._connection_laplacian(
                count,
                local_edges,
                canonical_displacement[edge_mask],
                local_weight,
            )
            grouped_laplacians.setdefault(count, []).append((graph_id, laplacian))

        graph_transports: list[Tensor | None] = [None] * num_graphs
        for count, indexed_laplacians in grouped_laplacians.items():
            laplacian_batch = torch.stack(
                [laplacian for _, laplacian in indexed_laplacians]
            )
            transport_batch = self._heat_transport(laplacian_batch)
            for batch_row, (graph_id, _) in enumerate(indexed_laplacians):
                graph_transports[graph_id] = F.pad(
                    transport_batch[batch_row],
                    (0, max_nodes - count, 0, max_nodes - count),
                )
        if any(transport is None for transport in graph_transports):
            raise RuntimeError("failed to build a transport for every graph")
        padded = torch.stack([transport for transport in graph_transports if transport is not None])
        node_ids = torch.arange(max_nodes, device=batch_index.device)
        real_nodes = node_ids.unsqueeze(0) < counts.unsqueeze(1)
        return padded, real_nodes
