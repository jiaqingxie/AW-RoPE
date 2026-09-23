"""Antisymmetric edge displacement fields for AW-RoPE."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class AntisymmetricEdgeField(nn.Module):
    """Learn ``a_uv = psi(u, v, e_uv) - psi(v, u, e_vu)`` locally.

    When reverse edge attributes are omitted they are assumed to be symmetric.
    The construction is antisymmetric by design and does not use node ordering,
    graph spectra, or eigendecomposition.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 64,
        *,
        max_displacement: Optional[float] = None,
    ) -> None:
        super().__init__()
        if node_dim <= 0 or edge_dim < 0 or hidden_dim <= 0:
            raise ValueError("node_dim and hidden_dim must be positive; edge_dim must be non-negative")
        self.edge_dim = edge_dim
        self.max_displacement = max_displacement
        self.scorer = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        reverse_edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a long tensor with shape (2, E)")
        source, target = edge_index
        if self.edge_dim:
            if edge_attr is None or edge_attr.shape != (edge_index.shape[1], self.edge_dim):
                raise ValueError(f"edge_attr must have shape (E, {self.edge_dim})")
            reverse_edge_attr = edge_attr if reverse_edge_attr is None else reverse_edge_attr
            if reverse_edge_attr.shape != edge_attr.shape:
                raise ValueError("reverse_edge_attr must have the same shape as edge_attr")
            forward_input = torch.cat((node_features[source], node_features[target], edge_attr), dim=-1)
            reverse_input = torch.cat((node_features[target], node_features[source], reverse_edge_attr), dim=-1)
        else:
            if edge_attr is not None or reverse_edge_attr is not None:
                raise ValueError("edge attributes were provided but edge_dim=0")
            forward_input = torch.cat((node_features[source], node_features[target]), dim=-1)
            reverse_input = torch.cat((node_features[target], node_features[source]), dim=-1)
        displacement = self.scorer(forward_input).squeeze(-1) - self.scorer(reverse_input).squeeze(-1)
        if self.max_displacement is not None:
            displacement = self.max_displacement * torch.tanh(displacement / self.max_displacement)
        return displacement

