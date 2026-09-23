"""Exact dense Holonomy-RoPE reference implementation.

This module deliberately preserves the original, non-scalable construction.
For every rotary frequency it builds a dense Hermitian U(1) connection
Laplacian, evaluates its heat kernel, normalizes every non-zero pairwise entry
to unit modulus, and uses the resulting non-factorized transport directly in
dense softmax attention.

The graph input contains every undirected edge exactly once.  An entry
``edge_index[:, e] = (u, v)`` with displacement ``a[e]`` contributes

    C_l[u, v] = w[e] exp(i omega_l a[e])
    C_l[v, u] = conj(C_l[u, v]).

Consequently ``L_l = D - C_l`` is Hermitian.  This convention is intentionally
different from the bidirectional directed-edge convention used by the sparse
AW-RoPE operators in :mod:`aw_rope.core`.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn


HeatKernelMethod = Literal["eigh", "matrix_exp"]


def build_rope_frequencies(
    rotary_dim: int,
    base: float = 10_000.0,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return standard RoPE frequencies ``base ** (-2 l / rotary_dim)``."""
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be a positive even integer, got {rotary_dim}")
    if base <= 0.0:
        raise ValueError("base must be positive")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point dtype")
    index = torch.arange(0, rotary_dim, 2, device=device, dtype=dtype)
    return base ** (-index / rotary_dim)


def _real_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.float64
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    raise TypeError(f"expected a floating-point dtype, got {dtype}")


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _validate_undirected_edges_once(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
    if edge_index.numel():
        minimum = int(edge_index.min())
        maximum = int(edge_index.max())
        if minimum < 0 or maximum >= num_nodes:
            raise ValueError(
                f"edge_index contains a node outside [0, {num_nodes}): "
                f"[{minimum}, {maximum}]"
            )

    seen: set[tuple[int, int]] = set()
    for source, target in edge_index.detach().cpu().t().tolist():
        if source == target:
            raise ValueError("self-loops are not part of the undirected-edge-once input convention")
        undirected = (min(source, target), max(source, target))
        if undirected in seen:
            raise ValueError(
                "edge_index must contain every undirected edge exactly once; "
                f"found a duplicate or reverse duplicate for {undirected}"
            )
        seen.add(undirected)


def build_connection_laplacians(
    num_nodes: int,
    edge_index: Tensor,
    edge_displacement: Tensor,
    frequencies: Tensor,
    edge_weight: Optional[Tensor] = None,
    *,
    validate_undirected_once: bool = True,
) -> tuple[Tensor, Tensor]:
    """Build one dense Hermitian U(1) connection Laplacian per frequency.

    Parameters follow the undirected-edge-once convention documented at the
    module level.  The result has shape ``(F, N, N)`` and costs
    ``O(F N^2)`` memory.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if validate_undirected_once:
        _validate_undirected_edges_once(edge_index, num_nodes)
    else:
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a long tensor with shape (2, E)")

    num_edges = edge_index.shape[1]
    if edge_displacement.ndim != 1 or edge_displacement.numel() != num_edges:
        raise ValueError("edge_displacement must have shape (E,)")
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must be a non-empty tensor with shape (F,)")
    if not edge_displacement.dtype.is_floating_point or not frequencies.dtype.is_floating_point:
        raise TypeError("edge_displacement and frequencies must be floating-point tensors")

    device = frequencies.device
    real_dtype = _real_compute_dtype(torch.promote_types(edge_displacement.dtype, frequencies.dtype))
    complex_dtype = _complex_dtype(real_dtype)
    edge_index = edge_index.to(device=device)
    displacement = edge_displacement.to(device=device, dtype=real_dtype)
    omega = frequencies.to(device=device, dtype=real_dtype)

    if edge_weight is None:
        weight = torch.ones(num_edges, device=device, dtype=real_dtype)
    else:
        if edge_weight.ndim != 1 or edge_weight.numel() != num_edges:
            raise ValueError("edge_weight must have shape (E,)")
        weight = edge_weight.to(device=device, dtype=real_dtype)
        if not bool(torch.all(torch.isfinite(weight))):
            raise ValueError("edge_weight must be finite")
        if bool(torch.any(weight < 0)):
            raise ValueError("edge_weight must be non-negative")

    angles = omega[:, None] * displacement[None, :]
    rho = torch.complex(torch.cos(angles), torch.sin(angles)).to(complex_dtype)
    values = rho * weight[None, :]

    source, target = edge_index
    forward_index = source * num_nodes + target
    reverse_index = target * num_nodes + source
    adjacency_flat = torch.zeros(
        (omega.numel(), num_nodes * num_nodes),
        device=device,
        dtype=complex_dtype,
    )
    adjacency_flat.index_add_(1, forward_index, values)
    adjacency_flat.index_add_(1, reverse_index, values.conj())
    connection_adjacency = adjacency_flat.view(omega.numel(), num_nodes, num_nodes)

    degree = torch.zeros(num_nodes, device=device, dtype=real_dtype)
    degree.index_add_(0, source, weight)
    degree.index_add_(0, target, weight)
    degree_matrix = torch.diag(degree).to(complex_dtype)
    laplacian = degree_matrix.unsqueeze(0) - connection_adjacency
    return laplacian, rho


def connection_heat_kernel(
    laplacian: Tensor,
    diffusion_time: float | Tensor = 1.0,
    *,
    method: HeatKernelMethod = "eigh",
    return_eigensystem: bool = True,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Evaluate ``exp(-t L)`` as a full matrix analytic function.

    ``method='eigh'`` is the canonical Hermitian reference implementation.
    ``method='matrix_exp'`` evaluates the same exact finite-matrix function
    without differentiating through eigenvectors, which is preferable when a
    learnable connection can create repeated or nearly repeated eigenvalues.
    Neither method is an AW truncation or low-rank approximation.
    """
    if laplacian.ndim != 3 or laplacian.shape[-1] != laplacian.shape[-2]:
        raise ValueError("laplacian must have shape (F, N, N)")
    if not torch.is_complex(laplacian):
        raise TypeError("laplacian must be complex")
    if method not in ("eigh", "matrix_exp"):
        raise ValueError("method must be 'eigh' or 'matrix_exp'")

    time = torch.as_tensor(
        diffusion_time,
        device=laplacian.device,
        dtype=laplacian.real.dtype,
    )
    if time.numel() != 1 or not bool(torch.isfinite(time)) or bool(time < 0):
        raise ValueError("diffusion_time must be a finite non-negative scalar")

    eigenvalues: Optional[Tensor] = None
    eigenvectors: Optional[Tensor] = None
    if method == "eigh":
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        heat_eigenvalues = torch.exp(-time * eigenvalues)
        heat_kernel = (
            eigenvectors * heat_eigenvalues.unsqueeze(-2)
        ) @ eigenvectors.conj().transpose(-1, -2)
    else:
        heat_kernel = torch.matrix_exp(-time * laplacian)
        if return_eigensystem:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

    if not return_eigensystem:
        eigenvalues, eigenvectors = None, None
    return heat_kernel, eigenvalues, eigenvectors


def heat_kernel_to_transport(
    heat_kernel: Tensor,
    *,
    eps: float = 1e-12,
    soft_normalization: bool = False,
) -> Tensor:
    """Extract pairwise phase ``H_uv / |H_uv|`` from a heat kernel."""
    if not torch.is_complex(heat_kernel):
        raise TypeError("heat_kernel must be complex")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    magnitude = heat_kernel.abs()
    if soft_normalization:
        return heat_kernel / torch.sqrt(magnitude.square() + eps**2)
    return torch.where(
        magnitude > eps,
        heat_kernel / magnitude.clamp_min(eps),
        torch.zeros_like(heat_kernel),
    )


class ExactHolonomyTransport(nn.Module):
    """Original dense Holonomy-RoPE transport, with no scalable approximation."""

    def __init__(
        self,
        rotary_dim: int,
        *,
        base: float = 10_000.0,
        diffusion_time: float = 1.0,
        eps: float = 1e-12,
        soft_normalization: bool = False,
        heat_kernel_method: HeatKernelMethod = "eigh",
        validate_undirected_once: bool = True,
    ) -> None:
        super().__init__()
        if diffusion_time < 0.0:
            raise ValueError("diffusion_time must be non-negative")
        if heat_kernel_method not in ("eigh", "matrix_exp"):
            raise ValueError("heat_kernel_method must be 'eigh' or 'matrix_exp'")
        self.rotary_dim = rotary_dim
        self.num_frequencies = rotary_dim // 2
        self.diffusion_time = diffusion_time
        self.eps = eps
        self.soft_normalization = soft_normalization
        self.heat_kernel_method = heat_kernel_method
        self.validate_undirected_once = validate_undirected_once
        self.register_buffer(
            "frequencies",
            build_rope_frequencies(rotary_dim=rotary_dim, base=base),
            persistent=False,
        )

    def forward(
        self,
        num_nodes: int,
        edge_index: Tensor,
        edge_displacement: Tensor,
        edge_weight: Optional[Tensor] = None,
        *,
        return_auxiliary: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        compute_dtype = _real_compute_dtype(edge_displacement.dtype)
        frequencies = self.frequencies.to(
            device=edge_displacement.device,
            dtype=compute_dtype,
        )
        laplacian, edge_transport = build_connection_laplacians(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_displacement=edge_displacement,
            frequencies=frequencies,
            edge_weight=edge_weight,
            validate_undirected_once=self.validate_undirected_once,
        )
        heat_kernel, eigenvalues, eigenvectors = connection_heat_kernel(
            laplacian,
            diffusion_time=self.diffusion_time,
            method=self.heat_kernel_method,
            return_eigensystem=return_auxiliary,
        )
        transport = heat_kernel_to_transport(
            heat_kernel,
            eps=self.eps,
            soft_normalization=self.soft_normalization,
        )
        if not return_auxiliary:
            return transport
        if eigenvalues is None or eigenvectors is None:
            raise RuntimeError("return_auxiliary=True requires an eigensystem")
        auxiliary = {
            "frequencies": frequencies,
            "edge_transport": edge_transport,
            "connection_laplacian": laplacian,
            "heat_kernel": heat_kernel,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        }
        return transport, auxiliary


class HolonomyRoPEAttention(nn.Module):
    """Dense exact pairwise Holonomy-RoPE softmax attention.

    ``x`` has shape ``(B, N, D)``.  One graph topology is shared by every item
    in the batch, so the transport has shape ``(rotary_dim / 2, N, N)``.  A
    batch of unrelated variable-size graphs should call this reference module
    once per graph (or pad graphs and supply separately precomputed transports).

    Boolean ``attention_mask`` values use ``True = allowed`` and must be
    broadcastable to ``(B, num_heads, N, N)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        rotary_dim: Optional[int] = None,
        rope_base: float = 10_000.0,
        diffusion_time: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        soft_phase_normalization: bool = False,
        heat_kernel_method: HeatKernelMethod = "eigh",
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        rotary_dim = self.head_dim if rotary_dim is None else rotary_dim
        if rotary_dim <= 0 or rotary_dim > self.head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be positive, even, and no larger than head_dim")
        self.rotary_dim = rotary_dim
        self.num_rotary_pairs = rotary_dim // 2

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.holonomy_transport = ExactHolonomyTransport(
            rotary_dim=rotary_dim,
            base=rope_base,
            diffusion_time=diffusion_time,
            soft_normalization=soft_phase_normalization,
            heat_kernel_method=heat_kernel_method,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, _ = x.shape
        return x.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

    def _to_complex(self, x: Tensor) -> Tensor:
        paired = x.reshape(*x.shape[:-1], self.num_rotary_pairs, 2)
        if paired.dtype == torch.float64:
            return torch.complex(paired[..., 0], paired[..., 1])
        return torch.complex(paired[..., 0].float(), paired[..., 1].float())

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_displacement: Tensor,
        edge_weight: Optional[Tensor] = None,
        *,
        attention_mask: Optional[Tensor] = None,
        precomputed_transport: Optional[Tensor] = None,
        return_attention: bool = False,
        return_transport: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"x must have shape (B, N, {self.embed_dim})")
        _, num_nodes, _ = x.shape
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        if precomputed_transport is None:
            transport = self.holonomy_transport(
                num_nodes=num_nodes,
                edge_index=edge_index,
                edge_displacement=edge_displacement,
                edge_weight=edge_weight,
            )
            if not isinstance(transport, Tensor):
                raise RuntimeError("unexpected auxiliary transport result")
        else:
            transport = precomputed_transport
        expected_shape = (self.num_rotary_pairs, num_nodes, num_nodes)
        if tuple(transport.shape) != expected_shape:
            raise ValueError(
                f"transport shape {tuple(transport.shape)} does not equal expected {expected_shape}"
            )

        query_complex = self._to_complex(query[..., : self.rotary_dim])
        key_complex = self._to_complex(key[..., : self.rotary_dim])
        transport = transport.to(device=x.device, dtype=query_complex.dtype)
        rotary_scores = torch.einsum(
            "bhif,fij,bhjf->bhij",
            query_complex.conj(),
            transport,
            key_complex,
        ).real

        if self.rotary_dim < self.head_dim:
            tail_scores = torch.einsum(
                "bhid,bhjd->bhij",
                query[..., self.rotary_dim :].float(),
                key[..., self.rotary_dim :].float(),
            )
            scores = rotary_scores + tail_scores
        else:
            scores = rotary_scores
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            mask = attention_mask.to(device=scores.device)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask.to(dtype=scores.dtype)

        attention = self.attn_dropout(torch.softmax(scores, dim=-1))
        output = torch.einsum("bhij,bhjd->bhid", attention, value.float())
        output = output.transpose(1, 2).contiguous().view(x.shape[0], num_nodes, self.embed_dim)
        output = self.out_proj(output.to(dtype=x.dtype))

        outputs: list[Tensor] = [output]
        if return_attention:
            outputs.append(attention)
        if return_transport:
            outputs.append(transport)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
