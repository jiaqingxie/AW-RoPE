"""Complete analytic-walk row action. Independent of GraphGPS imports.

This is NOT the historical heat-kernel/unit-pair-phase Full-HPE.  The
finite and complete backends have identical parameters and consume Q/K.
"""
from __future__ import annotations

import torch


RUNTIME_REVISION = "exact-aw-cusolver-real-block-v3"


def configure_cuda_solver():
    """Avoid the failing MAGMA LU dispatch, including in solve's backward.

    The cuSOLVER preference routes small batched LU/solve to cuBLAS where
    appropriate. It is a backend family preference, not a claim that every
    kernel is implemented by cuSOLVER. Fail closed if it cannot be selected.
    """
    preference = torch.backends.cuda.preferred_linalg_library
    if "cusolver" not in str(preference()).lower():
        preference("cusolver")
    current = str(preference())
    if "cusolver" not in current.lower():
        raise RuntimeError("Exact-AW requires the cuSOLVER CUDA linalg preference")
    return current


def solve_complex_system(matrix, rhs, *, backend="auto"):
    """Equivalent real-block CUDA solve; native complex solve on CPU.

    CUDA explicitly selects the cuSOLVER/cuBLAS backend family: merely using
    real blocks still allowed the failing default MAGMA batched LU path.
    Representation and backend selection do not change the operator or its
    parameters. Both representations retain ordinary PyTorch autograd.
    Explicit backends are for equivalence/regression tests only.
    """
    if matrix.is_cuda:
        configure_cuda_solver()
    if backend == "auto":
        backend = "real-block" if matrix.is_cuda else "native"
    if backend == "native":
        return torch.linalg.solve(matrix, rhs)
    if backend != "real-block":
        raise ValueError("unknown Exact-AW solve backend: " + str(backend))
    block = torch.cat((torch.cat((matrix.real, -matrix.imag), dim=-1),
                       torch.cat((matrix.imag, matrix.real), dim=-1)), dim=-2)
    right = torch.cat((rhs.real, rhs.imag), dim=-2)
    result = torch.linalg.solve(block, right)
    size = matrix.shape[-1]
    return torch.complex(result[..., :size, :], result[..., size:, :])


def exact_walk_resolvent(x, edge_index, displacement, frequencies, *, z,
                         batch=None, edge_weight=None, solve_backend="auto"):
    """Solve (I-z*T)Y=X separately for each graph/channel, without inversion.

    PyG batch nodes must be contiguous. Parallel edges add, zero-degree
    rows act as identity. Padding is isolated and discarded after solving.
    Float64/32 assemble complex128/64. CUDA solves the equivalent 2N real
    system in float64/32; CPU uses a native complex solve by default.
    """
    if x.ndim != 2 or x.shape[1] % 2 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("expected float32/64 N by even-width features")
    n, width = x.shape
    if not n:
        return x.clone()
    if edge_index.dtype != torch.long or edge_index.shape[0] != 2:
        raise ValueError("expected 2 by E long edges")
    if displacement.shape != (edge_index.shape[1],) or frequencies.shape != (width//2,):
        raise ValueError("edge displacement/channel frequency shape mismatch")
    z = torch.as_tensor(z, dtype=x.dtype, device=x.device)
    if z.numel() != 1 or not bool(torch.isfinite(z)) or not bool((z >= 0) & (z < 1)):
        raise ValueError("z must be finite and in [0,1)")
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=x.device)
    batch = batch.to(device=x.device)
    if batch.dtype != torch.long or batch.shape != (n,):
        raise ValueError("batch must have one long graph index per node")
    if bool((batch[1:] < batch[:-1]).any()) or int(batch[0]) != 0:
        raise ValueError("batch nodes must be contiguous starting at graph zero")
    counts = torch.bincount(batch)
    if bool((counts == 0).any()):
        raise ValueError("empty graph IDs are not supported")
    graphs, size, channels = counts.numel(), int(counts.max()), width//2
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    local = torch.arange(n, device=x.device) - starts[batch]
    source, target = edge_index.to(device=x.device)
    if bool((batch[source] != batch[target]).any()):
        raise ValueError("cross-graph edge")
    weight = x.new_ones(source.numel()) if edge_weight is None else edge_weight.to(x)
    if weight.shape != source.shape or bool((weight < 0).any()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("edge weights must be finite and nonnegative")
    degree = x.new_zeros(n).index_add(0, source, weight)
    weight = weight / degree[source].clamp_min(torch.finfo(x.dtype).tiny)
    angles = displacement.to(x)[:, None] * frequencies.to(x)[None, :]
    values = torch.complex(angles.cos(), angles.sin()) * weight[:, None]
    channel = torch.arange(channels, device=x.device)[None, :]
    indices = (((batch[source, None] * channels + channel) * size
                + local[source, None]) * size + local[target, None])
    transition = values.new_zeros(graphs*channels*size*size).index_add(
        0, indices.flatten(), values.flatten()).reshape(graphs, channels, size, size)
    complex_x = torch.view_as_complex(x.reshape(n, channels, 2).contiguous())
    rhs_indices = (batch[:, None]*channels + channel)*size + local[:, None]
    rhs = complex_x.new_zeros(graphs*channels*size).index_add(
        0, rhs_indices.flatten(), complex_x.flatten()).reshape(graphs, channels, size, 1)
    matrix = torch.eye(size, dtype=values.dtype, device=x.device) - z*transition
    solution = solve_complex_system(matrix, rhs, backend=solve_backend).squeeze(-1)
    gathered = solution.flatten()[rhs_indices]
    return torch.view_as_real(gathered).reshape_as(x)


def install_exact_backend(aw_class, graphrope_class):
    """Opt-in only: retain AW construction/state/RNG, replace transport action."""
    if torch.cuda.is_available():
        configure_cuda_solver()
    original_transport = aw_class._transport_single
    original_forward = graphrope_class.forward

    def transport(self, x, edge_index, displacement, edge_weight, reverse_edge,
                  frequencies=None):
        if self.method != "aw" or self.normalize_resolvent:
            raise ValueError("Exact-AW protocol supports ordinary unnormalized AW only")
        return exact_walk_resolvent(x, edge_index, displacement,
            self.frequencies if frequencies is None else frequencies,
            z=self.z, edge_weight=edge_weight,
            batch=getattr(self, "_exact_aw_batch", None))

    def forward(self, batch):
        module = getattr(self, "aw_rope", None)
        if module is not None:
            module._exact_aw_batch = getattr(batch, "batch", None)
        try:
            return original_forward(self, batch)
        finally:
            if module is not None:
                module._exact_aw_batch = None

    aw_class._transport_single = transport
    graphrope_class.forward = forward
    return original_transport, original_forward
