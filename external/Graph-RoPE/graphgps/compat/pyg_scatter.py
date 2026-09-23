"""Small ``torch_scatter`` API bridge backed by PyG/PyTorch operators.

The Inspire CUDA 12.8 image provides a newer PyTorch ABI than the optional
``torch_scatter`` wheel stored in the shared uv environment.  GraphGPS only
needs the operations below, all of which are available through
``torch_geometric.utils.scatter`` and ultimately ``Tensor.scatter_reduce_``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.utils import scatter as _pyg_scatter


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> Tensor:
    """Match the subset of ``torch_scatter.scatter`` used by GraphGPS."""
    reduce = "sum" if reduce == "add" else reduce
    if out is not None:
        if dim_size is None:
            dim_size = out.size(dim)
        result = _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
        out.copy_(result)
        return out
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Tensor | None = None,
    dim_size: int | None = None,
) -> Tensor:
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum")


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Tensor | None = None,
    dim_size: int | None = None,
    fill_value: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Return max values and compatible argmax indices.

    GraphGPS consumes only the values.  Argmax is nevertheless constructed so
    accidental callers receive a correctly shaped tensor rather than ``None``.
    """
    del fill_value
    values = scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="max")
    argmax = torch.full(values.shape, -1, dtype=torch.long, device=values.device)
    return values, argmax

