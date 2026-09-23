"""Analytic Walk Rotary Position Encodings.

Public operators are loaded lazily. Dataset and training dependencies remain
optional when importing the core package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import (
        AWRoPE,
        MultiScaleAWRoPE,
        build_reverse_edge_index,
        complex_walk_step,
        edge_displacement_from_positions,
        minimum_resolvent_steps,
        non_backtracking_walk_resolvent,
        resolvent_tail_bound,
        rotate_pairs,
        rotary_edge_scores,
        standard_frequencies,
        truncated_complex_walk_resolvent,
        truncated_walk_resolvent,
        walk_step,
    )
    from .fields import AntisymmetricEdgeField
    from .exact_holonomy import (
        ExactHolonomyTransport,
        HolonomyRoPEAttention,
        build_connection_laplacians,
        build_rope_frequencies,
        connection_heat_kernel,
        heat_kernel_to_transport,
    )
    from .static_holonomy import (
        STATIC_FIELD_PROTOCOL,
        canonical_undirected_edges,
        cycle_flux,
        static_topology_edge_displacement,
        topology_return_descriptor,
    )

__all__ = [
    "AWRoPE",
    "AntisymmetricEdgeField",
    "ExactHolonomyTransport",
    "HolonomyRoPEAttention",
    "MultiScaleAWRoPE",
    "STATIC_FIELD_PROTOCOL",
    "build_reverse_edge_index",
    "build_connection_laplacians",
    "build_rope_frequencies",
    "canonical_undirected_edges",
    "complex_walk_step",
    "connection_heat_kernel",
    "cycle_flux",
    "edge_displacement_from_positions",
    "non_backtracking_walk_resolvent",
    "minimum_resolvent_steps",
    "heat_kernel_to_transport",
    "rotate_pairs",
    "rotary_edge_scores",
    "resolvent_tail_bound",
    "standard_frequencies",
    "static_topology_edge_displacement",
    "topology_return_descriptor",
    "truncated_complex_walk_resolvent",
    "truncated_walk_resolvent",
    "walk_step",
]

_FIELD_EXPORTS = {"AntisymmetricEdgeField"}
_EXACT_EXPORTS = {
    "ExactHolonomyTransport",
    "HolonomyRoPEAttention",
    "build_connection_laplacians",
    "build_rope_frequencies",
    "connection_heat_kernel",
    "heat_kernel_to_transport",
}
_STATIC_HOLONOMY_EXPORTS = {
    "STATIC_FIELD_PROTOCOL",
    "canonical_undirected_edges",
    "cycle_flux",
    "static_topology_edge_displacement",
    "topology_return_descriptor",
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _FIELD_EXPORTS:
        module_name = ".fields"
    elif name in _STATIC_HOLONOMY_EXPORTS:
        module_name = ".static_holonomy"
    elif name in _EXACT_EXPORTS:
        module_name = ".exact_holonomy"
    else:
        module_name = ".core"
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
