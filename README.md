# AW-RoPE

Analytic Walk Rotary Position Encodings for graphs, implemented in PyTorch.
AW-RoPE assigns antisymmetric phases to edges and transports rotary feature
pairs along graph walks. The implementation includes learned edge fields,
ordinary and non-backtracking walks, and multi-scale resolvent mixtures.

For a phase-weighted transition operator `T`, the two implementations compute:

```text
Exact:   Y = (I - z T)^-1 X
Sparse:  Y = sum(k=0,...,K) z^k T^k X
```

The sparse recurrence avoids a dense positional matrix. Its transport cost
is `O(K (|V| + |E|) d)`; backbone attention and edge-field computation have
their own costs. Exact uses a differentiable linear solve. Both can act at
the same query/key interface.

## Installation

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python examples/minimal.py
python -m pytest tests/test_core.py tests/test_fields.py
```

For graph datasets and GraphGPS integration, install the additional dependencies:

```bash
python -m pip install -e '.[data,wire,test]'
```

Some upstream GraphGPS configurations also require PyG compiled extensions or
additional model dependencies. See [the installation guide](docs/installation.md).
The checked-in `uv.lock` provides the existing dependency resolution for `uv` users.

## Minimal use

```python
import torch
from aw_rope import AWRoPE, edge_displacement_from_positions

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
positions = torch.arange(3, dtype=torch.float32)
displacement = edge_displacement_from_positions(positions, edge_index)
x = torch.randn(3, 64)

rope = AWRoPE(dim=64, num_steps=16, initial_z=0.8, learnable_z=True)
y = rope(x, edge_index, displacement)
assert y.shape == x.shape
```

`AntisymmetricEdgeField` learns displacements from features. Set
`non_backtracking=True` to exclude immediate edge reversals, or use
`MultiScaleAWRoPE` for a mixture of propagation scales.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/aw_rope/core.py` | Sparse transport, rotary pairs and truncation bounds |
| `src/aw_rope/fields.py` | Learned antisymmetric edge fields |
| `src/aw_rope/experiments/` | Dataset loaders, GIN models and training utilities |
| `scripts/exact_aw_operator.py` | Complete analytic-walk solve and backend adapter |
| `external/Graph-RoPE/` | Vendored GraphGPS integration and attention implementations |
| `configs/` and `experiments/` | Smoke configuration and experiment protocols |
| `scripts/` | Constructed tasks, operator diagnostics and scaling measurements |
| `examples/` | Small executable examples |
| `tests/` | Operator, gradient, integration and dataset regression checks |

See [reproduction instructions](docs/reproduction.md) for GraphGPS, Exact/Sparse
diagnostics, constructed tasks and dataset conventions. Run the complete
repository test suite with `python -m pytest` after installing integration
dependencies. Tests use temporary synthetic data and require no private results.

## Method boundaries

Truncation guarantees compare the same parameters under a contraction condition;
they do not equate separately trained models. Dense Exact can be faster on small
graphs. Constructed-task separations apply to their stated model and readout
classes; nonlocal dynamic flat fields remain explicit controls.

The legacy `ExactHolonomyTransport` API implements a connection heat kernel.
It is retained for compatibility and is separate from Exact AW-RoPE.

## Third-party code

The integration builds on [Graph-RoPE](https://github.com/cederikhoefs/Graph-RoPE)
and [GraphGPS](https://github.com/rampasek/GraphGPS).
Upstream copyright and license notices are preserved in
[`external/Graph-RoPE/LICENSE`](external/Graph-RoPE/LICENSE).
See [third-party notices](THIRD_PARTY_NOTICES.md).
