# Reproduction

Run commands from the repository root in the environment described in
[installation.md](installation.md). Data, trained weights and measured results
are separate from this code distribution.

## Operator checks

```bash
python examples/minimal.py
python -m pytest tests/test_core.py tests/test_fields.py \
  tests/test_exact_aw_operator.py tests/test_exact_aw_runtime.py \
  tests/test_exact_aw_theory.py
python scripts/exact_aw_diagnostics.py --output runs/exact-diagnostics --device cpu
```

These cover sparse/dense agreement, gradients, truncation and backend
equivalence. The Exact implementation is `scripts/exact_aw_operator.py`;
the legacy heat-kernel implementation has a separate API.

## Constructed tasks

Small CPU runs can check the complete training path:

```bash
python scripts/route_phase_prediction.py --arm aw --steps 10 \
  --train-pairs 16 --val-pairs 8 --test-pairs 8 --batch-pairs 4 \
  --eval-every 5 --out runs/route-smoke
python scripts/full_network_field_prediction.py --arm unrestricted \
  --steps 10 --out runs/full-network-smoke
python scripts/inductive_field_prediction.py --arm unrestricted \
  --steps 10 --out runs/inductive-smoke
```

These commands are smoke checks, not full reported protocols. Each program's
`--help` exposes its seeds, training budget, readout and controls. The route
endpoint and complete-attention readouts have different theorem scopes. The
inductive task uses disjoint graph sizes for training, validation and testing;
its deterministic static-field obstruction does not cover arbitrary-depth
networks or input-dependent nonlocal flat fields.

## GIN and prepared datasets

```bash
python -m aw_rope.experiments.train --config configs/smoke/synthetic-nb.json \
  --data-root data --output-root runs/gin-smoke
```

This command needs the corresponding prepared WIRE synthetic assets; it does
not synthesize a substitute dataset. Inspect `load_dataset` in
`src/aw_rope/experiments/datasets.py` for dataset-specific paths.
LRGB processed splits use `data/lrgb/<dataset>/processed/{train,val,test}.pt`.
COCO-SP uses 81 node classes and Macro-F1. The loader preserves official split
ordering and supports both two-item and three-item PyG serialized stores.

The fixed COCO GIN protocol is in `experiments/coco_gin_20260916.json`.
`scripts/coco_gin_20260916.py --help` exposes its data preflight, individual
trials and complete-cohort reporting. Keep every declared seed and select
checkpoints using validation; test-selected exploratory protocols must be
identified separately.

## GraphGPS / Graph Transformer

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/external/Graph-RoPE${PYTHONPATH:+:$PYTHONPATH}"
python external/Graph-RoPE/main.py \
  --cfg external/Graph-RoPE/configs/AW-RoPE/wire-synthetic-common.yaml \
  dataset.dir data out_dir runs/graphgps-aw \
  gt.graphrope.method aw gt.graphrope.aw.num_steps 16 gt.graphrope.aw.z 0.8
```

Use `none`, `wire`, `aw`, `aw-nb` or `aw-nb-ms` for the positional method.
The common configuration can still supply LapPE as a backbone input even
when AW transport itself needs no eigendecomposition. Preserve data splits,
input features and initialization when comparing methods. See the
[integration notes](../external/Graph-RoPE/configs/AW-RoPE/README.md).

For matched Exact training, `scripts/exact_aw_entry.py` installs the complete
solve at the same interface. Set `AW_EXACT_ARM` to `exact-aw` or `sparse-aw`,
`AW_EXACT_PHASE` to `train`, and `AW_EXACT_REPORT` to an output JSON path;
pass the same GraphGPS arguments. The frozen scientific settings are in
`experiments/exact_aw_synthetic_20260906.json`.

## Scaling measurements

`scripts/exact_sparse_scaling.py` reads
`experiments/exact_sparse_scaling_20260907.json`. Its `plan`, `worker`, `run`
and `aggregate` actions separate individual measurements from scheduling.
The `run` action expects the declared four CUDA devices. Measurements cover
layer forward and backward, not end-to-end training. Keep CUDA OOM, timeout
and software errors distinct and retain all configured cells.
