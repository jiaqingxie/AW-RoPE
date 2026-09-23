# AW-RoPE on the official WIRE Graph Transformer

`wire-synthetic-common.yaml` is one shared experiment definition for NoPE,
WIRE, AW-RoPE, non-backtracking AW-RoPE, and multiscale non-backtracking
AW-RoPE.  The official `GraphRoPE` QKV projection, attention implementation,
output projection, Graph Transformer residual blocks, normalisation, FFN,
pooling head, data splits, and optimizer are unchanged between methods.

Only `gt.graphrope.method` changes:

| value | Q/K positional operation | needs spectrum |
|---|---|---|
| `none` | identity | no (strict config still supplies common input LapPE) |
| `wire` | official WIRE spectral rotation | yes |
| `aw` | truncated analytic walk resolvent | no |
| `aw-nb` | non-backtracking analytic walk resolvent | no |
| `aw-nb-ms` | learned multiscale non-backtracking resolvent mixture | no |

The main config uses `aw.field_hidden_dim: 1` for a parameter-matched primary
comparison: with `m=5`, the full 4-layer WIRE model has 39,644 parameters and
AW has 39,575 (a 0.17% difference).  Wider edge fields belong in a separately
labelled capacity ablation, not the primary table.

Example direct runs from the AW-ROPE project root:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/external/Graph-RoPE${PYTHONPATH:+:$PYTHONPATH}"

python external/Graph-RoPE/main.py \
  --cfg external/Graph-RoPE/configs/AW-RoPE/wire-synthetic-common.yaml \
  gt.graphrope.method wire

python external/Graph-RoPE/main.py \
  --cfg external/Graph-RoPE/configs/AW-RoPE/wire-synthetic-common.yaml \
  gt.graphrope.method aw-nb gt.graphrope.aw.num_steps 16 gt.graphrope.aw.z 0.6
```

The local WIRE assets are loaded by `WIRESyntheticDataset`; the loader never
downloads.  It uses 9,000/1,000 examples from the official 10,000 training
graphs for search and preserves the official 1,000-graph test split.  Refit on
all 10,000 training graphs before final test evaluation.

For WIRE's spectral dimension sweep, change these three values together:

```text
gt.graphrope.t_dim
posenc_LapRoPE.dim_pe
posenc_LapRoPE.eigen.max_freqs
```

When the common LapPE node input dimension is also swept, change
`posenc_LapPE.dim_pe` and `posenc_LapPE.eigen.max_freqs` as well.  Synthetic
normalized RMSE is the reported RMSE divided by 25 for monochromatic-subgraph
tasks and by 10 for Watts-Strogatz SPD.
