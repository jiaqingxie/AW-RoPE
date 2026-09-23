# Graph-RoPE integration

This directory contains the vendored
[Graph-RoPE](https://github.com/cederikhoefs/Graph-RoPE) implementation,
which builds on [GraphGPS](https://github.com/rampasek/GraphGPS), together
with AW-RoPE query/key transport and dataset-loader adaptations.

The implementation entry point is `main.py`. Model and dataset settings
live under `configs/`. See [AW-RoPE integration notes](configs/AW-RoPE/README.md)
and the repository's [reproduction guide](../../docs/reproduction.md).

Upstream licensing is preserved in [LICENSE](LICENSE). Consult the upstream
repositories for their papers, citations and framework documentation.
