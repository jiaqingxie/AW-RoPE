# Installation

The core operator needs Python >=3.10 and PyTorch >=2.4,<2.8. Install a PyTorch
build appropriate for your CPU or CUDA environment, then run:

```bash
python -m pip install -e '.[test]'
python examples/minimal.py
python -m pytest tests/test_core.py tests/test_fields.py
```

Dataset and GraphGPS support use separate dependency groups:

```bash
python -m pip install -e '.[data,wire,test]'
```

The `data` extra includes PyG 2.5.3, OGB, NumPy, SciPy and scikit-learn.
The `wire` extra includes Performer, YACS, TensorBoardX and TorchMetrics.
Some vendored upstream models use `torch-scatter`, `torch-sparse` or
`torch-cluster`; install matching wheels using the
[PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
Do not mix extension wheels built for different PyTorch/CUDA versions.

For GraphGPS commands, expose the vendored package and experiment adapters:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/external/Graph-RoPE${PYTHONPATH:+:$PYTHONPATH}"
python external/Graph-RoPE/main.py --help
```

Tests configure these import paths through `pyproject.toml`. All installation
commands are run before experiments; training does not install packages.
Dataset assets are supplied separately under a configurable data root.

With `uv`, the existing lockfile can be used via
`uv sync --extra data --extra wire --extra test`.
