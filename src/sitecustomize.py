"""Keep the shared PyG environment portable across Inspire CUDA images.

The shared venv intentionally inherits the container's PyTorch.  Its optional
PyG C++ extensions were built for the local Torch 2.4 ABI, while the official
NGC 25.02 image supplies a Torch 2.7 development build.  Importing those old
extensions under the NGC build aborts before PyG can select its native-PyTorch
fallbacks.  Python imports ``sitecustomize`` after processing the editable
``src`` path, so reject only the incompatible optional extensions up front.

No training functionality used by AW-RoPE requires these packages.  Scatter
and sparse propagation use native PyTorch/PyG, and the point-cloud adapter
constructs kNN graphs with ``torch.cdist`` and ``topk``.
"""

from aw_rope._pyg_abi_guard import install


install()
