"""Disable optional PyG wheels when their build-time Torch ABI is absent."""

from __future__ import annotations

import importlib.abc
import os
import sys


OPTIONAL_PYG_EXTENSIONS = frozenset(
    {
        "pyg_lib",
        "torch_cluster",
        "torch_scatter",
        "torch_sparse",
        "torch_spline_conv",
    }
)


class RejectIncompatiblePyGExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        if fullname.partition(".")[0] in OPTIONAL_PYG_EXTENSIONS:
            raise ModuleNotFoundError(
                f"optional PyG extension {fullname!r} is disabled because its "
                "binary ABI does not match the container PyTorch",
                name=fullname,
            )
        return None


def needs_native_pytorch_fallback() -> bool:
    if os.environ.get("AW_ROPE_FORCE_PURE_PYG") == "1":
        return True
    try:
        import torch
    except Exception:
        return False
    # The reference extension environment uses pt24cu121 wheels.
    version = torch.__version__.split("+")[0].split(".")[:2]
    return tuple(int(part) for part in version) != (2, 4)


def install() -> bool:
    if not needs_native_pytorch_fallback():
        return False
    if not any(isinstance(finder, RejectIncompatiblePyGExtensions) for finder in sys.meta_path):
        sys.meta_path.insert(0, RejectIncompatiblePyGExtensions())
    os.environ["AW_ROPE_PYG_EXTENSION_MODE"] = "native-pytorch-fallback"
    return True
