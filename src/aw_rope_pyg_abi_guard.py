"""Early, stdlib-only import guard for optional PyG binary extensions."""

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


def _incompatible() -> bool:
    if os.environ.get("AW_ROPE_FORCE_PURE_PYG") == "1":
        return True
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    version = torch.__version__.split("+")[0].split(".")[:2]
    # The reference optional-extension environment uses pt24cu121 wheels.
    return tuple(int(part) for part in version) != (2, 4)


class RejectIncompatiblePyGExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        if fullname.partition(".")[0] in OPTIONAL_PYG_EXTENSIONS and _incompatible():
            os.environ["AW_ROPE_PYG_EXTENSION_MODE"] = "native-pytorch-fallback"
            raise ModuleNotFoundError(
                f"optional PyG extension {fullname!r} is disabled because its "
                "binary ABI does not match the container PyTorch",
                name=fullname,
            )
        return None


def install() -> None:
    if not any(isinstance(finder, RejectIncompatiblePyGExtensions) for finder in sys.meta_path):
        sys.meta_path.insert(0, RejectIncompatiblePyGExtensions())
