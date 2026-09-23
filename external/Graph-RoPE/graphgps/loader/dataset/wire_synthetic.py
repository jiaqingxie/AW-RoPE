"""Offline adapters for the synthetic datasets used by the WIRE paper."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset


class WIRESyntheticDataset(InMemoryDataset):
    """Load an already prepared WIRE synthetic dataset without downloading.

    The local assets contain 10,000 official training graphs and 1,000 test
    graphs.  We deterministically reserve the last 1,000 training examples for
    hyperparameter validation.  A final-fit run can override the resulting
    split indices and train on all 10,000 official training examples.
    """

    _MONOCHROMATIC = {
        f"monochromatic-{deleted}": f"monochromatic_subgraphs/deleted_{deleted}.pt"
        for deleted in (0, 5, 10, 15)
    }
    _FILES = {
        **_MONOCHROMATIC,
        "watts-strogatz-spd": "watts_strogatz_spd/data.pt",
    }

    def __init__(self, root: str, name: str, validation_size: int = 1_000) -> None:
        if name not in self._FILES:
            raise ValueError(f"unknown WIRE synthetic dataset {name!r}")
        # root=None prevents PyG from invoking download/process hooks.  These
        # datasets must already exist before a CPU/GPU experiment starts.
        super().__init__(root=None)
        path = Path(root) / "synthetic" / self._FILES[name]
        if not path.is_file():
            raise FileNotFoundError(
                f"WIRE dataset is not prepared locally: {path}. "
                "Dataset downloads are intentionally disabled in the training loader."
            )
        (data, slices), metadata = torch.load(path, map_location="cpu", weights_only=False)
        self._data = data
        self.slices = slices
        self.name = name
        self.metadata = metadata
        self._wire_processed_dir = path.parent / ".graphrope_cache" / path.stem
        self._wire_processed_dir.mkdir(parents=True, exist_ok=True)

        official_train = [int(index) for index in metadata["train"]]
        official_test = [int(index) for index in metadata["test"]]
        if not 0 < validation_size < len(official_train):
            raise ValueError("validation_size must be smaller than the official training split")
        self.split_idxs = [
            official_train[:-validation_size],
            official_train[-validation_size:],
            official_test,
        ]
        self.official_train_idxs = official_train
        self.target_scale = 25.0 if name.startswith("monochromatic-") else 10.0

    @property
    def processed_dir(self) -> str:
        """Dataset-specific location for official spectral preprocessing."""
        return str(self._wire_processed_dir)
