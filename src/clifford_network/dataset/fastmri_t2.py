"""fastMRI T2 dataset adapter for complex autoencoder experiments.

The dataset expects preprocessed `.npy` arrays split into training, validation,
and test directories, then flattens each complex array into an input-target pair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FastMriT2AutoencoderDataset(Dataset):
    """Loads flattened complex-valued fastMRI T2 arrays for reconstruction."""

    def __init__(self, root: str | Path, split: str, max_items: int | None = None) -> None:
        """Collect `.npy` files for one split, optionally limiting item count."""
        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(f"fastMRI T2 split directory does not exist: {split_root}")
        self.paths = sorted(split_root.glob("*.npy"))
        if max_items is not None:
            self.paths = self.paths[:max_items]
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found under {split_root}")

    def __len__(self) -> int:
        """Return the number of files in this split."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one array and return it as both input and reconstruction target."""
        array = np.load(self.paths[index])
        value = torch.as_tensor(array, dtype=torch.complex64).reshape(-1)
        return value, value


def build_fastmri_t2_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset, int, None]:
    """Build train, validation, and test datasets plus the flattened input size."""
    dataset_config = config.get("dataset", {})
    root = dataset_config["root"]
    max_items = dataset_config.get("max_items")
    train_dataset = FastMriT2AutoencoderDataset(root, "training", max_items=max_items)
    validation_dataset = FastMriT2AutoencoderDataset(root, "validation", max_items=max_items)
    test_dataset = FastMriT2AutoencoderDataset(root, "test", max_items=max_items)
    sample, _ = train_dataset[0]
    return train_dataset, validation_dataset, test_dataset, int(sample.numel()), None
