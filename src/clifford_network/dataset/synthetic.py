"""Synthetic complex-valued datasets for smoke tests and quick experiments.

These small deterministic datasets avoid external downloads while exercising the
classification and autoencoder training paths with complex tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticSpec:
    """Describes the shape, class count, and task for a synthetic dataset."""

    input_size: int
    num_classes: int | None
    task: str


class SyntheticClassificationDataset(Dataset):
    """Creates labeled complex samples from a fixed random linear teacher."""

    def __init__(self, n_samples: int, input_size: int, num_classes: int, seed: int) -> None:
        """Generate deterministic inputs and teacher-derived class targets."""
        generator = torch.Generator().manual_seed(seed)
        self.values = torch.randn(n_samples, input_size, generator=generator, dtype=torch.complex64)
        teacher = torch.randn(input_size, num_classes, generator=generator, dtype=torch.complex64)
        logits = (self.values @ teacher).real
        self.targets = logits.argmax(dim=1)

    def __len__(self) -> int:
        """Return the number of generated classification samples."""
        return self.values.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one complex sample and its class target."""
        return self.values[index], self.targets[index]


class SyntheticAutoencoderDataset(Dataset):
    """Creates complex samples whose targets are exact reconstructions."""

    def __init__(self, n_samples: int, input_size: int, seed: int) -> None:
        """Generate deterministic complex inputs for reconstruction."""
        generator = torch.Generator().manual_seed(seed)
        self.values = torch.randn(n_samples, input_size, generator=generator, dtype=torch.complex64)

    def __len__(self) -> int:
        """Return the number of generated autoencoder samples."""
        return self.values.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one complex sample as both input and target."""
        value = self.values[index]
        return value, value
