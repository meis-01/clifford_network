"""FFT-transformed torchvision datasets for complex classification experiments.

This module wraps MNIST and CIFAR-10 samples, converts images into orthonormal
2-D Fourier coefficients, and exposes flattened complex tensors for classifiers.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset


def _load_torchvision_dataset(name: str, root: Path, train: bool, download: bool):
    """Load the requested torchvision dataset with a tensor transform."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for MNIST FFT and CIFAR10 FFT datasets.") from exc

    transform = transforms.ToTensor()
    if name == "mnist_fft":
        return datasets.MNIST(root=str(root), train=train, download=download, transform=transform)
    if name == "cifar10_fft":
        return datasets.CIFAR10(root=str(root), train=train, download=download, transform=transform)
    raise ValueError(f"Unsupported FFT vision dataset '{name}'.")


class FFTVisionDataset(Dataset):
    """Wraps an image dataset and returns flattened complex FFT features."""

    def __init__(self, dataset: Dataset, max_items: int | None = None) -> None:
        """Optionally cap the wrapped dataset to the first `max_items` samples."""
        if max_items is not None:
            dataset = Subset(dataset, range(min(max_items, len(dataset))))
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the number of wrapped samples."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert one image to complex FFT features and return its class label."""
        image, label = self.dataset[index]
        if image.ndim == 3:
            fft_image = torch.fft.fft2(image.to(torch.float32), dim=(-2, -1), norm="ortho")
        else:
            fft_image = torch.fft.fft2(image.unsqueeze(0).to(torch.float32), dim=(-2, -1), norm="ortho")
        return fft_image.reshape(-1).to(torch.complex64), torch.tensor(label, dtype=torch.long)


def _optional_int(value) -> int | None:
    """Convert optional config values to integers."""
    return None if value is None else int(value)


def _split_train_validation(
    dataset: Dataset,
    *,
    seed: int,
    validation_samples: int | None,
    validation_fraction: float,
    max_train: int | None,
    max_eval: int | None,
) -> tuple[Dataset, Dataset]:
    """Create deterministic, disjoint train and validation subsets."""
    if len(dataset) < 2:
        raise ValueError("FFT vision datasets need at least two training samples to create a validation split.")

    if validation_samples is None:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be greater than 0 and less than 1.")
        validation_count = max(1, int(round(len(dataset) * validation_fraction)))
    else:
        validation_count = validation_samples

    if max_eval is not None:
        validation_count = min(validation_count, max_eval)
    validation_count = min(validation_count, len(dataset) - 1)
    if validation_count < 1:
        raise ValueError("FFT vision validation split is empty.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    if max_train is not None:
        train_indices = train_indices[:max_train]
    if not train_indices:
        raise ValueError("FFT vision training split is empty after applying max_train.")

    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def build_fft_vision_datasets(config: dict, seed: int) -> tuple[Dataset, Dataset, Dataset, int, int]:
    """Build FFT vision train, validation, and test datasets from config."""
    dataset_config = config.get("dataset", {})
    name = dataset_config["name"].lower()
    root = Path(dataset_config.get("root", "data"))
    download = bool(dataset_config.get("download", False))
    max_train = _optional_int(dataset_config.get("max_train"))
    max_eval = _optional_int(dataset_config.get("max_eval"))
    validation_samples = _optional_int(dataset_config.get("validation_samples"))
    validation_fraction = float(dataset_config.get("validation_fraction", 0.1))

    train_raw = _load_torchvision_dataset(name, root, train=True, download=download)
    test_raw = _load_torchvision_dataset(name, root, train=False, download=download)

    train_subset, validation_subset = _split_train_validation(
        train_raw,
        seed=seed,
        validation_samples=validation_samples,
        validation_fraction=validation_fraction,
        max_train=max_train,
        max_eval=max_eval,
    )
    train_dataset = FFTVisionDataset(train_subset)
    validation_dataset = FFTVisionDataset(validation_subset)
    test_dataset = FFTVisionDataset(test_raw, max_items=max_eval)

    sample, _ = train_dataset[0]
    return train_dataset, validation_dataset, test_dataset, int(sample.numel()), 10
