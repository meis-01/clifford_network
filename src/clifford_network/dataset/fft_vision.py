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


def build_fft_vision_datasets(config: dict, seed: int) -> tuple[Dataset, Dataset, Dataset, int, int]:
    """Build FFT vision train, validation, and test datasets from config."""
    del seed
    dataset_config = config.get("dataset", {})
    name = dataset_config["name"].lower()
    root = Path(dataset_config.get("root", "data"))
    download = bool(dataset_config.get("download", False))
    max_train = dataset_config.get("max_train")
    max_eval = dataset_config.get("max_eval")

    train_raw = _load_torchvision_dataset(name, root, train=True, download=download)
    test_raw = _load_torchvision_dataset(name, root, train=False, download=download)

    train_dataset = FFTVisionDataset(train_raw, max_items=max_train)
    # TODO: Split the evaluation data into separate validation and test sets.
    # Currently both validation_dataset and test_dataset are built from test_raw,
    # which makes model selection and final evaluation use overlapping data.
    # Use a deterministic split with the provided seed.
    validation_dataset = FFTVisionDataset(test_raw, max_items=max_eval)
    test_dataset = FFTVisionDataset(test_raw, max_items=max_eval)

    sample, _ = train_dataset[0]
    return train_dataset, validation_dataset, test_dataset, int(sample.numel()), 10
