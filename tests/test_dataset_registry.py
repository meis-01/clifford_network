from __future__ import annotations

import torch
from torch.utils.data import Dataset

from clifford_network.dataset import fft_vision
from clifford_network.dataset.registry import build_dataloaders


class _TinyImageDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.full((3, 32, 32), index / 10), index % 100


def test_cifar100_fft_is_registered_and_builds_complex_loaders(monkeypatch) -> None:
    loaded_names: list[str] = []

    def load_dataset(name, _root, train, _download):
        loaded_names.append(name)
        return _TinyImageDataset(4 if train else 2)

    monkeypatch.setattr(fft_vision, "_load_torchvision_dataset", load_dataset)

    loaders, spec = build_dataloaders(
        {
            "dataset": {
                "name": "cifar100_fft",
                "max_train": 3,
                "max_eval": 2,
                "validation_samples": 1,
            },
            "experiment": {"task": "classification"},
            "training": {"batch_size": 2},
        },
        seed=0,
    )

    features, labels = next(iter(loaders["train"]))
    assert spec.input_size == 3 * 32 * 32
    assert spec.num_classes == 100
    assert spec.task == "classification"
    assert loaded_names == ["cifar100_fft", "cifar100_fft"]
    assert features.shape == (2, 3 * 32 * 32)
    assert features.dtype == torch.complex64
    assert labels.dtype == torch.long
