"""Dataset selection and dataloader construction.

The registry maps experiment config names to synthetic, FFT vision, or fastMRI
datasets and returns both PyTorch dataloaders and model-relevant dataset metadata.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from clifford_network.dataset.fastmri_t2 import build_fastmri_t2_datasets
from clifford_network.dataset.fft_vision import build_fft_vision_datasets
from clifford_network.dataset.synthetic import SyntheticAutoencoderDataset, SyntheticClassificationDataset


@dataclass(frozen=True)
class DataSpec:
    """Describes the flattened input shape, class count, and task for a dataset."""

    input_size: int
    num_classes: int | None
    task: str


def _seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy RNGs inside DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader(dataset, config: dict, shuffle: bool, *, seed: int) -> DataLoader:
    """Create a DataLoader using shared training settings from the config."""
    training_config = config.get("training", {})
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 128)),
        shuffle=shuffle,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=bool(training_config.get("pin_memory", False)),
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def build_dataloaders(config: dict, *, seed: int) -> tuple[dict[str, DataLoader], DataSpec]:
    """Build train, validation, and test dataloaders for the configured dataset."""
    dataset_config = config.get("dataset", {})
    name = dataset_config.get("name", "synthetic_classification").lower()
    task = config.get("experiment", {}).get("task", "classification").lower()

    if name == "synthetic_classification":
        input_size = int(dataset_config.get("input_size", 32))
        num_classes = int(dataset_config.get("num_classes", 2))
        train = SyntheticClassificationDataset(int(dataset_config.get("train_samples", 256)), input_size, num_classes, seed)
        validation = SyntheticClassificationDataset(int(dataset_config.get("validation_samples", 128)), input_size, num_classes, seed + 1)
        test = SyntheticClassificationDataset(int(dataset_config.get("test_samples", 128)), input_size, num_classes, seed + 2)
        spec = DataSpec(input_size=input_size, num_classes=num_classes, task="classification")
    elif name == "synthetic_autoencoder":
        input_size = int(dataset_config.get("input_size", 32))
        train = SyntheticAutoencoderDataset(int(dataset_config.get("train_samples", 256)), input_size, seed)
        validation = SyntheticAutoencoderDataset(int(dataset_config.get("validation_samples", 128)), input_size, seed + 1)
        test = SyntheticAutoencoderDataset(int(dataset_config.get("test_samples", 128)), input_size, seed + 2)
        spec = DataSpec(input_size=input_size, num_classes=None, task="autoencoder")
    elif name in {"mnist_fft", "cifar10_fft"}:
        train, validation, test, input_size, num_classes = build_fft_vision_datasets(config, seed)
        spec = DataSpec(input_size=input_size, num_classes=num_classes, task="classification")
    elif name == "fastmri_t2":
        train, validation, test, input_size, num_classes = build_fastmri_t2_datasets(config)
        spec = DataSpec(input_size=input_size, num_classes=num_classes, task="autoencoder")
    else:
        raise ValueError(f"Unknown dataset '{name}'.")

    if spec.task != task:
        raise ValueError(f"Experiment task '{task}' does not match dataset task '{spec.task}'.")

    return {
        "train": _loader(train, config, shuffle=True, seed=seed),
        "validation": _loader(validation, config, shuffle=False, seed=seed + 1),
        "test": _loader(test, config, shuffle=False, seed=seed + 2),
    }, spec
