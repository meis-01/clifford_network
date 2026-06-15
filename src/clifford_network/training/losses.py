"""Loss functions and loss selection for experiment tasks.

The module provides a complex-valued reconstruction loss and a small factory
that maps task names to the appropriate PyTorch-compatible loss callable.
"""

from __future__ import annotations

import torch
from torch import nn


def complex_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error over complex magnitudes."""
    return torch.mean(torch.abs(prediction - target) ** 2)


def build_loss(task: str) -> nn.Module | callable:
    """Return the loss function used for the requested task."""
    if task == "classification":
        return nn.CrossEntropyLoss()
    if task == "autoencoder":
        return complex_mse_loss
    raise ValueError(f"Unsupported task '{task}'.")
