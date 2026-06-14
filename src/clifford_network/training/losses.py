from __future__ import annotations

import torch
from torch import nn


def complex_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target) ** 2)


def build_loss(task: str) -> nn.Module | callable:
    if task == "classification":
        return nn.CrossEntropyLoss()
    if task == "autoencoder":
        return complex_mse_loss
    raise ValueError(f"Unsupported task '{task}'.")
