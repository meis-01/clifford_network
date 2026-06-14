from __future__ import annotations

import torch


def complex_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target) ** 2)


def complex_nmse(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    numerator = torch.sum(torch.abs(prediction - target) ** 2, dim=(-3, -2, -1))
    denominator = torch.sum(torch.abs(target) ** 2, dim=(-3, -2, -1)).clamp_min(eps)
    return torch.mean(numerator / denominator)
