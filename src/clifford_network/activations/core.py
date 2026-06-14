from __future__ import annotations

import torch
from torch import nn


class SplitTanh(nn.Module):
    """Applies tanh independently to real and imaginary components."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.tanh(values.real), torch.tanh(values.imag))


class ModTanh(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        magnitude = values.abs()
        phase = torch.angle(values)
        return torch.polar(torch.tanh(magnitude), phase)


class ZReLU(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mask = (values.real > 0) & (values.imag > 0)
        return torch.where(mask, values, torch.zeros_like(values))


class ModReLU(nn.Module):
    def __init__(self, channels: int | None = None, bias: float = 0.0) -> None:
        super().__init__()
        if channels is None:
            self.bias = nn.Parameter(torch.tensor(float(bias)))
        else:
            self.bias = nn.Parameter(torch.full((channels,), float(bias)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        magnitude = values.abs()
        bias = self.bias
        while bias.ndim < values.ndim:
            bias = bias.unsqueeze(0)
        scale = torch.relu(magnitude + bias) / (magnitude + 1.0e-8)
        return values * scale
