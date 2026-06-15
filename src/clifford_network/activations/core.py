"""Complex-valued activation layers used by the experiment models.

The classes here implement small PyTorch modules for common complex nonlinear
operators, preserving phase or gating values according to each activation's
definition.
"""

from __future__ import annotations

import torch
from torch import nn


class SplitTanh(nn.Module):
    """Applies tanh independently to real and imaginary components."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply real-valued tanh to both complex components separately."""
        return torch.complex(torch.tanh(values.real), torch.tanh(values.imag))


class ModTanh(nn.Module):
    """Applies tanh to magnitude while preserving phase."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Shrink magnitudes through tanh and reconstruct values with original phase."""
        magnitude = values.abs()
        phase = torch.angle(values)
        return torch.polar(torch.tanh(magnitude), phase)


class ZReLU(nn.Module):
    """Keeps values only when real and imaginary parts are both positive."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Zero out complex values outside the first quadrant."""
        mask = (values.real > 0) & (values.imag > 0)
        return torch.where(mask, values, torch.zeros_like(values))


class ModReLU(nn.Module):
    """Applies a learnable biased ReLU gate to complex magnitudes."""

    def __init__(self, channels: int | None = None, bias: float = 0.0) -> None:
        """Create a scalar or per-channel bias for the magnitude gate."""
        super().__init__()
        if channels is None:
            self.bias = nn.Parameter(torch.tensor(float(bias)))
        else:
            self.bias = nn.Parameter(torch.full((channels,), float(bias)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Scale each complex value by the positive part of its biased magnitude."""
        magnitude = values.abs()
        bias = self.bias
        while bias.ndim < values.ndim:
            bias = bias.unsqueeze(0)
        scale = torch.relu(magnitude + bias) / (magnitude + 1.0e-8)
        return values * scale
