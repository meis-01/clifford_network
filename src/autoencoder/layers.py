from __future__ import annotations

import torch
from torch import nn


class ComplexModReLU(nn.Module):
    def __init__(self, channels: int, eps: float = 1.0e-8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        if x.ndim == 4:
            bias = self.bias.view(1, -1, 1, 1)
        elif x.ndim == 2:
            bias = self.bias.view(1, -1)
        else:
            raise ValueError(f"Unsupported tensor shape for ComplexModReLU: {x.shape}")
        activated = torch.relu(magnitude + bias)
        return activated * x / magnitude.clamp_min(self.eps)


class ComplexCardioid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.cos(torch.angle(x))) * x


def build_complex_activation(name: str, channels: int) -> nn.Module:
    normalized = name.lower()
    if normalized == "modrelu":
        return ComplexModReLU(channels)
    if normalized == "cardioid":
        return ComplexCardioid()
    if normalized in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unknown complex activation: {name}")

class ComplexdenseBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activation: str,
        use_bias: bool,
    ):
        super().__init__()
        self.linear = nn.Linear(
            in_features, out_features, bias=use_bias, dtype=torch.cfloat
        )
        self.activation = build_complex_activation(activation, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class ComplexConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        activation: str,
        use_bias: bool,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=use_bias,
            dtype=torch.complex64,
        )
        self.activation = build_complex_activation(activation, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class ComplexConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: str,
        use_bias: bool,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias,
            dtype=torch.complex64,
        )
        self.activation = build_complex_activation(activation, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))
