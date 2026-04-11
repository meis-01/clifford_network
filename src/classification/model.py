from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.block(x))


class RealValuedClassifier(nn.Module):
    def __init__(self, in_channels: int = 2, channels: list[int] | tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers = []
        current_channels = in_channels
        for next_channels in channels:
            layers.append(ConvBlock(current_channels, next_channels))
            current_channels = next_channels

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(current_channels, current_channels // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(current_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features).squeeze(-1)


class ModReLU(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        scale = torch.relu(magnitude + self.bias) / torch.clamp(magnitude, min=self.eps)
        return scale.to(dtype=x.real.dtype) * x


class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dtype=torch.complex64)
        self.act1 = ModReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, dtype=torch.complex64)
        self.act2 = ModReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


class ComplexValuedClassifier(nn.Module):
    def __init__(self, in_channels: int = 2, channels: list[int] | tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        current_channels = in_channels
        self.encoder = nn.ModuleList()
        for next_channels in channels:
            self.encoder.append(ComplexConvBlock(current_channels, next_channels))
            current_channels = next_channels

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(current_channels * 2, current_channels),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(current_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexValuedClassifier expects native complex input tensors.")
        for block in self.encoder:
            x = block(x)

        pooled = self.pool(x).flatten(1)
        features = torch.cat([pooled.real, pooled.imag], dim=1)
        return self.head(features).squeeze(-1)


def build_classifier(representation: str, in_channels: int, channels: list[int] | tuple[int, ...], dropout: float) -> nn.Module:
    representation = str(representation).lower()
    if representation == "real":
        return RealValuedClassifier(in_channels=in_channels, channels=channels, dropout=dropout)
    if representation == "complex":
        return ComplexValuedClassifier(in_channels=in_channels, channels=channels, dropout=dropout)
    raise ValueError(f"Unsupported model representation: {representation}")


KSpaceClassifier = RealValuedClassifier