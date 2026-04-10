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


class KSpaceClassifier(nn.Module):
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