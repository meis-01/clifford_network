from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.autoencoder.layers import ComplexConvBlock, ComplexConvTransposeBlock


class ComplexAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: list[int] | tuple[int, ...] = (16, 32, 64, 128),
        latent_channels: int = 128,
        activation: str = "modrelu",
        use_bias: bool = True,
    ):
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one value.")

        encoder_layers: list[nn.Module] = []
        current_channels = in_channels
        for output_channels in channels:
            encoder_layers.append(
                ComplexConvBlock(
                    current_channels,
                    int(output_channels),
                    stride=2,
                    activation=activation,
                    use_bias=use_bias,
                )
            )
            current_channels = int(output_channels)
        encoder_layers.append(
            ComplexConvBlock(
                current_channels,
                latent_channels,
                stride=1,
                activation=activation,
                use_bias=use_bias,
            )
        )
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        current_channels = latent_channels
        for output_channels in reversed(channels):
            decoder_layers.append(
                ComplexConvTransposeBlock(
                    current_channels,
                    int(output_channels),
                    activation=activation,
                    use_bias=use_bias,
                )
            )
            current_channels = int(output_channels)
        decoder_layers.append(
            nn.Conv2d(
                current_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                dtype=torch.complex64,
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError(f"ComplexAutoencoder expects a complex tensor, got dtype={x.dtype}")
        return self.decode(self.encode(x))


def build_autoencoder(config: dict[str, Any]) -> ComplexAutoencoder:
    model_config = config["model"]
    return ComplexAutoencoder(
        in_channels=int(model_config["in_channels"]),
        channels=[int(value) for value in model_config["channels"]],
        latent_channels=int(model_config["latent_channels"]),
        activation=str(model_config["activation"]),
        use_bias=bool(model_config["use_bias"]),
    )
