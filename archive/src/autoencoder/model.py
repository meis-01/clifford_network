from __future__ import annotations

from typing import Any

import torch
from torch import nn, tensor

from src.autoencoder.layers import ComplexdenseBlock
from src.autoencoder.weight_initializations import initialize_model


class ComplexAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: list[int] | tuple[int, ...] = (16, 32, 64, 128),
        latent_channels: int = 128,
        activation: str = "modrelu",
        use_bias: bool = True,
        weight_init: str = "xavier",
        image_size: tuple[int, int] = (320, 320),
    ):
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one value.")

        input_features = in_channels * image_size[0] * image_size[1]
        self.flatten = nn.Flatten()

        encoder_layers: list[nn.Module] = []
        current_features = input_features
        for output_features in channels:
            encoder_layers.append(
                ComplexdenseBlock(
                    current_features,
                    int(output_features),
                    activation=activation,
                    use_bias=use_bias,
                )
            )
            current_features = int(output_features)
        encoder_layers.append(
            ComplexdenseBlock(
                current_features,
                latent_channels,
                activation=activation,
                use_bias=use_bias,
            )
        )
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        current_features = latent_channels
        for output_features in reversed(channels):
            decoder_layers.append(
                ComplexdenseBlock(
                    current_features,
                    int(output_features),
                    activation=activation,
                    use_bias=use_bias,
                )
            )
            current_features = int(output_features)
        decoder_layers.append(
            nn.Linear(
                current_features,
                input_features,
                bias=use_bias,
                dtype=torch.complex64,
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)
        initialize_model(self, method=weight_init)
        
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError(f"ComplexAutoencoder expects a complex tensor, got dtype={x.dtype}")
        original_shape = x.shape
        x = self.flatten(x)
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat.view(original_shape)


def build_autoencoder(config: dict[str, Any]) -> ComplexAutoencoder:
    model_config = config["model"]
    return ComplexAutoencoder(
        in_channels=int(model_config.get("in_channels", 1)),
        channels=[int(value) for value in model_config.get("channels", [16, 32, 64, 128])],
        latent_channels=int(model_config.get("latent_channels", 128)),
        activation=str(model_config.get("activation", "modrelu")),
        use_bias=bool(model_config.get("use_bias", True)),
        weight_init=str(model_config.get("weight_init", "xavier")),
        image_size=tuple(int(v) for v in model_config.get("image_size", (320, 320))),
    )
