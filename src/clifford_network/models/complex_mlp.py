from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from clifford_network.activations import build_activation
from clifford_network.models.layers import ComplexLinear


class ComplexMLPClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        num_classes: int,
        activation: str,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        in_features = input_size
        for layer_idx in range(depth):
            layers[f"linear_{layer_idx:03d}"] = ComplexLinear(in_features, hidden_size)
            layers[f"activation_{layer_idx:03d}"] = build_activation(activation, channels=hidden_size)
            in_features = hidden_size
        layers["head"] = ComplexLinear(in_features, num_classes)
        self.network = nn.Sequential(layers)

    def forward_complex(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_complex(values).real


class ComplexMLPAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        activation: str,
        latent_size: int | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        latent = latent_size or hidden_size
        encoder_depth = max(1, depth // 2)
        decoder_depth = max(1, depth - encoder_depth)
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        in_features = input_size
        for layer_idx in range(encoder_depth):
            out_features = latent if layer_idx == encoder_depth - 1 else hidden_size
            layers[f"encoder_linear_{layer_idx:03d}"] = ComplexLinear(in_features, out_features)
            layers[f"encoder_activation_{layer_idx:03d}"] = build_activation(activation, channels=out_features)
            in_features = out_features

        for layer_idx in range(decoder_depth):
            out_features = input_size if layer_idx == decoder_depth - 1 else hidden_size
            layers[f"decoder_linear_{layer_idx:03d}"] = ComplexLinear(in_features, out_features)
            if layer_idx != decoder_depth - 1:
                layers[f"decoder_activation_{layer_idx:03d}"] = build_activation(activation, channels=out_features)
            in_features = out_features

        self.network = nn.Sequential(layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)
