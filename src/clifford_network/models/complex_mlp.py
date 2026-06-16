"""Complex multilayer perceptron model definitions.

This module provides the classifier and autoencoder architectures used by the
experiments, assembled from complex linear layers and configurable activations.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from clifford_network.activations import build_activation
from clifford_network.models.layers import ComplexLinear


class ComplexMLPClassifier(nn.Module):
    """Complex MLP classifier that returns real logits from a complex network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        num_classes: int,
        activation: str,
    ) -> None:
        """Build a stack of complex hidden layers followed by a complex head."""
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
        """Return the complex-valued logits before taking the real component."""
        return self.network(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return real-valued logits for classification losses."""
        return self.forward_complex(values).abs()


class ComplexMLPAutoencoder(nn.Module):
    """Complex MLP autoencoder with configurable encoder and decoder depth."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        activation: str,
        latent_size: int | None = None,
    ) -> None:
        """Build an encoder-decoder stack that reconstructs complex inputs."""
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
        """Return complex reconstructions for the provided inputs."""
        return self.network(values)
