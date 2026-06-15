"""Public model factory exports.

The models package exposes a single config-driven builder for classifier and
autoencoder variants built from complex linear layers and activations.
"""

from clifford_network.models.registry import build_model

__all__ = ["build_model"]
