"""Public activation factory exports.

Importing from this package exposes the activation registry and builder used by
model definitions to construct complex-valued nonlinearities from config names.
"""

from clifford_network.activations.registry import ACTIVATION_REGISTRY, build_activation

__all__ = ["ACTIVATION_REGISTRY", "build_activation"]
