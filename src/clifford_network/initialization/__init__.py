"""Public initialization registry exports.

Importing from this package exposes config-name lookup and helper functions for
initializing complex tensors or every complex parameter in a model.
"""

from clifford_network.initialization.registry import (
    INITIALIZATION_REGISTRY,
    initialize_model,
    initialize_tensor,
)

__all__ = ["INITIALIZATION_REGISTRY", "initialize_model", "initialize_tensor"]
