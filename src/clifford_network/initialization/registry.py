"""Registry and application helpers for complex initializers.

The registry converts config strings into initializer functions and provides
utilities for copying sampled values into tensors or full model parameters.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import torch

from clifford_network.initialization.baselines import (
    he_complex,
    random_complex,
    trabelsi,
    unitary_phase,
    xavier_complex,
)
from clifford_network.initialization.structured_preserve import structured_preserve

Initializer = Callable[..., torch.Tensor]

INITIALIZATION_REGISTRY: dict[str, Initializer] = {
    "random": random_complex,
    "xavier": xavier_complex,
    "he": he_complex,
    "unitary": unitary_phase,
    "trabelsi": trabelsi,
    "structured_preserve": structured_preserve,
}


def initialize_tensor(tensor: torch.Tensor, method: str, **kwargs: float) -> None:
    """Fill a complex tensor in-place using the named initializer."""
    method_key = method.lower()
    if method_key not in INITIALIZATION_REGISTRY:
        available = ", ".join(sorted(INITIALIZATION_REGISTRY))
        raise ValueError(f"Unknown initialization method '{method}'. Available: {available}.")

    initializer = INITIALIZATION_REGISTRY[method_key]
    values = initializer(tuple(tensor.shape), device=tensor.device, dtype=tensor.dtype, **kwargs)
    with torch.no_grad():
        tensor.copy_(values)


def initialize_model(model: torch.nn.Module, method: str, **kwargs: float) -> None:
    """Initialize complex model weights and zero complex bias vectors.

    For ``structured_preserve``, ``gain`` is the maximum structured gain. It
    is reached linearly at the final complex weight layer, starting from 1.0
    at the first complex weight layer.
    """
    complex_parameters = [parameter for parameter in model.parameters() if parameter.is_complex()]
    weight_parameters = [parameter for parameter in complex_parameters if parameter.ndim >= 2]
    method_key = method.lower()
    maximum_gain = float(kwargs.get("gain", 1.0))
    if method_key == "structured_preserve" and (not math.isfinite(maximum_gain) or maximum_gain < 1.0):
        raise ValueError(f"structured_preserve maximum gain must be finite and at least 1.0, got {maximum_gain}.")

    weight_index = 0
    for parameter in complex_parameters:
        if parameter.ndim >= 2:
            tensor_kwargs = dict(kwargs)
            if method_key == "structured_preserve" and "gain" in tensor_kwargs:
                progress = 1.0 if len(weight_parameters) == 1 else weight_index / (len(weight_parameters) - 1)
                tensor_kwargs["gain"] = 1.0 + progress * (maximum_gain - 1.0)
            initialize_tensor(parameter, method, **tensor_kwargs)
            weight_index += 1
        else:
            with torch.no_grad():
                parameter.zero_()
