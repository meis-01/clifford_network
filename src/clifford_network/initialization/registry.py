from __future__ import annotations

from collections.abc import Callable

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
    method_key = method.lower()
    if method_key not in INITIALIZATION_REGISTRY:
        available = ", ".join(sorted(INITIALIZATION_REGISTRY))
        raise ValueError(f"Unknown initialization method '{method}'. Available: {available}.")

    initializer = INITIALIZATION_REGISTRY[method_key]
    values = initializer(tuple(tensor.shape), device=tensor.device, dtype=tensor.dtype, **kwargs)
    with torch.no_grad():
        tensor.copy_(values)


def initialize_model(model: torch.nn.Module, method: str, **kwargs: float) -> None:
    for parameter in model.parameters():
        if not parameter.is_complex():
            continue
        if parameter.ndim >= 2:
            initialize_tensor(parameter, method, **kwargs)
        else:
            with torch.no_grad():
                parameter.zero_()
