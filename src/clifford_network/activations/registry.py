from __future__ import annotations

from torch import nn

from clifford_network.activations.core import ModReLU, ModTanh, SplitTanh, ZReLU

ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "split_tanh": SplitTanh,
    "complex_tanh": SplitTanh,
    "tanh": SplitTanh,
    "modtanh": ModTanh,
    "mod_tanh": ModTanh,
    "modrelu": ModReLU,
    "mod_relu": ModReLU,
    "zrelu": ZReLU,
    "z_relu": ZReLU,
}


def build_activation(name: str, channels: int | None = None) -> nn.Module:
    key = name.lower()
    if key not in ACTIVATION_REGISTRY:
        available = ", ".join(sorted(ACTIVATION_REGISTRY))
        raise ValueError(f"Unknown activation '{name}'. Available: {available}.")

    activation_type = ACTIVATION_REGISTRY[key]
    if activation_type is ModReLU:
        return activation_type(channels=channels)
    return activation_type()
