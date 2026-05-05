import math
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_relu(z: torch.Tensor) -> torch.Tensor:
    return torch.complex(F.relu(z.real), F.relu(z.imag))


def complex_tanh(z: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.tanh(z.real), torch.tanh(z.imag))


def z_relu(z: torch.Tensor) -> torch.Tensor:
    mask = (z.real > 0) & (z.imag > 0)
    return z * mask.type_as(z.real)


def mod_relu(z: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
    magnitude = torch.abs(z)
    scale = F.relu(magnitude + bias) / (magnitude + 1e-9)
    return z * scale


def mod_tanh(z: torch.Tensor) -> torch.Tensor:
    magnitude = torch.abs(z)
    phase = torch.angle(z)
    return torch.polar(torch.tanh(magnitude), phase)


ACTIVATIONS = {
    "relu": complex_relu,
    "tanh": complex_tanh,
    "zrelu": z_relu,
    "modrelu": mod_relu,
    "modtanh": mod_tanh,
}


class ComplexLinear(nn.Module):
    """A complex-valued dense layer supporting weight initialization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.cfloat)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class ComplexMLP(nn.Module):
    """A deep complex dense network for forward activation and gradient analysis."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        n_layers: int,
        activation: str = "modrelu",
        use_output_layer: bool = True,
    ):
        super().__init__()
        self.activation_name = activation
        self.activation = ACTIVATIONS.get(activation, complex_relu)
        self.layers = nn.ModuleList()

        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        self.layers.append(ComplexLinear(in_features, hidden_size))
        for _ in range(n_layers - 1):
            self.layers.append(ComplexLinear(hidden_size, hidden_size))

        if use_output_layer:
            self.output_layer = ComplexLinear(hidden_size, hidden_size)
        else:
            self.output_layer = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        pre_activations: List[torch.Tensor] = []
        post_activations: List[torch.Tensor] = []

        for layer in self.layers:
            x = layer(x)
            pre_activations.append(x)
            x = self.activation(x)
            post_activations.append(x)

        if self.output_layer is not None:
            x = self.output_layer(x)
            pre_activations.append(x)
            x = self.activation(x)
            post_activations.append(x)

        return x, pre_activations, post_activations


def get_activation_stats(tensors: List[torch.Tensor]) -> List[dict]:
    stats = []
    for z in tensors:
        mag = torch.abs(z).detach().cpu()
        stats.append(
            {
                "mean": float(mag.mean()),
                "std": float(mag.std()),
                "min": float(mag.min()),
                "max": float(mag.max()),
            }
        )
    return stats


def get_gradient_norms(model: nn.Module) -> List[float]:
    norms = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        norms.append(float(param.grad.detach().norm()))
    return norms
