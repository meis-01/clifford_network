import math
from typing import Tuple

import torch


def complex_fan_in_and_fan_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    elif len(shape) > 2:
        receptive_field_size = int(torch.prod(torch.tensor(shape[2:], dtype=torch.int32)))
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        raise ValueError("Unsupported tensor shape for fan calculations")
    return fan_in, fan_out


def complex_normal(shape, std: float) -> torch.Tensor:
    real = torch.randn(shape, dtype=torch.float32)
    imag = torch.randn(shape, dtype=torch.float32)
    return torch.complex(real * std, imag * std)


def complex_unitary(shape) -> torch.Tensor:
    phase = torch.rand(shape, dtype=torch.float32) * 2 * math.pi
    return torch.exp(1j * phase)


def init_complex_weights(tensor: torch.Tensor, method: str = "xavier") -> None:
    method = method.lower()
    fan_in, fan_out = complex_fan_in_and_fan_out(tuple(tensor.shape))

    if method == "random":
        std = 1.0 / math.sqrt(max(1, fan_in))
        values = complex_normal(tensor.shape, std)
    elif method == "xavier":
        std = math.sqrt(2.0 / (fan_in + fan_out))
        values = complex_normal(tensor.shape, std)
    elif method == "he":
        std = math.sqrt(2.0 / max(1, fan_in))
        values = complex_normal(tensor.shape, std)
    elif method == "unitary":
        values = complex_unitary(tensor.shape)
    elif method == "trabelsi":
        sigma = math.sqrt(2.0 / (fan_in + fan_out))
        rayleigh = torch.sqrt(torch._standard_gamma(torch.ones(tensor.shape)) * 2.0) * sigma
        phase = torch.rand(tensor.shape, dtype=torch.float32) * 2 * math.pi
        values = rayleigh * torch.exp(1j * phase)
    elif method == "structured_preserve":
        if len(tensor.shape) != 2:
            raise ValueError("structured_preserve initialization only supports 2D tensors")
        # Structured signal preservation initialization
        # D_{i,j} = e^{iθ} if i ≡ j (mod N_{ℓ-1}) and θ ∈ (−π,π), 0 otherwise
        # Then W = D + Z where Z_{ij} ∼ N(0, σ_z²) with σ_z = α/√(N_{ℓ-1}), α=0.085

        out_features, in_features = tensor.shape
        alpha = 0.0085  # empirical value
        sigma_z = alpha / math.sqrt(in_features)

        # Initialize D matrix (structured part)
        D = torch.zeros_like(tensor, dtype=torch.cfloat)

        # For each output neuron i, connect to input neuron j = i % in_features
        for i in range(out_features):
            j = i % in_features
            # Random phase θ ∈ (−π, π)
            theta = torch.rand(1) * 2 * math.pi - math.pi
            D[i, j] = torch.exp(1j * theta)

        # Add noise matrix Z
        Z = complex_normal(tensor.shape, sigma_z)

        values = D + Z
    else:
        raise ValueError(f"Unknown initialization method '{method}'")

    with torch.no_grad():
        tensor.copy_(values)


def initialize_model(model: torch.nn.Module, method: str = "xavier") -> None:
    for param in model.parameters():
        if param.ndim > 1 and param.dtype == torch.complex64:
            init_complex_weights(param, method=method)
        elif param.ndim == 1 and param.dtype == torch.complex64:
            with torch.no_grad():
                param.zero_()
