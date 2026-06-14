from __future__ import annotations

import math

import torch


def complex_fan_in_and_fan_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return shape[1], shape[0]
    if len(shape) > 2:
        receptive_field_size = math.prod(shape[2:])
        return shape[1] * receptive_field_size, shape[0] * receptive_field_size
    raise ValueError(f"Unsupported tensor shape for fan calculation: {shape}")


def complex_normal(
    shape: tuple[int, ...],
    std: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError(f"Expected a complex dtype, got {dtype}.")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    buffer = torch.randn(*shape, 2, device=device, dtype=real_dtype) * std
    return torch.view_as_complex(buffer.contiguous())


def random_complex(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fan_in, _ = complex_fan_in_and_fan_out(shape)
    return complex_normal(shape, 1.0 / math.sqrt(max(1, fan_in)), device=device, dtype=dtype)


def xavier_complex(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fan_in, fan_out = complex_fan_in_and_fan_out(shape)
    return complex_normal(shape, math.sqrt(2.0 / max(1, fan_in + fan_out)), device=device, dtype=dtype)


def he_complex(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fan_in, _ = complex_fan_in_and_fan_out(shape)
    return complex_normal(shape, math.sqrt(2.0 / max(1, fan_in)), device=device, dtype=dtype)


def unitary_phase(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    phase = torch.rand(shape, device=device, dtype=real_dtype) * (2.0 * math.pi)
    return torch.polar(torch.ones(shape, device=device, dtype=real_dtype), phase).to(dtype)


def trabelsi(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fan_in, fan_out = complex_fan_in_and_fan_out(shape)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    sigma = math.sqrt(2.0 / max(1, fan_in + fan_out))
    gamma = torch.distributions.Gamma(
        concentration=torch.tensor(2.0, device=device, dtype=real_dtype),
        rate=torch.tensor(1.0, device=device, dtype=real_dtype),
    ).sample(shape)
    magnitude = gamma.sqrt() * sigma
    phase = torch.rand(shape, device=device, dtype=real_dtype) * (2.0 * math.pi)
    return torch.polar(magnitude, phase).to(dtype)
