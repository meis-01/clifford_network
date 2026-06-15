"""Structured-preserving complex initialization.

This initializer seeds each output row with a phase-only identity-like path and
adds small complex Gaussian noise, aiming to preserve signal flow at depth.
"""

from __future__ import annotations

import math

import torch

from clifford_network.initialization.baselines import complex_normal


def structured_preserve(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float = 0.0085,
) -> torch.Tensor:
    """Create a 2-D structured identity-phase matrix plus small complex noise."""
    if len(shape) != 2:
        raise ValueError(f"structured_preserve supports only 2-D weights, got {shape}.")

    out_features, in_features = shape
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    row_idx = torch.arange(out_features, device=device)
    col_idx = row_idx % in_features
    phases = torch.rand(out_features, device=device, dtype=real_dtype) * (2.0 * math.pi) - math.pi
    values = torch.zeros(out_features, in_features, device=device, dtype=dtype)
    values[row_idx, col_idx] = torch.polar(torch.ones_like(phases), phases).to(dtype)

    sigma_z = alpha / math.sqrt(in_features)
    return values + complex_normal(shape, sigma_z, device=device, dtype=dtype)
