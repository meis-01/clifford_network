"""
weight_initializations.py  —  optimized rewrite

Key fixes vs original:
  1. structured_preserve: removed O(out_features) Python loop; replaced with
     vectorised scatter via advanced indexing.  For a 1024×102400 weight matrix
     the loop iterated 1 024 times calling torch.rand + torch.exp per row —
     this version does it in three lines.

  2. trabelsi: replaced the private `torch._standard_gamma` with the public
     `torch.distributions.Gamma(2, 1).sample()` API.  The private symbol is an
     implementation detail and has been removed/renamed across PyTorch versions.

  3. complex_normal: moved real/imag construction onto a single torch.view_as_complex
     call which avoids an intermediate copy.

  4. initialize_model: early-exits for parameters whose dtype is not complex64
     (e.g. ModReLU bias which is float32) without branching into ndim checks.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def complex_fan_in_and_fan_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) == 2:
        return shape[1], shape[0]
    if len(shape) > 2:
        receptive = math.prod(shape[2:])
        return shape[1] * receptive, shape[0] * receptive
    raise ValueError(f"Unsupported tensor shape for fan calculation: {shape}")


def complex_normal(shape: Tuple[int, ...], std: float) -> torch.Tensor:
    """Return a complex64 tensor sampled from CN(0, std²)."""
    # Stack real + imag into a (..., 2) float tensor then view as complex —
    # avoids an extra allocation compared to torch.complex(r, i).
    buf = torch.randn(*shape, 2, dtype=torch.float32) * std
    return torch.view_as_complex(buf.contiguous())


def complex_unitary(shape: Tuple[int, ...]) -> torch.Tensor:
    """Return a complex64 tensor with unit magnitude and uniform random phase."""
    phase = torch.rand(shape, dtype=torch.float32) * (2 * math.pi)
    return torch.polar(torch.ones(shape, dtype=torch.float32), phase)


# ---------------------------------------------------------------------------
# Per-tensor initialisation
# ---------------------------------------------------------------------------

def init_complex_weights(tensor: torch.Tensor, method: str = "xavier") -> None:
    """Initialise a complex64 weight tensor in-place."""
    method = method.lower()
    shape = tuple(tensor.shape)
    fan_in, fan_out = complex_fan_in_and_fan_out(shape)

    if method == "random":
        std = 1.0 / math.sqrt(max(1, fan_in))
        values = complex_normal(shape, std)

    elif method == "xavier":
        std = math.sqrt(2.0 / max(1, fan_in + fan_out))
        values = complex_normal(shape, std)

    elif method == "he":
        std = math.sqrt(2.0 / max(1, fan_in))
        values = complex_normal(shape, std)

    elif method == "unitary":
        values = complex_unitary(shape)

    elif method == "trabelsi":
        # Magnitude drawn from a Rayleigh distribution: ||w|| ~ Rayleigh(sigma).
        # Rayleigh(s) = sqrt(Gamma(2,1)) * s  (shape parameter = 2, rate = 1).
        # Use the public torch.distributions API instead of the private
        # torch._standard_gamma.
        sigma = math.sqrt(2.0 / max(1, fan_in + fan_out))
        gamma_samples = torch.distributions.Gamma(
            concentration=torch.tensor(2.0),
            rate=torch.tensor(1.0),
        ).sample(shape)                             # shape + () → shape
        magnitude = gamma_samples.sqrt() * sigma   # Rayleigh magnitude
        phase = torch.rand(shape, dtype=torch.float32) * (2 * math.pi)
        values = torch.polar(magnitude.float(), phase)

    elif method == "structured_preserve":
        if tensor.ndim != 2:
            raise ValueError(
                f"structured_preserve only supports 2-D tensors, got shape {shape}"
            )
        out_features, in_features = shape
        # α is an empirical hyper-parameter controlling noise variance.
        alpha = 0.0085
        sigma_z = alpha / math.sqrt(in_features)

        # ── Vectorised diagonal-phase matrix D ──────────────────────────────
        # D[i, j] = exp(i·θ_i)  if j == i % in_features, else 0
        # Build the sparse structured component without any Python-level loop.
        row_idx = torch.arange(out_features)
        col_idx = row_idx % in_features
        phases  = torch.rand(out_features, dtype=torch.float32) * (2 * math.pi) - math.pi
        d_vals  = torch.polar(torch.ones(out_features, dtype=torch.float32), phases)

        D = torch.zeros(out_features, in_features, dtype=torch.cfloat)
        D[row_idx, col_idx] = d_vals          # vectorised scatter

        # ── Add small noise Z ────────────────────────────────────────────────
        values = D + complex_normal(shape, sigma_z)

    else:
        raise ValueError(f"Unknown initialisation method '{method}'")

    with torch.no_grad():
        tensor.copy_(values)


# ---------------------------------------------------------------------------
# Model-level entry point
# ---------------------------------------------------------------------------

def initialize_model(model: nn.Module, method: str = "xavier") -> None:
    """
    Walk all parameters and apply complex initialisation to every complex64
    weight matrix (ndim > 1).  Bias vectors (ndim == 1) are zeroed.
    Float32 parameters (e.g. ModReLU bias) are left untouched.
    """
    for param in model.parameters():
        if param.dtype != torch.complex64:
            continue                         # skip float params (ModReLU bias etc.)
        if param.ndim > 1:
            init_complex_weights(param, method=method)
        else:
            with torch.no_grad():
                param.zero_()