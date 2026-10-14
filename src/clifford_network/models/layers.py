"""Reusable complex-valued neural network layers.

Currently this module provides a dense complex linear layer with native complex
parameters, initialization, and matrix multiplication semantics.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ComplexLinear(nn.Module):
    """Fully connected layer with complex-valued weights and optional bias."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Allocate complex parameters and initialize them with a simple normal rule."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.complex64)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset weights with fan-in scaling and zero the optional bias."""
        std = 1.0 / math.sqrt(max(1, self.in_features))
        with torch.no_grad():
            self.weight.real.normal_(0.0, std)
            self.weight.imag.normal_(0.0, std)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the complex affine transformation to a batch of inputs."""
        output = values @ self.weight.transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output



# Complex batch normalization is based on the paper:
# "Deep Complex Networks" by Chiheb Trabelsi et al., 2018.s

class ComplexBatchNorm(nn.Module):
    """Whitening-based batch normalization for complex-valued tensors.

    Normalizes real and imaginary components jointly using the inverse
    square root of their 2x2 covariance matrix.

    Expects input of shape (..., num_features), with complex dtype.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()

        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between 0 and 1.")

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            # Symmetric 2x2 affine scaling matrix.
            self.gamma_rr = nn.Parameter(
                torch.full((num_features,), 1.0 / math.sqrt(2.0))
            )
            self.gamma_ri = nn.Parameter(torch.zeros(num_features))
            self.gamma_ii = nn.Parameter(
                torch.full((num_features,), 1.0 / math.sqrt(2.0))
            )
            self.beta_r = nn.Parameter(torch.zeros(num_features))
            self.beta_i = nn.Parameter(torch.zeros(num_features))

        if track_running_stats:
            self.register_buffer(
                "running_mean_r", torch.zeros(num_features)
            )
            self.register_buffer(
                "running_mean_i", torch.zeros(num_features)
            )
            self.register_buffer(
                "running_var_r", torch.full((num_features,), 0.5)
            )
            self.register_buffer(
                "running_var_i", torch.full((num_features,), 0.5)
            )
            self.register_buffer(
                "running_covar_ri", torch.zeros(num_features)
            )
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError(
                "ComplexBatchNorm expects a complex-valued tensor."
            )
        if x.ndim < 2 or x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected input shape (..., {self.num_features}), "
                f"got {tuple(x.shape)}."
            )

        xr, xi = x.real, x.imag
        dims = tuple(range(x.ndim - 1))

        if self.training or not self.track_running_stats:
            mean_r = xr.mean(dim=dims)
            mean_i = xi.mean(dim=dims)

            cr = xr - mean_r
            ci = xi - mean_i

            var_r = cr.square().mean(dim=dims)
            var_i = ci.square().mean(dim=dims)
            cov_ri = (cr * ci).mean(dim=dims)

            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    m = self.momentum

                    self.running_mean_r.lerp_(mean_r.detach(), m)
                    self.running_mean_i.lerp_(mean_i.detach(), m)
                    self.running_var_r.lerp_(var_r.detach(), m)
                    self.running_var_i.lerp_(var_i.detach(), m)
                    self.running_covar_ri.lerp_(cov_ri.detach(), m)
        else:
            mean_r = self.running_mean_r
            mean_i = self.running_mean_i
            cr = xr - mean_r
            ci = xi - mean_i

            var_r = self.running_var_r
            var_i = self.running_var_i
            cov_ri = self.running_covar_ri

        # Regularized covariance matrix:
        # V = [[var_r + eps, cov_ri],
        #      [cov_ri, var_i + eps]]
        a = var_r + self.eps
        b = cov_ri
        d = var_i + self.eps

        # Analytical inverse square root of a symmetric 2x2 matrix.
        det_sqrt = torch.sqrt(
            (a * d - b.square()).clamp_min(self.eps**2)
        )
        trace_factor = torch.sqrt(
            (a + d + 2.0 * det_sqrt).clamp_min(self.eps)
        )

        denom = (
            (a + det_sqrt) * (d + det_sqrt) - b.square()
        ).clamp_min(self.eps**2)

        inv_rr = trace_factor * (d + det_sqrt) / denom
        inv_ri = -trace_factor * b / denom
        inv_ii = trace_factor * (a + det_sqrt) / denom

        # Jointly whiten real and imaginary components.
        yr = inv_rr * cr + inv_ri * ci
        yi = inv_ri * cr + inv_ii * ci

        if self.affine:
            out_r = self.gamma_rr * yr + self.gamma_ri * yi
            out_i = self.gamma_ri * yr + self.gamma_ii * yi

            out_r = out_r + self.beta_r
            out_i = out_i + self.beta_i
        else:
            out_r, out_i = yr, yi

        return torch.complex(out_r, out_i)
