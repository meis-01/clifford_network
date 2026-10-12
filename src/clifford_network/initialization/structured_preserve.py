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
    alpha: float = 0.02,
    gain: float = 1.0,
) -> torch.Tensor:
    """Create a gained structured identity matrix plus small complex noise.

    A gain of ``1.0`` exactly recovers the original, ungained initializer.
    """
    if len(shape) != 2:
        raise ValueError(f"structured_preserve supports only 2-D weights, got {shape}.")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"structured_preserve gain must be finite and positive, got {gain}.")

    out_features, in_features = shape
    row_idx = torch.arange(out_features, device=device)
    col_idx = row_idx % in_features
    values = torch.zeros(out_features, in_features, device=device, dtype=dtype)
    values[row_idx, col_idx] = gain * torch.ones(out_features, device=device, dtype=dtype)

    sigma_z = alpha / math.sqrt(in_features)
    return values + complex_normal(shape, sigma_z, device=device, dtype=dtype)




"""Structured-preserving complex initialization.

This initializer seeds each output row with a structured identity-like path
and adds complex Gaussian noise.

The noise scale ``sigma`` is passed directly to ``complex_normal``.
Thus, conceptually,

    W = D + Z,

where D is the structured identity-like matrix and
Z ~ CN(0, sigma^2).
"""

# from __future__ import annotations

# import math

# import torch

# from clifford_network.initialization.baselines import complex_normal


# def structured_preserve(
#     shape: tuple[int, ...],
#     *,
#     device: torch.device,
#     dtype: torch.dtype,
#     sigma: float = 0.0085,
#     gain: float = 1.0,
# ) -> torch.Tensor:
#     """Create a structured identity matrix plus complex Gaussian noise.

#     Parameters
#     ----------
#     shape:
#         Weight matrix shape ``(out_features, in_features)``.

#     device:
#         Torch device on which the weights are created.

#     dtype:
#         Complex torch dtype.

#     sigma:
#         Complex Gaussian noise scale passed directly to
#         ``complex_normal``.

#         Conceptually,

#             Z ~ CN(0, sigma^2).

#         Unlike the previous ``alpha`` parameterization, sigma is
#         NOT divided by ``sqrt(in_features)``.

#     gain:
#         Gain applied to the structured identity-like component.

#         ``gain=1.0`` gives the standard identity-like path.
#     """

#     # --------------------------------------------------------
#     # Validate shape
#     # --------------------------------------------------------

#     if len(shape) != 2:
#         raise ValueError(
#             "structured_preserve supports only 2-D weights, "
#             f"got {shape}."
#         )

#     # --------------------------------------------------------
#     # Validate sigma
#     # --------------------------------------------------------

#     if not math.isfinite(sigma) or sigma < 0.0:
#         raise ValueError(
#             "structured_preserve sigma must be finite and "
#             f"non-negative, got {sigma}."
#         )

#     # --------------------------------------------------------
#     # Validate gain
#     # --------------------------------------------------------

#     if not math.isfinite(gain) or gain <= 0.0:
#         raise ValueError(
#             "structured_preserve gain must be finite and "
#             f"positive, got {gain}."
#         )

#     # --------------------------------------------------------
#     # Weight dimensions
#     # --------------------------------------------------------

#     out_features, in_features = shape

#     # --------------------------------------------------------
#     # Structured identity-like matrix
#     # --------------------------------------------------------

#     row_idx = torch.arange(
#         out_features,
#         device=device,
#     )

#     col_idx = row_idx % in_features

#     values = torch.zeros(
#         out_features,
#         in_features,
#         device=device,
#         dtype=dtype,
#     )

#     values[row_idx, col_idx] = (
#         gain
#         * torch.ones(
#             out_features,
#             device=device,
#             dtype=dtype,
#         )
#     )

#     # --------------------------------------------------------
#     # Add complex Gaussian noise
#     # --------------------------------------------------------
#     #
#     # IMPORTANT:
#     # sigma is now used directly.
#     #
#     # OLD:
#     #     sigma_z = alpha / sqrt(in_features)
#     #
#     # NEW:
#     #     sigma_z = sigma
#     #
#     # Therefore:
#     #
#     #     W = D + Z
#     #
#     # where Z ~ CN(0, sigma^2).
#     # --------------------------------------------------------

#     noise = complex_normal(
#         shape,
#         sigma,
#         device=device,
#         dtype=dtype,
#     )

#     return values + noise