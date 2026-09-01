
"""Simple multidimensional complex Fourier-function approximation dataset.

This module provides a synthetic complex-valued regression benchmark for
training and evaluating complex-valued neural networks.

The mathematical function is

    f(z) = (1 / sqrt(K)) * sum_k a_k
           * exp(i * omega_k^T Re(z))

where

    z       in C^d
    a_k     in C
    omega_k in R^d

The same function parameters are used for the train, validation, and test
datasets. Only the input samples are different.

The dataset returns:

    input  : complex tensor of shape [input_dim]
    target : complex tensor of shape [1]
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class ComplexFourierFunctionDataset(Dataset):
    """Synthetic multidimensional complex Fourier-function dataset."""

    def __init__(
        self,
        n_samples: int,
        input_dim: int,
        n_frequencies: int,
        *,
        input_range: tuple[float, float] = (-1.0, 1.0),
        frequency_range: tuple[float, float] = (-3.0, 3.0),
        omega: torch.Tensor | None = None,
        coefficients: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        """Create a complex-valued Fourier regression dataset."""

        super().__init__()

        if n_samples < 1:
            raise ValueError("n_samples must be positive.")

        if input_dim < 1:
            raise ValueError("input_dim must be positive.")

        if n_frequencies < 1:
            raise ValueError("n_frequencies must be positive.")

        if input_range[0] >= input_range[1]:
            raise ValueError("input_range must satisfy low < high.")

        if frequency_range[0] >= frequency_range[1]:
            raise ValueError("frequency_range must satisfy low < high.")

        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_frequencies = n_frequencies

        generator = torch.Generator().manual_seed(seed)

        # ---------------------------------------------------------------
        # Generate complex inputs
        #
        # z = x + i y
        #
        # x,y in R^d
        # ---------------------------------------------------------------

        input_low, input_high = input_range

        real = (
            torch.rand(
                n_samples,
                input_dim,
                generator=generator,
            )
            * (input_high - input_low)
            + input_low
        )

        imag = (
            torch.rand(
                n_samples,
                input_dim,
                generator=generator,
            )
            * (input_high - input_low)
            + input_low
        )

        self.x = torch.complex(
            real,
            imag,
        ).to(torch.complex64)

        # ---------------------------------------------------------------
        # Fourier frequencies
        #
        # omega_k in R^d
        #
        # Shape:
        #     [K, d]
        # ---------------------------------------------------------------

        if omega is None:

            freq_low, freq_high = frequency_range

            self.omega = (
                torch.rand(
                    n_frequencies,
                    input_dim,
                    generator=generator,
                )
                * (freq_high - freq_low)
                + freq_low
            ).float()

        else:

            self.omega = (
                omega.detach()
                .clone()
                .float()
            )

        # ---------------------------------------------------------------
        # Complex Fourier coefficients
        #
        # a_k in C
        #
        # Shape:
        #     [K]
        # ---------------------------------------------------------------

        if coefficients is None:

            coefficient_real = torch.randn(
                n_frequencies,
                generator=generator,
            )

            coefficient_imag = torch.randn(
                n_frequencies,
                generator=generator,
            )

            coefficients = torch.complex(
                coefficient_real,
                coefficient_imag,
            )

            # Normalize the sum.
            coefficients = coefficients / torch.sqrt(
                torch.tensor(
                    float(n_frequencies)
                )
            )

            self.coefficients = (
                coefficients.to(torch.complex64)
            )

        else:

            self.coefficients = (
                coefficients.detach()
                .clone()
                .to(torch.complex64)
            )

        # ---------------------------------------------------------------
        # Check dimensions
        # ---------------------------------------------------------------

        if self.omega.shape != (
            n_frequencies,
            input_dim,
        ):
            raise ValueError(
                f"omega must have shape "
                f"({n_frequencies}, {input_dim}), "
                f"got {tuple(self.omega.shape)}."
            )

        if self.coefficients.shape != (
            n_frequencies,
        ):
            raise ValueError(
                f"coefficients must have shape "
                f"({n_frequencies},), "
                f"got {tuple(self.coefficients.shape)}."
            )

        # ---------------------------------------------------------------
        # Generate target values
        # ---------------------------------------------------------------

        self.y = self._fourier_function(self.x)

    def _fourier_function(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the multidimensional complex Fourier function."""

        # ---------------------------------------------------------------
        # Take real part of complex input.
        #
        # z = x + iy
        #
        # x has shape [N, d]
        # ---------------------------------------------------------------

        x = z.real

        # ---------------------------------------------------------------
        # Calculate
        #
        # omega_k^T x
        #
        # Shape:
        #
        #     [N, d] @ [d, K]
        #
        #          = [N, K]
        # ---------------------------------------------------------------

        phase = x @ self.omega.T

        # ---------------------------------------------------------------
        # Fourier basis functions
        #
        # exp(i * omega_k^T x)
        #
        # Shape:
        #
        #     [N, K]
        # ---------------------------------------------------------------

        basis = torch.exp(
            1j * phase
        )

        # ---------------------------------------------------------------
        # Weighted Fourier sum
        #
        # f(z) =
        #
        #     1/sqrt(K)
        #     sum_k a_k exp(i omega_k^T x)
        # ---------------------------------------------------------------

        target = torch.sum(
            basis * self.coefficients,
            dim=1,
        )

        return target.to(torch.complex64)

    def __len__(self) -> int:
        """Return the number of samples."""

        return self.n_samples

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one complex input and one complex target."""

        return (
            self.x[index],
            self.y[index].unsqueeze(0),
        )


def _optional_int(
    value: Any,
) -> int | None:
    """Convert an optional value to int."""

    if value is None:
        return None

    return int(value)


def _create_fourier_parameters(
    *,
    input_dim: int,
    n_frequencies: int,
    frequency_range: tuple[float, float],
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Create the fixed parameters of the Fourier function."""

    generator = torch.Generator().manual_seed(seed)

    freq_low, freq_high = frequency_range

    # ---------------------------------------------------------------
    # Fourier frequencies
    #
    # omega shape = [K, d]
    # ---------------------------------------------------------------

    omega = (
        torch.rand(
            n_frequencies,
            input_dim,
            generator=generator,
        )
        * (freq_high - freq_low)
        + freq_low
    ).float()

    # ---------------------------------------------------------------
    # Complex coefficients
    #
    # a shape = [K]
    # ---------------------------------------------------------------

    coefficient_real = torch.randn(
        n_frequencies,
        generator=generator,
    )

    coefficient_imag = torch.randn(
        n_frequencies,
        generator=generator,
    )

    coefficients = torch.complex(
        coefficient_real,
        coefficient_imag,
    )

    coefficients = coefficients / torch.sqrt(
        torch.tensor(
            float(n_frequencies)
        )
    )

    return (
        omega,
        coefficients.to(torch.complex64),
    )


def build_function_approximation_datasets(
    config: dict,
    seed: int,
) -> tuple[
    Dataset,
    Dataset,
    Dataset,
    int,
    int,
]:
    """Build train, validation, and test datasets.

    All three datasets represent the same mathematical function.
    Only the sampled input points are different.
    """

    dataset_config = config.get(
        "dataset",
        {}
    )

    input_dim = int(
        dataset_config.get(
            "input_dim",
            20,
        )
    )

    n_frequencies = int(
        dataset_config.get(
            "n_frequencies",
            10,
        )
    )

    train_samples = int(
        dataset_config.get(
            "train_samples",
            20000,
        )
    )

    validation_samples = int(
        dataset_config.get(
            "validation_samples",
            5000,
        )
    )

    test_samples = int(
        dataset_config.get(
            "test_samples",
            5000,
        )
    )

    input_low = float(
        dataset_config.get(
            "input_low",
            -1.0,
        )
    )

    input_high = float(
        dataset_config.get(
            "input_high",
            1.0,
        )
    )

    frequency_low = float(
        dataset_config.get(
            "frequency_low",
            -3.0,
        )
    )

    frequency_high = float(
        dataset_config.get(
            "frequency_high",
            3.0,
        )
    )

    frequency_range = (
        frequency_low,
        frequency_high,
    )

    # ---------------------------------------------------------------
    # Generate ONE fixed target function.
    #
    # The same omega and coefficients are used in all splits.
    # ---------------------------------------------------------------

    omega, coefficients = _create_fourier_parameters(
        input_dim=input_dim,
        n_frequencies=n_frequencies,
        frequency_range=frequency_range,
        seed=seed,
    )

    common_kwargs = {
        "input_dim": input_dim,
        "n_frequencies": n_frequencies,
        "input_range": (
            input_low,
            input_high,
        ),
        "frequency_range": frequency_range,
        "omega": omega,
        "coefficients": coefficients,
    }

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    train_dataset = ComplexFourierFunctionDataset(
        n_samples=train_samples,
        seed=seed + 1,
        **common_kwargs,
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    validation_dataset = ComplexFourierFunctionDataset(
        n_samples=validation_samples,
        seed=seed + 2,
        **common_kwargs,
    )

    # ---------------------------------------------------------------
    # Test
    # ---------------------------------------------------------------

    test_dataset = ComplexFourierFunctionDataset(
        n_samples=test_samples,
        seed=seed + 3,
        **common_kwargs,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        input_dim,
        1,
    )












# """Mathematical complex-valued function approximation datasets.

# This module provides a synthetic complex-valued regression benchmark for
# training deep complex-valued neural networks.

# The benchmark function is

#     f(z) = sum_k a_k tanh(
#                 1/sqrt(d) * sum_j w[k,j] z[j] + b[k]
#             )
#             + c * exp(
#                 i * sum_j omega[j] |z[j]|^2
#             )

# where

#     z       in C^d
#     w[k,j]  in C
#     b[k]    in C
#     a[k]    in C
#     omega   in R^d
#     c       in C

# The same mathematical function parameters are used for train, validation,
# and test sets. Only the sampled input points differ.

# The dataset returns:

#     input  : complex tensor of shape [input_dim]
#     target : complex tensor of shape [1]
# """

# from __future__ import annotations

# from typing import Any

# import torch
# from torch.utils.data import Dataset


# class ComplexFourierFunctionDataset(Dataset):
#     """Synthetic multidimensional complex nonlinear-function dataset."""

#     def __init__(
#         self,
#         n_samples: int,
#         input_dim: int,
#         n_frequencies: int,
#         *,
#         input_range: tuple[float, float] = (-1.0, 1.0),
#         frequency_range: tuple[float, float] = (-3.0, 3.0),
#         omega: torch.Tensor | None = None,
#         coefficients: torch.Tensor | None = None,
#         weights: torch.Tensor | None = None,
#         biases: torch.Tensor | None = None,
#         c: torch.Tensor | None = None,
#         seed: int = 0,
#     ) -> None:
#         """Create a complex-valued nonlinear function approximation dataset."""

#         super().__init__()

#         if n_samples < 1:
#             raise ValueError("n_samples must be positive.")

#         if input_dim < 1:
#             raise ValueError("input_dim must be positive.")

#         if n_frequencies < 1:
#             raise ValueError("n_frequencies must be positive.")

#         if input_range[0] >= input_range[1]:
#             raise ValueError("input_range must satisfy low < high.")

#         if frequency_range[0] >= frequency_range[1]:
#             raise ValueError("frequency_range must satisfy low < high.")

#         self.n_samples = n_samples
#         self.input_dim = input_dim
#         self.n_frequencies = n_frequencies

#         generator = torch.Generator().manual_seed(seed)

#         # ---------------------------------------------------------------
#         # Generate complex input
#         #
#         # z = x + i y
#         #
#         # x,y in R^d
#         # ---------------------------------------------------------------

#         input_low, input_high = input_range

#         real = (
#             torch.rand(
#                 n_samples,
#                 input_dim,
#                 generator=generator,
#             )
#             * (input_high - input_low)
#             + input_low
#         )

#         imag = (
#             torch.rand(
#                 n_samples,
#                 input_dim,
#                 generator=generator,
#             )
#             * (input_high - input_low)
#             + input_low
#         )

#         self.x = torch.complex(
#             real,
#             imag,
#         ).to(torch.complex64)

#         # ---------------------------------------------------------------
#         # Complex weights w[k,j]
#         # ---------------------------------------------------------------

#         if weights is None:

#             weight_real = torch.randn(
#                 n_frequencies,
#                 input_dim,
#                 generator=generator,
#             )

#             weight_imag = torch.randn(
#                 n_frequencies,
#                 input_dim,
#                 generator=generator,
#             )

#             weights = torch.complex(
#                 weight_real,
#                 weight_imag,
#             )

#             # Keep the nonlinear arguments at a reasonable scale.
#             weights = weights / torch.sqrt(
#                 torch.tensor(float(input_dim))
#             )

#             self.weights = weights.to(torch.complex64)

#         else:
#             self.weights = (
#                 weights.detach()
#                 .clone()
#                 .to(torch.complex64)
#             )

#         # ---------------------------------------------------------------
#         # Complex biases b[k]
#         # ---------------------------------------------------------------

#         if biases is None:

#             bias_real = torch.randn(
#                 n_frequencies,
#                 generator=generator,
#             )

#             bias_imag = torch.randn(
#                 n_frequencies,
#                 generator=generator,
#             )

#             biases = torch.complex(
#                 bias_real,
#                 bias_imag,
#             )

#             self.biases = (
#                 0.5 * biases
#             ).to(torch.complex64)

#         else:
#             self.biases = (
#                 biases.detach()
#                 .clone()
#                 .to(torch.complex64)
#             )

#         # ---------------------------------------------------------------
#         # Complex coefficients a[k]
#         # ---------------------------------------------------------------

#         if coefficients is None:

#             coefficient_real = torch.randn(
#                 n_frequencies,
#                 generator=generator,
#             )

#             coefficient_imag = torch.randn(
#                 n_frequencies,
#                 generator=generator,
#             )

#             coefficients = torch.complex(
#                 coefficient_real,
#                 coefficient_imag,
#             )

#             # Normalize with number of terms.
#             coefficients = coefficients / torch.sqrt(
#                 torch.tensor(float(n_frequencies))
#             )

#             self.coefficients = (
#                 coefficients.to(torch.complex64)
#             )

#         else:
#             self.coefficients = (
#                 coefficients.detach()
#                 .clone()
#                 .to(torch.complex64)
#             )

#         # ---------------------------------------------------------------
#         # omega for the nonlinear phase term
#         #
#         # omega in R^d
#         # ---------------------------------------------------------------

#         if omega is None:

#             freq_low, freq_high = frequency_range

#             self.omega = (
#                 torch.rand(
#                     input_dim,
#                     generator=generator,
#                 )
#                 * (freq_high - freq_low)
#                 + freq_low
#             ).to(torch.float32)

#         else:
#             self.omega = (
#                 omega.detach()
#                 .clone()
#                 .float()
#             )

#         # ---------------------------------------------------------------
#         # Complex coefficient c
#         # ---------------------------------------------------------------

#         if c is None:

#             c_real = torch.randn(
#                 1,
#                 generator=generator,
#             )

#             c_imag = torch.randn(
#                 1,
#                 generator=generator,
#             )

#             c_value = torch.complex(
#                 c_real,
#                 c_imag,
#             )

#             self.c = (
#                 0.5 * c_value
#             ).to(torch.complex64).squeeze(0)

#         else:
#             self.c = (
#                 c.detach()
#                 .clone()
#                 .to(torch.complex64)
#             ).squeeze()

#         # ---------------------------------------------------------------
#         # Check dimensions
#         # ---------------------------------------------------------------

#         expected_weight_shape = (
#             n_frequencies,
#             input_dim,
#         )

#         if self.weights.shape != expected_weight_shape:
#             raise ValueError(
#                 f"weights must have shape "
#                 f"{expected_weight_shape}, "
#                 f"got {tuple(self.weights.shape)}."
#             )

#         if self.biases.shape != (n_frequencies,):
#             raise ValueError(
#                 f"biases must have shape "
#                 f"({n_frequencies},), "
#                 f"got {tuple(self.biases.shape)}."
#             )

#         if self.coefficients.shape != (n_frequencies,):
#             raise ValueError(
#                 f"coefficients must have shape "
#                 f"({n_frequencies},), "
#                 f"got {tuple(self.coefficients.shape)}."
#             )

#         if self.omega.shape != (input_dim,):
#             raise ValueError(
#                 f"omega must have shape "
#                 f"({input_dim},), "
#                 f"got {tuple(self.omega.shape)}."
#             )

#         # ---------------------------------------------------------------
#         # Generate exact mathematical target
#         # ---------------------------------------------------------------

#         self.y = self._fourier_function(self.x)

#     def _fourier_function(
#         self,
#         z: torch.Tensor,
#     ) -> torch.Tensor:
#         """Evaluate the multidimensional complex nonlinear function."""

#         # ---------------------------------------------------------------
#         # z:
#         #
#         #     [N, d]
#         #
#         # First calculate
#         #
#         #     sum_j w[k,j] z[j]
#         #
#         # resulting in
#         #
#         #     [N, K]
#         # ---------------------------------------------------------------

#         linear_argument = (
#             z @ self.weights.T
#         )

#         # ---------------------------------------------------------------
#         # Divide by sqrt(d)
#         # ---------------------------------------------------------------

#         linear_argument = (
#             linear_argument
#             / torch.sqrt(
#                 torch.tensor(
#                     float(self.input_dim),
#                     dtype=torch.float32,
#                 )
#             )
#         )

#         # ---------------------------------------------------------------
#         # Add complex bias
#         # ---------------------------------------------------------------

#         linear_argument = (
#             linear_argument
#             + self.biases
#         )

#         # ---------------------------------------------------------------
#         # Nonlinear complex tanh
#         #
#         # NOTE:
#         # This is the mathematical target function.
#         # It is NOT the network's split-tanh activation.
#         #
#         # This makes the target a genuinely complex nonlinear function.
#         # ---------------------------------------------------------------

#         nonlinear_terms = torch.tanh(
#             linear_argument
#         )

#         # ---------------------------------------------------------------
#         # Weighted sum
#         #
#         # sum_k a[k] * tanh(...)
#         # ---------------------------------------------------------------

#         first_term = torch.sum(
#             nonlinear_terms
#             * self.coefficients,
#             dim=1,
#         )

#         # ---------------------------------------------------------------
#         # Nonlinear phase term
#         #
#         # q(z) = sum_j omega[j] |z_j|^2
#         # ---------------------------------------------------------------

#         magnitude_squared = (
#             torch.abs(z) ** 2
#         )

#         phase = torch.sum(
#             magnitude_squared
#             * self.omega,
#             dim=1,
#         )

#         # ---------------------------------------------------------------
#         # c * exp(i q(z))
#         # ---------------------------------------------------------------

#         second_term = (
#             self.c
#             * torch.exp(
#                 1j * phase
#             )
#         )

#         # ---------------------------------------------------------------
#         # Complete function
#         #
#         # f(z) =
#         #
#         # sum_k a_k tanh(...)
#         # +
#         # c exp(i q(z))
#         # ---------------------------------------------------------------

#         target = (
#             first_term
#             + second_term
#         )

#         return target.to(torch.complex64)

#     def __len__(self) -> int:
#         """Return number of samples."""

#         return self.n_samples

#     def __getitem__(
#         self,
#         index: int,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Return one complex input and one complex target."""

#         return (
#             self.x[index],
#             self.y[index].unsqueeze(0),
#         )


# def _optional_int(
#     value: Any,
# ) -> int | None:
#     """Convert an optional configuration value to int."""

#     if value is None:
#         return None

#     return int(value)


# def _create_fourier_parameters(
#     *,
#     input_dim: int,
#     n_frequencies: int,
#     frequency_range: tuple[float, float],
#     seed: int,
# ) -> tuple[
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
# ]:
#     """Create the shared parameters of the mathematical function."""

#     generator = torch.Generator().manual_seed(seed)

#     freq_low, freq_high = frequency_range

#     # ---------------------------------------------------------------
#     # Complex weights
#     # ---------------------------------------------------------------

#     weight_real = torch.randn(
#         n_frequencies,
#         input_dim,
#         generator=generator,
#     )

#     weight_imag = torch.randn(
#         n_frequencies,
#         input_dim,
#         generator=generator,
#     )

#     weights = torch.complex(
#         weight_real,
#         weight_imag,
#     )

#     weights = weights / torch.sqrt(
#         torch.tensor(float(input_dim))
#     )

#     # ---------------------------------------------------------------
#     # Complex biases
#     # ---------------------------------------------------------------

#     bias_real = torch.randn(
#         n_frequencies,
#         generator=generator,
#     )

#     bias_imag = torch.randn(
#         n_frequencies,
#         generator=generator,
#     )

#     biases = torch.complex(
#         bias_real,
#         bias_imag,
#     )

#     biases = 0.5 * biases

#     # ---------------------------------------------------------------
#     # Complex coefficients
#     # ---------------------------------------------------------------

#     coefficient_real = torch.randn(
#         n_frequencies,
#         generator=generator,
#     )

#     coefficient_imag = torch.randn(
#         n_frequencies,
#         generator=generator,
#     )

#     coefficients = torch.complex(
#         coefficient_real,
#         coefficient_imag,
#     )

#     coefficients = coefficients / torch.sqrt(
#         torch.tensor(float(n_frequencies))
#     )

#     # ---------------------------------------------------------------
#     # Nonlinear phase frequencies
#     # ---------------------------------------------------------------

#     omega = (
#         torch.rand(
#             input_dim,
#             generator=generator,
#         )
#         * (freq_high - freq_low)
#         + freq_low
#     ).float()

#     # ---------------------------------------------------------------
#     # Complex coefficient c
#     # ---------------------------------------------------------------

#     c_real = torch.randn(
#         1,
#         generator=generator,
#     )

#     c_imag = torch.randn(
#         1,
#         generator=generator,
#     )

#     c = torch.complex(
#         c_real,
#         c_imag,
#     )

#     c = (0.5 * c).to(torch.complex64).squeeze(0)

#     return (
#         weights.to(torch.complex64),
#         biases.to(torch.complex64),
#         coefficients.to(torch.complex64),
#         omega,
#         c,
#     )


# def build_function_approximation_datasets(
#     config: dict,
#     seed: int,
# ) -> tuple[
#     Dataset,
#     Dataset,
#     Dataset,
#     int,
#     int,
# ]:
#     """Build train, validation, and test Fourier-function datasets.

#     The mathematical function is fixed for the complete experiment.

#     Train, validation, and test datasets use the same function parameters,
#     but different randomly sampled input points.
#     """

#     dataset_config = config.get(
#         "dataset",
#         {},
#     )

#     input_dim = int(
#         dataset_config.get(
#             "input_dim",
#             20,
#         )
#     )

#     n_frequencies = int(
#         dataset_config.get(
#             "n_frequencies",
#             50,
#         )
#     )

#     train_samples = int(
#         dataset_config.get(
#             "train_samples",
#             20000,
#         )
#     )

#     validation_samples = int(
#         dataset_config.get(
#             "validation_samples",
#             5000,
#         )
#     )

#     test_samples = int(
#         dataset_config.get(
#             "test_samples",
#             5000,
#         )
#     )

#     input_low = float(
#         dataset_config.get(
#             "input_low",
#             -1.0,
#         )
#     )

#     input_high = float(
#         dataset_config.get(
#             "input_high",
#             1.0,
#         )
#     )

#     frequency_low = float(
#         dataset_config.get(
#             "frequency_low",
#             -3.0,
#         )
#     )

#     frequency_high = float(
#         dataset_config.get(
#             "frequency_high",
#             3.0,
#         )
#     )

#     frequency_range = (
#         frequency_low,
#         frequency_high,
#     )

#     # ---------------------------------------------------------------
#     # Create ONE fixed mathematical function.
#     #
#     # These parameters remain identical for train/validation/test.
#     # ---------------------------------------------------------------

#     (
#         weights,
#         biases,
#         coefficients,
#         omega,
#         c,
#     ) = _create_fourier_parameters(
#         input_dim=input_dim,
#         n_frequencies=n_frequencies,
#         frequency_range=frequency_range,
#         seed=seed,
#     )

#     common_kwargs = {
#         "input_dim": input_dim,
#         "n_frequencies": n_frequencies,
#         "input_range": (
#             input_low,
#             input_high,
#         ),
#         "frequency_range": frequency_range,
#         "weights": weights,
#         "biases": biases,
#         "coefficients": coefficients,
#         "omega": omega,
#         "c": c,
#     }

#     # ---------------------------------------------------------------
#     # Different input samples for each split.
#     #
#     # Same function, different points.
#     # ---------------------------------------------------------------

#     train_dataset = ComplexFourierFunctionDataset(
#         n_samples=train_samples,
#         seed=seed + 1,
#         **common_kwargs,
#     )

#     validation_dataset = ComplexFourierFunctionDataset(
#         n_samples=validation_samples,
#         seed=seed + 2,
#         **common_kwargs,
#     )

#     test_dataset = ComplexFourierFunctionDataset(
#         n_samples=test_samples,
#         seed=seed + 3,
#         **common_kwargs,
#     )

#     return (
#         train_dataset,
#         validation_dataset,
#         test_dataset,
#         input_dim,
#         1,
#     )

















# """Mathematical complex-valued function approximation datasets.

# This module provides synthetic complex-valued regression problems for
# training deep complex-valued neural networks.

# The main benchmark is a multidimensional Fourier-type function

#     f(z) = sum_k c_k exp(i * (omega_k^T Re(z) + nu_k^T Im(z)))

# where

#     z      in C^d
#     omega  in R^d
#     nu     in R^d
#     c_k    in C

# The same Fourier function parameters are used for train, validation,
# and test sets. Only the sampled input points differ.

# The dataset returns:

#     input  : complex tensor of shape [input_dim]
#     target : complex scalar
# """

# from __future__ import annotations

# from typing import Any

# import torch
# from torch.utils.data import Dataset


# class ComplexFourierFunctionDataset(Dataset):
#     """Synthetic multidimensional complex Fourier-function dataset."""

#     def __init__(
#         self,
#         n_samples: int,
#         input_dim: int,
#         n_frequencies: int,
#         *,
#         input_range: tuple[float, float] = (-1.0, 1.0),
#         frequency_range: tuple[float, float] = (-3.0, 3.0),
#         omega: torch.Tensor | None = None,
#         nu: torch.Tensor | None = None,
#         coefficients: torch.Tensor | None = None,
#         seed: int = 0,
#     ) -> None:
#         """Create a complex Fourier-function regression dataset."""

#         super().__init__()

#         if n_samples < 1:
#             raise ValueError("n_samples must be positive.")

#         if input_dim < 1:
#             raise ValueError("input_dim must be positive.")

#         if n_frequencies < 1:
#             raise ValueError("n_frequencies must be positive.")

#         if input_range[0] >= input_range[1]:
#             raise ValueError("input_range must satisfy low < high.")

#         if frequency_range[0] >= frequency_range[1]:
#             raise ValueError("frequency_range must satisfy low < high.")

#         self.n_samples = n_samples
#         self.input_dim = input_dim
#         self.n_frequencies = n_frequencies

#         generator = torch.Generator().manual_seed(seed)

#         # ---------------------------------------------------------
#         # Generate complex input
#         #
#         # z = x + i y
#         #
#         # x, y in R^d
#         # ---------------------------------------------------------

#         input_low, input_high = input_range

#         real = (
#             torch.rand(
#                 n_samples,
#                 input_dim,
#                 generator=generator,
#             )
#             * (input_high - input_low)
#             + input_low
#         )

#         imag = (
#             torch.rand(
#                 n_samples,
#                 input_dim,
#                 generator=generator,
#             )
#             * (input_high - input_low)
#             + input_low
#         )

#         self.x = torch.complex(
#             real,
#             imag,
#         ).to(torch.complex64)

#         # ---------------------------------------------------------
#         # Fourier frequencies
#         #
#         # omega, nu in R^(K x d)
#         # ---------------------------------------------------------

#         if omega is None:
#             freq_low, freq_high = frequency_range

#             self.omega = (
#                 torch.rand(
#                     n_frequencies,
#                     input_dim,
#                     generator=generator,
#                 )
#                 * (freq_high - freq_low)
#                 + freq_low
#             ).to(torch.float32)

#         else:
#             self.omega = omega.detach().clone().float()

#         if nu is None:
#             freq_low, freq_high = frequency_range

#             self.nu = (
#                 torch.rand(
#                     n_frequencies,
#                     input_dim,
#                     generator=generator,
#                 )
#                 * (freq_high - freq_low)
#                 + freq_low
#             ).to(torch.float32)

#         else:
#             self.nu = nu.detach().clone().float()

#         # ---------------------------------------------------------
#         # Complex coefficients
#         #
#         # c_k in C
#         # ---------------------------------------------------------

#         if coefficients is None:

#             coefficient_real = torch.randn(
#                 n_frequencies,
#                 generator=generator,
#             )

#             coefficient_imag = torch.randn(
#                 n_frequencies,
#                 generator=generator,
#             )

#             coefficients = torch.complex(
#                 coefficient_real,
#                 coefficient_imag,
#             )

#             # Normalize so that target magnitude does not grow
#             # with the number of Fourier modes.
#             coefficients = coefficients / torch.sqrt(
#                 torch.tensor(float(n_frequencies))
#             )

#             self.coefficients = coefficients.to(torch.complex64)

#         else:
#             self.coefficients = (
#                 coefficients.detach()
#                 .clone()
#                 .to(torch.complex64)
#             )

#         # ---------------------------------------------------------
#         # Check parameter dimensions
#         # ---------------------------------------------------------

#         expected_shape = (
#             n_frequencies,
#             input_dim,
#         )

#         if self.omega.shape != expected_shape:
#             raise ValueError(
#                 f"omega must have shape {expected_shape}, "
#                 f"got {tuple(self.omega.shape)}."
#             )

#         if self.nu.shape != expected_shape:
#             raise ValueError(
#                 f"nu must have shape {expected_shape}, "
#                 f"got {tuple(self.nu.shape)}."
#             )

#         if self.coefficients.shape != (n_frequencies,):
#             raise ValueError(
#                 "coefficients must have shape "
#                 f"({n_frequencies},), "
#                 f"got {tuple(self.coefficients.shape)}."
#             )

#         # ---------------------------------------------------------
#         # Generate exact mathematical target
#         # ---------------------------------------------------------

#         self.y = self._fourier_function(self.x)

#     def _fourier_function(self, z: torch.Tensor) -> torch.Tensor:
#         """Evaluate the multidimensional Fourier-type function."""

#         # z = x + i*y
#         x = z.real
#         y = z.imag

#         # ---------------------------------------------------------
#         # omega^T x
#         #
#         # x:
#         #       [N, d]
#         #
#         # omega:
#         #       [K, d]
#         #
#         # result:
#         #       [N, K]
#         # ---------------------------------------------------------

#         phase_real = x @ self.omega.T

#         # ---------------------------------------------------------
#         # nu^T y
#         # ---------------------------------------------------------

#         phase_imag = y @ self.nu.T

#         # ---------------------------------------------------------
#         # Total phase
#         #
#         # theta_k =
#         # omega_k^T x + nu_k^T y
#         # ---------------------------------------------------------

#         phase = phase_real + phase_imag

#         # ---------------------------------------------------------
#         # Fourier basis
#         #
#         # exp(i theta)
#         # ---------------------------------------------------------

#         basis = torch.exp(
#             1j * phase
#         )

#         # ---------------------------------------------------------
#         # Sum over Fourier modes
#         #
#         # f(z) = sum_k c_k exp(i theta_k)
#         #
#         # [N,K] * [K] -> [N,K]
#         # sum -> [N]
#         # ---------------------------------------------------------

#         target = torch.sum(
#             basis * self.coefficients,
#             dim=1,
#         )

#         return target.to(torch.complex64)

#     def __len__(self) -> int:
#         """Return number of samples."""
#         return self.n_samples

#     def __getitem__(
#         self,
#         index: int,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Return one complex input and its complex target."""

#         return self.x[index], self.y[index].unsqueeze(0)


# def _optional_int(value: Any) -> int | None:
#     """Convert an optional configuration value to int."""

#     if value is None:
#         return None

#     return int(value)


# def _create_fourier_parameters(
#     *,
#     input_dim: int,
#     n_frequencies: int,
#     frequency_range: tuple[float, float],
#     seed: int,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Create the shared Fourier function parameters."""

#     generator = torch.Generator().manual_seed(seed)

#     freq_low, freq_high = frequency_range

#     omega = (
#         torch.rand(
#             n_frequencies,
#             input_dim,
#             generator=generator,
#         )
#         * (freq_high - freq_low)
#         + freq_low
#     ).float()

#     nu = (
#         torch.rand(
#             n_frequencies,
#             input_dim,
#             generator=generator,
#         )
#         * (freq_high - freq_low)
#         + freq_low
#     ).float()

#     coefficient_real = torch.randn(
#         n_frequencies,
#         generator=generator,
#     )

#     coefficient_imag = torch.randn(
#         n_frequencies,
#         generator=generator,
#     )

#     coefficients = torch.complex(
#         coefficient_real,
#         coefficient_imag,
#     )

#     coefficients = coefficients / torch.sqrt(
#         torch.tensor(float(n_frequencies))
#     )

#     return (
#         omega,
#         nu,
#         coefficients.to(torch.complex64),
#     )


# def build_function_approximation_datasets(
#     config: dict,
#     seed: int,
# ) -> tuple[Dataset, Dataset, Dataset, int, int]:
#     """Build train, validation, and test Fourier-function datasets.

#     Returns
#     -------
#     train_dataset:
#         Training dataset.

#     validation_dataset:
#         Validation dataset.

#     test_dataset:
#         Test dataset.

#     input_dim:
#         Number of complex input dimensions.

#     output_dim:
#         Number of complex output dimensions.
#         For this benchmark output_dim = 1.
#     """

#     dataset_config = config.get("dataset", {})

#     input_dim = int(
#         dataset_config.get("input_dim", 20)
#     )

#     n_frequencies = int(
#         dataset_config.get("n_frequencies", 50)
#     )

#     train_samples = int(
#         dataset_config.get("train_samples", 20000)
#     )

#     validation_samples = int(
#         dataset_config.get("validation_samples", 5000)
#     )

#     test_samples = int(
#         dataset_config.get("test_samples", 5000)
#     )

#     input_low = float(
#         dataset_config.get("input_low", -1.0)
#     )

#     input_high = float(
#         dataset_config.get("input_high", 1.0)
#     )

#     frequency_low = float(
#         dataset_config.get("frequency_low", -3.0)
#     )

#     frequency_high = float(
#         dataset_config.get("frequency_high", 3.0)
#     )

#     frequency_range = (
#         frequency_low,
#         frequency_high,
#     )

#     # ---------------------------------------------------------
#     # IMPORTANT:
#     #
#     # One mathematical function is created.
#     #
#     # Train, validation, and test all use the SAME
#     # omega, nu and coefficients.
#     # ---------------------------------------------------------

#     omega, nu, coefficients = _create_fourier_parameters(
#         input_dim=input_dim,
#         n_frequencies=n_frequencies,
#         frequency_range=frequency_range,
#         seed=seed,
#     )

#     common_kwargs = {
#         "input_dim": input_dim,
#         "n_frequencies": n_frequencies,
#         "input_range": (
#             input_low,
#             input_high,
#         ),
#         "frequency_range": frequency_range,
#         "omega": omega,
#         "nu": nu,
#         "coefficients": coefficients,
#     }

#     # ---------------------------------------------------------
#     # Different random points for each split
#     # ---------------------------------------------------------

#     train_dataset = ComplexFourierFunctionDataset(
#         n_samples=train_samples,
#         seed=seed + 1,
#         **common_kwargs,
#     )

#     validation_dataset = ComplexFourierFunctionDataset(
#         n_samples=validation_samples,
#         seed=seed + 2,
#         **common_kwargs,
#     )

#     test_dataset = ComplexFourierFunctionDataset(
#         n_samples=test_samples,
#         seed=seed + 3,
#         **common_kwargs,
#     )

#     return (
#         train_dataset,
#         validation_dataset,
#         test_dataset,
#         input_dim,
#         1,
#     )


