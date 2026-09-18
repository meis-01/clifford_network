
"""
64-dimensional complex function approximation dataset.

Mathematical function:

    f : C^d -> C^d

    f(z) = z^2
           + |z|^2 z
           + 0.2 U(|z|^2 z)

where

    z in C^d

and U is a fixed complex unitary matrix.

For the main experiment:

    d = 64

The same U is used for the training, validation, and test
datasets. Therefore, all three datasets represent exactly
the same mathematical function; only the sampled inputs
are different.

Dataset shapes:

    input  :  [input_dim]
    target :  [input_dim]

For input_dim = 64:

    input  :  C^64
    target :  C^64
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ComplexFourierFunctionDataset(Dataset):
    """Synthetic 64-dimensional complex-valued function dataset."""

    def __init__(
        self,
        n_samples: int,
        input_dim: int,
        *,
        input_range: tuple[float, float] = (-1.0, 1.0),
        seed: int = 0,
        unitary_matrix: torch.Tensor | None = None,
    ) -> None:

        super().__init__()

        if n_samples < 1:
            raise ValueError("n_samples must be positive.")

        if input_dim < 1:
            raise ValueError("input_dim must be positive.")

        if input_range[0] >= input_range[1]:
            raise ValueError(
                "input_range must satisfy low < high."
            )

        self.n_samples = n_samples
        self.input_dim = input_dim

        # ---------------------------------------------------------------
        # Random generator for this dataset's input samples
        # ---------------------------------------------------------------

        generator = torch.Generator().manual_seed(seed)

        # ---------------------------------------------------------------
        # Generate complex inputs
        #
        # z = x + iy
        #
        # Shape:
        #
        #     [n_samples, input_dim]
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
        # Use the SAME unitary matrix for train/validation/test.
        #
        # If no matrix is provided, generate one.
        # ---------------------------------------------------------------

        if unitary_matrix is None:
            self.U = self._generate_unitary_matrix(
                input_dim=input_dim,
                generator=generator,
            )
        else:
            if unitary_matrix.shape != (
                input_dim,
                input_dim,
            ):
                raise ValueError(
                    "unitary_matrix must have shape "
                    f"({input_dim}, {input_dim}), "
                    f"got {unitary_matrix.shape}."
                )

            self.U = unitary_matrix.to(
                torch.complex64
            )

        # ---------------------------------------------------------------
        # Generate target
        # ---------------------------------------------------------------

        self.y = self._function(self.x)

    @staticmethod
    def _generate_unitary_matrix(
        input_dim: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Generate a fixed complex unitary matrix U.

        U satisfies approximately

            U^H U = I.
        """

        real = torch.randn(
            input_dim,
            input_dim,
            generator=generator,
        )

        imag = torch.randn(
            input_dim,
            input_dim,
            generator=generator,
        )

        matrix = torch.complex(
            real,
            imag,
        ).to(torch.complex64)

        # ---------------------------------------------------------------
        # Complex QR decomposition
        # ---------------------------------------------------------------

        q, r = torch.linalg.qr(matrix)

        # ---------------------------------------------------------------
        # Normalize the arbitrary phase introduced by QR.
        # ---------------------------------------------------------------

        diagonal = torch.diagonal(r)

        phase = diagonal / (
            torch.abs(diagonal) + 1e-12
        )

        q = q * phase.conj().unsqueeze(0)

        return q.to(torch.complex64)

    def _function(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate

            f(z)
            =
            z^2
            + |z|^2 z
            + 0.2 U(|z|^2 z)

        Input:
            z: [N, input_dim]

        Output:
            target: [N, input_dim]
        """

        # ---------------------------------------------------------------
        # Element-wise quadratic term
        #
        # z^2
        # ---------------------------------------------------------------

        quadratic = z**2

        # ---------------------------------------------------------------
        # Element-wise nonlinear cubic term
        #
        # |z|^2 z
        # ---------------------------------------------------------------

        cubic = (
            torch.abs(z) ** 2
        ) * z

        # ---------------------------------------------------------------
        # Cross-dimensional interaction
        #
        # U(|z|^2 z)
        #
        # cubic:
        #     [N, d]
        #
        # U:
        #     [d, d]
        #
        # Therefore:
        #
        #     cubic @ U.T
        #
        # gives:
        #
        #     [N, d]
        # ---------------------------------------------------------------

        mixed_cubic = cubic @ self.U.T

        # ---------------------------------------------------------------
        # Final function
        # ---------------------------------------------------------------

        target = (
            quadratic
            + cubic
            + 0.2 * mixed_cubic
        )

        return target.to(torch.complex64)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return (
            self.x[index],
            self.y[index],
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

    dataset_config = config.get(
        "dataset",
        {}
    )

    # ---------------------------------------------------------------
    # Input dimension
    # ---------------------------------------------------------------

    input_dim = int(
        dataset_config.get(
            "input_dim",
            64,
        )
    )

    # ---------------------------------------------------------------
    # Number of samples
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Input range
    # ---------------------------------------------------------------

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

    input_range = (
        input_low,
        input_high,
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # Generate ONE fixed unitary matrix U.
    #
    # The same U is then passed to train, validation and test.
    # ---------------------------------------------------------------

    matrix_generator = torch.Generator().manual_seed(
        seed
    )

    unitary_matrix = (
        ComplexFourierFunctionDataset
        ._generate_unitary_matrix(
            input_dim=input_dim,
            generator=matrix_generator,
        )
    )

    # ---------------------------------------------------------------
    # Training dataset
    # ---------------------------------------------------------------

    train_dataset = ComplexFourierFunctionDataset(
        n_samples=train_samples,
        input_dim=input_dim,
        input_range=input_range,
        seed=seed + 1,
        unitary_matrix=unitary_matrix,
    )

    # ---------------------------------------------------------------
    # Validation dataset
    # ---------------------------------------------------------------

    validation_dataset = ComplexFourierFunctionDataset(
        n_samples=validation_samples,
        input_dim=input_dim,
        input_range=input_range,
        seed=seed + 2,
        unitary_matrix=unitary_matrix,
    )

    # ---------------------------------------------------------------
    # Test dataset
    # ---------------------------------------------------------------

    test_dataset = ComplexFourierFunctionDataset(
        n_samples=test_samples,
        input_dim=input_dim,
        input_range=input_range,
        seed=seed + 3,
        unitary_matrix=unitary_matrix,
    )

    # ---------------------------------------------------------------
    # Optional sanity check:
    #
    # Verify that all datasets use exactly the same function.
    # ---------------------------------------------------------------

    assert torch.equal(
        train_dataset.U,
        validation_dataset.U,
    )

    assert torch.equal(
        train_dataset.U,
        test_dataset.U,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        input_dim,
        input_dim,
    )
