"""
Flexible complex-valued function approximation dataset.

Input:
    C^input_dim

Output:
    C^output_dim

Example:
    input_dim=64
    output_dim=1

    C^64 -> C^1

Mathematical function:

    h(z) = |z|^2 z

    sin_split(z)
        = sin(Re(z)) + i sin(Im(z))

    cos_split(z)
        = cos(Re(z)) + i cos(Im(z))

    g(z)
        = quadratic_scale * z^2
        + cubic_scale * h(z)
        + mixed_cubic_scale * U h(z)
        + sin_scale * sin_split(z)
        + cos_scale * cos_split(z)

    f(z) = P g(z)

where U is a fixed unitary matrix of shape

    [input_dim, input_dim]

and P is a fixed projection matrix of shape

    [input_dim, output_dim].

The same U and P are used for train,
validation, and test datasets.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ComplexFunctionDataset(Dataset):
    """Flexible complex-valued function approximation dataset."""

    def __init__(
        self,
        n_samples: int,
        input_dim: int,
        output_dim: int,
        *,
        input_range: tuple[float, float] = (-1.0, 1.0),
        seed: int = 0,
        unitary_matrix: torch.Tensor | None = None,
        projection_matrix: torch.Tensor | None = None,

        # ---------------------------------------------------------------
        # Function coefficients
        # ---------------------------------------------------------------

        quadratic_scale: float = 1.0,
        cubic_scale: float = 1.0,
        mixed_cubic_scale: float = 0.2,
        sin_scale: float = 0.3,
        cos_scale: float = 0.3,
    ) -> None:

        super().__init__()

        # ---------------------------------------------------------------
        # Validate arguments
        # ---------------------------------------------------------------

        if n_samples < 1:
            raise ValueError(
                "n_samples must be positive."
            )

        if input_dim < 1:
            raise ValueError(
                "input_dim must be positive."
            )

        if output_dim < 1:
            raise ValueError(
                "output_dim must be positive."
            )

        if output_dim > input_dim:
            raise ValueError(
                "output_dim cannot be larger than input_dim "
                "for this projection-based dataset."
            )

        if input_range[0] >= input_range[1]:
            raise ValueError(
                "input_range must satisfy low < high."
            )

        # ---------------------------------------------------------------
        # Store configuration
        # ---------------------------------------------------------------

        self.n_samples = n_samples
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.quadratic_scale = float(
            quadratic_scale
        )

        self.cubic_scale = float(
            cubic_scale
        )

        self.mixed_cubic_scale = float(
            mixed_cubic_scale
        )

        self.sin_scale = float(
            sin_scale
        )

        self.cos_scale = float(
            cos_scale
        )

        # ---------------------------------------------------------------
        # Generator for this dataset's samples
        # ---------------------------------------------------------------

        generator = torch.Generator().manual_seed(seed)

        # ---------------------------------------------------------------
        # Generate complex input
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
        # Fixed unitary matrix U
        #
        # Shape:
        #
        #     [input_dim, input_dim]
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
        # Fixed projection matrix P
        #
        # Shape:
        #
        #     [input_dim, output_dim]
        # ---------------------------------------------------------------

        if projection_matrix is None:

            self.P = self._generate_projection_matrix(
                input_dim=input_dim,
                output_dim=output_dim,
                generator=generator,
            )

        else:

            if projection_matrix.shape != (
                input_dim,
                output_dim,
            ):
                raise ValueError(
                    "projection_matrix must have shape "
                    f"({input_dim}, {output_dim}), "
                    f"got {projection_matrix.shape}."
                )

            self.P = projection_matrix.to(
                torch.complex64
            )

        # ---------------------------------------------------------------
        # Generate target
        # ---------------------------------------------------------------

        self.y = self._function(self.x)

    # ===================================================================
    # Generate unitary matrix
    # ===================================================================

    @staticmethod
    def _generate_unitary_matrix(
        input_dim: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Generate a fixed complex unitary matrix U.

        Shape:
            [input_dim, input_dim]
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
        # Normalize arbitrary QR phases
        # ---------------------------------------------------------------

        diagonal = torch.diagonal(r)

        phase = diagonal / (
            torch.abs(diagonal) + 1e-12
        )

        q = q * phase.conj().unsqueeze(0)

        return q.to(torch.complex64)

    # ===================================================================
    # Generate projection matrix
    # ===================================================================

    @staticmethod
    def _generate_projection_matrix(
        input_dim: int,
        output_dim: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Generate a fixed complex projection matrix.

        Shape:
            [input_dim, output_dim]

        Each column is normalized.
        """

        real = torch.randn(
            input_dim,
            output_dim,
            generator=generator,
        )

        imag = torch.randn(
            input_dim,
            output_dim,
            generator=generator,
        )

        matrix = torch.complex(
            real,
            imag,
        ).to(torch.complex64)

        # ---------------------------------------------------------------
        # Normalize each output direction
        # ---------------------------------------------------------------

        norms = torch.linalg.vector_norm(
            matrix,
            dim=0,
            keepdim=True,
        )

        matrix = matrix / (
            norms + 1e-12
        )

        return matrix.to(torch.complex64)

    # ===================================================================
    # Split sine
    # ===================================================================

    @staticmethod
    def _split_sin(
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Split-complex sine.

        sin_split(z)
            = sin(Re(z)) + i sin(Im(z))

        Input:
            z: [N, input_dim]

        Output:
            [N, input_dim]
        """

        return torch.complex(
            torch.sin(z.real),
            torch.sin(z.imag),
        )

    # ===================================================================
    # Split cosine
    # ===================================================================

    @staticmethod
    def _split_cos(
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Split-complex cosine.

        cos_split(z)
            = cos(Re(z)) + i cos(Im(z))

        Input:
            z: [N, input_dim]

        Output:
            [N, input_dim]
        """

        return torch.complex(
            torch.cos(z.real),
            torch.cos(z.imag),
        )

    # ===================================================================
    # Mathematical target function
    # ===================================================================

    def _function(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the complex-valued target function.

        Mathematical definition:

            h(z) = |z|^2 z

            g(z)
                = a_q z^2
                + a_c h(z)
                + a_m U h(z)
                + a_s sin_split(z)
                + a_cos cos_split(z)

            f(z) = P g(z)

        where:

            a_q   = quadratic_scale
            a_c   = cubic_scale
            a_m   = mixed_cubic_scale
            a_s   = sin_scale
            a_cos = cos_scale

        Input:
            z: [N, input_dim]

        Output:
            target: [N, output_dim]
        """

        # ---------------------------------------------------------------
        # Quadratic component
        # ---------------------------------------------------------------

        quadratic = z**2

        # ---------------------------------------------------------------
        # Cubic component
        #
        # h(z) = |z|^2 z
        # ---------------------------------------------------------------

        cubic = (
            torch.abs(z) ** 2
        ) * z

        # ---------------------------------------------------------------
        # Mixed cubic component
        #
        # U h(z)
        # ---------------------------------------------------------------

        mixed_cubic = (
            cubic @ self.U.T
        )

        # ---------------------------------------------------------------
        # Split trigonometric components
        # ---------------------------------------------------------------

        sine = self._split_sin(z)

        cosine = self._split_cos(z)

        # ---------------------------------------------------------------
        # Combine all components
        # ---------------------------------------------------------------

        features = (
            self.quadratic_scale * quadratic
            + self.cubic_scale * cubic
            + self.mixed_cubic_scale * mixed_cubic
            + self.sin_scale * sine
            + self.cos_scale * cosine
        )

        # ---------------------------------------------------------------
        # Projection
        #
        # C^input_dim -> C^output_dim
        # ---------------------------------------------------------------

        target = features @ self.P

        return target.to(torch.complex64)

    # ===================================================================
    # Dataset interface
    # ===================================================================

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


# =========================================================================
# Dataset builder
# =========================================================================

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
        {},
    )

    # ---------------------------------------------------------------------
    # Input dimension
    # ---------------------------------------------------------------------

    input_dim = int(
        dataset_config.get(
            "input_dim",
            64,
        )
    )

    # ---------------------------------------------------------------------
    # Output dimension
    # ---------------------------------------------------------------------

    output_dim = int(
        dataset_config.get(
            "output_dim",
            1,
        )
    )

    if output_dim > input_dim:
        raise ValueError(
            "output_dim cannot be larger than input_dim."
        )

    # ---------------------------------------------------------------------
    # Number of samples
    # ---------------------------------------------------------------------

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

    # ---------------------------------------------------------------------
    # Input range
    # ---------------------------------------------------------------------

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

    # ---------------------------------------------------------------------
    # Function coefficients
    # ---------------------------------------------------------------------

    quadratic_scale = float(
        dataset_config.get(
            "quadratic_scale",
            1.0,
        )
    )

    cubic_scale = float(
        dataset_config.get(
            "cubic_scale",
            1.0,
        )
    )

    mixed_cubic_scale = float(
        dataset_config.get(
            "mixed_cubic_scale",
            0.2,
        )
    )

    sin_scale = float(
        dataset_config.get(
            "sin_scale",
            0.3,
        )
    )

    cos_scale = float(
        dataset_config.get(
            "cos_scale",
            0.3,
        )
    )

    # ---------------------------------------------------------------------
    # Generate ONE fixed unitary matrix
    # ---------------------------------------------------------------------

    matrix_generator = torch.Generator().manual_seed(
        seed
    )

    unitary_matrix = (
        ComplexFunctionDataset
        ._generate_unitary_matrix(
            input_dim=input_dim,
            generator=matrix_generator,
        )
    )

    # ---------------------------------------------------------------------
    # Generate ONE fixed projection matrix
    # ---------------------------------------------------------------------

    projection_matrix = (
        ComplexFunctionDataset
        ._generate_projection_matrix(
            input_dim=input_dim,
            output_dim=output_dim,
            generator=matrix_generator,
        )
    )

    # ---------------------------------------------------------------------
    # Training dataset
    # ---------------------------------------------------------------------

    train_dataset = ComplexFunctionDataset(
        n_samples=train_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        input_range=input_range,
        seed=seed + 1,
        unitary_matrix=unitary_matrix,
        projection_matrix=projection_matrix,
        quadratic_scale=quadratic_scale,
        cubic_scale=cubic_scale,
        mixed_cubic_scale=mixed_cubic_scale,
        sin_scale=sin_scale,
        cos_scale=cos_scale,
    )

    # ---------------------------------------------------------------------
    # Validation dataset
    # ---------------------------------------------------------------------

    validation_dataset = ComplexFunctionDataset(
        n_samples=validation_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        input_range=input_range,
        seed=seed + 2,
        unitary_matrix=unitary_matrix,
        projection_matrix=projection_matrix,
        quadratic_scale=quadratic_scale,
        cubic_scale=cubic_scale,
        mixed_cubic_scale=mixed_cubic_scale,
        sin_scale=sin_scale,
        cos_scale=cos_scale,
    )

    # ---------------------------------------------------------------------
    # Test dataset
    # ---------------------------------------------------------------------

    test_dataset = ComplexFunctionDataset(
        n_samples=test_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        input_range=input_range,
        seed=seed + 3,
        unitary_matrix=unitary_matrix,
        projection_matrix=projection_matrix,
        quadratic_scale=quadratic_scale,
        cubic_scale=cubic_scale,
        mixed_cubic_scale=mixed_cubic_scale,
        sin_scale=sin_scale,
        cos_scale=cos_scale,
    )

    # ---------------------------------------------------------------------
    # Verify same underlying function
    # ---------------------------------------------------------------------

    assert torch.equal(
        train_dataset.U,
        validation_dataset.U,
    )

    assert torch.equal(
        train_dataset.U,
        test_dataset.U,
    )

    assert torch.equal(
        train_dataset.P,
        validation_dataset.P,
    )

    assert torch.equal(
        train_dataset.P,
        test_dataset.P,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        input_dim,
        output_dim,
    )