
"""Complex-valued nonlinear channel equalization dataset.

This module implements a reproducible synthetic benchmark for
complex-valued neural networks.

The task is nonlinear channel equalization:

    C^5 -> C

The received signal is generated from:

    1. Random 16-QAM transmitted symbols
    2. Complex multipath channel
    3. Memoryless cubic nonlinearity
    4. Complex AWGN noise

The neural network receives a window of five consecutive received
complex-valued samples and predicts the delayed transmitted symbol.

Mathematical setup
------------------

Let x[n] be the transmitted complex symbol.

The linear channel is

    u[n] = sum_k h[k] x[n-k]

with

    h = [0.34 + 0.21j,
         0.87 - 0.13j,
         0.34 + 0.09j].

A memoryless cubic nonlinearity is then applied:

    v[n] = u[n] + 0.2 u[n] |u[n]|^2.

Complex AWGN is added:

    r[n] = v[n] + eta[n].

The network input is

    X[n] = [r[n-5], r[n-4], ..., r[n-1]]

and the target is

    y[n] = x[n-2].

Therefore:

    input dimension  = 5
    output dimension = 1

Both input and output are complex-valued.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ComplexChannelEqualizationDataset(Dataset):
    """Synthetic complex nonlinear channel equalization dataset."""

    def __init__(
        self,
        n_samples: int,
        *,
        window: int = 5,
        snr_db: float = 20.0,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if n_samples < 1:
            raise ValueError(
                "n_samples must be positive."
            )

        if window < 1:
            raise ValueError(
                "window must be positive."
            )

        self.n_samples = n_samples
        self.window = window
        self.snr_db = snr_db

        rng = np.random.default_rng(seed)

        # ---------------------------------------------------------
        # 1. Generate random 16-QAM transmitted symbols
        # ---------------------------------------------------------

        tx = self._qam16_symbols(
            n_samples + window,
            rng,
        )

        # ---------------------------------------------------------
        # 2. Complex multipath channel
        # ---------------------------------------------------------

        channel_taps = np.array(
            [
                0.34 + 0.21j,
                0.87 - 0.13j,
                0.34 + 0.09j,
            ],
            dtype=np.complex64,
        )

        linear_out = np.convolve(
            tx,
            channel_taps,
            mode="full",
        )[: len(tx)]

        # ---------------------------------------------------------
        # 3. Memoryless nonlinear distortion
        #
        #       f(x) = x + 0.2*x*|x|^2
        # ---------------------------------------------------------

        nonlinear_out = (
            linear_out
            + 0.2
            * linear_out
            * np.abs(linear_out) ** 2
        )

        # ---------------------------------------------------------
        # 4. Add complex AWGN
        # ---------------------------------------------------------

        signal_power = np.mean(
            np.abs(nonlinear_out) ** 2
        )

        noise_power = (
            signal_power
            / (10.0 ** (snr_db / 10.0))
        )

        noise = np.sqrt(
            noise_power / 2.0
        ) * (
            rng.standard_normal(len(nonlinear_out))
            + 1j
            * rng.standard_normal(len(nonlinear_out))
        )

        rx = nonlinear_out + noise

        # ---------------------------------------------------------
        # 5. Build windowed input/output pairs
        #
        # Input:
        #
        #   [rx[i-window], ..., rx[i-1]]
        #
        # Target:
        #
        #   tx[i-window//2]
        #
        # For window=5:
        #
        #   input  = [rx[i-5], ..., rx[i-1]]
        #   target = tx[i-2]
        # ---------------------------------------------------------

        X = np.empty(
            (n_samples, window),
            dtype=np.complex64,
        )

        y = np.empty(
            n_samples,
            dtype=np.complex64,
        )

        delay = window // 2

        for i in range(
            window,
            n_samples + window,
        ):
            sample_idx = i - window

            X[sample_idx] = rx[
                i - window : i
            ]

            y[sample_idx] = tx[
                i - delay
            ]

        # ---------------------------------------------------------
        # Convert to PyTorch complex tensors
        # ---------------------------------------------------------

        self.x = torch.from_numpy(
            X
        ).to(torch.complex64)

        self.y = torch.from_numpy(
            y
        ).to(torch.complex64)

        self.channel_taps = torch.from_numpy(
            channel_taps
        ).to(torch.complex64)

    @staticmethod
    def _qam16_symbols(
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate normalized random 16-QAM symbols."""

        real = rng.choice(
            [-3, -1, 1, 3],
            size=n,
        )

        imag = rng.choice(
            [-3, -1, 1, 3],
            size=n,
        )

        symbols = (
            real + 1j * imag
        ) / np.sqrt(10.0)

        return symbols.astype(
            np.complex64
        )

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


def build_channel_equalization_datasets(
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

    The three splits use the same channel model and SNR,
    but independent random symbol/noise realizations.

    Returns
    -------
    train_dataset
        Training dataset.

    validation_dataset
        Validation dataset.

    test_dataset
        Test dataset.

    input_dim
        Number of complex received samples per input.

    output_dim
        Number of complex output values.
    """

    dataset_config = config.get(
        "dataset",
        {},
    )

    window = int(
        dataset_config.get(
            "window",
            5,
        )
    )

    snr_db = float(
        dataset_config.get(
            "snr_db",
            20.0,
        )
    )

    train_samples = int(
        dataset_config.get(
            "train_samples",
            8000,
        )
    )

    validation_samples = int(
        dataset_config.get(
            "validation_samples",
            1000,
        )
    )

    test_samples = int(
        dataset_config.get(
            "test_samples",
            1000,
        )
    )

    # ---------------------------------------------------------
    # Same channel/SNR configuration for every split.
    # Different seeds give independent samples.
    # ---------------------------------------------------------

    train_dataset = ComplexChannelEqualizationDataset(
        n_samples=train_samples,
        window=window,
        snr_db=snr_db,
        seed=seed,
    )

    validation_dataset = ComplexChannelEqualizationDataset(
        n_samples=validation_samples,
        window=window,
        snr_db=snr_db,
        seed=seed + 1,
    )

    test_dataset = ComplexChannelEqualizationDataset(
        n_samples=test_samples,
        window=window,
        snr_db=snr_db,
        seed=seed + 2,
    )

    # One complex value is predicted.
    input_dim = window
    output_dim = 1

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        input_dim,
        output_dim,
    )
