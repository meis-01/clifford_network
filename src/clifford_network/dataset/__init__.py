"""Dataset registry exports.

The dataset package provides synthetic, FFT-transformed vision, and fastMRI T2
dataset builders behind a single dataloader construction API.
"""

from clifford_network.dataset.registry import DataSpec, build_dataloaders

__all__ = ["DataSpec", "build_dataloaders"]
