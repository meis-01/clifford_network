"""Deterministic seed setup.

This module keeps Python, NumPy, and PyTorch random number generators aligned so
experiments can be reproduced across runs and devices.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU PyTorch, and available CUDA generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
