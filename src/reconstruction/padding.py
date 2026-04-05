from __future__ import annotations

import numpy as np


def compute_symmetric_padding(current_size: int, target_size: int) -> tuple[int, int]:
    if target_size <= current_size:
        return 0, 0
    total = target_size - current_size
    left = total // 2
    right = total - left
    return left, right


def pad_kspace_phase(kspace: np.ndarray, target_phase: int) -> np.ndarray:
    left, right = compute_symmetric_padding(kspace.shape[-1], target_phase)
    if left == 0 and right == 0:
        return kspace
    pad_spec = [(0, 0)] * kspace.ndim
    pad_spec[-1] = (left, right)
    return np.pad(kspace, pad_spec, mode="constant")
