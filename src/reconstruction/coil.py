from __future__ import annotations

import numpy as np


def rss_combine(image_coils: np.ndarray, coil_axis: int = 0) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(image_coils) ** 2, axis=coil_axis))


def sensitivity_combine(image_coils: np.ndarray, sensitivity_maps: np.ndarray, coil_axis: int = 0) -> np.ndarray:
    return np.sum(image_coils * np.conj(sensitivity_maps), axis=coil_axis)
