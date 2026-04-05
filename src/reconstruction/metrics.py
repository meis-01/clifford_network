from __future__ import annotations

import numpy as np


def compute_trace(diffusion_images: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(diffusion_images, axis=axis)


def compute_adc(signal_low: np.ndarray, signal_high: np.ndarray, b_low: float, b_high: float, eps: float = 1e-8) -> np.ndarray:
    delta_b = b_high - b_low
    if delta_b <= 0:
        raise ValueError("b_high must be greater than b_low")
    return -np.log((signal_high + eps) / (signal_low + eps)) / delta_b


def synthesize_b1500(signal_b0: np.ndarray, adc_map: np.ndarray, b_value: float = 1500.0) -> np.ndarray:
    return signal_b0 * np.exp(-b_value * adc_map)
