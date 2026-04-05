from __future__ import annotations

import numpy as np

from src.reconstruction.header import RegriddingParameters


def trapezoidal_density_profile(length: int, params: RegriddingParameters) -> np.ndarray:
    ramp_up = int(params.ramp_up_time or 0)
    ramp_down = int(params.ramp_down_time or 0)
    profile = np.ones(length, dtype=float)
    if ramp_up > 0:
        profile[: min(length, ramp_up)] = np.linspace(0.0, 1.0, min(length, ramp_up), endpoint=False)
    if ramp_down > 0:
        tail = min(length, ramp_down)
        profile[-tail:] = np.linspace(1.0, 0.0, tail, endpoint=False)
    return np.clip(profile, 1e-6, None)


def regrid_trapezoidal(kspace: np.ndarray, params: RegriddingParameters) -> np.ndarray:
    if all(
        value in (None, 0)
        for value in (
            params.ramp_up_time,
            params.ramp_down_time,
            params.flat_top_time,
            params.acquisition_delay_time,
        )
    ):
        return kspace

    profile = trapezoidal_density_profile(kspace.shape[-2], params)
    shape = (1,) * (kspace.ndim - 2) + (profile.shape[0], 1)
    return kspace / profile.reshape(shape)
