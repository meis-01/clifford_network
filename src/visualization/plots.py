from __future__ import annotations

import numpy as np

try:
    import holoviews as hv
except ImportError:  # pragma: no cover - optional dependency
    hv = None


def _require_holoviews() -> None:
    if hv is None:
        raise RuntimeError("HoloViews is required for visualization")


def plot_kspace_magnitude(kspace: np.ndarray):
    _require_holoviews()
    image = np.log1p(np.abs(kspace))
    return hv.Image(image).opts(cmap="viridis", colorbar=True)


def plot_rss_image(image: np.ndarray):
    _require_holoviews()
    return hv.Image(image).opts(cmap="gray", colorbar=True)
