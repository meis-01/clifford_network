from __future__ import annotations

from typing import Any

from src.reconstruction.t2.prostate_t2_recon import t2_reconstruction


def reconstruct_t2(
    kspace: Any,
    calib_data: Any,
    hdr: str,
    slice_idx: int | None = None,
    kernel_size: tuple[int, int] | None = None,
    return_type: str | None = None,
):
    """Compatibility wrapper around the T2 reconstruction pipeline."""
    _ = kernel_size  # Kept for API compatibility with historical callers.
    result = t2_reconstruction(kspace_data=kspace, calib_data=calib_data, hdr=hdr)

    if return_type == "rss":
        if slice_idx is None:
            slice_idx = 0
        return result["rss"][slice_idx]
    return result


def reconstruct_t2_rss(
    kspace: Any,
    calib_data: Any,
    hdr: str,
    slice_idx: int = 0,
    kernel_size: tuple[int, int] = (5, 5),
):
    """Return an RSS image slice for a given input volume."""
    return reconstruct_t2(
        kspace=kspace,
        calib_data=calib_data,
        hdr=hdr,
        slice_idx=slice_idx,
        kernel_size=kernel_size,
        return_type="rss",
    )
