from __future__ import annotations

import numpy as np


def compute_symmetric_padding(current_size: int, target_size: int) -> tuple[int, int]:
    """Compute left/right padding required to reach target size."""
    if target_size < current_size:
        raise ValueError("target_size must be >= current_size")
    delta = target_size - current_size
    left = delta // 2
    right = delta - left
    return left, right


def pad_kspace_phase(kspace: np.ndarray, target_phase: int) -> np.ndarray:
    """Pad k-space along the last (phase-encode) axis."""
    left, right = compute_symmetric_padding(kspace.shape[-1], target_phase)
    pad_spec = [(0, 0)] * kspace.ndim
    pad_spec[-1] = (left, right)
    return np.pad(kspace, tuple(pad_spec), mode="constant")

# def resize_kspace(kspace: np.ndarray, header: ParsedIsmrmrdHeader) -> np.ndarray:
#     _, _, ro, pe = kspace.shape
#     nx, ny, _ = header.recon_matrix

#     out = np.zeros((kspace.shape[0], kspace.shape[1], nx, ny), dtype=kspace.dtype)

#     cx = header.center
#     cy = pe // 2

#     cx_out = nx // 2
#     cy_out = ny // 2

#     x_in0 = max(cx - cx_out, 0)
#     x_in1 = min(cx + cx_out, ro)

#     y_in0 = max(cy - cy_out, 0)
#     y_in1 = min(cy + cy_out, pe)

#     x_out0 = max(cx_out - cx, 0)
#     y_out0 = max(cy_out - cy, 0)

#     out[:, :, x_out0:x_out0+(x_in1-x_in0), y_out0:y_out0+(y_in1-y_in0)] = \
#         kspace[:, :, x_in0:x_in1, y_in0:y_in1]

#     return out