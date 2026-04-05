from __future__ import annotations

import numpy as np

from src.reconstruction.coil import rss_combine
from src.reconstruction.grappa import Grappa
from src.reconstruction.header import parse_ismrmrd_header
from src.reconstruction.padding import pad_kspace_phase
from src.reconstruction.regridding import regrid_trapezoidal


def reconstruct_t2_rss(
    kspace: np.ndarray,
    calib_data: np.ndarray,
    hdr: bytes | str,
    slice_idx: int = 0,
    kernel_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    header = parse_ismrmrd_header(hdr)
    target_phase = header.encoded_matrix[1]
    num_avg = kspace.shape[0]
    images: list[np.ndarray] = []

    for avg_idx in range(num_avg):
        undersampled = np.moveaxis(kspace[avg_idx, slice_idx], 0, -1)
        calibration = np.moveaxis(calib_data[slice_idx], 0, -1)

        grappa = Grappa(undersampled, kernel_size=kernel_size, coil_axis=-1)
        weights = grappa.compute_weights(calibration)
        filled = grappa.apply_weights(undersampled, weights)
        filled = regrid_trapezoidal(filled, header.regridding)

        filled = np.moveaxis(filled, -1, 0)
        filled = pad_kspace_phase(filled, target_phase)

        image_coils = np.fft.ifftshift(filled, axes=(-2, -1))
        image_coils = np.fft.ifft2(image_coils, axes=(-2, -1), norm="ortho")
        image_coils = np.fft.fftshift(image_coils, axes=(-2, -1))
        images.append(rss_combine(image_coils, coil_axis=0))

    return np.mean(np.stack(images, axis=0), axis=0)
