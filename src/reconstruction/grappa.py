from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class KernelGeometry:
    patches: np.ndarray
    patch_indices: np.ndarray
    holes_x: dict[int, np.ndarray]
    holes_y: dict[int, np.ndarray]
    has_holes: bool


class Grappa:
    def __init__(
        self,
        kspace: np.ndarray,
        kernel_size: tuple[int, int] = (5, 5),
        coil_axis: int = -1,
        lamda: float = 0.01,
    ) -> None:
        self.kspace = kspace
        self.kernel_size = kernel_size
        self.coil_axis = coil_axis
        self.lamda = lamda
        self.kernel_geometry = self.get_kernel_geometries()

    def get_kernel_geometries(self) -> KernelGeometry:
        kspace = np.moveaxis(self.kspace, self.coil_axis, -1)
        mask = np.ascontiguousarray(np.abs(kspace[..., 0]) > 0)
        if mask.all():
            empty = np.array([], dtype=int)
            return KernelGeometry(
                patches=np.empty((0, *self.kernel_size, kspace.shape[-1]), dtype=bool),
                patch_indices=empty,
                holes_x={},
                holes_y={},
                has_holes=False,
            )

        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        padded_mask = np.pad(mask, ((kx2, kx2), (ky2, ky2)), mode="constant")
        patches = sliding_window_view(padded_mask, (kx, ky)).reshape(-1, kx, ky)
        patch_grid_shape = sliding_window_view(padded_mask, (kx, ky)).shape[:2]
        unique_patches, inverse_indices = np.unique(patches, return_inverse=True, axis=0)

        valid_patches = np.argwhere(~unique_patches[:, kx2, ky2]).squeeze()
        invalid_patches = np.argwhere(np.all(unique_patches == 0, axis=(1, 2))).squeeze()
        valid_patches = np.setdiff1d(np.atleast_1d(valid_patches), np.atleast_1d(invalid_patches), assume_unique=False)
        tiled_patches = np.tile(unique_patches[..., None], (1, 1, 1, kspace.shape[-1]))

        holes_x: dict[int, np.ndarray] = {}
        holes_y: dict[int, np.ndarray] = {}
        for patch_index in valid_patches:
            locations = np.argwhere(inverse_indices == patch_index).ravel()
            grid_x, grid_y = np.unravel_index(locations, patch_grid_shape)
            holes_x[int(patch_index)] = np.atleast_1d(grid_x + kx2)
            holes_y[int(patch_index)] = np.atleast_1d(grid_y + ky2)

        return KernelGeometry(
            patches=tiled_patches,
            patch_indices=np.asarray(valid_patches, dtype=int),
            holes_x=holes_x,
            holes_y=holes_y,
            has_holes=True,
        )

    def compute_weights(self, calib: np.ndarray) -> dict[int, np.ndarray]:
        if not self.kernel_geometry.has_holes:
            return {}

        calib = np.moveaxis(calib, self.coil_axis, -1)
        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        num_coils = calib.shape[-1]
        calib = np.pad(calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")
        source_patches = sliding_window_view(calib, (kx, ky, num_coils)).reshape(-1, kx, ky, num_coils)

        weights: dict[int, np.ndarray] = {}
        for patch_index in self.kernel_geometry.patch_indices:
            patch_mask = self.kernel_geometry.patches[patch_index, ...]
            source = source_patches[:, patch_mask]
            target = source_patches[:, kx2, ky2, :]
            source_h_source = source.conj().T @ source
            source_h_target = source.conj().T @ target
            reg = self.lamda * np.linalg.norm(source_h_source) / max(source_h_source.shape[0], 1)
            weights[int(patch_index)] = np.linalg.solve(
                source_h_source + reg * np.eye(source_h_source.shape[0]),
                source_h_target,
            ).T
        return weights

    def apply_weights(self, kspace: np.ndarray, weights: dict[int, np.ndarray]) -> np.ndarray:
        if not self.kernel_geometry.has_holes:
            return kspace

        kspace = np.moveaxis(kspace, self.coil_axis, -1)
        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        padded = np.pad(kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")
        recon = np.zeros_like(padded)

        for patch_index in self.kernel_geometry.patch_indices:
            patch_mask = self.kernel_geometry.patches[patch_index, ...]
            patch_weights = weights[int(patch_index)]
            for x_coord, y_coord in zip(
                self.kernel_geometry.holes_x[int(patch_index)],
                self.kernel_geometry.holes_y[int(patch_index)],
            ):
                source = padded[x_coord - kx2 : x_coord + kx2 + 1, y_coord - ky2 : y_coord + ky2 + 1, :]
                source = source[patch_mask]
                recon[x_coord, y_coord, :] = (patch_weights @ source[:, None]).squeeze()

        result = recon + padded
        return np.moveaxis(result[kx2:-kx2, ky2:-ky2, :], -1, self.coil_axis)
