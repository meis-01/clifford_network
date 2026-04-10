import numpy as np
from typing import Dict, Tuple
from skimage.util import view_as_windows


class Grappa:
    def __init__(self, kspace: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), coil_axis: int = -1):
        self.kspace = np.moveaxis(kspace, coil_axis, -1)
        self.kernel_size = kernel_size
        self.coil_axis = coil_axis
        self.lamda = 0.01

        self.kernel_var_dict = self.get_kernel_geometries()

    def get_kernel_geometries(self):
        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        nc = self.kspace.shape[-1]

        kspace = np.pad(self.kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
        mask = np.abs(kspace[..., 0]) > 0

        patches = view_as_windows(mask, (kx, ky))
        Psh = patches.shape[:2]
        patches = patches.reshape(-1, kx, ky)

        unique_patches, iidx = np.unique(patches, axis=0, return_inverse=True)

        # valid: center is zero (hole), but patch not empty
        valid = np.where(~unique_patches[:, kx2, ky2])[0]
        non_empty = np.where(np.any(unique_patches, axis=(1, 2)))[0]
        valid = np.intersect1d(valid, non_empty)

        unique_patches = np.tile(unique_patches[..., None], (1, 1, 1, nc))

        holes_x, holes_y = {}, {}

        for ii in valid:
            idx = np.unravel_index(np.where(iidx == ii)[0], Psh)
            x = idx[0] + kx2
            y = idx[1] + ky2
            holes_x[ii] = x
            holes_y[ii] = y

        return {
            "patches": unique_patches,
            "patch_indices": valid,
            "holes_x": holes_x,
            "holes_y": holes_y
        }

    def compute_weights(self, calib: np.ndarray) -> Dict[int, np.ndarray]:
        calib = np.moveaxis(calib, self.coil_axis, -1)

        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        nc = calib.shape[-1]

        calib = np.pad(calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')

        A = view_as_windows(calib, (kx, ky, nc)).reshape(-1, kx, ky, nc)

        weights = {}

        for ii in self.kernel_var_dict["patch_indices"]:
            mask = self.kernel_var_dict["patches"][ii]

            S = A[:, mask]                  # sources
            T = A[:, kx2, ky2, :]          # targets

            ShS = S.conj().T @ S
            ShT = S.conj().T @ T

            lam = self.lamda * np.linalg.norm(ShS) / ShS.shape[0]

            W = np.linalg.solve(
                ShS + lam * np.eye(ShS.shape[0]),
                ShT
            ).T

            weights[ii] = W

        return weights

    def apply_weights(self, kspace: np.ndarray, weights: Dict[int, np.ndarray]) -> np.ndarray:
        kspace = np.moveaxis(kspace, self.coil_axis, -1)

        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        adjx, adjy = kx % 2, ky % 2

        padded = np.pad(kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
        recon = padded.copy()

        for ii in self.kernel_var_dict["patch_indices"]:
            mask = self.kernel_var_dict["patches"][ii]

            xs = self.kernel_var_dict["holes_x"][ii]
            ys = self.kernel_var_dict["holes_y"][ii]

            for x, y in zip(xs, ys):
                patch = padded[x-kx2:x+kx2+adjx, y-ky2:y+ky2+adjy, :]
                S = patch[mask]

                recon[x, y, :] = (weights[ii] @ S[:, None]).squeeze()

        recon = recon[kx2:-kx2, ky2:-ky2, :]
        return np.moveaxis(recon, -1, self.coil_axis)