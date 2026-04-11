import numpy as np
from skimage.util import view_as_windows


class Grappa:
    def __init__(self, kspace: np.ndarray, kernel_size: tuple[int, int] = (5, 5), coil_axis: int = -1):
        self.kspace = np.moveaxis(kspace, coil_axis, -1)
        self.kernel_size = kernel_size
        self.coil_axis = coil_axis
        self.lamda = 0.01
        self.kernel_var_dict = self.get_kernel_geometries()

    def get_kernel_geometries(self) -> dict[str, object]:
        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        nc = self.kspace.shape[-1]

        kspace = np.pad(self.kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")
        mask = np.abs(kspace[..., 0]) > 0

        patches = view_as_windows(mask, (kx, ky))
        patch_shape = patches.shape[:]
        patches = patches.reshape(-1, kx, ky)

        unique_patches, inverse_indices = np.unique(patches, axis=0, return_inverse=True)
        valid = np.argwhere(~unique_patches[:, kx2, ky2]).squeeze()
        invalid = np.argwhere(np.all(unique_patches == 0, axis=(1, 2)))
        valid = np.setdiff1d(np.atleast_1d(valid), invalid, assume_unique=True)
        unique_patches = np.tile(unique_patches[..., None], (1, 1, 1, nc))

        holes_x: dict[int, np.ndarray] = {}
        holes_y: dict[int, np.ndarray] = {}
        for index in valid:
            coordinates = np.unravel_index(np.where(inverse_indices == index)[0], patch_shape[:2])
            holes_x[index] = coordinates[0] + kx2
            holes_y[index] = coordinates[1] + ky2

        return {
            "patches": unique_patches,
            "patch_indices": valid,
            "holes_x": holes_x,
            "holes_y": holes_y,
        }

    def compute_weights(self, calib: np.ndarray) -> dict[int, np.ndarray]:
        calib = np.moveaxis(calib, self.coil_axis, -1)
        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        nc = calib.shape[-1]
        calib = np.pad(calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")
        patches = view_as_windows(calib, (kx, ky, nc)).reshape(-1, kx, ky, nc)

        weights: dict[int, np.ndarray] = {}
        for index in self.kernel_var_dict["patch_indices"]:
            mask = self.kernel_var_dict["patches"][index]
            sources = patches[:, mask]
            targets = patches[:, kx2, ky2, :]
            shs = sources.conj().T @ sources
            sht = sources.conj().T @ targets
            lam = self.lamda * np.linalg.norm(shs) / shs.shape[0]
            weights[index] = np.linalg.solve(shs + lam * np.eye(shs.shape[0]), sht).T
        return weights

    def apply_weights(self, kspace: np.ndarray, weights: dict[int, np.ndarray]) -> np.ndarray:
        kspace = np.moveaxis(kspace, self.coil_axis, -1)
        kx, ky = self.kernel_size
        kx2, ky2 = kx // 2, ky // 2
        adjx, adjy = kx % 2, ky % 2

        padded = np.pad(kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")
        recon = padded.copy()

        for index in self.kernel_var_dict["patch_indices"]:
            mask = self.kernel_var_dict["patches"][index]
            xs = self.kernel_var_dict["holes_x"][index]
            ys = self.kernel_var_dict["holes_y"][index]

            for x, y in zip(xs, ys, strict=False):
                patch = padded[x - kx2:x + kx2 + adjx, y - ky2:y + ky2 + adjy, :]
                sources = patch[mask]
                recon[x, y, :] = (weights[index] @ sources[:, None]).squeeze()

        recon = recon[kx2:-kx2, ky2:-ky2, :]
        return np.moveaxis(recon, -1, self.coil_axis)