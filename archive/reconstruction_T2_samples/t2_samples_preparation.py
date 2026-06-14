from __future__ import annotations

import json
import xml.etree.ElementTree as etree
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

import pandas as pd
from numpy.fft import fftshift, ifftshift, ifftn
from skimage.util import view_as_windows


@dataclass(frozen=True)
class T2SliceConfig:
    labels_csv: Path
    data_root: Path
    output_dir: Path
    output_size: tuple[int, int] = (224, 224)
    compressed_coils: int = 12
    kernel_size: tuple[int, int] = (5, 5)
    positive_threshold: float = 2.0
    patient_id_column: str = "fastmri_pt_id"
    slice_column: str = "slice"
    split_column: str = "data_split"
    label_column: str = "PIRADS"
    folder_column: str = "folder"
    file_column: str = "fastmri_rawfile"
    drop_missing: bool = True
    limit: int | None = None
    cols: Sequence[str] = (
        'sample_id', 'fastmri_pt_id','slice','slice_index',
        'source_folder','source_file','coil','coil_index',
        'total_coils','data_split','label','raw_path','feature_path'
    )


def ifftnd(kspace, axes) -> np.ndarray:
    """
    Compute the n-dimensional inverse Fourier transform of the k-space data along the specified axes.

    Parameters:
    -----------
    kspace: np.ndarray
        The input k-space data.
    axes: list or tuple, optional
        The list of axes along which to compute the inverse Fourier transform. Default is [-1].

    Returns:
    --------
    img: ndarray
        The output image after inverse Fourier transform.
    """

    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)   
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))    

    return img

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


def et_query(root: etree.Element, qlist: Sequence[str], namespace: str = "http://www.ismrm.org/ISMRMRD") -> str:
    prefix = "ismrmrd_namespace"
    ns = {prefix: namespace}
    query = "."
    for element in qlist:
        query += f"//{prefix}:{element}"
    value = root.find(query, ns)
    if value is None:
        raise RuntimeError("Element not found")
    return str(value.text)


def encoded_readout_from_header(hdr: str | bytes) -> int | None:
    try:
        root = etree.fromstring(hdr)
        return int(et_query(root, ["encoding", "encodedSpace", "matrixSize", "x"]))
    except Exception:
        return None


def center_crop_or_pad_2d(array: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    target_rows, target_cols = output_size
    rows, cols = array.shape[-2:]
    row_start = max((rows - target_rows) // 2, 0)
    col_start = max((cols - target_cols) // 2, 0)
    cropped = array[..., row_start:row_start + min(rows, target_rows), col_start:col_start + min(cols, target_cols)]

    pad_rows = max(target_rows - cropped.shape[-2], 0)
    pad_cols = max(target_cols - cropped.shape[-1], 0)
    if pad_rows == 0 and pad_cols == 0:
        return cropped

    pad_width = [(0, 0)] * cropped.ndim
    pad_width[-2] = (pad_rows // 2, pad_rows - pad_rows // 2)
    pad_width[-1] = (pad_cols // 2, pad_cols - pad_cols // 2)
    return np.pad(cropped, pad_width, mode="constant")


def coil_first(slice_kspace: np.ndarray) -> np.ndarray:
    if slice_kspace.ndim != 3:
        raise ValueError(f"Expected a 3D k-space slice, got shape {slice_kspace.shape}")
    coil_axis = int(np.argmin(slice_kspace.shape))
    return np.moveaxis(slice_kspace, coil_axis, 0).astype(np.complex64)


def coil_first_to_grid(kspace: np.ndarray) -> np.ndarray:
    return np.moveaxis(kspace, 0, -1)


def grid_to_coil_first(kspace: np.ndarray) -> np.ndarray:
    return np.moveaxis(kspace, -1, 0)


def pad_phase_to_encoded_readout(kspace: np.ndarray, hdr: str | bytes) -> np.ndarray:
    encoded_readout = encoded_readout_from_header(hdr)
    if encoded_readout is None:
        return kspace

    phase = kspace.shape[-1]
    pad_total = max(encoded_readout - phase, 0)
    if pad_total == 0:
        return kspace

    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(kspace, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")


def compress_over_coils(kspace: np.ndarray, k: int = 8) -> np.ndarray:
    """
    kspace shape: (A, S, C, Nx, Ny)
    returns shape: (A, S, k, Nx, Ny)
    """
    acquisitions, slices, coils, readout, phase = kspace.shape
    if k > coils:
        raise ValueError(f"Cannot compress {coils} coils to {k} channels.")

    x = kspace.transpose(2, 0, 1, 3, 4).reshape(coils, -1)
    basis, _, _ = np.linalg.svd(x, full_matrices=False)
    projection = basis[:, :k]
    return np.einsum("ck,ascxy->askxy", np.conj(projection), kspace).astype(np.complex64)


def t2_complex_slice_from_h5(
    h5_path: Path,
    slice_index: int,
    output_size: tuple[int, int],
    compressed_coils: int,
    kernel_size: tuple[int, int],
) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        kspace = handle["kspace"][:]
        calibration_data = handle["calibration_data"][:]
        hdr = handle["ismrmrd_header"][()]

    if kspace.ndim != 5:
        raise ValueError(f"Expected T2 kspace shape (average, slice, ..., ..., ...), got {kspace.shape}")
    if slice_index < 0 or slice_index >= kspace.shape[1]:
        raise IndexError(f"slice_index={slice_index} is outside available range 0..{kspace.shape[1] - 1}")

    average_index = kspace.shape[0] // 2
    kspace_slice = coil_first(kspace[average_index, slice_index, ...])
    calibration_slice = coil_first(calibration_data[slice_index, ...])

    reference_grid = coil_first_to_grid(kspace_slice)
    calibration_grid = coil_first_to_grid(calibration_slice)
    grappa = Grappa(reference_grid, kernel_size=kernel_size, coil_axis=-1)
    weights = grappa.compute_weights(calibration_grid)
    filled_grid = grappa.apply_weights(reference_grid, weights)
    filled = grid_to_coil_first(filled_grid)
    filled = pad_phase_to_encoded_readout(filled, hdr)
    recon = ifftnd(filled, axes=(1, 2))
    if compressed_coils > 0:
        compressed = compress_over_coils(recon[None, None, ...], k=compressed_coils)[0, 0]
    else:
        compressed = recon
    if all(size > 0 for size in output_size):
        sized = center_crop_or_pad_2d(compressed, output_size)
    else:
        sized = compressed
    return np.nan_to_num(sized.astype(np.complex64), nan=0.0, posinf=0.0, neginf=0.0)
    # return np.nan_to_num(filled.astype(np.complex64), nan=0.0, posinf=0.0, neginf=0.0)

def build_t2_manifest(config: T2SliceConfig) -> pd.DataFrame:
    labels = pd.read_csv(config.labels_csv)
    required = {
        config.patient_id_column,
        config.slice_column,
        config.split_column,
        config.label_column,
        config.folder_column,
        config.file_column,
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Labels CSV is missing required columns: {sorted(missing)}")

    if config.limit is not None:
        labels = labels.head(config.limit).copy()

    records = []
    for _, row in labels.iterrows():
        raw_path = (config.data_root / Path(row[config.folder_column]) / Path(row[config.file_column])).resolve()
        if config.drop_missing and not raw_path.exists():
            continue

        patient_id = row[config.patient_id_column]
        slice_number = int(row[config.slice_column])
        sample_id = f"t2_pt_id_{patient_id:02d}_slice_{slice_number:02d}"
        feature_name = f"{sample_id}.npy"
        records.append(
            {
                "sample_id": sample_id,
                "fastmri_pt_id": patient_id,
                "slice": slice_number,
                "slice_index": slice_number - 1,
                "data_split": row[config.split_column],
                "label": int(float(row[config.label_column]) > config.positive_threshold),
                "raw_path": str(raw_path),
                "feature_path": str((config.output_dir / "features" / feature_name).resolve()),
                "source_folder": row[config.folder_column],
                "source_file": row[config.file_column],
            }
        )

    return pd.DataFrame.from_records(records)


def generate_t2_slices(config: T2SliceConfig) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = config.output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_t2_manifest(config)
    print(f"Generated manifest with {len(manifest)} entries. Starting feature extraction...")
    processed = []
    for index, row in manifest.iterrows():
        try:
            
            feature_path = Path(row["feature_path"])
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            feature = t2_complex_slice_from_h5(
                Path(row["raw_path"]),
                int(row["slice_index"]),
                config.output_size,
                config.compressed_coils,
                config.kernel_size,
            )
            slice_name = feature_path.stem
            save_dir = feature_path.parent / Path(f"{row["data_split"]}")
            save_dir.mkdir(parents=True, exist_ok=True)
            for coil in range(feature.shape[0]):
                file_name = save_dir / f"{slice_name}_coil_{coil+1:02d}.npy"
                np.save(file_name, feature[coil])
                print(f"Feature saved to {file_name.parent.stem}/{file_name.stem}")
                coil_row = row.copy()
                coil_row["feature_path"] = str(file_name)
                coil_row["coil"] = coil + 1
                coil_row["coil_index"] = coil
                coil_row["total_coils"] = feature.shape[0]
                processed.append(coil_row)
                processed_manifest = pd.DataFrame(processed)
                processed_manifest = processed_manifest[list(config.cols)]
                processed_manifest.to_csv(config.output_dir / "manifest.csv", index=False)

        except Exception as e:
            print(f"Error processing sample {index + 1}/{len(manifest)}: {e}")
            print("#" * 40)
            if not config.drop_missing:
                processed.append(row)
    with (config.output_dir / "processing_config.json").open("w", encoding="utf-8") as handle:
        payload = asdict(config)
        payload.update(
            {
                "labels_csv": str(config.labels_csv),
                "data_root": str(config.data_root),
                "output_dir": str(config.output_dir),
            }
        )
        json.dump(payload, handle, indent=2)
    return processed_manifest
