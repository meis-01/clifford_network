from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.data.mri_data import load_file_T2, load_file_dwi, zero_pad_kspace_hdr
from src.reconstruction.dwi.regridding import trapezoidal_regridding
from src.reconstruction.grappa import Grappa


def _resize_2d(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    tensor = torch.as_tensor(image, dtype=torch.float32)[None, None, ...]
    resized = F.interpolate(tensor, size=output_size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def _normalize_channel(image: np.ndarray, mode: str) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "minmax":
        min_value = float(image.min())
        max_value = float(image.max())
        if max_value > min_value:
            return (image - min_value) / (max_value - min_value)
        return np.zeros_like(image)

    mean_value = float(image.mean())
    std_value = float(image.std())
    if std_value < 1e-6:
        return image - mean_value
    return (image - mean_value) / std_value


def _cache_key(modality: str, volume_path: str, slice_index: int, feature_config: dict) -> str:
    digest = hashlib.sha1(f"{modality}|{volume_path}|{slice_index}|{feature_config}".encode("utf-8")).hexdigest()
    return digest[:20]


def _select_averages(kspace: np.ndarray, average_indices: Iterable[int] | None) -> np.ndarray:
    if average_indices is None:
        return kspace
    indices = [int(index) for index in average_indices]
    return kspace[indices, ...]


def _collapse_kspace(
    kspace: np.ndarray,
    output_size: tuple[int, int],
    coil_reduction: str,
    average_reduction: str,
    log_scale: bool,
    normalization: str,
) -> np.ndarray:
    magnitude = np.abs(kspace)
    if coil_reduction == "rss":
        collapsed = np.sqrt(np.sum(np.square(magnitude), axis=1))
    elif coil_reduction == "mean":
        collapsed = magnitude.mean(axis=1)
    else:
        raise ValueError(f"Unsupported coil reduction: {coil_reduction}")

    if average_reduction == "mean":
        collapsed = collapsed.mean(axis=0)
    elif average_reduction == "max":
        collapsed = collapsed.max(axis=0)
    else:
        raise ValueError(f"Unsupported average reduction: {average_reduction}")

    collapsed = np.fft.fftshift(collapsed, axes=(-2, -1))
    if log_scale:
        collapsed = np.log1p(collapsed)

    resized = _resize_2d(collapsed, output_size)
    return _normalize_channel(resized, normalization)


def _compute_t2_filled_kspace(volume_path: str, slice_index: int, kernel_size: tuple[int, int]) -> np.ndarray:
    kspace, calibration_data, hdr, _, _ = load_file_T2(volume_path)
    calibration_slice = calibration_data[slice_index, ...]
    reference = np.transpose(kspace[0, slice_index, ...], (2, 0, 1))
    grappa = Grappa(reference, kernel_size=kernel_size, coil_axis=1)
    weights = grappa.compute_weights(np.transpose(calibration_slice, (2, 0, 1)))

    filled = []
    for average in range(kspace.shape[0]):
        kspace_slice = np.transpose(kspace[average, slice_index, ...], (2, 0, 1))
        post_grappa = grappa.apply_weights(kspace_slice, weights)
        post_grappa = np.moveaxis(np.moveaxis(post_grappa, 0, 1), 1, 2)
        padded = zero_pad_kspace_hdr(hdr, post_grappa[None, ...])[0]
        filled.append(padded)

    return np.stack(filled, axis=0)


def _compute_dwi_filled_kspace(volume_path: str, slice_index: int, kernel_size: tuple[int, int]) -> np.ndarray:
    kspace, calibration_data, _, hdr = load_file_dwi(volume_path)
    calibration_slice = trapezoidal_regridding(calibration_data[slice_index, ...], hdr)
    reference = trapezoidal_regridding(kspace[0, slice_index, ...], hdr)
    grappa = Grappa(np.transpose(reference, (2, 0, 1)), kernel_size=kernel_size, coil_axis=1)
    weights = grappa.compute_weights(np.transpose(calibration_slice, (2, 0, 1)))

    filled = []
    for average in range(kspace.shape[0]):
        kspace_slice = trapezoidal_regridding(kspace[average, slice_index, ...], hdr)
        post_grappa = grappa.apply_weights(np.transpose(kspace_slice, (2, 0, 1)), weights)
        post_grappa = np.moveaxis(np.moveaxis(post_grappa, 0, 1), 1, 2)
        filled.append(post_grappa)

    return np.stack(filled, axis=0)


def _extract_channel(
    modality: str,
    volume_path: str,
    slice_index: int,
    feature_config: dict,
) -> np.ndarray:
    cache_dir = feature_config.get("cache_dir")
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = _cache_key(modality, volume_path, slice_index, feature_config) + ".npy"
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            return np.load(cache_path)

    if modality == "t2":
        filled_kspace = _compute_t2_filled_kspace(volume_path, slice_index, feature_config["kernel_size"])
        average_indices = feature_config.get("t2_average_indices")
    elif modality == "dwi":
        filled_kspace = _compute_dwi_filled_kspace(volume_path, slice_index, feature_config["kernel_size"])
        average_indices = feature_config.get("dwi_average_indices")
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    channel = _collapse_kspace(
        _select_averages(filled_kspace, average_indices),
        output_size=feature_config["output_size"],
        coil_reduction=feature_config["coil_reduction"],
        average_reduction=feature_config["average_reduction"],
        log_scale=bool(feature_config["log_scale"]),
        normalization=feature_config["normalization"],
    )
    if cache_path is not None:
        np.save(cache_path, channel)
    return channel


def extract_paired_kspace_channels(t2_path: str, dwi_path: str, slice_index: int, feature_config: dict) -> np.ndarray:
    t2_channel = _extract_channel("t2", t2_path, slice_index, feature_config)
    dwi_channel = _extract_channel("dwi", dwi_path, slice_index, feature_config)
    return np.stack([t2_channel, dwi_channel], axis=0).astype(np.float32)