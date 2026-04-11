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
from src.reconstruction.utils import fftnd, ifftnd


SUPPORTED_FEATURE_DOMAINS = {"kspace", "reconstruction"}
SUPPORTED_FEATURE_REPRESENTATIONS = {"real", "complex"}
SUPPORTED_COIL_COMBINATIONS = {"sense"}
SUPPORTED_AVERAGE_REDUCTIONS = {"mean", "max"}


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


def get_input_channels(feature_config: dict) -> int:
    representation = str(feature_config["representation"]).lower()
    if representation in {"real", "complex"}:
        return 2
    raise ValueError(f"Unsupported feature representation: {representation}")


def _validate_feature_config(feature_config: dict) -> None:
    domain = str(feature_config["domain"]).lower()
    representation = str(feature_config["representation"]).lower()
    coil_combination = str(feature_config["coil_combination"]).lower()
    average_reduction = str(feature_config["average_reduction"]).lower()

    if domain not in SUPPORTED_FEATURE_DOMAINS:
        raise ValueError(f"Unsupported feature domain: {domain}")
    if representation not in SUPPORTED_FEATURE_REPRESENTATIONS:
        raise ValueError(f"Unsupported feature representation: {representation}")
    if coil_combination not in SUPPORTED_COIL_COMBINATIONS:
        raise ValueError(f"Unsupported coil combination: {coil_combination}")
    if average_reduction not in SUPPORTED_AVERAGE_REDUCTIONS:
        raise ValueError(f"Unsupported average reduction: {average_reduction}")


def _coil_images_from_kspace(multicoil_kspace: np.ndarray) -> np.ndarray:
    images = np.zeros_like(multicoil_kspace, dtype=np.complex64)
    for average_index in range(multicoil_kspace.shape[0]):
        images[average_index] = ifftnd(multicoil_kspace[average_index], [1, 2])
    return images


def _estimate_sensitivity_maps(reference_kspace: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    reference_image = ifftnd(reference_kspace.astype(np.complex64), [1, 2])
    denominator = np.sqrt(np.sum(np.abs(reference_image) ** 2, axis=0, keepdims=True))
    denominator = np.maximum(denominator, eps)
    return reference_image / denominator


def _sense_combine(coil_images: np.ndarray, sensitivity_maps: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    numerator = np.sum(coil_images * sensitivity_maps.conj()[None, ...], axis=1)
    denominator = np.sum(np.abs(sensitivity_maps) ** 2, axis=0)
    denominator = np.maximum(denominator, eps)
    return numerator / denominator[None, ...]


def _build_physical_single_channel(
    multicoil_kspace: np.ndarray,
    sensitivity_maps: np.ndarray,
    domain: str,
) -> np.ndarray:
    coil_images = _coil_images_from_kspace(multicoil_kspace)
    combined_image = _sense_combine(coil_images, sensitivity_maps)
    if domain == "reconstruction":
        return combined_image.astype(np.complex64)

    combined_kspace = np.stack([fftnd(image, [0, 1]) for image in combined_image], axis=0)
    return combined_kspace.astype(np.complex64)


def _collapse_real_representation(
    data: np.ndarray,
    output_size: tuple[int, int],
    average_reduction: str,
    log_scale: bool,
    normalization: str,
) -> np.ndarray:
    magnitude = np.abs(data)
    if average_reduction == "mean":
        collapsed = magnitude.mean(axis=0)
    elif average_reduction == "max":
        collapsed = magnitude.max(axis=0)
    else:
        raise ValueError(f"Unsupported average reduction: {average_reduction}")

    if log_scale:
        collapsed = np.log1p(collapsed)

    resized = _resize_2d(collapsed, output_size)
    return _normalize_channel(resized, normalization)[None, ...]


def _collapse_complex_representation(
    data: np.ndarray,
    output_size: tuple[int, int],
    average_reduction: str,
    normalization: str,
) -> np.ndarray:
    if average_reduction == "mean":
        collapsed = data.mean(axis=0)
    else:
        raise ValueError("Complex-valued averaging currently supports only 'mean' to preserve a valid complex average.")

    real = _normalize_channel(_resize_2d(collapsed.real, output_size), normalization)
    imag = _normalize_channel(_resize_2d(collapsed.imag, output_size), normalization)
    return (real + 1j * imag).astype(np.complex64)


def _represent_volume(filled_data: np.ndarray, feature_config: dict) -> np.ndarray:
    _validate_feature_config(feature_config)
    representation = feature_config["representation"]

    if representation == "real":
        return _collapse_real_representation(
            filled_data,
            output_size=feature_config["output_size"],
            average_reduction=feature_config["average_reduction"],
            log_scale=bool(feature_config["log_scale"]),
            normalization=feature_config["normalization"],
        )

    return _collapse_complex_representation(
        filled_data,
        output_size=feature_config["output_size"],
        average_reduction=feature_config["average_reduction"],
        normalization=feature_config["normalization"],
    )


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


def _compute_dwi_filled_kspace(volume_path: str, slice_index: int, kernel_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    kspace, calibration_data, coil_sens_maps, hdr = load_file_dwi(volume_path)
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

    return np.stack(filled, axis=0), coil_sens_maps[slice_index, ...].astype(np.complex64)


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
        selected_kspace = _select_averages(filled_kspace, average_indices)
        reference_kspace = selected_kspace.mean(axis=0)
        sensitivity_maps = _estimate_sensitivity_maps(reference_kspace)
    elif modality == "dwi":
        filled_kspace, sensitivity_maps = _compute_dwi_filled_kspace(volume_path, slice_index, feature_config["kernel_size"])
        average_indices = feature_config.get("dwi_average_indices")
        selected_kspace = _select_averages(filled_kspace, average_indices)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    physical_volume = _build_physical_single_channel(selected_kspace, sensitivity_maps, feature_config["domain"])
    channel = _represent_volume(physical_volume, feature_config)
    if cache_path is not None:
        np.save(cache_path, channel)
    return channel


def extract_paired_input_channels(t2_path: str, dwi_path: str, slice_index: int, feature_config: dict) -> np.ndarray:
    t2_channel = _extract_channel("t2", t2_path, slice_index, feature_config)
    dwi_channel = _extract_channel("dwi", dwi_path, slice_index, feature_config)
    if str(feature_config["representation"]).lower() == "complex":
        return np.stack([t2_channel, dwi_channel], axis=0).astype(np.complex64)
    return np.concatenate([t2_channel, dwi_channel], axis=0).astype(np.float32)


def extract_paired_kspace_channels(t2_path: str, dwi_path: str, slice_index: int, feature_config: dict) -> np.ndarray:
    return extract_paired_input_channels(t2_path, dwi_path, slice_index, feature_config)