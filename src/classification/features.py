from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Any

import numpy as np

from src.data.mri_data import compress_over_coils, load_file_T2, load_file_dwi, zero_pad_kspace_hdr
from src.reconstruction.dwi.regridding import trapezoidal_regridding
from src.reconstruction.grappa import Grappa
from src.reconstruction.utils import fftnd, ifftnd


SUPPORTED_FEATURE_DOMAINS = {"kspace", "reconstruction"}
SUPPORTED_FEATURE_REPRESENTATIONS = {"real", "complex"}
SUPPORTED_COIL_COMBINATIONS = {"sense"}
SUPPORTED_AVERAGE_REDUCTIONS = {"mean", "max"}
SUPPORTED_AVERAGE_SELECTIONS = {"all", "middle"}
SUPPORTED_MODALITIES = {"t2", "dwi"}
SUPPORTED_NORMALIZATIONS = {"zscore", "minmax", "none", "train_quantile"}


def _center_crop_or_pad_2d(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    target_rows, target_cols = output_size
    rows, cols = image.shape[-2:]

    row_start = max((rows - target_rows) // 2, 0)
    col_start = max((cols - target_cols) // 2, 0)
    cropped = image[..., row_start:row_start + min(rows, target_rows), col_start:col_start + min(cols, target_cols)]

    pad_rows = max(target_rows - cropped.shape[-2], 0)
    pad_cols = max(target_cols - cropped.shape[-1], 0)
    if pad_rows == 0 and pad_cols == 0:
        return cropped

    pad_before_rows = pad_rows // 2
    pad_after_rows = pad_rows - pad_before_rows
    pad_before_cols = pad_cols // 2
    pad_after_cols = pad_cols - pad_before_cols
    pad_width = [(0, 0)] * cropped.ndim
    pad_width[-2] = (pad_before_rows, pad_after_rows)
    pad_width[-1] = (pad_before_cols, pad_after_cols)
    return np.pad(cropped, pad_width, mode="constant")


def _normalize_channel(image: np.ndarray, mode: str, quantiles: tuple[float, float] | None = None) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return image
    if mode == "train_quantile":
        if quantiles is None:
            raise ValueError("train_quantile normalization requires training-split quantile statistics.")
        lower, upper = quantiles
        if upper <= lower:
            return image - lower
        return np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
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


def _select_middle_average(kspace: np.ndarray) -> np.ndarray:
    return kspace[kspace.shape[0] // 2:kspace.shape[0] // 2 + 1, ...]


def _configured_modalities(feature_config: dict) -> list[str]:
    modalities = feature_config.get("modalities", ["t2", "dwi"])
    if isinstance(modalities, str):
        modalities = [modalities]
    normalized = [str(modality).lower() for modality in modalities]
    unsupported = set(normalized) - SUPPORTED_MODALITIES
    if unsupported:
        raise ValueError(f"Unsupported modalities: {sorted(unsupported)}")
    if not normalized:
        raise ValueError("At least one modality must be configured.")
    return normalized


def get_input_channels(feature_config: dict) -> int:
    representation = str(feature_config["representation"]).lower()
    modality_count = len(_configured_modalities(feature_config))
    coil_compression_channels = feature_config.get("coil_compression_channels")
    if representation == "complex" and coil_compression_channels is not None:
        return modality_count * int(coil_compression_channels)
    if representation in {"real", "complex"}:
        return modality_count
    raise ValueError(f"Unsupported feature representation: {representation}")


def _validate_feature_config(feature_config: dict) -> None:
    domain = str(feature_config["domain"]).lower()
    representation = str(feature_config["representation"]).lower()
    coil_combination = str(feature_config["coil_combination"]).lower()
    average_reduction = str(feature_config["average_reduction"]).lower()
    average_selection = str(feature_config.get("average_selection", "all")).lower()
    normalization = str(feature_config["normalization"]).lower()
    _configured_modalities(feature_config)

    if domain not in SUPPORTED_FEATURE_DOMAINS:
        raise ValueError(f"Unsupported feature domain: {domain}")
    if representation not in SUPPORTED_FEATURE_REPRESENTATIONS:
        raise ValueError(f"Unsupported feature representation: {representation}")
    if coil_combination not in SUPPORTED_COIL_COMBINATIONS:
        raise ValueError(f"Unsupported coil combination: {coil_combination}")
    if average_reduction not in SUPPORTED_AVERAGE_REDUCTIONS:
        raise ValueError(f"Unsupported average reduction: {average_reduction}")
    if average_selection not in SUPPORTED_AVERAGE_SELECTIONS:
        raise ValueError(f"Unsupported average selection: {average_selection}")
    if normalization not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(f"Unsupported normalization: {normalization}")
    if feature_config.get("coil_compression_channels") is not None and (domain != "kspace" or representation != "complex"):
        raise ValueError("coil_compression_channels is only supported for complex k-space features.")


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

    sized = _center_crop_or_pad_2d(collapsed, output_size)
    return np.nan_to_num(sized.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)[None, ...]


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

    sized = _center_crop_or_pad_2d(collapsed, output_size)
    return np.nan_to_num(sized.astype(np.complex64), nan=0.0, posinf=0.0, neginf=0.0)


def _coil_first_kspace(filled_kspace: np.ndarray) -> np.ndarray:
    if filled_kspace.ndim != 4:
        raise ValueError(f"Expected filled k-space with shape (average, readout, coils, phase), got {filled_kspace.shape}")
    return np.moveaxis(filled_kspace, 2, 1)


def _compress_coils_svd(kspace: np.ndarray, target_coils: int) -> np.ndarray:
    if kspace.ndim != 3:
        raise ValueError(f"Expected k-space with shape (coils, readout, phase), got {kspace.shape}")

    coils, readout, phase = kspace.shape
    if target_coils > coils:
        raise ValueError(f"Cannot compress {coils} coils to {target_coils} channels.")

    expanded = kspace[None, None, ...]
    compressed = compress_over_coils(expanded, k=target_coils)
    return compressed[0, 0].astype(np.complex64)


def _size_complex_channels(
    channels: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    sized = _center_crop_or_pad_2d(channels, output_size)
    return np.nan_to_num(sized.astype(np.complex64), nan=0.0, posinf=0.0, neginf=0.0)


def _represent_compressed_kspace_channels(filled_kspace: np.ndarray, feature_config: dict) -> np.ndarray:
    _validate_feature_config(feature_config)
    target_coils = int(feature_config["coil_compression_channels"])
    selected = _coil_first_kspace(filled_kspace)
    selected = _select_middle_average(selected) if str(feature_config.get("average_selection", "all")).lower() == "middle" else selected
    if selected.shape[0] != 1:
        raise ValueError("Compressed complex k-space channels expect exactly one selected average.")

    compressed = _compress_coils_svd(selected[0], target_coils)
    return _size_complex_channels(
        compressed,
        output_size=feature_config["output_size"],
    )


def _stats_quantiles(stats: dict, channel_index: int, component: str | None = None) -> tuple[float, float]:
    if "magnitude_lower" in stats and "magnitude_upper" in stats:
        return float(stats["magnitude_lower"]), float(stats["magnitude_upper"])
    if component is None:
        return float(stats["lower"][channel_index]), float(stats["upper"][channel_index])
    return (
        float(stats[f"{component}_lower"][channel_index]),
        float(stats[f"{component}_upper"][channel_index]),
    )


def _normalize_complex_magnitude(data: np.ndarray, mode: str, quantiles: tuple[float, float] | None = None) -> np.ndarray:
    magnitude = np.abs(data).astype(np.float32)
    normalized_magnitude = _normalize_channel(magnitude, mode, quantiles)
    scale = np.divide(
        normalized_magnitude,
        magnitude,
        out=np.zeros_like(normalized_magnitude, dtype=np.float32),
        where=magnitude > 1e-8,
    )
    return (data * scale).astype(np.complex64)


def _normalize_input_channels(data: np.ndarray, feature_config: dict) -> np.ndarray:
    mode = str(feature_config["normalization"]).lower()
    if mode == "none":
        return data

    stats = feature_config.get("normalization_stats")
    if mode == "train_quantile" and stats is None:
        raise ValueError("train_quantile normalization requires normalization_stats computed from the training split.")
    normalized = np.empty_like(data)
    if np.iscomplexobj(data):
        if mode == "train_quantile":
            return _normalize_complex_magnitude(data, mode, _stats_quantiles(stats, 0))
        for channel_index in range(data.shape[0]):
            real = _normalize_channel(data[channel_index].real, mode)
            imag = _normalize_channel(data[channel_index].imag, mode)
            normalized[channel_index] = real + 1j * imag
        return normalized.astype(np.complex64)

    for channel_index in range(data.shape[0]):
        normalized[channel_index] = _normalize_channel(
            data[channel_index],
            mode,
            _stats_quantiles(stats, channel_index) if mode == "train_quantile" else None,
        )
    return normalized.astype(np.float32)


def _sample_flat_values(values: np.ndarray, values_per_sample: int) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size <= values_per_sample:
        return flat.astype(np.float32)
    indices = np.linspace(0, flat.size - 1, values_per_sample, dtype=np.int64)
    return flat[indices].astype(np.float32)


def _trim_sample_lists(sample_lists: list[list[np.ndarray]], max_values: int) -> None:
    for channel_samples in sample_lists:
        total = sum(sample.size for sample in channel_samples)
        if total <= max_values * 2:
            continue
        merged = np.concatenate(channel_samples)
        indices = np.linspace(0, merged.size - 1, max_values, dtype=np.int64)
        channel_samples[:] = [merged[indices].astype(np.float32)]


def _finalize_quantiles(sample_lists: list[list[np.ndarray]], lower_q: float, upper_q: float) -> tuple[list[float], list[float]]:
    lowers = []
    uppers = []
    for channel_samples in sample_lists:
        if not channel_samples:
            raise ValueError("No values were collected for quantile normalization.")
        merged = np.concatenate(channel_samples)
        lowers.append(float(np.quantile(merged, lower_q)))
        uppers.append(float(np.quantile(merged, upper_q)))
    return lowers, uppers


def _selected_uncompressed_kspace_for_stats(
    modality: str,
    volume_path: str,
    slice_index: int,
    feature_config: dict,
) -> np.ndarray:
    if modality == "t2":
        filled_kspace = _compute_t2_filled_kspace(volume_path, slice_index, feature_config["kernel_size"])
    elif modality == "dwi":
        filled_kspace, _ = _compute_dwi_filled_kspace(volume_path, slice_index, feature_config["kernel_size"])
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    selected = _coil_first_kspace(filled_kspace)
    if str(feature_config.get("average_selection", "all")).lower() == "middle":
        selected = _select_middle_average(selected)
    return selected


def build_train_quantile_stats(manifest: Any, feature_config: dict) -> dict[str, Any]:
    lower_q = float(feature_config.get("quantile_lower", 0.05))
    upper_q = float(feature_config.get("quantile_upper", 0.95))
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("Expected quantile_lower and quantile_upper to satisfy 0 <= lower < upper <= 1.")

    stats_config = dict(feature_config)
    stats_config["normalization"] = "none"
    stats_config.pop("normalization_stats", None)

    channel_count = get_input_channels(stats_config)
    max_values = int(feature_config.get("quantile_max_samples_per_channel", 1_000_000))
    values_per_sample = int(feature_config.get("quantile_values_per_sample", 4096))
    modalities = _configured_modalities(stats_config)

    if max_values <= 0 or values_per_sample <= 0:
        raise ValueError("Quantile sample limits must be positive integers.")

    if str(stats_config["representation"]).lower() == "complex":
        magnitude_samples: list[list[np.ndarray]] = [[]]
        use_uncompressed_stats = stats_config.get("coil_compression_channels") is not None
        for _, row in manifest.iterrows():
            if use_uncompressed_stats:
                for modality in modalities:
                    selected = _selected_uncompressed_kspace_for_stats(
                        modality,
                        row[f"{modality}_path"],
                        int(row["slice_index"]),
                        stats_config,
                    )
                    magnitude_samples[0].append(_sample_flat_values(np.abs(selected), values_per_sample))
            else:
                modality_paths = {modality: row[f"{modality}_path"] for modality in modalities}
                data = extract_input_channels(modality_paths, int(row["slice_index"]), stats_config)
                magnitude_samples[0].append(_sample_flat_values(np.abs(data), values_per_sample))
            _trim_sample_lists(magnitude_samples, max_values)

        lower, upper = _finalize_quantiles(magnitude_samples, lower_q, upper_q)
        return {
            "mode": "train_quantile",
            "source": "uncompressed_kspace" if use_uncompressed_stats else "represented_features",
            "target": "complex_magnitude",
            "scope": "global",
            "quantile_lower": lower_q,
            "quantile_upper": upper_q,
            "magnitude_lower": lower[0],
            "magnitude_upper": upper[0],
            "channels": channel_count,
        }

    real_samples: list[list[np.ndarray]] = [[] for _ in range(channel_count)]

    for _, row in manifest.iterrows():
        modality_paths = {modality: row[f"{modality}_path"] for modality in modalities}
        data = extract_input_channels(modality_paths, int(row["slice_index"]), stats_config)
        for channel_index in range(channel_count):
            real_samples[channel_index].append(_sample_flat_values(data[channel_index], values_per_sample))

        _trim_sample_lists(real_samples, max_values)

    lower, upper = _finalize_quantiles(real_samples, lower_q, upper_q)
    return {
        "mode": "train_quantile",
        "source": "represented_features",
        "target": "real_channels",
        "scope": "per_channel",
        "quantile_lower": lower_q,
        "quantile_upper": upper_q,
        "channels": channel_count,
        "lower": lower,
        "upper": upper,
    }


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
        if feature_config.get("coil_compression_channels") is not None:
            channel = _represent_compressed_kspace_channels(filled_kspace, feature_config)
            if cache_path is not None:
                np.save(cache_path, channel)
            return channel
        average_indices = feature_config.get("t2_average_indices")
        selected_kspace = _select_averages(filled_kspace, average_indices)
        reference_kspace = selected_kspace.mean(axis=0)
        sensitivity_maps = _estimate_sensitivity_maps(reference_kspace)
    elif modality == "dwi":
        filled_kspace, sensitivity_maps = _compute_dwi_filled_kspace(volume_path, slice_index, feature_config["kernel_size"])
        if feature_config.get("coil_compression_channels") is not None:
            channel = _represent_compressed_kspace_channels(filled_kspace, feature_config)
            if cache_path is not None:
                np.save(cache_path, channel)
            return channel
        average_indices = feature_config.get("dwi_average_indices")
        selected_kspace = _select_averages(filled_kspace, average_indices)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    physical_volume = _build_physical_single_channel(selected_kspace, sensitivity_maps, feature_config["domain"])
    channel = _represent_volume(physical_volume, feature_config)
    if cache_path is not None:
        np.save(cache_path, channel)
    return channel


def extract_input_channels(modality_paths: dict[str, str], slice_index: int, feature_config: dict) -> np.ndarray:
    channels = [
        _extract_channel(modality, modality_paths[modality], slice_index, feature_config)
        for modality in _configured_modalities(feature_config)
    ]
    if str(feature_config["representation"]).lower() == "complex":
        if feature_config.get("coil_compression_channels") is not None:
            data = np.concatenate(channels, axis=0).astype(np.complex64)
        else:
            data = np.stack(channels, axis=0).astype(np.complex64)
        return _normalize_input_channels(data, feature_config)
    data = np.concatenate(channels, axis=0).astype(np.float32)
    return _normalize_input_channels(data, feature_config)


def extract_paired_input_channels(t2_path: str, dwi_path: str, slice_index: int, feature_config: dict) -> np.ndarray:
    return extract_input_channels({"t2": t2_path, "dwi": dwi_path}, slice_index, feature_config)


def extract_paired_kspace_channels(t2_path: str, dwi_path: str, slice_index: int, feature_config: dict) -> np.ndarray:
    return extract_paired_input_channels(t2_path, dwi_path, slice_index, feature_config)
