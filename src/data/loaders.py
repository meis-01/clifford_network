from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _attrs_to_dict(handle: h5py.File) -> dict[str, Any]:
    return {key: value for key, value in handle.attrs.items()}


@dataclass
class T2Volume:
    volume_id: str
    kspace: np.ndarray
    calibration_data: np.ndarray | None
    reconstruction_rss: np.ndarray | None
    ismrmrd_header: bytes | str | None
    metadata: dict[str, Any]


@dataclass
class DwiVolume:
    volume_id: str
    kspace: np.ndarray
    calibration_data: np.ndarray | None
    coil_sensitivity_maps: np.ndarray | None
    ismrmrd_header: bytes | str | None
    metadata: dict[str, Any]


def load_t2_volume(path: Path) -> T2Volume:
    with h5py.File(path, "r") as handle:
        return T2Volume(
            volume_id=path.stem,
            kspace=handle["kspace"][:],
            calibration_data=handle["calibration_data"][:] if "calibration_data" in handle else None,
            reconstruction_rss=handle["reconstruction_rss"][:] if "reconstruction_rss" in handle else None,
            ismrmrd_header=handle["ismrmrd_header"][()] if "ismrmrd_header" in handle else None,
            metadata=_attrs_to_dict(handle),
        )


def load_dwi_volume(path: Path) -> DwiVolume:
    with h5py.File(path, "r") as handle:
        return DwiVolume(
            volume_id=path.stem,
            kspace=handle["kspace"][:],
            calibration_data=handle["calibration_data"][:] if "calibration_data" in handle else None,
            coil_sensitivity_maps=handle["coil_sens_maps"][:] if "coil_sens_maps" in handle else None,
            ismrmrd_header=handle["ismrmrd_header"][()] if "ismrmrd_header" in handle else None,
            metadata=_attrs_to_dict(handle),
        )
