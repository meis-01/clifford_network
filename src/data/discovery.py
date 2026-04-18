from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VolumeRecord:
    path: Path
    modality: str


def _infer_modality(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    hint = f"{parent}/{name}"
    if "t2" in hint or "axt2" in hint:
        return "t2"
    if "diff" in hint or "dwi" in hint:
        return "dwi"
    return "unknown"


def discover_volume_files(root: str | Path) -> list[VolumeRecord]:
    """Recursively discover HDF5 volume files under a dataset root."""
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return []

    records = [
        VolumeRecord(path=path, modality=_infer_modality(path))
        for path in sorted(base.rglob("*.h5"))
        if path.is_file()
    ]
    return records
