from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VolumeRecord:
    volume_id: str
    path: Path
    modality: str


def detect_modality(path: Path) -> str:
    text = str(path).lower()
    if "diff" in text or "dwi" in text:
        return "dwi"
    if "t2" in text or "axt2" in text:
        return "t2"
    return "unknown"


def discover_volume_files(data_root: Path) -> list[VolumeRecord]:
    records: list[VolumeRecord] = []
    for path in sorted(data_root.rglob("*.h5")):
        records.append(
            VolumeRecord(
                volume_id=path.stem,
                path=path,
                modality=detect_modality(path),
            )
        )
    return records
