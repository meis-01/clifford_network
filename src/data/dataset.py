from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SliceSample:
    image: np.ndarray
    label: int
    volume_id: str
    slice_index: int
    modality: str
    metadata: dict[str, Any]


class FastMRISliceDataset(Sequence[dict[str, Any]]):
    def __init__(self, samples: list[SliceSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "image": sample.image,
            "label": sample.label,
            "volume_id": sample.volume_id,
            "slice_index": sample.slice_index,
            "modality": sample.modality,
            "metadata": sample.metadata,
        }
