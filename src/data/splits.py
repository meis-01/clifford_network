from __future__ import annotations

import random
from collections.abc import Iterable


def volume_wise_split(
    volume_ids: Iterable[str],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 7,
) -> dict[str, list[str]]:
    ids = sorted(set(volume_ids))
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1")

    rng = random.Random(seed)
    rng.shuffle(ids)

    train_end = int(len(ids) * train_fraction)
    val_end = train_end + int(len(ids) * val_fraction)
    return {
        "train": ids[:train_end],
        "val": ids[train_end:val_end],
        "test": ids[val_end:],
    }
