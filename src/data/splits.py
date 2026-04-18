from __future__ import annotations

from collections.abc import Sequence
import random


def volume_wise_split(
    volume_ids: Sequence[str],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 7,
) -> dict[str, list[str]]:
    """Create deterministic disjoint train/val/test splits at the volume level."""
    unique_ids = list(dict.fromkeys(volume_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    total = len(unique_ids)
    train_count = int(total * train_fraction)
    val_count = int(total * val_fraction)

    train = unique_ids[:train_count]
    val = unique_ids[train_count : train_count + val_count]
    test = unique_ids[train_count + val_count :]

    return {"train": train, "val": val, "test": test}
