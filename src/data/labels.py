from __future__ import annotations

from pathlib import Path

import pandas as pd


def map_binary_label(raw_label: int) -> int:
    return 0 if int(raw_label) == 1 else 1


def load_slice_labels(
    csv_path: Path,
    volume_column: str = "volume_id",
    slice_column: str = "slice_index",
    label_column: str = "label",
) -> dict[tuple[str, int], int]:
    frame = pd.read_csv(csv_path)
    expected = {volume_column, slice_column, label_column}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required label columns in {csv_path}: {sorted(missing)}")

    lookup: dict[tuple[str, int], int] = {}
    for row in frame[[volume_column, slice_column, label_column]].itertuples(index=False):
        lookup[(str(row[0]), int(row[1]))] = map_binary_label(int(row[2]))
    return lookup
