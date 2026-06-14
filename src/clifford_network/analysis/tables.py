from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_final_metric_table(history: pd.DataFrame, output_dir: str | Path) -> Path | None:
    if history.empty:
        return None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metric_cols = [column for column in history.columns if column.startswith("validation_") or column.startswith("train_")]
    table = (
        history.sort_values("epoch")
        .groupby(["initialization", "depth", "seed"], as_index=False)
        .tail(1)[["initialization", "depth", "seed", *metric_cols]]
        .sort_values(["depth", "initialization", "seed"])
    )
    output = output_path / "final_metrics.csv"
    table.to_csv(output, index=False)
    return output
