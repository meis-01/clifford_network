"""Load and summarize experiment result CSVs.

This module scans result directories for per-run histories and layer statistics,
then combines them into DataFrames suitable for plotting and reporting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_histories(results_dir: str | Path) -> pd.DataFrame:
    """Concatenate all `history.csv` files under a results directory."""
    frames = []
    for path in Path(results_dir).glob("**/history.csv"):
        frame = pd.read_csv(path)
        frame["run_dir"] = str(path.parent)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_layer_stats(results_dir: str | Path) -> pd.DataFrame:
    """Concatenate all `layer_stats.csv` files under a results directory."""
    frames = []
    for path in Path(results_dir).glob("**/layer_stats.csv"):
        frame = pd.read_csv(path)
        frame["run_dir"] = str(path.parent)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_final_metrics(history: pd.DataFrame) -> pd.DataFrame:
    """Return the last epoch row for each initialization, depth, seed, and run."""
    if history.empty:
        return history
    group_cols = ["initialization", "depth", "seed", "run_dir"]
    final_rows = history.sort_values("epoch").groupby(group_cols, as_index=False).tail(1)
    return final_rows.reset_index(drop=True)
