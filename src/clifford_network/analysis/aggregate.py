from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_histories(results_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(results_dir).glob("**/history.csv"):
        frame = pd.read_csv(path)
        frame["run_dir"] = str(path.parent)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_layer_stats(results_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(results_dir).glob("**/layer_stats.csv"):
        frame = pd.read_csv(path)
        frame["run_dir"] = str(path.parent)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_final_metrics(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    group_cols = ["initialization", "depth", "seed", "run_dir"]
    final_rows = history.sort_values("epoch").groupby(group_cols, as_index=False).tail(1)
    return final_rows.reset_index(drop=True)
