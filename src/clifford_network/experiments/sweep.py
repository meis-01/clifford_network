"""Sweep execution across expanded experiment configurations.

The sweep runner executes each concrete config produced by the config expander
and writes a combined summary table for the experiment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clifford_network.experiments.config import expand_sweep
from clifford_network.experiments.runner import run_experiment


def run_sweep(config: dict) -> pd.DataFrame:
    """Run every expanded config and return the collected summary DataFrame."""
    summaries = [run_experiment(run_config) for run_config in expand_sweep(config)]
    frame = pd.DataFrame.from_records(summaries)
    if summaries:
        experiment_name = summaries[0]["experiment"]
        output_root = Path(config["experiment"].get("output_dir", "results/runs")) / experiment_name
        output_root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_root / "sweep_summary.csv", index=False)
    return frame
