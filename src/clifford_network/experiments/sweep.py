"""Sweep execution across expanded experiment configurations.

The sweep runner executes each concrete config produced by the config expander
and writes a combined summary table for the experiment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from clifford_network.experiments.config import expand_sweep
from clifford_network.experiments.runner import run_experiment

LOGGER = logging.getLogger(__name__)


def run_sweep(config: dict) -> pd.DataFrame:
    """Run every expanded config and return the collected summary DataFrame."""
    run_configs = expand_sweep(config)
    if not run_configs:
        LOGGER.warning("Sweep expansion produced no runs.")
        return pd.DataFrame()

    methods = sorted({run_config["initialization"]["method"] for run_config in run_configs})
    depths = sorted({int(run_config["model"]["depth"]) for run_config in run_configs})
    seeds = sorted({int(run_config["training"]["seed"]) for run_config in run_configs})
    LOGGER.info(
        "Sweep plan: experiment=%s dataset=%s task=%s methods=%s depths=%s seeds=%s total_runs=%s",
        config["experiment"]["name"],
        config.get("dataset", {}).get("name", "unknown"),
        config["experiment"]["task"],
        methods,
        depths,
        seeds,
        len(run_configs),
    )

    summaries = []
    for run_index, run_config in enumerate(run_configs, start=1):
        LOGGER.info(
            "Sweep run %s/%s: init=%s depth=%s seed=%s",
            run_index,
            len(run_configs),
            run_config["initialization"]["method"],
            run_config["model"]["depth"],
            run_config["training"]["seed"],
        )
        summaries.append(run_experiment(run_config))

    frame = pd.DataFrame.from_records(summaries)
    if summaries:
        experiment_name = summaries[0]["experiment"]
        output_root = Path(config["experiment"].get("output_dir", "results/runs")) / experiment_name
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "sweep_summary.csv"
        frame.to_csv(summary_path, index=False)
        LOGGER.info("Sweep summary written: %s", summary_path)
    return frame
