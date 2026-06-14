from __future__ import annotations

from pathlib import Path

from clifford_network.experiments.config import load_config
from clifford_network.experiments.runner import run_experiment


def test_single_smoke_run(tmp_path: Path) -> None:
    config = load_config("configs/smoke/synthetic_classification.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)
    config["initialization"]["method"] = "structured_preserve"
    config["initialization"].pop("methods", None)
    config["model"]["depth"] = 2
    config["model"].pop("depths", None)
    config["training"]["seed"] = 0
    config["training"].pop("seeds", None)
    config["training"]["epochs"] = 1

    summary = run_experiment(config)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "layer_stats.csv").exists()
    assert summary["best_validation_loss"] > 0
