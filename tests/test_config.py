from __future__ import annotations

from clifford_network.experiments.config import expand_sweep, load_config


def test_smoke_config_expands_initialization_depth_seed_grid() -> None:
    config = load_config("configs/smoke/synthetic_classification.yaml")
    runs = expand_sweep(config)
    assert len(runs) == 4
    assert {run["initialization"]["method"] for run in runs} == {"structured_preserve", "trabelsi"}
    assert {run["model"]["depth"] for run in runs} == {2, 4}
