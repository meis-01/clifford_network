from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from clifford_network.utils.io import load_yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "experiment",
        "task": "classification",
        "output_dir": "results/runs",
    },
    "model": {
        "name": "complex_mlp",
        "hidden_size": 128,
        "depth": 4,
        "activation": "split_tanh",
    },
    "initialization": {
        "method": "structured_preserve",
    },
    "training": {
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 1.0e-3,
        "optimizer": "adam",
        "device": "auto",
        "seed": 0,
        "save_checkpoint": False,
    },
    "monitoring": {
        "enabled": True,
        "saturation_threshold": 0.95,
        "vanishing_threshold": 1.0e-5,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    raw = load_yaml(path)
    config = deep_merge(DEFAULT_CONFIG, raw)
    config["_config_path"] = str(Path(path).resolve())
    return config


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def expand_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    init_config = config.get("initialization", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    methods = as_list(init_config.get("methods", init_config.get("method", "structured_preserve")))
    depths = as_list(model_config.get("depths", model_config.get("depth", 4)))
    seeds = as_list(training_config.get("seeds", training_config.get("seed", 0)))

    expanded: list[dict[str, Any]] = []
    for method in methods:
        for depth in depths:
            for seed in seeds:
                run_config = deepcopy(config)
                run_config["initialization"]["method"] = method
                run_config["model"]["depth"] = int(depth)
                run_config["training"]["seed"] = int(seed)
                run_config["initialization"].pop("methods", None)
                run_config["model"].pop("depths", None)
                run_config["training"].pop("seeds", None)
                expanded.append(run_config)
    return expanded
