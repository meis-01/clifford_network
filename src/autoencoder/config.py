from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "features_root": "F:/fastmri_prostate/t2_samples/features",
        "manifest_csv": "F:/fastmri_prostate/t2_samples/manifest.csv",
        "split_dirs": {
            "train": "training",
            "val": "validation",
            "test": "test",
        },
        "glob": "*.npy",
        "use_manifest": True,
        "image_size": [320, 320],
        "normalization": "sample_rms",
        "eps": 1.0e-8,
    },
    "model": {
        "in_channels": 1,
        "channels": [16, 32, 64, 128],
        "latent_channels": 128,
        "activation": "modrelu",
        "use_bias": True,
    },
    "training": {
        "batch_size": 4,
        "num_workers": 0,
        "epochs": 50,
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-5,
        "grad_clip_norm": 1.0,
        "patience": 10,
        "device": "auto",
        "seed": 7,
        "output_dir": "runs/complex_t2_autoencoder",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_optional_path(value: Any, base_dir: Path) -> Path | None:
    if value in (None, "", False):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_base_dir(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return Path.cwd().resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    config = _deep_merge(DEFAULT_CONFIG, user_config)
    base_dir = _resolve_base_dir(config_path)

    data_config = config["data"]
    training_config = config["training"]

    data_config["features_root"] = _resolve_optional_path(data_config.get("features_root"), base_dir)
    data_config["manifest_csv"] = _resolve_optional_path(data_config.get("manifest_csv"), base_dir)
    data_config["image_size"] = tuple(int(value) for value in data_config["image_size"])
    data_config["normalization"] = str(data_config["normalization"]).lower()
    data_config["glob"] = str(data_config["glob"])

    training_config["output_dir"] = _resolve_optional_path(training_config["output_dir"], base_dir)

    model_config = config["model"]
    model_config["channels"] = [int(value) for value in model_config["channels"]]
    model_config["in_channels"] = int(model_config["in_channels"])
    model_config["latent_channels"] = int(model_config["latent_channels"])
    model_config["activation"] = str(model_config["activation"]).lower()

    return config
