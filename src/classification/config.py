from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "paired_manifest_csv": None,
        "t2_labels_csv": "data/labels/t2_slice_level_labels.csv",
        "dwi_labels_csv": "data/labels/dwi_slice_level_labels.csv",
        "t2_root": "data",
        "dwi_root": "data",
        "patient_id_column": "fastmri_pt_id",
        "slice_column": "slice",
        "split_column": "data_split",
        "label_column": "PIRADS",
        "folder_column": "folder",
        "file_column": "fastmri_rawfile",
        "positive_threshold": 2,
        "allow_label_disagreement": True,
        "drop_missing": True,
        "split_names": {
            "train": "training",
            "val": "validation",
            "test": "test",
        },
    },
    "features": {
        "domain": "kspace",
        "representation": "real",
        "output_size": [224, 224],
        "kernel_size": [5, 5],
        "coil_combination": "sense",
        "average_reduction": "mean",
        "log_scale": True,
        "normalization": "zscore",
        "cache_dir": None,
        "t2_average_indices": None,
        "dwi_average_indices": None,
    },
    "model": {
        "channels": [32, 64, 128, 256],
        "dropout": 0.3,
    },
    "training": {
        "batch_size": 8,
        "num_workers": 0,
        "epochs": 20,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "patience": 5,
        "monitor": "val_auc",
        "device": "auto",
        "amp": True,
        "seed": 7,
        "output_dir": "runs/kspace_joint_classifier",
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
    repo_root = config_path.parent.parent
    if (repo_root / "pyproject.toml").exists() and (repo_root / "src").exists():
        return repo_root
    return Path.cwd().resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    config = _deep_merge(DEFAULT_CONFIG, user_config)
    base_dir = _resolve_base_dir(config_path)

    data_config = config["data"]
    features_config = config["features"]
    training_config = config["training"]

    for key in ("paired_manifest_csv", "t2_labels_csv", "dwi_labels_csv", "t2_root", "dwi_root"):
        data_config[key] = _resolve_optional_path(data_config.get(key), base_dir)

    features_config["cache_dir"] = _resolve_optional_path(features_config.get("cache_dir"), base_dir)
    training_config["output_dir"] = _resolve_optional_path(training_config["output_dir"], base_dir)
    features_config["domain"] = str(features_config["domain"]).lower()
    features_config["representation"] = str(features_config["representation"]).lower()
    features_config["coil_combination"] = str(features_config["coil_combination"]).lower()
    features_config["average_reduction"] = str(features_config["average_reduction"]).lower()
    features_config["normalization"] = str(features_config["normalization"]).lower()
    features_config["output_size"] = tuple(int(value) for value in features_config["output_size"])
    features_config["kernel_size"] = tuple(int(value) for value in features_config["kernel_size"])

    return config