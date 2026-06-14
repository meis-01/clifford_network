"""
dataset.py  —  patched to support data.max_samples for quick debug runs.

Only change vs original: ComplexCoilImageDataset.__init__ now reads
config["data"].get("max_samples") and slices self.paths when set.

Everything else (normalisation, loading, DataLoader creation) is identical.

Usage — add to your YAML config:
    data:
      max_samples: 50    # cap every split at 50 files; null = use all
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _split_feature_paths(config: dict[str, Any], split_name: str) -> list[Path]:
    data_config = config["data"]
    features_root = Path(data_config["features_root"])
    split_dir = str(data_config["split_dirs"][split_name])
    glob_pattern = str(data_config["glob"])
    paths = sorted((features_root / split_dir).glob(glob_pattern))
    if paths:
        return paths

    manifest_csv = data_config.get("manifest_csv")
    if bool(data_config.get("use_manifest", True)) and manifest_csv is not None and Path(manifest_csv).exists():
        manifest = pd.read_csv(manifest_csv)
        if "data_split" not in manifest or "feature_path" not in manifest:
            raise ValueError("Manifest must contain data_split and feature_path columns.")
        split_manifest = manifest[manifest["data_split"].astype(str) == split_dir]
        paths = [Path(v) for v in split_manifest["feature_path"].tolist()]
        return [p for p in paths if p.exists()]

    return []


def _load_complex_npy(path: Path) -> torch.Tensor:
    array = np.load(path)
    if not np.iscomplexobj(array):
        raise ValueError(f"Expected a native complex .npy array at {path}, got dtype={array.dtype}")
    tensor = torch.from_numpy(array.astype(np.complex64, copy=False))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected shape (H, W) or (C, H, W) at {path}, got {tuple(tensor.shape)}")
    return tensor


def _normalize(image: torch.Tensor, mode: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "none":
        return image, torch.tensor(1.0, dtype=torch.float32)
    if mode == "sample_rms":
        scale = torch.sqrt(torch.mean(torch.abs(image) ** 2)).clamp_min(eps)
        return image / scale.to(image.real.dtype), scale.detach().to(torch.float32)
    if mode == "sample_max":
        scale = torch.amax(torch.abs(image)).clamp_min(eps)
        return image / scale.to(image.real.dtype), scale.detach().to(torch.float32)
    raise ValueError(f"Unknown normalization mode: {mode}")


class ComplexCoilImageDataset(Dataset):
    def __init__(self, config: dict[str, Any], split_name: str):
        self.config = config
        self.split_name = split_name
        all_paths = _split_feature_paths(config, split_name)

        # ── subset support ────────────────────────────────────────────────
        max_samples = config["data"].get("max_samples")
        if max_samples is not None:
            max_samples = int(max_samples)
            all_paths = all_paths[:max_samples]
        # ─────────────────────────────────────────────────────────────────

        self.paths = all_paths
        self.image_size = tuple(config["data"]["image_size"])
        self.normalization = str(config["data"]["normalization"])
        self.eps = float(config["data"]["eps"])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        image = _load_complex_npy(path)
        if tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"{path} has image size {tuple(image.shape[-2:])}, expected {self.image_size}"
            )
        image, scale = _normalize(image, self.normalization, self.eps)
        return {
            "image": image,
            "target": image.clone(),
            "scale": scale,
            "path": str(path),
        }


def build_dataloaders(
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, DataLoader]:
    training_config = config["training"]
    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        dataset = ComplexCoilImageDataset(config, split_name)
        max_samples = config["data"].get("max_samples")
        if logger is not None:
            subset_note = f" (capped at {max_samples})" if max_samples else ""
            logger.info(
                "Discovered %d files%s for split=%s",
                len(dataset),
                subset_note,
                split_name,
            )
            if len(dataset) > 0:
                logger.info("First %s sample: %s", split_name, dataset.paths[0])
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=int(training_config["batch_size"]),
            shuffle=split_name == "train" and len(dataset) > 0,
            num_workers=int(training_config["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )
        if logger is not None:
            logger.info(
                "Built %s dataloader: batches=%d batch_size=%d",
                split_name,
                len(loaders[split_name]),
                int(training_config["batch_size"]),
            )
    return loaders