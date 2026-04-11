from __future__ import annotations

import random
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.classification.features import extract_paired_input_channels
from src.classification.manifest import split_manifest


def _apply_shared_augmentation(image: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[1])
    rotations = random.randint(0, 3)
    if rotations:
        image = torch.rot90(image, k=rotations, dims=[1, 2])
    return image


class PairedKspaceDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, config: dict[str, Any], split_name: str):
        self.manifest = split_manifest(manifest, config, split_name)
        self.feature_config = config["features"]
        self.split_name = split_name
        self.augment = split_name == "train"

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        image = extract_paired_input_channels(
            t2_path=row["t2_path"],
            dwi_path=row["dwi_path"],
            slice_index=int(row["slice_index"]),
            feature_config=self.feature_config,
        )
        tensor = torch.from_numpy(image)
        if self.augment:
            tensor = _apply_shared_augmentation(tensor)

        return {
            "image": tensor,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "patient_id": int(row["fastmri_pt_id"]),
            "slice": int(row["slice"]),
        }


def build_dataloaders(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, DataLoader]:
    training_config = config["training"]
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])

    loaders: dict[str, DataLoader] = {}
    for split_name in config["data"]["split_names"]:
        dataset = PairedKspaceDataset(manifest, config, split_name)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train" and len(dataset) > 0,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders