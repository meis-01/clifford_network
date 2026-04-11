from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


@dataclass
class EpochResult:
    loss: float
    auc: float
    accuracy: float
    predictions: pd.DataFrame


def build_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(max(negatives / positives, 1.0), dtype=torch.float32)


def _compute_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        use_autocast = amp_enabled and not torch.is_complex(images)
        with torch.amp.autocast(device_type=autocast_device, enabled=use_autocast):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_training:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        total_loss += float(loss.item()) * len(labels_np)
        labels_list.append(labels_np)
        probs_list.append(probabilities)

        for sample_id, label_value, probability in zip(batch["sample_id"], labels_np, probabilities, strict=False):
            records.append(
                {
                    "sample_id": sample_id,
                    "label": float(label_value),
                    "probability": float(probability),
                }
            )

    labels_all = np.concatenate(labels_list) if labels_list else np.array([], dtype=np.float32)
    probs_all = np.concatenate(probs_list) if probs_list else np.array([], dtype=np.float32)
    average_loss = total_loss / max(len(labels_all), 1)
    predictions = (probs_all >= 0.5).astype(np.int64)
    accuracy = float((predictions == labels_all).mean()) if len(labels_all) else float("nan")

    return EpochResult(
        loss=average_loss,
        auc=_compute_auc(labels_all, probs_all),
        accuracy=accuracy,
        predictions=pd.DataFrame.from_records(records),
    )