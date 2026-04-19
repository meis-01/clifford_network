from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.autoencoder.losses import complex_mse_loss, complex_nmse


@dataclass(frozen=True)
class EpochResult:
    loss: float
    nmse: float
    predictions: pd.DataFrame


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_nmse = 0.0
    total_samples = 0
    rows: list[dict[str, Any]] = []

    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.complex64, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.complex64, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            prediction = model(image)
            loss = complex_mse_loss(prediction, target)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            nmse_value = complex_nmse(prediction, target)
            batch_size = int(image.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_nmse += float(nmse_value.detach().cpu()) * batch_size
            total_samples += batch_size
            errors = torch.mean(torch.abs(prediction - target) ** 2, dim=(-3, -2, -1)).detach().cpu()
            for path, sample_error in zip(batch["path"], errors.tolist(), strict=False):
                rows.append({"path": str(path), "mse": float(sample_error)})

    if total_samples == 0:
        return EpochResult(loss=float("nan"), nmse=float("nan"), predictions=pd.DataFrame(rows))
    return EpochResult(
        loss=total_loss / total_samples,
        nmse=total_nmse / total_samples,
        predictions=pd.DataFrame(rows),
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "best_val_loss": best_val_loss,
        },
        path,
    )
