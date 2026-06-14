from __future__ import annotations

from dataclasses import dataclass
import logging
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
    logger: logging.Logger | None = None,
    phase: str = "epoch",
    log_interval: int = 0,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_nmse = 0.0
    total_samples = 0
    rows: list[dict[str, Any]] = []
    total_batches = len(loader)
    if logger is not None:
        logger.info("Starting %s: batches=%d training=%s", phase, total_batches, is_train)

    for batch_index, batch in enumerate(loader, start=1):
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

        if logger is not None and log_interval > 0 and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches):
            running_loss = total_loss / max(total_samples, 1)
            running_nmse = total_nmse / max(total_samples, 1)
            logger.info(
                "%s batch %d/%d: running_loss=%.6f running_nmse=%.6f samples=%d",
                phase,
                batch_index,
                total_batches,
                running_loss,
                running_nmse,
                total_samples,
            )

    if total_samples == 0:
        if logger is not None:
            logger.warning("Finished %s with no samples.", phase)
        return EpochResult(loss=float("nan"), nmse=float("nan"), predictions=pd.DataFrame(rows))
    if logger is not None:
        logger.info(
            "Finished %s: loss=%.6f nmse=%.6f samples=%d",
            phase,
            total_loss / total_samples,
            total_nmse / total_samples,
            total_samples,
        )
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
