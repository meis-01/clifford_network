"""Batch-level metric helpers.

This module converts task outputs and losses into plain float metrics that can
be averaged per epoch and written to CSV histories.
"""

from __future__ import annotations

import torch


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy from class logits and integer targets."""
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def batch_metrics(task: str, prediction: torch.Tensor, target: torch.Tensor, loss: torch.Tensor) -> dict[str, float]:
    """Build a task-specific metric dictionary for one batch."""
    if task == "classification":
        return {
            "loss": float(loss.detach().item()),
            "accuracy": classification_accuracy(prediction.detach(), target.detach()),
        }
    if task == "autoencoder":
        return {
            "loss": float(loss.detach().item()),
            "mse": float(loss.detach().item()),
        }
    raise ValueError(f"Unsupported task '{task}'.")
