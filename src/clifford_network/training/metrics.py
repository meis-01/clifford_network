from __future__ import annotations

import torch


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def batch_metrics(task: str, prediction: torch.Tensor, target: torch.Tensor, loss: torch.Tensor) -> dict[str, float]:
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
