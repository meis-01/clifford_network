from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.base import torch, torch_available


@dataclass
class Trainer:
    model: Any
    optimizer: Any
    loss_fn: Any
    device: str = "cpu"

    def train_step(self, batch: dict[str, Any]) -> float:
        if not torch_available():
            raise RuntimeError("PyTorch is required for training")

        self.model.train()
        inputs = batch["image"].to(self.device)
        targets = batch["label"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())
