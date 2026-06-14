from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from clifford_network.training.metrics import batch_metrics
from clifford_network.training.monitoring import LayerMonitor


@dataclass
class EpochResult:
    metrics: dict[str, float]
    layer_records: list[dict]


def _move_batch(batch, device: torch.device):
    inputs, targets = batch
    return inputs.to(device), targets.to(device)


def run_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    task: str,
    epoch: int,
    split: str,
    monitor: LayerMonitor | None,
) -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)
    totals: dict[str, float] = {}
    n_batches = 0
    layer_records: list[dict] = []

    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = _move_batch(batch, device)
        should_monitor = is_training and batch_idx == 0 and monitor is not None

        if should_monitor:
            monitor.begin_step()

        with torch.set_grad_enabled(is_training):
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if should_monitor:
                    layer_records.extend(monitor.collect(epoch=epoch, split=split))
                optimizer.step()

        metrics = batch_metrics(task, predictions, targets, loss)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1

    return EpochResult(metrics={key: value / max(1, n_batches) for key, value in totals.items()}, layer_records=layer_records)
