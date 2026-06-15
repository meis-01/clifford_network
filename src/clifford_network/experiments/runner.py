"""Single-experiment training orchestration.

This module wires together config values, datasets, model construction,
initialization, optimization, monitoring, and result persistence for one run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from clifford_network.dataset import build_dataloaders
from clifford_network.initialization import initialize_model
from clifford_network.models import build_model
from clifford_network.training.losses import build_loss
from clifford_network.training.monitoring import LayerMonitor, MonitorConfig
from clifford_network.training.runner import run_epoch
from clifford_network.utils.device import resolve_device
from clifford_network.utils.io import save_json, save_records, save_yaml
from clifford_network.utils.seed import set_seed


def _run_dir(config: dict[str, Any]) -> Path:
    """Build the deterministic output directory for a resolved run config."""
    experiment = config["experiment"]
    output_root = Path(experiment.get("output_dir", "results/runs"))
    name = experiment["name"]
    method = config["initialization"]["method"]
    depth = config["model"]["depth"]
    seed = config["training"]["seed"]
    return output_root / name / f"init={method}__depth={depth}__seed={seed}"


def _build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """Construct the configured optimizer for a model."""
    training = config["training"]
    name = training.get("optimizer", "adam").lower()
    learning_rate = float(training.get("learning_rate", 1.0e-3))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run training, validation, testing, and artifact writing for one config."""
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = resolve_device(config["training"].get("device"))
    task = config["experiment"]["task"].lower()

    dataloaders, data_spec = build_dataloaders(config, seed=seed)
    model = build_model(
        config,
        input_size=data_spec.input_size,
        num_classes=data_spec.num_classes,
        task=task,
        depth=int(config["model"]["depth"]),
    ).to(device)

    init_kwargs = {}
    if "alpha" in config["initialization"]:
        init_kwargs["alpha"] = float(config["initialization"]["alpha"])
    initialize_model(model, config["initialization"]["method"], **init_kwargs)

    loss_fn = build_loss(task)
    optimizer = _build_optimizer(config, model)
    monitor = LayerMonitor(
        model,
        MonitorConfig(
            enabled=bool(config["monitoring"].get("enabled", True)),
            saturation_threshold=float(config["monitoring"].get("saturation_threshold", 0.95)),
            vanishing_threshold=float(config["monitoring"].get("vanishing_threshold", 1.0e-5)),
        ),
    )

    run_dir = _run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config_resolved.yaml")

    history: list[dict[str, Any]] = []
    layer_records: list[dict[str, Any]] = []
    epochs = int(config["training"].get("epochs", 10))
    best_validation_loss = float("inf")

    try:
        for epoch in range(1, epochs + 1):
            train_result = run_epoch(
                model=model,
                dataloader=dataloaders["train"],
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                task=task,
                epoch=epoch,
                split="train",
                monitor=monitor,
            )
            validation_result = run_epoch(
                model=model,
                dataloader=dataloaders["validation"],
                loss_fn=loss_fn,
                optimizer=None,
                device=device,
                task=task,
                epoch=epoch,
                split="validation",
                monitor=None,
            )
            row = {
                "epoch": epoch,
                "initialization": config["initialization"]["method"],
                "depth": config["model"]["depth"],
                "seed": seed,
            }
            row.update({f"train_{key}": value for key, value in train_result.metrics.items()})
            row.update({f"validation_{key}": value for key, value in validation_result.metrics.items()})
            history.append(row)
            layer_records.extend(train_result.layer_records)
            best_validation_loss = min(best_validation_loss, validation_result.metrics["loss"])
    finally:
        monitor.close()

    test_result = run_epoch(
        model=model,
        dataloader=dataloaders["test"],
        loss_fn=loss_fn,
        optimizer=None,
        device=device,
        task=task,
        epoch=epochs,
        split="test",
        monitor=None,
    )

    save_records(history, run_dir / "history.csv")
    save_records(layer_records, run_dir / "layer_stats.csv")
    summary = {
        "experiment": config["experiment"]["name"],
        "task": task,
        "dataset": config["dataset"]["name"],
        "initialization": config["initialization"]["method"],
        "depth": config["model"]["depth"],
        "seed": seed,
        "best_validation_loss": best_validation_loss,
        "final_validation_loss": history[-1]["validation_loss"],
        "test_metrics": test_result.metrics,
        "run_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")

    if bool(config["training"].get("save_checkpoint", False)):
        torch.save({"model_state_dict": model.state_dict(), "config": config}, run_dir / "model.pt")

    return summary
