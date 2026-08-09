"""Single-experiment training orchestration.

This module wires together config values, datasets, model construction,
initialization, optimization, monitoring, and result persistence for one run.
"""

from __future__ import annotations

import logging
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

LOGGER = logging.getLogger(__name__)


def _run_dir(config: dict[str, Any]) -> Path:
    """Build the deterministic output directory for a resolved run config."""
    experiment = config["experiment"]
    output_root = Path(experiment.get("output_dir", "results/runs"))
    name = experiment["name"]
    method = config["initialization"]["method"]
    depth = config["model"]["depth"]
    seed = config["training"]["seed"]
    return output_root / name / f"init={method}__depth={depth}__seed={seed}"


def _count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy model state to CPU so the best epoch can be restored later."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


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


def _format_metrics(metrics: dict[str, float]) -> str:
    """Format metric dictionaries for compact progress logging."""
    return ", ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run training, validation, testing, and artifact writing for one config."""
    seed = int(config["training"]["seed"])
    LOGGER.info(
        "Preparing run: experiment=%s dataset=%s task=%s init=%s depth=%s seed=%s",
        config["experiment"]["name"],
        config.get("dataset", {}).get("name", "unknown"),
        config["experiment"]["task"],
        config["initialization"]["method"],
        config["model"]["depth"],
        seed,
    )
    set_seed(seed)
    LOGGER.info("Random seed set: %s", seed)
    device = resolve_device(config["training"].get("device"))
    LOGGER.info("Resolved device: %s", device)
    task = config["experiment"]["task"].lower()

    LOGGER.info("Building dataloaders.")
    dataloaders, data_spec = build_dataloaders(config, seed=seed)
    LOGGER.info(
        "Dataset ready: input_size=%s num_classes=%s train_items=%s validation_items=%s test_items=%s",
        data_spec.input_size,
        data_spec.num_classes,
        len(dataloaders["train"].dataset),
        len(dataloaders["validation"].dataset),
        len(dataloaders["test"].dataset),
    )
    LOGGER.info(
        "Building model: name=%s hidden_size=%s depth=%s activation=%s",
        config["model"]["name"],
        config["model"]["hidden_size"],
        config["model"]["depth"],
        config["model"].get("activation", "split_tanh"),
    )
    model = build_model(
        config,
        input_size=data_spec.input_size,
        num_classes=data_spec.num_classes,
        task=task,
        depth=int(config["model"]["depth"]),
    ).to(device)
    LOGGER.info("Model ready: trainable_parameters=%s", _count_parameters(model))

    initialization = config["initialization"]
    method = initialization["method"]
    init_kwargs = {}
    if method.lower() == "structured_preserve":
        if "alpha" in initialization:
            init_kwargs["alpha"] = float(initialization["alpha"])
        if "gain" in initialization:
            init_kwargs["gain"] = float(initialization["gain"])
    LOGGER.info("Initializing model: method=%s kwargs=%s", method, init_kwargs or "{}")
    initialize_model(model, method, **init_kwargs)

    loss_fn = build_loss(task)
    optimizer = _build_optimizer(config, model)
    LOGGER.info(
        "Training setup: loss=%s optimizer=%s learning_rate=%s batch_size=%s epochs=%s",
        getattr(loss_fn, "__name__", loss_fn.__class__.__name__),
        config["training"].get("optimizer", "adam"),
        config["training"].get("learning_rate", 1.0e-3),
        config["training"].get("batch_size", 128),
        config["training"].get("epochs", 10),
    )
    monitor = LayerMonitor(
        model,
        MonitorConfig(
            enabled=bool(config["monitoring"].get("enabled", True)),
            saturation_threshold=float(config["monitoring"].get("saturation_threshold", 0.95)),
            vanishing_threshold=float(config["monitoring"].get("vanishing_threshold", 1.0e-5)),
        ),
    )
    LOGGER.info(
        "Monitoring setup: enabled=%s saturation_threshold=%s vanishing_threshold=%s",
        bool(config["monitoring"].get("enabled", True)),
        float(config["monitoring"].get("saturation_threshold", 0.95)),
        float(config["monitoring"].get("vanishing_threshold", 1.0e-5)),
    )

    run_dir = _run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config_resolved.yaml")
    LOGGER.info("Run output directory: %s", run_dir)

    history: list[dict[str, Any]] = []
    layer_records: list[dict[str, Any]] = []
    epochs = int(config["training"].get("epochs", 10))
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    try:
        for epoch in range(1, epochs + 1):
            LOGGER.info("Epoch %s/%s started.", epoch, epochs)
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
            validation_loss = validation_result.metrics["loss"]
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state_dict = _clone_state_dict(model)
            LOGGER.info(
                "Epoch %s/%s complete: train[%s] validation[%s] best_validation_loss=%.6g best_epoch=%s layer_records=%s",
                epoch,
                epochs,
                _format_metrics(train_result.metrics),
                _format_metrics(validation_result.metrics),
                best_validation_loss,
                best_epoch,
                len(train_result.layer_records),
            )
    finally:
        monitor.close()
        LOGGER.info("Monitoring hooks closed.")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        LOGGER.info("Restored best validation model from epoch %s before test evaluation.", best_epoch)

    LOGGER.info("Running final test evaluation.")
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
    LOGGER.info("Saved history rows=%s and layer_stat rows=%s.", len(history), len(layer_records))
    summary = {
        "experiment": config["experiment"]["name"],
        "task": task,
        "dataset": config["dataset"]["name"],
        "initialization": config["initialization"]["method"],
        "depth": config["model"]["depth"],
        "seed": seed,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "final_validation_loss": history[-1]["validation_loss"],
        "test_metrics": test_result.metrics,
        "run_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")
    LOGGER.info(
        "Run summary: best_validation_loss=%.6g final_validation_loss=%.6g test[%s]",
        best_validation_loss,
        history[-1]["validation_loss"],
        _format_metrics(test_result.metrics),
    )

    if bool(config["training"].get("save_checkpoint", False)):
        torch.save({"model_state_dict": model.state_dict(), "config": config}, run_dir / "model.pt")
        LOGGER.info("Checkpoint saved: %s", run_dir / "model.pt")

    return summary
