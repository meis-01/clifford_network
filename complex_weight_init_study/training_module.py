import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict

from complex_layers import ComplexMLP, get_activation_stats, get_gradient_norms
from initialization import initialize_model
from data_module import create_data_loaders
from logger import ExperimentLogger


class ActivationMonitor:
    """Monitors activation statistics during training."""

    def __init__(self, monitor_layers: List[int]):
        self.monitor_layers = monitor_layers
        self.activation_history = defaultdict(list)
        self.min_max_history = defaultdict(lambda: {"min": [], "max": []})

    def __call__(self, model: nn.Module, epoch: int):
        """Monitor activations for a single epoch."""
        model.eval()
        with torch.no_grad():
            x = torch.randn(
                100,
                model.layers[0].in_features,
                dtype=torch.cfloat,
                device=next(model.parameters()).device,
            )
            _, pre_acts, post_acts = model(x)

            for layer_idx in self.monitor_layers:
                if layer_idx < len(post_acts):
                    activations = post_acts[layer_idx]
                    mag = torch.abs(activations).flatten()

                    self.activation_history[layer_idx].append({
                        "epoch": epoch,
                        "mean": float(mag.mean()),
                        "std": float(mag.std()),
                        "min": float(mag.min()),
                        "max": float(mag.max()),
                        "values": mag.cpu().numpy(),
                    })
                    self.min_max_history[layer_idx]["min"].append(float(mag.min()))
                    self.min_max_history[layer_idx]["max"].append(float(mag.max()))

    def get_layer_stats(self, layer_idx: int) -> Dict[str, Any]:
        history = self.activation_history[layer_idx]
        if not history:
            return {"history": [], "min_over_epochs": None, "max_over_epochs": None}
        return {
            "history": history,
            "min_over_epochs": min(self.min_max_history[layer_idx]["min"]),
            "max_over_epochs": max(self.min_max_history[layer_idx]["max"]),
        }

    def latest_metrics(self) -> Dict[str, float]:
        """Return a flat dict of the most recent stats for every monitored layer."""
        metrics = {}
        for layer_idx in self.monitor_layers:
            history = self.activation_history[layer_idx]
            if not history:
                continue
            last = history[-1]
            values = last["values"]
            prefix = f"layer{layer_idx}"
            metrics[f"{prefix}_mean"]     = last["mean"]
            metrics[f"{prefix}_std"]      = last["std"]
            metrics[f"{prefix}_min"]      = last["min"]
            metrics[f"{prefix}_max"]      = last["max"]
            metrics[f"{prefix}_vanish"]   = float(np.mean(values < 0.01))
            metrics[f"{prefix}_saturate"] = float(np.mean(values > 10.0))
        return metrics


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class TrainingExperiment:
    """Manages training experiments with activation monitoring."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        monitor_layers: List[int] = None,
        device: str = "cpu",
        early_stopping_patience: int = 0,  # 0 = disabled
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor_layers = monitor_layers or [len(model.layers) - 1]
        self.monitor = ActivationMonitor(self.monitor_layers)

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.grad_norm_means: List[float] = []
        self.grad_norm_maxes: List[float] = []

        self.early_stopper = (
            EarlyStopping(patience=early_stopping_patience)
            if early_stopping_patience > 0
            else None
        )

    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, _, _ = self.model(inputs)
            loss = (
                self.criterion(outputs.real, targets.real)
                + self.criterion(outputs.imag, targets.imag)
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """Validate and return average loss."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:           # ← was train_loader (bug fix)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _, _ = self.model(inputs)
                loss = (
                    self.criterion(outputs.real, targets.real)
                    + self.criterion(outputs.imag, targets.imag)
                )
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _collect_grad_metrics(self) -> Dict[str, float]:
        """Collect gradient norm stats after a backward pass."""
        norms = get_gradient_norms(self.model)
        if not norms:
            return {}
        mean_norm = float(np.mean(norms))
        max_norm  = float(np.max(norms))
        self.grad_norm_means.append(mean_norm)
        self.grad_norm_maxes.append(max_norm)
        return {"grad_norm_mean": mean_norm, "grad_norm_max": max_norm}

    def train(
        self,
        epochs: int,
        monitor_every: int = 1,
        output_dir: str = "results",
        experiment_name: str = "exp",
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the model, log metrics, and optionally checkpoint the best weights."""

        logger = ExperimentLogger(output_dir, experiment_name)
        best_val_loss = float("inf")
        stopped_early = False

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss   = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # --- build metrics dict for this epoch ---
            metrics: Dict[str, Any] = {
                "train_loss": train_loss,
                "val_loss":   val_loss,
            }

            # gradient norms (computed every epoch — cheap after backward)
            metrics.update(self._collect_grad_metrics())

            # activation stats (computed every monitor_every epochs)
            if epoch % monitor_every == 0:
                self.monitor(self.model, epoch)
                metrics.update(self.monitor.latest_metrics())

            # log everything
            logger.log(epoch + 1, metrics)

            # optional checkpoint
            ckpt_dir = checkpoint_dir or output_dir
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(ckpt_dir, f"{experiment_name}_best.pt"),
                )

            self.scheduler.step(val_loss)

            # early stopping check
            if self.early_stopper and self.early_stopper.step(val_loss, epoch + 1):
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val loss {self.early_stopper.best_loss:.6f} "
                    f"at epoch {self.early_stopper.best_epoch})"
                )
                stopped_early = True
                break

        logger.save_summary({
            "activation":    self.model.activation_name,
            "stopped_early": stopped_early,
            "best_val_loss": best_val_loss,
        })

        return {
            "train_losses":    self.train_losses,
            "val_losses":      self.val_losses,
            "grad_norm_means": self.grad_norm_means,
            "grad_norm_maxes": self.grad_norm_maxes,
            "activation_monitor": self.monitor,
            "logger": logger,
            "stopped_early": stopped_early,
        }


def run_training_experiment(
    activation: str,
    init_method: str,
    in_features: Optional[int],
    hidden_size: int,
    n_layers: int,
    epochs: int = 50,
    batch_size: int = 128,
    device: str = "cpu",
    monitor_layers: Optional[List[int]] = None,
    dataset_name: str = "synthetic",
    dataset_root: str = "data",
    output_dir: str = "results",
    early_stopping_patience: int = 0,
) -> Dict[str, Any]:
    """Run a complete training experiment."""

    train_loader, val_loader = create_data_loaders(
        train_samples=5000,
        val_samples=1000,
        batch_size=batch_size,
        in_features=in_features,
        out_features=hidden_size,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )

    if dataset_name in {"mnist", "cifar10"} or in_features is None:
        in_features = train_loader.dataset.in_features

    model = ComplexMLP(
        in_features=in_features,
        hidden_size=hidden_size,
        n_layers=n_layers,
        activation=activation,
    )
    initialize_model(model, method=init_method)

    monitor_layers = monitor_layers or [n_layers - 1]
    experiment_name = f"{activation}_{init_method}"

    experiment = TrainingExperiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        monitor_layers=monitor_layers,
        device=device,
        early_stopping_patience=early_stopping_patience,
    )

    # output_dir and experiment_name now flow through to the logger and checkpointer
    results = experiment.train(
        epochs=epochs,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    return {
        "activation":  activation,
        "init_method": init_method,
        "model":       model,
        "results":     results,
    }