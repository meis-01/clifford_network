import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from complex_layers import ComplexMLP, get_activation_stats
from initialization import initialize_model
from data_module import create_data_loaders


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
            # Use a fixed input for monitoring
            x = torch.randn(100, model.layers[0].in_features, dtype=torch.cfloat, device=next(model.parameters()).device)

            _, pre_acts, post_acts = model(x)

            for layer_idx in self.monitor_layers:
                if layer_idx < len(post_acts):
                    activations = post_acts[layer_idx]
                    mag = torch.abs(activations).flatten()

                    # Store distribution statistics
                    self.activation_history[layer_idx].append({
                        "epoch": epoch,
                        "mean": float(mag.mean()),
                        "std": float(mag.std()),
                        "min": float(mag.min()),
                        "max": float(mag.max()),
                        "values": mag.cpu().numpy(),
                    })

                    # Track min/max over epochs
                    self.min_max_history[layer_idx]["min"].append(float(mag.min()))
                    self.min_max_history[layer_idx]["max"].append(float(mag.max()))

    def get_layer_stats(self, layer_idx: int) -> Dict[str, Any]:
        """Get statistics for a specific layer."""
        return {
            "history": self.activation_history[layer_idx],
            "min_over_epochs": min(self.min_max_history[layer_idx]["min"]),
            "max_over_epochs": max(self.min_max_history[layer_idx]["max"]),
        }


class TrainingExperiment:
    """Manages training experiments with activation monitoring."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        monitor_layers: List[int] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor_layers = monitor_layers or [len(model.layers) - 1]  # Monitor last layer by default
        self.monitor = ActivationMonitor(self.monitor_layers)

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs, _, _ = self.model(inputs)
            loss = self.criterion(outputs.real, targets.real) + self.criterion(outputs.imag, targets.imag)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """Validate and return average loss."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _, _ = self.model(inputs)
                loss = self.criterion(outputs.real, targets.real) + self.criterion(outputs.imag, targets.imag)
                total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def train(self, epochs: int, monitor_every: int = 1) -> Dict[str, Any]:
        """Train the model and monitor activations."""
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % monitor_every == 0:
                self.monitor(self.model, epoch)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "activation_monitor": self.monitor,
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
) -> Dict[str, Any]:
    """Run a complete training experiment."""

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_samples=5000,
        val_samples=1000,
        batch_size=batch_size,
        in_features=in_features,
        out_features=hidden_size,  # Match model output
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )

    if dataset_name in {"mnist", "cifar10"}:
        in_features = train_loader.dataset.in_features
    elif in_features is None:
        in_features = train_loader.dataset.in_features

    # Create model
    model = ComplexMLP(
        in_features=in_features,
        hidden_size=hidden_size,
        n_layers=n_layers,
        activation=activation,
    )

    # Initialize weights
    initialize_model(model, method=init_method)

    # Setup experiment
    monitor_layers = monitor_layers or [n_layers - 1]  # Monitor last layer
    experiment = TrainingExperiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        monitor_layers=monitor_layers,
        device=device,
    )

    # Train
    results = experiment.train(epochs=epochs)

    return {
        "activation": activation,
        "init_method": init_method,
        "model": model,
        "results": results,
    }