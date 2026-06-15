"""Layer-wise activation, gradient, and weight monitoring.

The monitor registers lightweight hooks on complex layers and activations,
records magnitude statistics for the first training batch of an epoch, and
returns CSV-ready rows for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from clifford_network.activations.core import ModReLU, ModTanh, SplitTanh, ZReLU
from clifford_network.models.layers import ComplexLinear


MONITORED_TYPES = (ComplexLinear, SplitTanh, ModTanh, ModReLU, ZReLU)


def _tensor_stats(values: torch.Tensor, prefix: str, saturation_threshold: float, vanishing_threshold: float) -> dict[str, float]:
    """Summarize magnitude mean, spread, extremes, saturation, and vanishing rates."""
    detached = values.detach()
    magnitude = detached.abs() if detached.is_complex() else detached.abs()
    return {
        f"{prefix}_mean": float(magnitude.mean().item()),
        f"{prefix}_std": float(magnitude.std(unbiased=False).item()),
        f"{prefix}_max": float(magnitude.max().item()),
        f"{prefix}_saturated_fraction": float((magnitude > saturation_threshold).float().mean().item()),
        f"{prefix}_vanishing_fraction": float((magnitude < vanishing_threshold).float().mean().item()),
    }


@dataclass
class MonitorConfig:
    """Configuration thresholds and enable flag for layer monitoring."""

    enabled: bool = True
    saturation_threshold: float = 0.95
    vanishing_threshold: float = 1.0e-5


class LayerMonitor:
    """Local hook-based activation, gradient, and weight monitoring."""

    def __init__(self, model: nn.Module, config: MonitorConfig) -> None:
        """Register forward hooks for supported modules when monitoring is enabled."""
        self.model = model
        self.config = config
        self._active = False
        self._outputs: dict[str, torch.Tensor] = {}
        self._activation_stats: dict[str, dict[str, float]] = {}
        self._handles: list[Any] = []

        if self.config.enabled:
            for name, module in model.named_modules():
                if isinstance(module, MONITORED_TYPES):
                    self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        """Create a forward hook that captures activation stats for one layer."""
        def hook(_module, _inputs, output):
            """Capture output magnitude stats when this monitor is active."""
            if not self._active or not torch.is_tensor(output):
                return
            self._activation_stats[name] = _tensor_stats(
                output,
                "activation",
                self.config.saturation_threshold,
                self.config.vanishing_threshold,
            )
            if output.requires_grad:
                output.retain_grad()
                self._outputs[name] = output

        return hook

    def begin_step(self) -> None:
        """Enable capture for the next monitored forward/backward step."""
        self._active = self.config.enabled
        self._outputs = {}
        self._activation_stats = {}

    def collect(self, *, epoch: int, split: str) -> list[dict[str, Any]]:
        """Collect captured layer statistics and reset active monitoring state."""
        records: list[dict[str, Any]] = []
        if not self.config.enabled:
            return records

        modules = dict(self.model.named_modules())
        for name, activation_stats in self._activation_stats.items():
            record: dict[str, Any] = {"epoch": epoch, "split": split, "layer": name}
            record.update(activation_stats)

            output = self._outputs.get(name)
            if output is not None and output.grad is not None:
                record.update(
                    _tensor_stats(
                        output.grad,
                        "gradient",
                        self.config.saturation_threshold,
                        self.config.vanishing_threshold,
                    )
                )

            module = modules.get(name)
            weight = getattr(module, "weight", None)
            if torch.is_tensor(weight):
                record.update(
                    _tensor_stats(
                        weight,
                        "weight",
                        self.config.saturation_threshold,
                        self.config.vanishing_threshold,
                    )
                )
            records.append(record)

        self._active = False
        return records

    def close(self) -> None:
        """Remove all registered hooks from monitored modules."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
