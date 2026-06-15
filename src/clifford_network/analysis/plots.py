"""Interactive plot generation for experiment outputs.

The functions in this module turn aggregated histories and layer statistics into
small HoloViews/Bokeh HTML artifacts that can be linked from reports.
"""

from __future__ import annotations

from pathlib import Path

import holoviews as hv
import pandas as pd

hv.extension("bokeh")


def _ensure_output_dir(path: str | Path) -> Path:
    """Create an output directory if needed and return it as a `Path`."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_training_curves(history: pd.DataFrame, output_dir: str | Path) -> Path | None:
    """Write validation-loss curves grouped by initialization, depth, and seed."""
    if history.empty:
        return None

    output = _ensure_output_dir(output_dir) / "training_curves.html"
    curves = []
    for (method, depth, seed), frame in history.groupby(["initialization", "depth", "seed"]):
        label = f"{method}, depth={depth}, seed={seed}"
        curves.append(hv.Curve(frame, kdims=["epoch"], vdims=["validation_loss"], label=label))
    plot = hv.Overlay(curves).opts(
        title="Validation loss by initialization and depth",
        xlabel="Epoch",
        ylabel="Validation loss",
        width=900,
        height=420,
        legend_position="right",
    )
    hv.save(plot, output, backend="bokeh")
    return output


def save_accuracy_curves(history: pd.DataFrame, output_dir: str | Path) -> Path | None:
    """Write validation-accuracy curves when classification metrics are present."""
    if history.empty or "validation_accuracy" not in history.columns:
        return None

    output = _ensure_output_dir(output_dir) / "accuracy_curves.html"
    curves = []
    for (method, depth, seed), frame in history.groupby(["initialization", "depth", "seed"]):
        label = f"{method}, depth={depth}, seed={seed}"
        curves.append(hv.Curve(frame, kdims=["epoch"], vdims=["validation_accuracy"], label=label))
    plot = hv.Overlay(curves).opts(
        title="Validation accuracy by initialization and depth",
        xlabel="Epoch",
        ylabel="Validation accuracy",
        width=900,
        height=420,
        legend_position="right",
    )
    hv.save(plot, output, backend="bokeh")
    return output


def save_layer_metric_curves(layer_stats: pd.DataFrame, output_dir: str | Path) -> Path | None:
    """Write per-layer activation-magnitude curves when layer stats are available."""
    if layer_stats.empty:
        return None

    output = _ensure_output_dir(output_dir) / "layer_metrics.html"
    value_dim = "activation_mean"
    if value_dim not in layer_stats.columns:
        return None

    curves = []
    for layer, frame in layer_stats.groupby("layer"):
        curves.append(hv.Curve(frame.sort_values("epoch"), kdims=["epoch"], vdims=[value_dim], label=layer))
    plot = hv.Overlay(curves).opts(
        title="Layer activation magnitude over training",
        xlabel="Epoch",
        ylabel="Mean activation magnitude",
        width=900,
        height=480,
        legend_position="right",
    )
    hv.save(plot, output, backend="bokeh")
    return output
