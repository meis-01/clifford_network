"""Interactive plot generation for experiment outputs.

The functions in this module turn aggregated histories and layer statistics into
small HoloViews/Bokeh HTML artifacts that can be linked from reports.
"""

from __future__ import annotations

from pathlib import Path
import re

import holoviews as hv
import pandas as pd

hv.extension("bokeh")


def _ensure_output_dir(path: str | Path) -> Path:
    """Create an output directory if needed and return it as a `Path`."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _add_run_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover initialization, depth, and seed from deterministic run directories."""
    if "run_dir" not in frame.columns:
        return frame

    pattern = re.compile(r"init=(?P<initialization>[^/\\]+)__depth=(?P<depth>\d+)__seed=(?P<seed>\d+)")
    parsed = frame["run_dir"].astype(str).str.extract(pattern)
    enriched = frame.copy()
    for column in ["initialization", "depth", "seed"]:
        if column not in enriched.columns:
            enriched[column] = parsed[column]
        else:
            enriched[column] = enriched[column].fillna(parsed[column])
    enriched["depth"] = pd.to_numeric(enriched["depth"], errors="coerce")
    enriched["seed"] = pd.to_numeric(enriched["seed"], errors="coerce")
    return enriched


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
    """Write initial post-tanh layer-flow diagnostics grouped by depth."""
    if layer_stats.empty:
        return None

    output = _ensure_output_dir(output_dir) / "layer_metrics.html"
    metric_specs = [
        ("activation_mean", "Post-tanh activation mean", "Mean activation magnitude", False),
        ("activation_saturated_fraction", "Post-tanh activation saturation", "Saturated fraction", False),
        ("activation_vanishing_fraction", "Post-tanh activation vanishing", "Vanishing fraction", False),
        ("gradient_mean", "Post-tanh gradient mean", "Mean gradient magnitude", True),
        ("gradient_vanishing_fraction", "Post-tanh gradient vanishing", "Vanishing fraction", False),
    ]
    metric_cols = [column for column, _title, _ylabel, _logy in metric_specs if column in layer_stats.columns]
    if not metric_cols:
        return None

    frame = _add_run_metadata(layer_stats)
    required = {"run_dir", "initialization", "depth", "epoch", "layer"}
    if not required.issubset(frame.columns):
        return None
    frame = frame.dropna(subset=["initialization", "depth", "epoch", "layer"]).copy()
    if frame.empty:
        return None

    if "split" in frame.columns:
        frame = frame[frame["split"] == "train"].copy()
    frame = frame[frame["layer"].astype(str).str.contains("activation", regex=False)].copy()
    if frame.empty:
        return None

    first_epoch = frame.groupby("run_dir")["epoch"].transform("min")
    frame = frame[frame["epoch"] == first_epoch].copy()
    frame = frame.sort_values(["run_dir", "epoch"], kind="stable")
    frame["layer_position"] = frame.groupby(["run_dir", "epoch"]).cumcount() + 1

    summary = (
        frame.groupby(["depth", "initialization", "layer_position"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["depth", "initialization", "layer_position"])
    )
    plots = []
    for depth, depth_frame in summary.groupby("depth"):
        for metric, title, ylabel, use_log_scale in metric_specs:
            if metric not in depth_frame.columns:
                continue
            metric_frame = depth_frame[["initialization", "layer_position", metric]].dropna().copy()
            if use_log_scale:
                metric_frame = metric_frame[metric_frame[metric] > 0]
            if metric_frame.empty:
                continue

            curves = []
            for method, method_frame in metric_frame.groupby("initialization"):
                curves.append(hv.Curve(method_frame, kdims=["layer_position"], vdims=[metric], label=str(method)))
            plots.append(
                hv.Overlay(curves).opts(
                    title=f"{title}, depth={int(depth)}",
                    xlabel="Post-tanh layer",
                    ylabel=ylabel,
                    width=900,
                    height=320,
                    legend_position="right",
                    logy=use_log_scale,
                )
            )

    if not plots:
        return None

    hv.save(hv.Layout(plots).cols(1).opts(shared_axes=False), output, backend="bokeh")
    return output
