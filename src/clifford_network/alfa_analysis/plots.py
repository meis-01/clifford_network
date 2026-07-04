"""HoloViews/Bokeh plots for the alfa activation analysis."""

from __future__ import annotations

from pathlib import Path

import holoviews as hv
import numpy as np

hv.extension("bokeh")


def _load_activation(path: Path) -> dict[str, object]:
    """Load raw activations and metadata from an alfa-analysis NPZ file."""
    with np.load(path, allow_pickle=False) as payload:
        return {
            "activations": payload["activations"],
            "dataset": str(payload["dataset"]),
            "dataset_label": str(payload["dataset_label"]),
            "alfa": float(payload["alfa"]),
            "depth": int(payload["depth"]),
            "hidden_size": int(payload["hidden_size"]),
            "num_samples": int(payload["num_samples"]),
            "input_size": int(payload["input_size"]),
            "activation": str(payload["activation"]),
        }


def _histogram(values: np.ndarray, *, bins: int, title: str, xlabel: str) -> hv.Histogram:
    """Create a HoloViews histogram with Bokeh-friendly styling."""
    counts, edges = np.histogram(values.reshape(-1), bins=bins)
    return hv.Histogram((edges, counts)).opts(
        title=title,
        xlabel=xlabel,
        ylabel="Count",
        width=360,
        height=260,
        tools=["hover"],
        line_color="#2f5f8f",
        fill_color="#2f5f8f",
        fill_alpha=0.82,
    )


def _component_plots(record: dict[str, object], *, bins: int) -> list[hv.Histogram]:
    """Build real, imaginary, and magnitude histograms for one run."""
    activations = np.asarray(record["activations"])
    dataset_label = str(record["dataset_label"])
    alfa = float(record["alfa"])
    depth = int(record["depth"])
    hidden_size = int(record["hidden_size"])
    title_prefix = f"{dataset_label}, alfa={alfa:g}, layer={depth}, hidden={hidden_size}"
    return [
        _histogram(activations.real, bins=bins, title=f"{title_prefix}: real", xlabel="Real activation"),
        _histogram(activations.imag, bins=bins, title=f"{title_prefix}: imaginary", xlabel="Imag activation"),
        _histogram(np.abs(activations), bins=bins, title=f"{title_prefix}: magnitude", xlabel="Activation magnitude"),
    ]


def save_activation_histograms(
    *,
    activation_paths: list[Path],
    output_dir: Path,
    bins: int,
) -> Path:
    """Save a combined HoloViews/Bokeh HTML histogram report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [_load_activation(Path(path)) for path in activation_paths]
    records.sort(key=lambda item: (str(item["dataset"]), float(item["alfa"])))

    plots: list[hv.Histogram] = []
    for record in records:
        plots.extend(_component_plots(record, bins=bins))

    layout = hv.Layout(plots).cols(3).opts(shared_axes=False)
    output_path = output_dir / "activation_histograms.html"
    hv.save(layout, output_path, backend="bokeh")
    return output_path
