"""HoloViews/Bokeh and paper-ready PDF plots for alfa activation analysis.

Plots:
1. Real-part histogram
2. Imaginary-part histogram
3. Magnitude histogram
4. Complex-plane distribution (Re(z), Im(z))
5. Paper-ready PDF comparing all alfa values at each fixed layer

The original HoloViews/Bokeh HTML reports are preserved.
"""

from __future__ import annotations

from pathlib import Path

import holoviews as hv
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

hv.extension("bokeh")


# ============================================================
# LOAD ACTIVATION
# ============================================================

def _load_activation(path: Path) -> dict[str, object]:
    """Load raw activations and metadata from an alfa-analysis NPZ file."""

    with np.load(path, allow_pickle=False) as payload:
        if "layer" in payload:
            layer = int(payload["layer"])
        else:
            layer = _layer_from_filename(path)

        return {
            "activations": payload["activations"],
            "dataset": str(payload["dataset"]),
            "dataset_label": str(payload["dataset_label"]),
            "alfa": float(payload["alfa"]),
            "depth": int(payload["depth"]),
            "layer": layer,
            "hidden_size": int(payload["hidden_size"]),
            "num_samples": int(payload["num_samples"]),
            "input_size": int(payload["input_size"]),
            "activation": str(payload["activation"]),
        }


# ============================================================
# LAYER FROM FILENAME
# ============================================================

def _layer_from_filename(path: Path) -> int:
    """Extract layer number from filenames such as layer=0250 or layer=1000."""

    name = path.stem

    marker = "layer="
    if marker in name:
        value = name.split(marker, 1)[1].split("_", 1)[0]
        return int(value)

    return int(name.rsplit("layer", 1)[-1])


# ============================================================
# HOLOVIEWS HISTOGRAM
# ============================================================

def _histogram(
    values: np.ndarray,
    *,
    bins: int,
    title: str,
    xlabel: str,
) -> hv.Histogram:
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


# ============================================================
# COMPLEX PLANE
# ============================================================

def _complex_plane_plot(
    activations: np.ndarray,
    *,
    title: str,
    max_points: int = 20000,
) -> hv.Scatter:
    """Plot complex activation values in the Re-Im plane."""

    z = np.asarray(activations).reshape(-1)

    if z.size > max_points:
        rng = np.random.default_rng(0)

        indices = rng.choice(
            z.size,
            size=max_points,
            replace=False,
        )

        z = z[indices]

    points = np.column_stack(
        (
            z.real.astype(np.float64),
            z.imag.astype(np.float64),
        )
    )

    scatter = hv.Scatter(
        points,
        kdims=["Real"],
        vdims=["Imaginary"],
    )

    return scatter.opts(
        title=title,
        xlabel="Real part",
        ylabel="Imaginary part",
        width=540,
        height=540,
        size=4,
        alpha=0.25,
        tools=[
            "hover",
            "box_zoom",
            "wheel_zoom",
            "reset",
            "pan",
        ],
        axiswise=True,
        framewise=True,
        padding=0.05,
    )


# ============================================================
# COMPONENT PLOTS
# ============================================================

def _component_plots(
    record: dict[str, object],
    *,
    bins: int,
    max_points: int,
) -> list[hv.Element]:
    """Build histograms and complex-plane plot for one activation layer."""

    activations = np.asarray(record["activations"])

    dataset_label = str(record["dataset_label"])
    alfa = float(record["alfa"])
    layer = int(record["layer"])
    depth = int(record["depth"])
    hidden_size = int(record["hidden_size"])

    title_prefix = (
        f"{dataset_label}, "
        f"α={alfa:g}, "
        f"layer={layer}, "
        f"depth={depth}, "
        f"hidden={hidden_size}"
    )

    real_hist = _histogram(
        activations.real,
        bins=bins,
        title=f"{title_prefix}: real",
        xlabel="Real activation",
    )

    imag_hist = _histogram(
        activations.imag,
        bins=bins,
        title=f"{title_prefix}: imaginary",
        xlabel="Imaginary activation",
    )

    magnitude_hist = _histogram(
        np.abs(activations),
        bins=bins,
        title=f"{title_prefix}: magnitude",
        xlabel="Activation magnitude",
    )

    complex_plane = _complex_plane_plot(
        activations,
        title=f"{title_prefix}: complex plane",
        max_points=max_points,
    )

    return [
        real_hist,
        imag_hist,
        magnitude_hist,
        complex_plane,
    ]


# ============================================================
# ORIGINAL HTML REPORT
# ============================================================

def save_activation_histograms(
    *,
    activation_paths: list[Path],
    output_dir: Path,
    bins: int,
    max_points: int = 20000,
) -> Path:
    """Save combined HoloViews/Bokeh activation distribution report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        _load_activation(Path(path))
        for path in activation_paths
    ]

    records.sort(
        key=lambda item: (
            str(item["dataset"]),
            float(item["alfa"]),
            int(item["layer"]),
        )
    )

    plots: list[hv.Element] = []

    for record in records:
        plots.extend(
            _component_plots(
                record,
                bins=bins,
                max_points=max_points,
            )
        )

    layout = hv.Layout(plots).cols(2).opts(
        shared_axes=False,
    )

    output_path = output_dir / "activation_distributions.html"

    hv.save(
        layout,
        output_path,
        backend="bokeh",
    )

    return output_path


# ============================================================
# ORIGINAL COMPLEX-PLANE HTML REPORT
# ============================================================

def save_complex_plane_plots(
    *,
    activation_paths: list[Path],
    output_dir: Path,
    max_points: int = 20000,
) -> Path:
    """Save separate HoloViews/Bokeh report containing only Re-Im plots."""

    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        _load_activation(Path(path))
        for path in activation_paths
    ]

    records.sort(
        key=lambda item: (
            str(item["dataset"]),
            float(item["alfa"]),
            int(item["layer"]),
        )
    )

    plots: list[hv.Scatter] = []

    for record in records:
        activations = np.asarray(record["activations"])

        dataset_label = str(record["dataset_label"])
        alfa = float(record["alfa"])
        layer = int(record["layer"])
        depth = int(record["depth"])

        title = (
            f"{dataset_label}: "
            f"α={alfa:g}, "
            f"layer={layer}/{depth}"
        )

        plots.append(
            _complex_plane_plot(
                activations,
                title=title,
                max_points=max_points,
            )
        )

    layout = hv.Layout(plots).cols(3).opts(
        shared_axes=False,
    )

    output_path = output_dir / "complex_activation_planes.html"

    hv.save(
        layout,
        output_path,
        backend="bokeh",
    )

    return output_path


# ============================================================
# PAPER PDF HELPERS
# ============================================================

def _density_histogram(
    values: np.ndarray,
    *,
    bins: int,
    x_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized histogram density."""

    values = np.asarray(values).reshape(-1)

    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.array([]), np.array([])

    counts, edges = np.histogram(
        values,
        bins=bins,
        range=x_range,
        density=True,
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, counts


def _format_alfa(alfa: float) -> str:
    """Format alfa values cleanly for legends."""

    if alfa == 0:
        return r"$\alpha=0$"

    if abs(alfa) < 1e-3 or abs(alfa) >= 1e3:
        return rf"$\alpha={alfa:.2e}$"

    return rf"$\alpha={alfa:g}$"


# ============================================================
# PAPER-READY PDF
# ============================================================

# ============================================================
# PAPER-READY PDF
# ============================================================

def save_fixed_layer_alpha_comparison_pdf(
    *,
    activation_paths: list[Path],
    output_path: Path,
    bins: int = 120,
) -> Path:
    """Create separate PDF files for each alfa value.

    For each alfa:
        one PDF file

    For each layer:
        one PDF page

    Each page contains:
        Left  = real activation histogram
        Right = imaginary activation histogram

    The histogram calculation is the same as the original
    HoloViews/Bokeh HTML histogram.
    """

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = [
        _load_activation(Path(path))
        for path in activation_paths
    ]

    if not records:
        raise ValueError(
            "No activation files were provided."
        )

    # --------------------------------------------------------
    # Group by alfa
    # --------------------------------------------------------

    alfa_groups: dict[
        float,
        list[dict[str, object]],
    ] = {}

    for record in records:
        alfa = float(
            record["alfa"]
        )

        alfa_groups.setdefault(
            alfa,
            [],
        ).append(record)

    # --------------------------------------------------------
    # Create one PDF for each alfa
    # --------------------------------------------------------

    generated_pdfs = []

    for alfa in sorted(alfa_groups):

        alfa_records = alfa_groups[alfa]

        # ----------------------------------------------------
        # Sort layers
        # ----------------------------------------------------

        alfa_records.sort(
            key=lambda record: (
                str(record["dataset"]),
                int(record["layer"]),
            )
        )

        # ----------------------------------------------------
        # File name
        # ----------------------------------------------------

        if abs(alfa) < 1e-3 or abs(alfa) >= 1e3:
            alfa_string = f"{alfa:.2e}"
        else:
            alfa_string = f"{alfa:g}"

        alfa_pdf_path = (
            output_dir
            / f"alpha_{alfa_string}.pdf"
        )

        # ----------------------------------------------------
        # Create PDF
        # ----------------------------------------------------

        with PdfPages(
            alfa_pdf_path
        ) as pdf:

            for record in alfa_records:

                activations = np.asarray(
                    record["activations"]
                ).reshape(-1)

                layer = int(
                    record["layer"]
                )

                # --------------------------------------------
                # Real and imaginary values
                # --------------------------------------------

                real = activations.real.astype(
                    np.float64
                )

                imag = activations.imag.astype(
                    np.float64
                )

                real = real[
                    np.isfinite(real)
                ]

                imag = imag[
                    np.isfinite(imag)
                ]

                # --------------------------------------------
                # Figure
                # --------------------------------------------

                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(11.0, 4.5),
                )

                ax_real = axes[0]
                ax_imag = axes[1]

                # ============================================
                # REAL ACTIVATION
                # ============================================

                if real.size > 0:

                    counts_real, edges_real = np.histogram(
                        real,
                        bins=bins,
                    )

                    ax_real.stairs(
                        counts_real,
                        edges_real,
                        linewidth=1.5,
                    )

                ax_real.set_xlabel(
                    "Real activation",
                    fontsize=11,
                )

                ax_real.set_ylabel(
                    "Count",
                    fontsize=11,
                )

                # No subplot title

                ax_real.grid(
                    True,
                    alpha=0.25,
                )

                # ============================================
                # IMAGINARY ACTIVATION
                # ============================================

                if imag.size > 0:

                    counts_imag, edges_imag = np.histogram(
                        imag,
                        bins=bins,
                    )

                    ax_imag.stairs(
                        counts_imag,
                        edges_imag,
                        linewidth=1.5,
                    )

                ax_imag.set_xlabel(
                    "Imaginary activation",
                    fontsize=11,
                )

                ax_imag.set_ylabel(
                    "Count",
                    fontsize=11,
                )

                # No subplot title

                ax_imag.grid(
                    True,
                    alpha=0.25,
                )

                # --------------------------------------------
                # ONLY ALFA AT TOP
                # --------------------------------------------

                fig.suptitle(
                    rf"$\alpha={alfa:g}$",
                    fontsize=14,
                    y=0.98,
                )

                # --------------------------------------------
                # Layout
                # --------------------------------------------

                fig.tight_layout(
                    rect=[
                        0,
                        0,
                        1,
                        0.94,
                    ]
                )

                # --------------------------------------------
                # Add page
                # --------------------------------------------

                pdf.savefig(
                    fig,
                    bbox_inches="tight",
                )

                plt.close(fig)

        generated_pdfs.append(
            alfa_pdf_path
        )

    # --------------------------------------------------------
    # Return output directory
    # --------------------------------------------------------

    return output_dir