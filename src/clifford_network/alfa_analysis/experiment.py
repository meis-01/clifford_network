
"""Run the structured-preserve alfa activation histogram experiment.

The experiment studies activation distributions in a very deep complex MLP
using structured-preserve initialization and different alfa values.

Input distributions:
    normal  -> complex normal CN(0, variance)
    uniform -> independent real/imaginary Uniform(low, high)

Examples:

    python -m clifford_network.alfa_analysis \
        --device cuda \
        --depth 10000 \
        --hidden-size 32 \
        --num-samples 3000 \
        --batch-size 3000 \
        --layers 1 10 20 2500 5000 7500 9000 10000 \
        --alfa-values 1e-5 1e-3 0.0085 0.1 1 10 \
        --input-distribution normal

or:

    python -m clifford_network.alfa_analysis \
        --device cuda \
        --depth 10000 \
        --hidden-size 32 \
        --num-samples 3000 \
        --batch-size 3000 \
        --layers 1 10 20 2500 5000 7500 9000 10000 \
        --alfa-values 1e-5 1e-3 0.0085 0.1 1 10 \
        --input-distribution uniform
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from clifford_network.initialization import initialize_model
from clifford_network.models import build_model
from clifford_network.utils.device import resolve_device
from clifford_network.utils.seed import set_seed
from clifford_network.alfa_analysis.plots import (
    save_activation_histograms,
    save_fixed_layer_alpha_comparison_pdf,
)


LOGGER = logging.getLogger(__name__)


# ============================================================
# Defaults
# ============================================================

ALFA_VALUES: tuple[float, ...] = (
    1.0e-5,
    1.0e-3,
    0.0085,
    1.0e-1,
    1.0,
    10.0,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results"
)


# ============================================================
# Dataset specification
# ============================================================

@dataclass(frozen=True)
class ClassificationSpec:
    """Dataset-shaped classifier dimensions for random complex inputs."""

    name: str
    label: str
    input_size: int
    num_classes: int = 10


DATASET_SPECS: dict[str, ClassificationSpec] = {
    "mnist": ClassificationSpec(
        name="mnist",
        label="MNIST",
        input_size=28 * 28,
    ),
}


# ============================================================
# Experiment configuration
# ============================================================

@dataclass(frozen=True)
class AlfaAnalysisConfig:
    """Configuration for the alfa activation analysis."""

    output_dir: Path = DEFAULT_OUTPUT_DIR

    dataset_names: tuple[str, ...] = ("mnist",)

    alfa_values: tuple[float, ...] = ALFA_VALUES

    num_samples: int = 3000

    depth: int = 1000

    hidden_size: int = 32

    activation: str = "split_tanh"

    batch_size: int = 3000

    bins: int = 120

    layers: tuple[int, ...] = ()

    max_plot_points: int = 20000

    seed: int = 0

    device: str = "auto"

    # --------------------------------------------------------
    # Input distribution
    # --------------------------------------------------------

    input_distribution: str = "normal"

    # For CN(0, variance)
    normal_variance: float = 1.0

    # For independent real/imaginary uniform distributions
    uniform_low: float = -1.0

    uniform_high: float = 1.0


# ============================================================
# Activation run
# ============================================================

@dataclass(frozen=True)
class ActivationRun:
    """Paths and metadata produced by one dataset/alfa run."""

    dataset: str

    alfa: float

    # Store ALL selected activation-layer paths.
    activation_paths: list[Path]

    rows: list[dict[str, float | int | str]]


# ============================================================
# Build model
# ============================================================

def _build_classifier(
    spec: ClassificationSpec,
    config: AlfaAnalysisConfig,
) -> nn.Module:
    """Build the repository's complex classifier."""

    model_config = {
        "experiment": {
            "task": "classification",
        },
        "model": {
            "name": "complex_mlp",
            "hidden_size": config.hidden_size,
            "activation": config.activation,
        },
    }

    return build_model(
        model_config,
        input_size=spec.input_size,
        num_classes=spec.num_classes,
        task="classification",
        depth=config.depth,
    )


# ============================================================
# Generate complex samples
# ============================================================

def _complex_samples(
    *,
    num_samples: int,
    input_size: int,
    distribution: str,
    seed: int,
    normal_variance: float = 1.0,
    uniform_low: float = -1.0,
    uniform_high: float = 1.0,
) -> torch.Tensor:
    """Generate complex-valued input samples.

    normal:
        z ~ CN(0, normal_variance)

        Re(z), Im(z) ~ N(0, normal_variance / 2)

        Therefore:

            E[|z|^2] = normal_variance

    uniform:
        Re(z), Im(z) independently follow:

            Uniform(uniform_low, uniform_high)
    """

    distribution = distribution.lower()

    if distribution not in {
        "normal",
        "uniform",
    }:
        raise ValueError(
            f"Unknown input distribution '{distribution}'. "
            "Choose 'normal' or 'uniform'."
        )

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    shape = (
        num_samples,
        input_size,
    )

    # --------------------------------------------------------
    # Complex normal CN(0, variance)
    # --------------------------------------------------------

    if distribution == "normal":

        if normal_variance <= 0:
            raise ValueError(
                "normal_variance must be positive."
            )

        # For CN(0, variance):
        #
        # Re(z), Im(z) ~ N(0, variance / 2)

        std = (
            normal_variance / 2.0
        ) ** 0.5

        real = (
            torch.randn(
                shape,
                dtype=torch.float32,
                generator=generator,
            )
            * std
        )

        imag = (
            torch.randn(
                shape,
                dtype=torch.float32,
                generator=generator,
            )
            * std
        )

    # --------------------------------------------------------
    # Complex uniform
    # --------------------------------------------------------

    else:

        if uniform_low >= uniform_high:
            raise ValueError(
                "uniform_low must be smaller than uniform_high."
            )

        real = torch.empty(
            shape,
            dtype=torch.float32,
        )

        imag = torch.empty(
            shape,
            dtype=torch.float32,
        )

        real.uniform_(
            uniform_low,
            uniform_high,
            generator=generator,
        )

        imag.uniform_(
            uniform_low,
            uniform_high,
            generator=generator,
        )

    return torch.complex(
        real,
        imag,
    )


# ============================================================
# Activation module name
# ============================================================

def _target_activation_name(
    layer: int,
) -> str:
    """Return the post-tanh activation module name."""

    if layer < 1:
        raise ValueError(
            "layer must be at least 1."
        )

    return (
        f"network.activation_{layer - 1:03d}"
    )


# ============================================================
# Capture activations
# ============================================================

def _capture_activations(
    model: nn.Module,
    samples: torch.Tensor,
    *,
    layers: tuple[int, ...],
    depth: int,
    batch_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Forward once and collect activations at selected layers."""

    if not layers:
        layers = (depth,)

    layers = tuple(
        dict.fromkeys(layers)
    )

    modules = dict(
        model.named_modules()
    )

    hooks = []

    captured: dict[
        int,
        list[torch.Tensor],
    ] = {
        layer: []
        for layer in layers
    }

    for layer in layers:

        target_name = (
            _target_activation_name(layer)
        )

        if target_name not in modules:

            available = ", ".join(
                name
                for name in modules
                if "activation" in name
            )

            raise ValueError(
                f"Could not find activation layer "
                f"'{target_name}'. "
                f"Available activations: {available}"
            )

        def make_hook(
            target_layer: int,
            target_name: str,
        ):
            def hook(
                _module: nn.Module,
                _inputs: tuple[
                    torch.Tensor,
                    ...,
                ],
                output: torch.Tensor,
            ) -> None:

                if not torch.is_tensor(
                    output
                ):
                    raise TypeError(
                        f"Expected tensor output from "
                        f"'{target_name}', got "
                        f"{type(output)!r}."
                    )

                captured[
                    target_layer
                ].append(
                    output.detach().cpu()
                )

            return hook

        hooks.append(
            modules[
                target_name
            ].register_forward_hook(
                make_hook(
                    layer,
                    target_name,
                )
            )
        )

    try:

        model.eval()

        effective_batch_size = (
            samples.shape[0]
            if batch_size <= 0
            else batch_size
        )

        with torch.inference_mode():

            for batch in samples.split(
                effective_batch_size
            ):

                _ = model(
                    batch.to(device)
                )

    finally:

        for handle in hooks:
            handle.remove()

    result: dict[
        int,
        torch.Tensor,
    ] = {}

    for layer in layers:

        if not captured[layer]:

            raise RuntimeError(
                f"No activations were captured "
                f"from layer {layer}."
            )

        result[layer] = torch.cat(
            captured[layer],
            dim=0,
        )

    return result


# ============================================================
# Complex-plane plots
# ============================================================

def _save_complex_plane_plots(
    activations_by_layer: dict[int, np.ndarray],
    *,
    alfa: float,
    dataset: str,
    output_dir: Path,
    max_points: int,
) -> list[Path]:
    """Save Re(z)-Im(z) scatter plots."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = []

    for layer, activations in (
        activations_by_layer.items()
    ):

        z = activations.reshape(-1)

        if z.size > max_points:

            rng = np.random.default_rng(
                0
            )

            idx = rng.choice(
                z.size,
                size=max_points,
                replace=False,
            )

            z = z[idx]

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        ax.scatter(
            z.real,
            z.imag,
            s=4,
            alpha=0.25,
            rasterized=True,
        )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        ax.axvline(
            0,
            linewidth=0.8,
        )

        ax.set_xlabel(
            "Real part"
        )

        ax.set_ylabel(
            "Imaginary part"
        )

        ax.set_title(
            f"Complex activation distribution | "
            f"{dataset} | α={alfa:g} | "
            f"layer={layer}"
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            True,
            alpha=0.2,
        )

        path = (
            output_dir
            / (
                f"{dataset}_alfa={alfa:g}_"
                f"layer={layer:04d}_"
                f"complex_plane.png"
            )
        )

        fig.tight_layout()

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        paths.append(path)

    return paths


# ============================================================
# Activation statistics
# ============================================================

def _component_stats(
    values: np.ndarray,
    *,
    component: str,
) -> dict[str, float | str]:
    """Summarize one flattened activation component."""

    flat = values.reshape(-1).astype(
        np.float64,
        copy=False,
    )

    quantiles = np.quantile(
        flat,
        [
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ],
    )

    row: dict[
        str,
        float | str,
    ] = {
        "component": component,
        "count": int(flat.size),
        "mean": float(
            np.mean(flat)
        ),
        "std": float(
            np.std(flat)
        ),
        "min": float(
            np.min(flat)
        ),
        "p01": float(
            quantiles[0]
        ),
        "p05": float(
            quantiles[1]
        ),
        "p50": float(
            quantiles[2]
        ),
        "p95": float(
            quantiles[3]
        ),
        "p99": float(
            quantiles[4]
        ),
        "max": float(
            np.max(flat)
        ),
    }

    if component in {
        "real",
        "imag",
    }:

        row[
            "abs_gt_0_98_fraction"
        ] = float(
            np.mean(
                np.abs(flat) > 0.98
            )
        )

        row[
            "abs_lt_1e_5_fraction"
        ] = float(
            np.mean(
                np.abs(flat) < 1.0e-5
            )
        )

    if component == "magnitude":

        row[
            "gt_0_98_fraction"
        ] = float(
            np.mean(
                flat > 0.98
            )
        )

        row[
            "lt_1e_5_fraction"
        ] = float(
            np.mean(
                flat < 1.0e-5
            )
        )

    return row


# ============================================================
# Activation CSV rows
# ============================================================

def _activation_rows(
    activations: np.ndarray,
    *,
    spec: ClassificationSpec,
    config: AlfaAnalysisConfig,
    alfa: float,
    layer: int,
) -> list[
    dict[str, float | int | str]
]:
    """Build CSV-ready summaries."""

    components = {
        "real": activations.real,
        "imag": activations.imag,
        "magnitude": np.abs(
            activations
        ),
    }

    rows: list[
        dict[str, float | int | str]
    ] = []

    for component, values in (
        components.items()
    ):

        row = _component_stats(
            values,
            component=component,
        )

        row.update(
            {
                "dataset": spec.name,
                "dataset_label": spec.label,
                "alfa": alfa,
                "depth": config.depth,
                "hidden_size": config.hidden_size,
                "num_samples": config.num_samples,
                "input_size": spec.input_size,
                "layer": layer,
                "activation": config.activation,
                "input_distribution": (
                    config.input_distribution
                ),
            }
        )

        rows.append(row)

    return rows


# ============================================================
# Save raw activations
# ============================================================

def _write_activation_npz(
    path: Path,
    activations: np.ndarray,
    *,
    spec: ClassificationSpec,
    config: AlfaAnalysisConfig,
    alfa: float,
    layer: int,
) -> None:
    """Persist raw complex activations and metadata."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,

        layer=np.array(
            layer,
            dtype=np.int64,
        ),

        activations=activations,

        dataset=spec.name,

        dataset_label=spec.label,

        alfa=np.array(
            alfa,
            dtype=np.float64,
        ),

        depth=np.array(
            config.depth,
            dtype=np.int64,
        ),

        hidden_size=np.array(
            config.hidden_size,
            dtype=np.int64,
        ),

        num_samples=np.array(
            config.num_samples,
            dtype=np.int64,
        ),

        input_size=np.array(
            spec.input_size,
            dtype=np.int64,
        ),

        activation=config.activation,

        input_distribution=(
            config.input_distribution
        ),

        normal_variance=np.array(
            config.normal_variance,
            dtype=np.float64,
        ),

        uniform_low=np.array(
            config.uniform_low,
            dtype=np.float64,
        ),

        uniform_high=np.array(
            config.uniform_high,
            dtype=np.float64,
        ),

        seed=np.array(
            config.seed,
            dtype=np.int64,
        ),
    )


# ============================================================
# Run one alfa
# ============================================================

def _run_one(
    *,
    spec: ClassificationSpec,
    samples: torch.Tensor,
    alfa: float,
    config: AlfaAnalysisConfig,
    device: torch.device,
    dataset_index: int,
) -> ActivationRun:
    """Run one dataset/alfa combination."""

    model_seed = (
        config.seed
        + 10_000
        + dataset_index
    )

    set_seed(model_seed)

    model = _build_classifier(
        spec,
        config,
    ).to(device)

    set_seed(model_seed)

    initialize_model(
        model,
        "structured_preserve",
        alpha=alfa,
    )

    LOGGER.info(
        "Forwarding dataset=%s "
        "alfa=%g depth=%s "
        "hidden_size=%s samples=%s "
        "input_size=%s distribution=%s "
        "device=%s",
        spec.name,
        alfa,
        config.depth,
        config.hidden_size,
        config.num_samples,
        spec.input_size,
        config.input_distribution,
        device,
    )

    layers = (
        config.layers
        if config.layers
        else (config.depth,)
    )

    activations_by_layer = (
        _capture_activations(
            model,
            samples,
            layers=layers,
            depth=config.depth,
            batch_size=config.batch_size,
            device=device,
        )
    )

    # --------------------------------------------------------
    # Save ALL selected layers
    # --------------------------------------------------------

    all_rows: list[
        dict[str, float | int | str]
    ] = []

    activation_paths: list[
        Path
    ] = []

    plot_data: dict[
        int,
        np.ndarray,
    ] = {}

    for layer, tensor in (
        activations_by_layer.items()
    ):

        activations = tensor.numpy()

        output_path = (
            config.output_dir
            / "activations"
            / (
                f"{spec.name}_"
                f"alfa={alfa:g}_"
                f"layer={layer:04d}.npz"
            )
        )

        _write_activation_npz(
            output_path,
            activations,
            spec=spec,
            config=config,
            alfa=alfa,
            layer=layer,
        )

        activation_paths.append(
            output_path
        )

        rows = _activation_rows(
            activations,
            spec=spec,
            config=config,
            alfa=alfa,
            layer=layer,
        )

        all_rows.extend(rows)

        plot_data[layer] = activations

    # --------------------------------------------------------
    # Static complex-plane PNGs
    # --------------------------------------------------------

    _save_complex_plane_plots(
        plot_data,
        alfa=alfa,
        dataset=spec.name,
        output_dir=(
            config.output_dir
            / "plots"
            / "complex_plane"
        ),
        max_points=config.max_plot_points,
    )

    return ActivationRun(
        dataset=spec.name,
        alfa=alfa,
        activation_paths=activation_paths,
        rows=all_rows,
    )


# ============================================================
# Metadata
# ============================================================

def _write_metadata(
    config: AlfaAnalysisConfig,
    runs: Iterable[ActivationRun],
) -> Path:
    """Write resolved experiment settings."""

    metadata_path = (
        config.output_dir
        / "metadata.json"
    )

    payload = {
        "datasets": list(
            config.dataset_names
        ),

        "alfa_values": list(
            config.alfa_values
        ),

        "num_samples": config.num_samples,

        "depth": config.depth,

        "hidden_size": config.hidden_size,

        "activation": config.activation,

        "batch_size": config.batch_size,

        "bins": config.bins,

        "layers": list(
            config.layers
        ),

        "max_plot_points": (
            config.max_plot_points
        ),

        "seed": config.seed,

        "device": config.device,

        # ----------------------------------------------------
        # Input distribution
        # ----------------------------------------------------

        "input_distribution": (
            config.input_distribution
        ),

        "normal_variance": (
            config.normal_variance
        ),

        "uniform_low": (
            config.uniform_low
        ),

        "uniform_high": (
            config.uniform_high
        ),

        "runs": [
            {
                "dataset": run.dataset,

                "alfa": run.alfa,

                "activation_paths": [
                    str(path)
                    for path
                    in run.activation_paths
                ],
            }
            for run in runs
        ],
    }

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return metadata_path


# ============================================================
# Main experiment
# ============================================================

def run_alfa_analysis(
    config: AlfaAnalysisConfig,
) -> dict[str, Path]:
    """Run all dataset/alfa combinations."""

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        config.device
    )

    runs: list[
        ActivationRun
    ] = []

    summary_rows: list[
        dict[str, float | int | str]
    ] = []

    for dataset_index, dataset_name in enumerate(
        config.dataset_names
    ):

        if dataset_name not in DATASET_SPECS:

            available = ", ".join(
                sorted(
                    DATASET_SPECS
                )
            )

            raise ValueError(
                f"Unknown dataset "
                f"'{dataset_name}'. "
                f"Available: {available}"
            )

        spec = DATASET_SPECS[
            dataset_name
        ]

        input_seed = (
            config.seed
            + dataset_index
        )

        # ----------------------------------------------------
        # Generate ONE fixed input dataset.
        #
        # The same samples are then used for every alfa.
        # ----------------------------------------------------

        samples = _complex_samples(
            num_samples=config.num_samples,
            input_size=spec.input_size,
            distribution=(
                config.input_distribution
            ),
            normal_variance=(
                config.normal_variance
            ),
            uniform_low=(
                config.uniform_low
            ),
            uniform_high=(
                config.uniform_high
            ),
            seed=input_seed,
        )

        # ----------------------------------------------------
        # Input statistics
        # ----------------------------------------------------

        LOGGER.info(
            "Input distribution=%s",
            config.input_distribution,
        )

        if (
            config.input_distribution
            == "normal"
        ):

            LOGGER.info(
                "Complex normal: "
                "CN(0, %.4g)",
                config.normal_variance,
            )

        else:

            LOGGER.info(
                "Complex uniform: "
                "Re/Im ~ U(%.4g, %.4g)",
                config.uniform_low,
                config.uniform_high,
            )

        # ----------------------------------------------------
        # Alfa sweep
        # ----------------------------------------------------

        for alfa in (
            config.alfa_values
        ):

            run = _run_one(
                spec=spec,
                samples=samples,
                alfa=alfa,
                config=config,
                device=device,
                dataset_index=dataset_index,
            )

            runs.append(run)

            summary_rows.extend(
                run.rows
            )

    # ========================================================
    # CSV summary
    # ========================================================

    summary_path = (
        config.output_dir
        / "activation_summary.csv"
    )

    pd.DataFrame.from_records(
        summary_rows
    ).to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata_path = _write_metadata(
        config,
        runs,
    )

    # ========================================================
    # HTML histogram report
    # ========================================================

    all_activation_paths: list[
        Path
    ] = []

    for run in runs:

        all_activation_paths.extend(
            run.activation_paths
        )

    LOGGER.info(
        "Building HTML activation report "
        "from %d activation files.",
        len(
            all_activation_paths
        ),
    )
    histogram_path = (
        save_activation_histograms(
            activation_paths=(
                all_activation_paths
            ),
            output_dir=(
                config.output_dir
                / "plots"
            ),
            bins=config.bins,
            max_points=(
                config.max_plot_points
            ),
        )
        )

    # ========================================================
    # Paper-ready PDF:
    # Compare all alfa values at each fixed layer
    # ========================================================

    LOGGER.info(
        "Building paper-ready alfa comparison PDF "
        "from %d activation files.",
        len(all_activation_paths),
    )

    paper_pdf_path = (
        save_fixed_layer_alpha_comparison_pdf(
            activation_paths=(
                all_activation_paths
            ),
            output_path=(
                config.output_dir
                / "plots"
                / "alpha_layer_comparison.pdf"
            ),
            bins=config.bins,
        )
    )

    return {
        "summary": summary_path,
        "metadata": metadata_path,
        "histograms": histogram_path,
        "paper_pdf": paper_pdf_path,
    }
    # histogram_path = (
    #     save_activation_histograms(
    #         activation_paths=(
    #             all_activation_paths
    #         ),
    #         output_dir=(
    #             config.output_dir
    #             / "plots"
    #         ),
    #         bins=config.bins,
    #         max_points=(
    #             config.max_plot_points
    #         ),
    #     )
    # )

    # return {
    #     "summary": summary_path,
    #     "metadata": metadata_path,
    #     "histograms": histogram_path,
    # }


# ============================================================
# CLI
# ============================================================

def _parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run alfa activation histograms "
            "with HoloViews/Bokeh."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(
            DATASET_SPECS
        ),
        default=["mnist"],
    )

    parser.add_argument(
        "--alfa-values",
        nargs="+",
        type=float,
        default=list(
            ALFA_VALUES
        ),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--activation",
        default="split_tanh",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help=(
            "1-based activation layers "
            "to plot and save."
        ),
    )

    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=20000,
        help=(
            "Maximum complex activation "
            "points per plane plot."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    # ========================================================
    # Input distribution arguments
    # ========================================================

    parser.add_argument(
        "--input-distribution",
        choices=[
            "normal",
            "uniform",
        ],
        default="normal",
        help=(
            "Complex input distribution. "
            "'normal' gives CN(0, variance). "
            "'uniform' gives independent "
            "real/imaginary Uniform(low, high)."
        ),
    )

    parser.add_argument(
        "--normal-variance",
        type=float,
        default=1.0,
        help=(
            "Variance of CN(0, variance) "
            "when using normal."
        ),
    )

    parser.add_argument(
        "--uniform-low",
        type=float,
        default=-1.0,
    )

    parser.add_argument(
        "--uniform-high",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser.parse_args(
        argv
    )


# ============================================================
# Entry point
# ============================================================

def main(
    argv: list[str] | None = None,
) -> None:
    """Run the alfa activation analysis."""

    args = _parse_args(
        argv
    )

    logging.basicConfig(
        level=(
            logging.WARNING
            if args.quiet
            else logging.INFO
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
        force=True,
    )

    config = AlfaAnalysisConfig(
        output_dir=args.output_dir,

        dataset_names=tuple(
            args.datasets
        ),

        alfa_values=tuple(
            args.alfa_values
        ),

        num_samples=args.num_samples,

        depth=args.depth,

        hidden_size=args.hidden_size,

        activation=args.activation,

        batch_size=args.batch_size,

        bins=args.bins,

        layers=(
            tuple(args.layers)
            if args.layers
            else ()
        ),

        max_plot_points=(
            args.max_plot_points
        ),

        seed=args.seed,

        device=args.device,

        input_distribution=(
            args.input_distribution
        ),

        normal_variance=(
            args.normal_variance
        ),

        uniform_low=(
            args.uniform_low
        ),

        uniform_high=(
            args.uniform_high
        ),
    )

    artifacts = run_alfa_analysis(
        config
    )

    for name, path in (
        artifacts.items()
    ):

        LOGGER.info(
            "%s: %s",
            name,
            path,
        )

