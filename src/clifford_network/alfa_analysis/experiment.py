"""Run the structured-preserve alfa activation histogram experiment.

The experiment mirrors the 1000-layer tanh activation histogram check from the
real-valued initialization paper, but uses this repository's complex
classification MLP and varies the alfa value passed to structured_preserve.
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

from clifford_network.alfa_analysis.plots import save_activation_histograms

LOGGER = logging.getLogger(__name__)

ALFA_VALUES: tuple[float, ...] = (1.0e-5, 1.0e-3, 0.0085, 1.0e-1, 1.0, 10.0)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ClassificationSpec:
    """Dataset-shaped classifier dimensions for random complex inputs."""

    name: str
    label: str
    input_size: int
    num_classes: int = 10


DATASET_SPECS: dict[str, ClassificationSpec] = {
    "mnist": ClassificationSpec(name="mnist", label="MNIST", input_size=28 * 28),
}


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
    seed: int = 0
    device: str = "auto"
    uniform_low: float = -1.0
    uniform_high: float = 1.0


@dataclass(frozen=True)
class ActivationRun:
    """Paths and metadata produced by one dataset/alfa run."""

    dataset: str
    alfa: float
    activation_path: Path
    rows: list[dict[str, float | int | str]]


def _build_classifier(spec: ClassificationSpec, config: AlfaAnalysisConfig) -> nn.Module:
    """Build the repository's complex classifier for a dataset-shaped input."""
    model_config = {
        "experiment": {"task": "classification"},
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


def _complex_uniform_samples(
    *,
    num_samples: int,
    input_size: int,
    low: float,
    high: float,
    seed: int,
) -> torch.Tensor:
    """Generate complex samples with independent real and imaginary uniforms."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.empty(num_samples, input_size, dtype=torch.float32)
    imag = torch.empty(num_samples, input_size, dtype=torch.float32)
    real.uniform_(low, high, generator=generator)
    imag.uniform_(low, high, generator=generator)
    return torch.complex(real, imag)


def _target_activation_name(depth: int) -> str:
    """Return the post-tanh activation module name for the requested layer."""
    if depth < 1:
        raise ValueError("depth must be at least 1.")
    return f"network.activation_{depth - 1:03d}"


def _capture_activation(
    model: nn.Module,
    samples: torch.Tensor,
    *,
    depth: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Forward samples and collect the post-activation values at layer depth."""
    target_name = _target_activation_name(depth)
    modules = dict(model.named_modules())
    if target_name not in modules:
        available = ", ".join(name for name in modules if "activation" in name)
        raise ValueError(f"Could not find activation layer '{target_name}'. Available activations: {available}")

    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not torch.is_tensor(output):
            raise TypeError(f"Expected tensor output from '{target_name}', got {type(output)!r}.")
        captured.append(output.detach().cpu())

    handle = modules[target_name].register_forward_hook(hook)
    try:
        model.eval()
        effective_batch_size = samples.shape[0] if batch_size <= 0 else batch_size
        with torch.inference_mode():
            for batch in samples.split(effective_batch_size):
                _ = model(batch.to(device))
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(f"No activations were captured from '{target_name}'.")
    return torch.cat(captured, dim=0)


def _component_stats(values: np.ndarray, *, component: str) -> dict[str, float | str]:
    """Summarize one flattened activation component."""
    flat = values.reshape(-1).astype(np.float64, copy=False)
    quantiles = np.quantile(flat, [0.01, 0.05, 0.5, 0.95, 0.99])
    row: dict[str, float | str] = {
        "component": component,
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(np.max(flat)),
    }
    if component in {"real", "imag"}:
        row["abs_gt_0_98_fraction"] = float(np.mean(np.abs(flat) > 0.98))
        row["abs_lt_1e_5_fraction"] = float(np.mean(np.abs(flat) < 1.0e-5))
    if component == "magnitude":
        row["gt_0_98_fraction"] = float(np.mean(flat > 0.98))
        row["lt_1e_5_fraction"] = float(np.mean(flat < 1.0e-5))
    return row


def _activation_rows(
    activations: np.ndarray,
    *,
    spec: ClassificationSpec,
    config: AlfaAnalysisConfig,
    alfa: float,
) -> list[dict[str, float | int | str]]:
    """Build CSV-ready summaries for complex layer activations."""
    components = {
        "real": activations.real,
        "imag": activations.imag,
        "magnitude": np.abs(activations),
    }
    rows: list[dict[str, float | int | str]] = []
    for component, values in components.items():
        row = _component_stats(values, component=component)
        row.update(
            {
                "dataset": spec.name,
                "dataset_label": spec.label,
                "alfa": alfa,
                "depth": config.depth,
                "hidden_size": config.hidden_size,
                "num_samples": config.num_samples,
                "input_size": spec.input_size,
                "layer": config.depth,
                "activation": config.activation,
            }
        )
        rows.append(row)
    return rows


def _write_activation_npz(
    path: Path,
    activations: np.ndarray,
    *,
    spec: ClassificationSpec,
    config: AlfaAnalysisConfig,
    alfa: float,
) -> None:
    """Persist raw complex activations and enough metadata to replot them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        activations=activations,
        dataset=spec.name,
        dataset_label=spec.label,
        alfa=np.array(alfa, dtype=np.float64),
        depth=np.array(config.depth, dtype=np.int64),
        hidden_size=np.array(config.hidden_size, dtype=np.int64),
        num_samples=np.array(config.num_samples, dtype=np.int64),
        input_size=np.array(spec.input_size, dtype=np.int64),
        activation=config.activation,
        uniform_low=np.array(config.uniform_low, dtype=np.float64),
        uniform_high=np.array(config.uniform_high, dtype=np.float64),
        seed=np.array(config.seed, dtype=np.int64),
    )


def _run_one(
    *,
    spec: ClassificationSpec,
    samples: torch.Tensor,
    alfa: float,
    config: AlfaAnalysisConfig,
    device: torch.device,
    dataset_index: int,
) -> ActivationRun:
    """Run one dataset/alfa combination and persist raw activations."""
    model_seed = config.seed + 10_000 + dataset_index
    set_seed(model_seed)
    model = _build_classifier(spec, config).to(device)
    set_seed(model_seed)
    initialize_model(model, "structured_preserve", alpha=alfa)

    LOGGER.info(
        "Forwarding dataset=%s alfa=%g depth=%s hidden_size=%s samples=%s input_size=%s device=%s",
        spec.name,
        alfa,
        config.depth,
        config.hidden_size,
        config.num_samples,
        spec.input_size,
        device,
    )
    activations = _capture_activation(
        model,
        samples,
        depth=config.depth,
        batch_size=config.batch_size,
        device=device,
    ).numpy()

    output_path = config.output_dir / "activations" / f"{spec.name}_alfa={alfa:g}.npz"
    _write_activation_npz(output_path, activations, spec=spec, config=config, alfa=alfa)
    rows = _activation_rows(activations, spec=spec, config=config, alfa=alfa)
    return ActivationRun(dataset=spec.name, alfa=alfa, activation_path=output_path, rows=rows)


def _write_metadata(config: AlfaAnalysisConfig, runs: Iterable[ActivationRun]) -> Path:
    """Write the resolved experiment settings and output paths."""
    metadata_path = config.output_dir / "metadata.json"
    payload = {
        "datasets": list(config.dataset_names),
        "alfa_values": list(config.alfa_values),
        "num_samples": config.num_samples,
        "depth": config.depth,
        "hidden_size": config.hidden_size,
        "activation": config.activation,
        "batch_size": config.batch_size,
        "bins": config.bins,
        "seed": config.seed,
        "device": config.device,
        "uniform_low": config.uniform_low,
        "uniform_high": config.uniform_high,
        "runs": [
            {
                "dataset": run.dataset,
                "alfa": run.alfa,
                "activation_path": str(run.activation_path),
            }
            for run in runs
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def run_alfa_analysis(config: AlfaAnalysisConfig) -> dict[str, Path]:
    """Run all requested dataset/alfa combinations and build Bokeh artifacts."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    runs: list[ActivationRun] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for dataset_index, dataset_name in enumerate(config.dataset_names):
        if dataset_name not in DATASET_SPECS:
            available = ", ".join(sorted(DATASET_SPECS))
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
        spec = DATASET_SPECS[dataset_name]
        input_seed = config.seed + dataset_index
        samples = _complex_uniform_samples(
            num_samples=config.num_samples,
            input_size=spec.input_size,
            low=config.uniform_low,
            high=config.uniform_high,
            seed=input_seed,
        )
        for alfa in config.alfa_values:
            run = _run_one(
                spec=spec,
                samples=samples,
                alfa=alfa,
                config=config,
                device=device,
                dataset_index=dataset_index,
            )
            runs.append(run)
            summary_rows.extend(run.rows)

    summary_path = config.output_dir / "activation_summary.csv"
    pd.DataFrame.from_records(summary_rows).to_csv(summary_path, index=False)
    metadata_path = _write_metadata(config, runs)
    histogram_path = save_activation_histograms(
        activation_paths=[run.activation_path for run in runs],
        output_dir=config.output_dir / "plots",
        bins=config.bins,
    )
    return {
        "summary": summary_path,
        "metadata": metadata_path,
        "histograms": histogram_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the alfa analysis."""
    parser = argparse.ArgumentParser(description="Run alfa activation histograms with HoloViews/Bokeh.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), default=["mnist"])
    parser.add_argument("--alfa-values", nargs="+", type=float, default=list(ALFA_VALUES))
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--activation", default="split_tanh")
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--uniform-low", type=float, default=-1.0)
    parser.add_argument("--uniform-high", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the alfa activation analysis from the command line."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    config = AlfaAnalysisConfig(
        output_dir=args.output_dir,
        dataset_names=tuple(args.datasets),
        alfa_values=tuple(args.alfa_values),
        num_samples=args.num_samples,
        depth=args.depth,
        hidden_size=args.hidden_size,
        activation=args.activation,
        batch_size=args.batch_size,
        bins=args.bins,
        seed=args.seed,
        device=args.device,
        uniform_low=args.uniform_low,
        uniform_high=args.uniform_high,
    )
    artifacts = run_alfa_analysis(config)
    for name, path in artifacts.items():
        LOGGER.info("%s: %s", name, path)
