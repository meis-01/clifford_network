"""Analysis artifact and Markdown report builders.

This module coordinates aggregation, plot generation, and table output, then can
summarize the generated artifact paths in a lightweight report document.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clifford_network.analysis.aggregate import collect_histories, collect_layer_stats
from clifford_network.analysis.plots import save_accuracy_curves, save_layer_metric_curves, save_training_curves
from clifford_network.analysis.tables import save_final_metric_table

LOGGER = logging.getLogger(__name__)


def build_analysis(results_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    """Generate CSV, plot, and table artifacts for a results directory."""
    results_path = Path(results_dir)
    output_path = Path(output_dir) if output_dir is not None else results_path / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Analysis output directory: %s", output_path)

    history = collect_histories(results_path)
    layer_stats = collect_layer_stats(results_path)
    LOGGER.info("Collected history rows=%s layer_stat rows=%s.", len(history), len(layer_stats))
    artifacts: dict[str, str] = {}

    history_path = output_path / "history_all.csv"
    layer_path = output_path / "layer_stats_all.csv"
    history.to_csv(history_path, index=False)
    layer_stats.to_csv(layer_path, index=False)
    artifacts["history"] = str(history_path)
    artifacts["layer_stats"] = str(layer_path)
    LOGGER.info("Wrote aggregate history: %s", history_path)
    LOGGER.info("Wrote aggregate layer stats: %s", layer_path)

    for name, path in {
        "training_curves": save_training_curves(history, output_path),
        "accuracy_curves": save_accuracy_curves(history, output_path),
        "layer_metrics": save_layer_metric_curves(layer_stats, output_path),
        "final_metrics": save_final_metric_table(history, output_path),
    }.items():
        if path is not None:
            artifacts[name] = str(path)
            LOGGER.info("Wrote analysis artifact %s: %s", name, path)
        else:
            LOGGER.info("Skipped analysis artifact %s because required data was unavailable.", name)

    return artifacts


def build_markdown_report(results_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Generate analysis artifacts and write a Markdown index of their paths."""
    LOGGER.info("Generating report index.")
    artifacts = build_analysis(results_dir, output_dir)
    output_path = Path(output_dir) if output_dir is not None else Path(results_dir) / "analysis"
    report_path = output_path / "report.md"
    lines = [
        "# Experiment Report",
        "",
        f"Results directory: `{Path(results_dir)}`",
        "",
        "## Generated Artifacts",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{path}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Markdown report written: %s", report_path)
    return report_path
