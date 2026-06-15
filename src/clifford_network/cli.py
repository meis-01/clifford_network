"""Command-line interface for running experiments and producing analysis artifacts.

This module wires the package's main workflows into subcommands: single runs,
sweeps, aggregate analysis, and Markdown report generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clifford_network.analysis.report import build_analysis, build_markdown_report
from clifford_network.experiments.config import load_config
from clifford_network.experiments.runner import run_experiment
from clifford_network.experiments.sweep import run_sweep


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and register all supported subcommands."""
    parser = argparse.ArgumentParser(prog="clifford-network")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single resolved experiment config.")
    run_parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")

    sweep_parser = subparsers.add_parser("sweep", help="Run initialization x depth x seed sweeps.")
    sweep_parser.add_argument("--config", required=True, help="Path to a YAML sweep config.")

    analyze_parser = subparsers.add_parser("analyze", help="Aggregate results and generate Bokeh plots.")
    analyze_parser.add_argument("--results-dir", required=True, help="Experiment results directory.")
    analyze_parser.add_argument("--output-dir", default=None, help="Optional analysis output directory.")

    report_parser = subparsers.add_parser("report", help="Generate analysis artifacts and a Markdown report.")
    report_parser.add_argument("--results-dir", required=True, help="Experiment results directory.")
    report_parser.add_argument("--output-dir", default=None, help="Optional report output directory.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and dispatch to the requested workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        summary = run_experiment(load_config(args.config))
        print(f"Run complete: {summary['run_dir']}")
    elif args.command == "sweep":
        frame = run_sweep(load_config(args.config))
        print(f"Sweep complete: {len(frame)} runs")
    elif args.command == "analyze":
        artifacts = build_analysis(Path(args.results_dir), args.output_dir)
        print("Analysis artifacts:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")
    elif args.command == "report":
        report_path = build_markdown_report(Path(args.results_dir), args.output_dir)
        print(f"Report written: {report_path}")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
