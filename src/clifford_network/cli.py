"""Command-line interface for running experiments and producing analysis artifacts.

This module wires the package's main workflows into subcommands: single runs,
sweeps, aggregate analysis, and Markdown report generation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from clifford_network.analysis.report import build_analysis, build_markdown_report
from clifford_network.experiments.config import load_config
from clifford_network.experiments.runner import run_experiment
from clifford_network.experiments.sweep import run_sweep

LOGGER = logging.getLogger(__name__)


def _add_logging_args(parser: argparse.ArgumentParser, *, with_defaults: bool) -> None:
    """Add common logging controls while keeping default output verbose."""
    default = False if with_defaults else argparse.SUPPRESS
    parser.add_argument("--quiet", action="store_true", default=default, help="Only show warnings and errors.")
    parser.add_argument("--verbose", action="store_true", default=default, help="Show debug-level diagnostic output.")


def _configure_logging(args: argparse.Namespace) -> None:
    """Configure CLI logging once after argument parsing."""
    if bool(getattr(args, "quiet", False)):
        level = logging.WARNING
    elif bool(getattr(args, "verbose", False)):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S", force=True)


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and register all supported subcommands."""
    parser = argparse.ArgumentParser(prog="clifford-network")
    _add_logging_args(parser, with_defaults=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single resolved experiment config.")
    _add_logging_args(run_parser, with_defaults=False)
    run_parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")

    sweep_parser = subparsers.add_parser("sweep", help="Run initialization x depth x seed sweeps.")
    _add_logging_args(sweep_parser, with_defaults=False)
    sweep_parser.add_argument("--config", required=True, help="Path to a YAML sweep config.")

    analyze_parser = subparsers.add_parser("analyze", help="Aggregate results and generate Bokeh plots.")
    _add_logging_args(analyze_parser, with_defaults=False)
    analyze_parser.add_argument("--results-dir", required=True, help="Experiment results directory.")
    analyze_parser.add_argument("--output-dir", default=None, help="Optional analysis output directory.")

    report_parser = subparsers.add_parser("report", help="Generate analysis artifacts and a Markdown report.")
    _add_logging_args(report_parser, with_defaults=False)
    report_parser.add_argument("--results-dir", required=True, help="Experiment results directory.")
    report_parser.add_argument("--output-dir", default=None, help="Optional report output directory.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and dispatch to the requested workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args)
    LOGGER.info("CLI command started: %s", args.command)

    if args.command == "run":
        LOGGER.info("Loading config: %s", Path(args.config).resolve())
        summary = run_experiment(load_config(args.config))
        LOGGER.info("Run complete: %s", summary["run_dir"])
    elif args.command == "sweep":
        LOGGER.info("Loading sweep config: %s", Path(args.config).resolve())
        frame = run_sweep(load_config(args.config))
        LOGGER.info("Sweep complete: %s runs", len(frame))
    elif args.command == "analyze":
        LOGGER.info("Building analysis for results directory: %s", Path(args.results_dir).resolve())
        artifacts = build_analysis(Path(args.results_dir), args.output_dir)
        for name, path in artifacts.items():
            LOGGER.info("Analysis artifact %s: %s", name, path)
    elif args.command == "report":
        LOGGER.info("Building report for results directory: %s", Path(args.results_dir).resolve())
        report_path = build_markdown_report(Path(args.results_dir), args.output_dir)
        LOGGER.info("Report written: %s", report_path)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
