#!/usr/bin/env python3
"""
Main script for running complex weight initialization training experiments.
This script trains models with different activation functions and initialization methods,
monitoring activation statistics in deeper layers to analyze saturation and vanishing signals.
"""

import argparse
import os
from typing import List, Optional

from training_module import run_training_experiment
from analysis_module import analyze_experiment, compare_initializations
from config_module import load_experiment_config, load_comparison_config


def run_single_experiment_from_config(config_path: str):
    """Run a single experiment from YAML config."""
    config = load_experiment_config(config_path)

    print("Running single experiment from config:")
    print(f"  Config: {config_path}")
    print(f"  Activation: {config.activation}")
    print(f"  Init method: {config.init_method}")
    print(f"  Layers: {config.n_layers}, Hidden size: {config.hidden_size}")
    print(f"  Dataset: {config.dataset_name} (root={config.dataset_root})")
    print(f"  Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"  Monitor layers: {config.monitor_layers}")
    print(f"  Output: {config.output_dir}")
    print("-" * 50)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    results = run_training_experiment(
        activation=config.activation,
        init_method=config.init_method,
        in_features=config.in_features,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        epochs=config.epochs,
        batch_size=config.batch_size,
        device=config.device,
        monitor_layers=config.monitor_layers,
        dataset_name=config.dataset_name,
        dataset_root=config.dataset_root,
    )

    experiment_name = f"{config.activation}_{config.init_method}"
    analyze_experiment(results, config.output_dir, experiment_name)

    print(f"✓ Completed experiment. Results saved to {config.output_dir}")


def run_comparison_from_config(config_path: str):
    """Run comparison experiments from YAML config."""
    config = load_comparison_config(config_path)

    print("Running comparison experiments from config:")
    print(f"  Config: {config_path}")
    print(f"  Activations: {config.activations}")
    print(f"  Init methods: {config.init_methods}")
    print(f"  Layers: {config.n_layers}, Hidden size: {config.hidden_size}")
    print(f"  Dataset: {config.dataset_name} (root={config.dataset_root})")
    print(f"  Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"  Monitor layers: {config.monitor_layers}")
    print(f"  Output: {config.output_dir}")
    print("-" * 50)

    os.makedirs(config.output_dir, exist_ok=True)

    all_results = []

    for activation in config.activations:
        for init_method in config.init_methods:
            print(f"\nRunning {activation} + {init_method}...")

            result = run_training_experiment(
                activation=activation,
                init_method=init_method,
                in_features=config.in_features,
                hidden_size=config.hidden_size,
                n_layers=config.n_layers,
                epochs=config.epochs,
                batch_size=config.batch_size,
                device=config.device,
                monitor_layers=config.monitor_layers,
                dataset_name=config.dataset_name,
                dataset_root=config.dataset_root,
            )

            all_results.append(result)
            print(f"✓ Completed {activation} + {init_method}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    for layer_idx in config.monitor_layers:
        compare_initializations(all_results, config.output_dir, layer_idx)

    print(f"\nComparison complete! Results saved to {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Complex Weight Initialization Training Experiments")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--create-configs",
        action="store_true",
        help="Create default configuration files"
    )

    args = parser.parse_args()

    if args.create_configs:
        from config_module import create_default_configs
        create_default_configs()
        return

    if not args.config:
        print("Usage:")
        print("  python train_experiments.py --create-configs  # Create default configs")
        print("  python train_experiments.py --config configs/single_experiment.yaml")
        print("  python train_experiments.py --config configs/comparison.yaml")
        print("\nAvailable config files:")
        if os.path.exists("configs"):
            for f in os.listdir("configs"):
                if f.endswith(".yaml"):
                    print(f"  - configs/{f}")
        return

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return

    # Determine config type based on content
    try:
        # Try loading as comparison config first
        config = load_comparison_config(args.config)
        run_comparison_from_config(args.config)
    except ValueError:
        # If that fails, try single experiment config
        try:
            config = load_experiment_config(args.config)
            run_single_experiment_from_config(args.config)
        except ValueError as e:
            print(f"Error loading config: {e}")
            return


if __name__ == "__main__":
    main()