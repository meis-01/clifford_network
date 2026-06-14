#!/usr/bin/env python3
"""
Quick comparison script for different initialization methods.
Runs multiple initialization methods with the same activation and compares results.
"""

import os
from training_module import run_training_experiment
from analysis_module import compare_initializations
from config_module import load_comparison_config


def main():
    # Default configuration - can be overridden by creating custom config files
    config_path = "configs/comparison.yaml"

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Run 'python train_experiments.py --create-configs' to create default configs")
        return

    try:
        config = load_comparison_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    print(f"Comparing initialization methods for {config.activation} activation")
    print(f"Layers: {config.n_layers}, Hidden size: {config.hidden_size}")
    print(f"Monitor layers: {config.monitor_layers}")
    print("-" * 50)

    results = []

    for init_method in config.init_methods:
        print(f"\nRunning {init_method} initialization...")

        result = run_training_experiment(
            activation=config.activation,
            init_method=init_method,
            in_features=config.in_features,
            hidden_size=config.hidden_size,
            n_layers=config.n_layers,
            epochs=config.epochs,
            batch_size=config.batch_size,
            device=config.device,
            monitor_layers=config.monitor_layers,
        )

        results.append(result)
        print(f"✓ Completed {init_method}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    for layer_idx in config.monitor_layers:
        compare_initializations(results, config.output_dir, layer_idx)

    print(f"\nComparison complete! Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()