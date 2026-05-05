import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import pandas as pd

from training_module import run_training_experiment


def plot_activation_evolution(results: Dict[str, Any], output_dir: str, experiment_name: str):
    """Plot how activation statistics evolve over training epochs."""
    monitor = results["results"]["activation_monitor"]

    for layer_idx in monitor.monitor_layers:
        layer_stats = monitor.get_layer_stats(layer_idx)
        history = layer_stats["history"]

        epochs = [h["epoch"] for h in history]
        means = [h["mean"] for h in history]
        stds = [h["std"] for h in history]
        mins = [h["min"] for h in history]
        maxs = [h["max"] for h in history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Activation Evolution - Layer {layer_idx}\n{experiment_name}")

        # Mean and std evolution
        axes[0, 0].plot(epochs, means, 'b-', label='Mean |z|', linewidth=2)
        axes[0, 0].fill_between(epochs,
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.3, color='blue', label='±1 std')
        axes[0, 0].set_title("Mean Activation Magnitude")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Magnitude")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Min/Max evolution
        axes[0, 1].plot(epochs, mins, 'r-', label='Min |z|', linewidth=2)
        axes[0, 1].plot(epochs, maxs, 'g-', label='Max |z|', linewidth=2)
        axes[0, 1].set_title("Min/Max Activation Magnitude")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Magnitude")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Saturation analysis (values near 0 or very large)
        saturation_ratios = []
        vanishing_ratios = []
        for h in history:
            values = h["values"]
            saturation_ratios.append(np.mean(values > 10))  # Values > 10 considered saturated
            vanishing_ratios.append(np.mean(values < 0.01))  # Values < 0.01 considered vanishing

        axes[1, 0].plot(epochs, saturation_ratios, 'orange', label='Saturation Ratio', linewidth=2)
        axes[1, 0].plot(epochs, vanishing_ratios, 'purple', label='Vanishing Ratio', linewidth=2)
        axes[1, 0].set_title("Saturation/Vanishing Analysis")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Ratio")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Final distribution
        if history:
            final_values = history[-1]["values"]
            axes[1, 1].hist(final_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(np.mean(final_values), color='red', linestyle='--', label=f'Mean: {np.mean(final_values):.3f}')
            axes[1, 1].axvline(np.median(final_values), color='green', linestyle='--', label=f'Median: {np.median(final_values):.3f}')
            axes[1, 1].set_title("Final Activation Distribution")
            axes[1, 1].set_xlabel("Activation Magnitude")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{experiment_name}_layer_{layer_idx}_evolution.png"), dpi=150, bbox_inches='tight')
        plt.close()


def plot_activation_distributions(results: Dict[str, Any], output_dir: str, experiment_name: str):
    """Plot activation distributions for monitored layers."""
    monitor = results["results"]["activation_monitor"]

    for layer_idx in monitor.monitor_layers:
        layer_stats = monitor.get_layer_stats(layer_idx)
        history = layer_stats["history"]

        if not history:
            continue

        # Plot distributions at different epochs
        epochs_to_plot = [0, len(history)//4, len(history)//2, len(history)-1]
        epochs_to_plot = [e for e in epochs_to_plot if e < len(history)]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        fig.suptitle(f"Activation Distributions - Layer {layer_idx}\n{experiment_name}")

        for i, epoch_idx in enumerate(epochs_to_plot):
            values = history[epoch_idx]["values"]
            epoch = history[epoch_idx]["epoch"]

            axes[i].hist(values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].axvline(np.mean(values), color='red', linestyle='--',
                          label=f'Mean: {np.mean(values):.3f}')
            axes[i].axvline(np.median(values), color='green', linestyle='--',
                          label=f'Median: {np.median(values):.3f}')
            axes[i].set_title(f"Epoch {epoch}")
            axes[i].set_xlabel("Activation Magnitude")
            axes[i].set_ylabel("Frequency")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{experiment_name}_layer_{layer_idx}_distributions.png"), dpi=150, bbox_inches='tight')
        plt.close()


def plot_training_curves(results: Dict[str, Any], output_dir: str, experiment_name: str):
    """Plot training and validation loss curves."""
    train_losses = results["results"]["train_losses"]
    val_losses = results["results"]["val_losses"]

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_title(f"Training Curves\n{experiment_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')  # Log scale for better visualization

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()


def compare_initializations(results_list: List[Dict[str, Any]], output_dir: str, layer_idx: int = -1):
    """Compare different initialization methods."""
    if not results_list:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect final statistics for the specified layer
    comparison_data = []
    for results in results_list:
        monitor = results["results"]["activation_monitor"]
        if layer_idx not in monitor.monitor_layers:
            continue

        layer_stats = monitor.get_layer_stats(layer_idx)
        history = layer_stats["history"]

        if history:
            final_stats = history[-1]
            comparison_data.append({
                "method": results["init_method"],
                "activation": results["activation"],
                "final_mean": final_stats["mean"],
                "final_std": final_stats["std"],
                "final_min": final_stats["min"],
                "final_max": final_stats["max"],
                "saturation_ratio": np.mean(final_stats["values"] > 10),
                "vanishing_ratio": np.mean(final_stats["values"] < 0.01),
                "final_loss": results["results"]["val_losses"][-1],
            })

    if not comparison_data:
        return

    df = pd.DataFrame(comparison_data)

    # Plot comparison of final activation statistics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Initialization Method Comparison - Layer {layer_idx}")

    methods = df["method"].unique()
    x = np.arange(len(methods))

    # Final mean activation
    means = [df[df["method"] == m]["final_mean"].iloc[0] for m in methods]
    axes[0, 0].bar(x, means, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Final Mean Activation")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=45)

    # Final std activation
    stds = [df[df["method"] == m]["final_std"].iloc[0] for m in methods]
    axes[0, 1].bar(x, stds, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title("Final Std Activation")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods, rotation=45)

    # Saturation ratio
    sat_ratios = [df[df["method"] == m]["saturation_ratio"].iloc[0] for m in methods]
    axes[0, 2].bar(x, sat_ratios, color='orange', edgecolor='black')
    axes[0, 2].set_title("Saturation Ratio")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(methods, rotation=45)

    # Vanishing ratio
    van_ratios = [df[df["method"] == m]["vanishing_ratio"].iloc[0] for m in methods]
    axes[1, 0].bar(x, van_ratios, color='purple', edgecolor='black')
    axes[1, 0].set_title("Vanishing Ratio")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods, rotation=45)

    # Min/Max range
    min_vals = [df[df["method"] == m]["final_min"].iloc[0] for m in methods]
    max_vals = [df[df["method"] == m]["final_max"].iloc[0] for m in methods]
    axes[1, 1].bar(x, max_vals, color='red', alpha=0.7, label='Max', edgecolor='black')
    axes[1, 1].bar(x, min_vals, color='blue', alpha=0.7, label='Min', edgecolor='black')
    axes[1, 1].set_title("Min/Max Range")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods, rotation=45)
    axes[1, 1].legend()

    # Final validation loss
    losses = [df[df["method"] == m]["final_loss"].iloc[0] for m in methods]
    axes[1, 2].bar(x, losses, color='gray', edgecolor='black')
    axes[1, 2].set_title("Final Validation Loss")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(methods, rotation=45)
    axes[1, 2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"initialization_comparison_layer_{layer_idx}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Save comparison table
    df.to_csv(os.path.join(output_dir, f"initialization_comparison_layer_{layer_idx}.csv"), index=False)


def analyze_experiment(results: Dict[str, Any], output_dir: str, experiment_name: str):
    """Run complete analysis for a single experiment."""
    plot_training_curves(results, output_dir, experiment_name)
    plot_activation_evolution(results, output_dir, experiment_name)
    plot_activation_distributions(results, output_dir, experiment_name)