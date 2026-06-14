"""
Weight Initialization Comparison Analysis Script

Analyzes and visualizes training results from different weight initialization schemes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_training_history(results_dir: Path, init_method: str) -> dict[str, Any] | None:
    """Load training history for an initialization method."""
    history_file = results_dir / f"init_{init_method}" / "history.json"
    if not history_file.exists():
        return None
    with open(history_file, "r") as f:
        return json.load(f)


def load_summary(results_dir: Path, init_method: str) -> dict[str, Any] | None:
    """Load summary for an initialization method."""
    summary_file = results_dir / f"init_{init_method}" / "summary.json"
    if not summary_file.exists():
        return None
    with open(summary_file, "r") as f:
        return json.load(f)


def create_comparison_plots(
    results_dir: Path,
    init_methods: list[str],
    output_dir: Path | None = None,
) -> None:
    """
    Create comparison plots across initialization methods.
    
    Args:
        results_dir: Base directory containing init_* subdirectories
        init_methods: List of initialization methods to compare
        output_dir: Directory to save plots. If None, uses results_dir/plots
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir or results_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all histories
    histories = {}
    summaries = {}
    
    for method in init_methods:
        history = load_training_history(results_dir, method)
        summary = load_summary(results_dir, method)
        
        if history and summary:
            histories[method] = history
            summaries[method] = summary
    
    if not histories:
        print(f"No training histories found in {results_dir}")
        return
    
    print(f"Found training data for {len(histories)} methods: {', '.join(histories.keys())}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Weight Initialization Comparison - Large Autoencoder", fontsize=16, fontweight="bold")
    
    # Plot 1: Training and Validation Loss
    ax = axes[0, 0]
    for method, history in histories.items():
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        
        ax.plot(epochs, train_loss, label=f"{method} (train)", linestyle="--", alpha=0.7)
        ax.plot(epochs, val_loss, label=f"{method} (val)", linestyle="-", linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation NMSE
    ax = axes[0, 1]
    for method, history in histories.items():
        epochs = [h["epoch"] for h in history]
        train_nmse = [h["train_nmse"] for h in history]
        val_nmse = [h["val_nmse"] for h in history]
        
        ax.plot(epochs, train_nmse, label=f"{method} (train)", linestyle="--", alpha=0.7)
        ax.plot(epochs, val_nmse, label=f"{method} (val)", linestyle="-", linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMSE")
    ax.set_title("Training and Validation NMSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Best Validation Loss Comparison
    ax = axes[1, 0]
    methods = list(summaries.keys())
    best_losses = [summaries[m]["best_val_loss"] for m in methods]
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    bars = ax.bar(methods, best_losses, color=colors)
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Best Validation Loss by Initialization Method")
    ax.tick_params(axis="x", rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    # Plot 4: Training Time and Convergence
    ax = axes[1, 1]
    x_pos = np.arange(len(methods))
    width = 0.35
    
    training_times = [summaries[m]["training_time_seconds"] / 60 for m in methods]  # Convert to minutes
    best_epochs = [summaries[m]["best_epoch"] for m in methods]
    
    bars1 = ax.bar(x_pos - width / 2, training_times, width, label="Training Time (min)", color="skyblue")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width / 2, best_epochs, width, label="Best Epoch", color="lightcoral")
    
    ax.set_xlabel("Initialization Method")
    ax.set_ylabel("Training Time (minutes)", color="skyblue")
    ax2.set_ylabel("Best Epoch", color="lightcoral")
    ax.set_title("Training Time and Convergence Speed")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    ax.tick_params(axis="y", labelcolor="skyblue")
    ax2.tick_params(axis="y", labelcolor="lightcoral")
    
    # Add legends
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_overview.png", dpi=300, bbox_inches="tight")
    print(f"Saved comparison overview to {output_dir / 'comparison_overview.png'}")
    plt.close()
    
    # Create detailed loss curves plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Detailed Loss Curves - Weight Initialization Comparison", fontsize=14, fontweight="bold")
    
    ax = axes[0]
    for method, history in histories.items():
        epochs = [h["epoch"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        ax.plot(epochs, val_loss, marker="o", markersize=3, label=method, linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for method, history in histories.items():
        epochs = [h["epoch"] for h in history]
        loss_gap = [h["loss_gap"] for h in history]
        ax.plot(epochs, loss_gap, marker="s", markersize=3, label=method, linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Gap (val - train)")
    ax.set_title("Generalization Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "detailed_loss_curves.png", dpi=300, bbox_inches="tight")
    print(f"Saved detailed loss curves to {output_dir / 'detailed_loss_curves.png'}")
    plt.close()


def create_summary_table(
    results_dir: Path,
    init_methods: list[str],
    output_file: Path | None = None,
) -> pd.DataFrame:
    """
    Create a summary table comparing all initialization methods.
    
    Args:
        results_dir: Base directory containing init_* subdirectories
        init_methods: List of initialization methods to compare
        output_file: Optional path to save the table as CSV
        
    Returns:
        DataFrame with comparison results
    """
    results_dir = Path(results_dir)
    
    rows = []
    
    for method in init_methods:
        summary = load_summary(results_dir, method)
        history = load_training_history(results_dir, method)
        
        if summary and history:
            final_metrics = summary.get("final_metrics", {})
            
            row = {
                "Method": method,
                "Best Val Loss": summary["best_val_loss"],
                "Best Epoch": summary["best_epoch"],
                "Total Epochs": summary["total_epochs"],
                "Training Time (min)": summary["training_time_seconds"] / 60,
                "Final Train Loss": final_metrics.get("train_loss", np.nan),
                "Final Val Loss": final_metrics.get("val_loss", np.nan),
                "Final Train NMSE": final_metrics.get("train_nmse", np.nan),
                "Final Val NMSE": final_metrics.get("val_nmse", np.nan),
                "Loss Gap": final_metrics.get("loss_gap", np.nan),
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by best validation loss
    df = df.sort_values("Best Val Loss")
    
    print("\nWeight Initialization Comparison Summary")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nSummary table saved to {output_file}")
    
    return df


def analyze_convergence(
    results_dir: Path,
    init_methods: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Analyze convergence properties for each initialization method.
    
    Args:
        results_dir: Base directory containing init_* subdirectories
        init_methods: List of initialization methods to compare
        
    Returns:
        Dictionary with convergence metrics for each method
    """
    results_dir = Path(results_dir)
    analysis = {}
    
    for method in init_methods:
        history = load_training_history(results_dir, method)
        summary = load_summary(results_dir, method)
        
        if not history or not summary:
            continue
        
        losses = [h["val_loss"] for h in history]
        nmses = [h["val_nmse"] for h in history]
        
        # Calculate metrics
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        # Find convergence point (when loss stops improving significantly)
        best_loss = min(losses)
        epochs_to_best = summary["best_epoch"]
        
        # Calculate learning stability (variance of last 10% of epochs)
        num_epochs = len(losses)
        last_n = max(1, num_epochs // 10)
        stability = np.std(losses[-last_n:]) if last_n > 0 else 0
        
        analysis[method] = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement_percent": improvement,
            "best_loss": best_loss,
            "epochs_to_convergence": epochs_to_best,
            "stability": stability,
            "total_epochs": num_epochs,
            "convergence_rate": epochs_to_best / num_epochs if num_epochs > 0 else 0,
        }
    
    print("\nConvergence Analysis")
    print("=" * 100)
    print(f"{'Method':<20} {'Initial Loss':<15} {'Final Loss':<15} {'Improvement %':<15} {'Convergence Epoch':<15}")
    print("-" * 100)
    
    for method in sorted(analysis.keys(), key=lambda m: analysis[m]["best_loss"]):
        metrics = analysis[method]
        print(
            f"{method:<20} {metrics['initial_loss']:<15.6f} {metrics['final_loss']:<15.6f} "
            f"{metrics['improvement_percent']:<15.2f} {metrics['epochs_to_convergence']:<15}"
        )
    
    print("=" * 100)
    
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize weight initialization comparison results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Base directory containing init_* subdirectories with training results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="List of initialization methods that were trained.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots and tables. Default: results_dir/plots",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    
    # Create comparison plots
    print("Creating comparison plots...")
    create_comparison_plots(args.results_dir, args.methods, args.output_dir)
    
    # Create summary table
    print("\nGenerating summary table...")
    output_dir = Path(args.output_dir or args.results_dir / "plots")
    create_summary_table(args.results_dir, args.methods, output_dir / "comparison_summary.csv")
    
    # Analyze convergence
    print("\nAnalyzing convergence properties...")
    analyze_convergence(args.results_dir, args.methods)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
