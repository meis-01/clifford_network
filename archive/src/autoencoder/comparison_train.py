"""
Weight Initialization Comparison Script

Trains a large complex autoencoder with different weight initialization schemes
and compares their performance on reconstruction tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.autoencoder.config import load_config
from src.autoencoder.dataset import build_dataloaders
from src.autoencoder.engine import run_epoch, save_checkpoint
from src.autoencoder.logging_utils import setup_logger
from src.autoencoder.model import build_autoencoder
from src.autoencoder.utils import ensure_output_dir, resolve_device, save_json, set_seed


# Available weight initialization methods
INITIALIZATION_METHODS = [
    "random",
    "xavier",
    "he",
    "unitary",
    "trabelsi",
    "structured_preserve",
]


def train_with_init(
    config_path: str | Path,
    init_method: str,
    log_level: str = "INFO",
) -> dict:
    """
    Train a model with a specific weight initialization method.
    
    Args:
        config_path: Path to the YAML config file
        init_method: Weight initialization method to use
        log_level: Logging level
        
    Returns:
        Dictionary with training results (history and metrics)
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    
    # Override weight initialization method
    config["model"]["weight_init"] = init_method
    
    training_config = config["training"]
    base_output_dir = Path(training_config["output_dir"])
    output_dir = ensure_output_dir(base_output_dir / f"init_{init_method}")
    
    logger = setup_logger(
        f"autoencoder.train.{init_method}",
        output_dir / "train.log",
        level=getattr(logging, log_level),
    )
    
    logger.info("=" * 80)
    logger.info(f"Training with weight initialization: {init_method}")
    logger.info("=" * 80)
    logger.info("Config path: %s", config_path.resolve())
    logger.info("Output directory: %s", output_dir.resolve())
    
    set_seed(int(training_config["seed"]))
    logger.info("Set random seed to %d.", int(training_config["seed"]))

    logger.info("Building dataloaders.")
    loaders = build_dataloaders(config, logger=logger)
    if len(loaders["train"].dataset) == 0 or len(loaders["val"].dataset) == 0:
        raise RuntimeError("Training requires non-empty train and validation split folders.")

    shutil.copy2(config_path, output_dir / "config.yaml")
    logger.info("Copied config to %s.", output_dir / "config.yaml")
    
    device = resolve_device(str(training_config["device"]))
    logger.info("Resolved device: %s.", device)
    
    model = build_autoencoder(config).to(device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Built model: %s", model.__class__.__name__)
    logger.info("Model parameters: %d", parameter_count)
    logger.info("Weight initialization method: %s", init_method)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    logger.info(
        "Built optimizer AdamW: learning_rate=%s weight_decay=%s",
        training_config["learning_rate"],
        training_config["weight_decay"],
    )

    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict] = []
    patience = int(training_config["patience"])
    log_interval = int(training_config.get("log_interval", 25))
    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"
    
    logger.info("Training for up to %d epochs with patience=%d.", int(training_config["epochs"]), patience)
    logger.info("Checkpoint paths: best=%s last=%s", best_checkpoint, last_checkpoint)

    start_time = time.time()
    
    for epoch in range(int(training_config["epochs"])):
        logger.info("Epoch %03d started.", epoch)
        
        train_result = run_epoch(
            model,
            loaders["train"],
            device=device,
            optimizer=optimizer,
            grad_clip_norm=float(training_config["grad_clip_norm"]),
            logger=logger,
            phase=f"train epoch {epoch:03d}",
            log_interval=log_interval,
        )
        
        val_result = run_epoch(
            model,
            loaders["val"],
            device=device,
            logger=logger,
            phase=f"val epoch {epoch:03d}",
            log_interval=log_interval,
        )
        
        loss_gap = val_result.loss - train_result.loss
        nmse_gap = val_result.nmse - train_result.nmse
        
        row = {
            "epoch": float(epoch),
            "train_loss": train_result.loss,
            "train_nmse": train_result.nmse,
            "val_loss": val_result.loss,
            "val_nmse": val_result.nmse,
            "loss_gap": loss_gap,
            "nmse_gap": nmse_gap,
        }
        history.append(row)
        
        logger.info(
            "Epoch summary: "
            "train_loss=%.6f val_loss=%.6f loss_gap=%.6f "
            "train_nmse=%.6f val_nmse=%.6f nmse_gap=%.6f",
            train_result.loss,
            val_result.loss,
            loss_gap,
            train_result.nmse,
            val_result.nmse,
            nmse_gap,
        )

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_epoch = epoch
            stale_epochs = 0
            logger.info("New best validation loss: %.6f at epoch %d", best_val_loss, best_epoch)
            save_checkpoint(best_checkpoint, model=model, optimizer=optimizer, epoch=epoch, config=config, best_val_loss=best_val_loss)
        else:
            stale_epochs += 1
            logger.info("Stale epochs: %d / %d", stale_epochs, patience)

        save_checkpoint(last_checkpoint, model=model, optimizer=optimizer, epoch=epoch, config=config, best_val_loss=best_val_loss)

        if stale_epochs >= patience:
            logger.info("Stopping early: stale_epochs >= patience (%d >= %d).", stale_epochs, patience)
            break

    elapsed_time = time.time() - start_time
    logger.info("Training completed in %.2f seconds (%.2f minutes)", elapsed_time, elapsed_time / 60)
    logger.info("Best validation loss: %.6f at epoch %d", best_val_loss, best_epoch)

    save_json({"history": history}, output_dir / "history.json")
    logger.info("Saved training history to %s", output_dir / "history.json")

    summary = {
        "init_method": init_method,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": len(history),
        "training_time_seconds": elapsed_time,
        "output_dir": str(output_dir.resolve()),
        "final_metrics": history[-1] if history else {},
    }
    save_json(summary, output_dir / "summary.json")
    logger.info("Saved summary to %s", output_dir / "summary.json")
    
    return summary


def run_comparison(
    config_path: str | Path,
    init_methods: list[str] | None = None,
    log_level: str = "INFO",
) -> None:
    """
    Train models with multiple weight initialization schemes and compare results.
    
    Args:
        config_path: Path to the YAML config file
        init_methods: List of initialization methods to compare. If None, uses all available methods.
        log_level: Logging level
    """
    config_path = Path(config_path)
    
    if init_methods is None:
        init_methods = INITIALIZATION_METHODS
    else:
        # Validate requested methods
        invalid = set(init_methods) - set(INITIALIZATION_METHODS)
        if invalid:
            raise ValueError(f"Invalid initialization methods: {invalid}")
    
    config = load_config(config_path)
    output_base = Path(config["training"]["output_dir"])
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Setup main logger
    main_logger = setup_logger(
        "autoencoder.comparison",
        output_base / "comparison.log",
        level=getattr(logging, log_level),
    )
    
    main_logger.info("=" * 80)
    main_logger.info("WEIGHT INITIALIZATION COMPARISON STUDY")
    main_logger.info("=" * 80)
    main_logger.info("Methods to compare: %s", ", ".join(init_methods))
    main_logger.info("Config: %s", config_path.resolve())
    main_logger.info("Output base directory: %s", output_base.resolve())
    
    results = {}
    
    for init_method in init_methods:
        main_logger.info("")
        main_logger.info("*" * 80)
        main_logger.info(f"Training with initialization method: {init_method}")
        main_logger.info("*" * 80)
        
        try:
            summary = train_with_init(config_path, init_method, log_level=log_level)
            results[init_method] = summary
            
            main_logger.info(f"✓ Training with {init_method} completed successfully")
            main_logger.info(f"  Best validation loss: {summary['best_val_loss']:.6f}")
            main_logger.info(f"  Best epoch: {summary['best_epoch']}")
            main_logger.info(f"  Total training time: {summary['training_time_seconds']:.2f}s")
            
        except Exception as e:
            main_logger.error(f"✗ Training with {init_method} failed: {e}", exc_info=True)
            results[init_method] = {"error": str(e)}
    
    # Save comparison summary
    comparison_summary = {
        "methods": init_methods,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    save_json(comparison_summary, output_base / "comparison_summary.json")
    main_logger.info("")
    main_logger.info("=" * 80)
    main_logger.info("COMPARISON SUMMARY")
    main_logger.info("=" * 80)
    
    # Print results table
    main_logger.info(f"{'Method':<25} {'Best Val Loss':<15} {'Best Epoch':<12} {'Status':<10}")
    main_logger.info("-" * 62)
    
    for method in init_methods:
        result = results[method]
        if "error" in result:
            main_logger.info(f"{method:<25} {'N/A':<15} {'N/A':<12} {'FAILED':<10}")
        else:
            best_loss = result["best_val_loss"]
            best_epoch = result["best_epoch"]
            main_logger.info(f"{method:<25} {best_loss:<15.6f} {best_epoch:<12} {'SUCCESS':<10}")
    
    main_logger.info("-" * 62)
    
    # Find best initialization method
    successful_results = {k: v for k, v in results.items() if "error" not in v}
    if successful_results:
        best_method = min(successful_results.items(), key=lambda x: x[1]["best_val_loss"])
        main_logger.info(f"\nBest initialization method: {best_method[0]} (loss: {best_method[1]['best_val_loss']:.6f})")
    
    main_logger.info(f"\nComparison summary saved to: {output_base / 'comparison_summary.json'}")
    main_logger.info("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare complex autoencoder with different weight initialization schemes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Weight initialization methods to compare. Options: {', '.join(INITIALIZATION_METHODS)}",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_comparison(args.config, init_methods=args.methods, log_level=args.log_level)


if __name__ == "__main__":
    main()
