from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch

from src.autoencoder.config import load_config
from src.autoencoder.dataset import build_dataloaders
from src.autoencoder.engine import run_epoch, save_checkpoint
from src.autoencoder.logging_utils import setup_logger
from src.autoencoder.model import build_autoencoder
from src.autoencoder.utils import ensure_output_dir, resolve_device, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a native-complex PyTorch autoencoder for T2 coil images.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run_training(config_path: str | Path, log_level: str = "INFO") -> None:
    config_path = Path(config_path)
    config = load_config(config_path)
    training_config = config["training"]
    output_dir = ensure_output_dir(training_config["output_dir"])
    logger = setup_logger("autoencoder.train", output_dir / "train.log", level=getattr(logging, log_level))
    logger.info("Starting native-complex autoencoder training.")
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
    logger.info("Native complex model dtype is torch.complex64; losses are real-valued for Wirtinger autograd.")
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
    history: list[dict[str, float]] = []
    patience = int(training_config["patience"])
    log_interval = int(training_config.get("log_interval", 25))
    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"
    logger.info("Training for up to %d epochs with patience=%d.", int(training_config["epochs"]), patience)
    logger.info("Checkpoint paths: best=%s last=%s", best_checkpoint, last_checkpoint)

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
            "val_minus_train_loss": loss_gap,
            "val_minus_train_nmse": nmse_gap,
            "best_val_loss_before_epoch": best_val_loss,
        }
        history.append(row)
        logger.info(
            "Epoch %03d summary: train_loss=%.6f train_nmse=%.6f val_loss=%.6f val_nmse=%.6f "
            "val_minus_train_loss=%+.6f val_minus_train_nmse=%+.6f",
            epoch,
            train_result.loss,
            train_result.nmse,
            val_result.loss,
            val_result.nmse,
            loss_gap,
            nmse_gap,
        )
        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                best_val_loss=best_val_loss,
            )
            logger.info("Saved new best checkpoint to %s with val_loss=%.6f.", best_checkpoint, best_val_loss)
        else:
            stale_epochs += 1
            logger.info(
                "No validation improvement. best_val_loss=%.6f best_epoch=%03d stale_epochs=%d/%d.",
                best_val_loss,
                best_epoch,
                stale_epochs,
                patience,
            )

        row["best_val_loss_after_epoch"] = best_val_loss
        row["best_epoch_after_epoch"] = float(best_epoch)
        row["stale_epochs_after_epoch"] = float(stale_epochs)
        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            best_val_loss=best_val_loss,
        )
        logger.info("Saved last checkpoint to %s.", last_checkpoint)

        save_json({"history": history}, output_dir / "history.json")
        logger.info("Wrote history to %s.", output_dir / "history.json")
        if stale_epochs >= patience:
            logger.info(
                "Early stopping at epoch %d because validation loss did not improve for %d epochs. "
                "Best epoch=%03d best_val_loss=%.6f.",
                epoch,
                patience,
                best_epoch,
                best_val_loss,
            )
            break

    logger.info("Loading best checkpoint for final validation/test evaluation: %s", best_checkpoint)
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_result = run_epoch(model, loaders["val"], device=device, logger=logger, phase="final val", log_interval=log_interval)
    test_result = (
        run_epoch(model, loaders["test"], device=device, logger=logger, phase="final test", log_interval=log_interval)
        if len(loaders["test"].dataset)
        else None
    )
    val_result.predictions.to_csv(output_dir / "val_reconstruction_errors.csv", index=False)
    logger.info("Wrote validation reconstruction errors to %s.", output_dir / "val_reconstruction_errors.csv")
    if test_result is not None:
        test_result.predictions.to_csv(output_dir / "test_reconstruction_errors.csv", index=False)
        logger.info("Wrote test reconstruction errors to %s.", output_dir / "test_reconstruction_errors.csv")

    save_json(
        {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "val_loss": val_result.loss,
            "val_nmse": val_result.nmse,
            "test_loss": None if test_result is None else test_result.loss,
            "test_nmse": None if test_result is None else test_result.nmse,
            "epochs_ran": len(history),
            "device": str(device),
        },
        output_dir / "summary.json",
    )
    logger.info("Wrote summary to %s.", output_dir / "summary.json")
    logger.info("Training complete.")


def main() -> None:
    args = build_parser().parse_args()
    run_training(args.config, args.log_level)


if __name__ == "__main__":
    main()
