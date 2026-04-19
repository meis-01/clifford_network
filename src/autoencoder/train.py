from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from src.autoencoder.config import load_config
from src.autoencoder.dataset import build_dataloaders
from src.autoencoder.engine import run_epoch, save_checkpoint
from src.autoencoder.model import build_autoencoder
from src.autoencoder.utils import ensure_output_dir, resolve_device, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a native-complex PyTorch autoencoder for T2 coil images.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    return parser


def run_training(config_path: str | Path) -> None:
    config_path = Path(config_path)
    config = load_config(config_path)
    training_config = config["training"]
    output_dir = ensure_output_dir(training_config["output_dir"])
    set_seed(int(training_config["seed"]))

    loaders = build_dataloaders(config)
    if len(loaders["train"].dataset) == 0 or len(loaders["val"].dataset) == 0:
        raise RuntimeError("Training requires non-empty train and validation split folders.")

    shutil.copy2(config_path, output_dir / "config.yaml")
    device = resolve_device(str(training_config["device"]))
    model = build_autoencoder(config).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    patience = int(training_config["patience"])
    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"

    for epoch in range(int(training_config["epochs"])):
        train_result = run_epoch(
            model,
            loaders["train"],
            device=device,
            optimizer=optimizer,
            grad_clip_norm=float(training_config["grad_clip_norm"]),
        )
        val_result = run_epoch(model, loaders["val"], device=device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_result.loss,
            "train_nmse": train_result.nmse,
            "val_loss": val_result.loss,
            "val_nmse": val_result.nmse,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_result.loss:.6f} train_nmse={train_result.nmse:.6f} "
            f"val_loss={val_result.loss:.6f} val_nmse={val_result.nmse:.6f}"
        )

        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            best_val_loss=best_val_loss,
        )
        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            stale_epochs = 0
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                best_val_loss=best_val_loss,
            )
        else:
            stale_epochs += 1

        save_json({"history": history}, output_dir / "history.json")
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_result = run_epoch(model, loaders["val"], device=device)
    test_result = run_epoch(model, loaders["test"], device=device) if len(loaders["test"].dataset) else None
    val_result.predictions.to_csv(output_dir / "val_reconstruction_errors.csv", index=False)
    if test_result is not None:
        test_result.predictions.to_csv(output_dir / "test_reconstruction_errors.csv", index=False)

    save_json(
        {
            "best_val_loss": best_val_loss,
            "val_loss": val_result.loss,
            "val_nmse": val_result.nmse,
            "test_loss": None if test_result is None else test_result.loss,
            "test_nmse": None if test_result is None else test_result.nmse,
            "epochs_ran": len(history),
            "device": str(device),
        },
        output_dir / "summary.json",
    )


def main() -> None:
    args = build_parser().parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
