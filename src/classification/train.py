from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from src.classification.config import load_config
from src.classification.dataset import build_dataloaders
from src.classification.engine import build_pos_weight, run_epoch
from src.classification.manifest import build_paired_manifest, split_manifest
from src.classification.model import KSpaceClassifier
from src.classification.utils import ensure_output_dir, resolve_device, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a paired T2/DWI k-space classifier.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    return parser


def _is_better(metric_name: str, current_value: float, best_value: float) -> bool:
    if metric_name == "val_loss":
        return current_value < best_value
    return current_value > best_value


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    training_config = config["training"]
    output_dir = ensure_output_dir(training_config["output_dir"])
    set_seed(int(training_config["seed"]))

    manifest = build_paired_manifest(config)
    if manifest.empty:
        raise RuntimeError(
            "No paired T2/DWI samples were found. Check the configured roots, CSV files, and whether both modalities exist for the same patient/slice pairs."
        )
    manifest.to_csv(output_dir / "paired_manifest.csv", index=False)
    shutil.copy2(args.config, output_dir / "config.yaml")

    loaders = build_dataloaders(manifest, config)
    train_manifest = split_manifest(manifest, config, "train")
    val_manifest = split_manifest(manifest, config, "val")
    if train_manifest.empty or val_manifest.empty:
        raise RuntimeError("Training requires non-empty train and validation splits in the paired manifest.")
    pos_weight = build_pos_weight(train_manifest["label"].to_numpy())

    device = resolve_device(training_config["device"])
    model = KSpaceClassifier(
        in_channels=2,
        channels=config["model"]["channels"],
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(enabled=bool(training_config["amp"]) and device.type == "cuda")

    best_metric_name = str(training_config["monitor"])
    best_metric_value = float("inf") if best_metric_name == "val_loss" else float("-inf")
    best_checkpoint = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    patience = int(training_config["patience"])
    stale_epochs = 0

    for epoch in range(int(training_config["epochs"])):
        train_result = run_epoch(
            model,
            loaders["train"],
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=bool(training_config["amp"]) and device.type == "cuda",
        )
        val_result = run_epoch(model, loaders["val"], device=device, criterion=criterion)

        row = {
            "epoch": float(epoch),
            "train_loss": train_result.loss,
            "train_auc": train_result.auc,
            "train_accuracy": train_result.accuracy,
            "val_loss": val_result.loss,
            "val_auc": val_result.auc,
            "val_accuracy": val_result.accuracy,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_result.loss:.4f} train_auc={train_result.auc:.4f} "
            f"val_loss={val_result.loss:.4f} val_auc={val_result.auc:.4f}"
        )

        current_metric = row[best_metric_name]
        if _is_better(best_metric_name, current_metric, best_metric_value):
            best_metric_value = current_metric
            stale_epochs = 0
            torch.save(model.state_dict(), best_checkpoint)
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    val_result = run_epoch(model, loaders["val"], device=device, criterion=criterion)
    test_result = run_epoch(model, loaders["test"], device=device, criterion=criterion)

    val_result.predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_result.predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    save_json(
        {
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "val_loss": val_result.loss,
            "val_auc": val_result.auc,
            "val_accuracy": val_result.accuracy,
            "test_loss": test_result.loss,
            "test_auc": test_result.auc,
            "test_accuracy": test_result.accuracy,
            "epochs_ran": len(history),
        },
        output_dir / "summary.json",
    )
    save_json({"history": history}, output_dir / "history.json")


if __name__ == "__main__":
    main()