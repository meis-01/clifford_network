from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.autoencoder.dataset import build_dataloaders
from src.autoencoder.engine import run_epoch
from src.autoencoder.logging_utils import setup_logger
from src.autoencoder.model import build_autoencoder
from src.autoencoder.utils import ensure_output_dir, resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained native-complex T2 autoencoder checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt or last_model.pt.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run_evaluation(
    checkpoint_path: str | Path,
    split: str,
    output_dir: str | Path | None,
    device_name: str,
    log_level: str = "INFO",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train first with: python -m src.autoencoder.train --config configs/t2_complex_autoencoder.yaml"
        )
    destination = ensure_output_dir(output_dir or checkpoint_path.parent)
    logger = setup_logger("autoencoder.evaluate", destination / f"evaluate_{split}.log", level=getattr(logging, log_level))
    logger.info("Starting native-complex autoencoder evaluation.")
    logger.info("Checkpoint path: %s", checkpoint_path.resolve())
    logger.info("Output directory: %s", destination.resolve())
    logger.info("Requested split: %s", split)

    logger.info("Loading checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(device_name)
    logger.info("Resolved device: %s.", device)

    logger.info("Building model and loading weights.")
    model = build_autoencoder(config).to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model parameters: %d", parameter_count)
    logger.info("Building dataloaders.")
    loaders = build_dataloaders(config, logger=logger)
    if len(loaders[split].dataset) == 0:
        raise RuntimeError(f"Split {split!r} is empty.")

    result = run_epoch(
        model,
        loaders[split],
        device=device,
        logger=logger,
        phase=f"evaluate {split}",
        log_interval=int(config["training"].get("log_interval", 25)),
    )
    result.predictions.to_csv(destination / f"{split}_reconstruction_errors.csv", index=False)
    save_json({f"{split}_loss": result.loss, f"{split}_nmse": result.nmse}, destination / f"{split}_metrics.json")
    logger.info("Wrote reconstruction errors to %s.", destination / f"{split}_reconstruction_errors.csv")
    logger.info("Wrote metrics to %s.", destination / f"{split}_metrics.json")
    logger.info("%s_loss=%.6f %s_nmse=%.6f", split, result.loss, split, result.nmse)
    logger.info("Evaluation complete.")


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args.checkpoint, args.split, args.output_dir, args.device, args.log_level)


if __name__ == "__main__":
    main()
