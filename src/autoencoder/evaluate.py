from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.autoencoder.dataset import build_dataloaders
from src.autoencoder.engine import run_epoch
from src.autoencoder.model import build_autoencoder
from src.autoencoder.utils import ensure_output_dir, resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained native-complex T2 autoencoder checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt or last_model.pt.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser


def run_evaluation(checkpoint_path: str | Path, split: str, output_dir: str | Path | None, device_name: str) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(device_name)

    model = build_autoencoder(config).to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loaders = build_dataloaders(config)
    if len(loaders[split].dataset) == 0:
        raise RuntimeError(f"Split {split!r} is empty.")

    result = run_epoch(model, loaders[split], device=device)
    destination = ensure_output_dir(output_dir or checkpoint_path.parent)
    result.predictions.to_csv(destination / f"{split}_reconstruction_errors.csv", index=False)
    save_json({f"{split}_loss": result.loss, f"{split}_nmse": result.nmse}, destination / f"{split}_metrics.json")
    print(f"{split}_loss={result.loss:.6f} {split}_nmse={result.nmse:.6f}")


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args.checkpoint, args.split, args.output_dir, args.device)


if __name__ == "__main__":
    main()
