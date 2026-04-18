from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.classification.config import load_config
from src.classification.dataset import build_dataloaders
from src.classification.engine import build_pos_weight, run_epoch
from src.classification.features import build_train_quantile_stats, get_input_channels
from src.classification.manifest import build_paired_manifest, split_manifest
from src.classification.model import build_classifier
from src.classification.utils import ensure_output_dir, resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a paired T2/DWI classifier checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to evaluate. Defaults to output_dir/best_model.pt.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_dir = ensure_output_dir(config["training"]["output_dir"])
    checkpoint = args.checkpoint or (output_dir / "best_model.pt")

    manifest = build_paired_manifest(config)
    if manifest.empty:
        raise RuntimeError(
            "No paired T2/DWI samples were found. Check the configured roots, CSV files, and whether both modalities exist for the same patient/slice pairs."
        )
    train_manifest = split_manifest(manifest, config, "train")
    if str(config["features"]["normalization"]).lower() == "train_quantile":
        config["features"]["normalization_stats"] = build_train_quantile_stats(train_manifest, config["features"])

    loaders = build_dataloaders(manifest, config)
    if split_manifest(manifest, config, args.split).empty:
        raise RuntimeError(f"The requested split '{args.split}' is empty in the paired manifest.")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=build_pos_weight(train_manifest["label"].to_numpy()))

    device = resolve_device(config["training"]["device"])
    model = build_classifier(
        representation=config["features"]["representation"],
        in_channels=get_input_channels(config["features"]),
        channels=config["model"]["channels"],
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    result = run_epoch(model, loaders[args.split], device=device, criterion=criterion.to(device))
    result.predictions.to_csv(output_dir / f"{args.split}_predictions_eval.csv", index=False)
    save_json(
        {
            "split": args.split,
            "loss": result.loss,
            "auc": result.auc,
            "accuracy": result.accuracy,
            "checkpoint": str(Path(checkpoint).resolve()),
        },
        output_dir / f"{args.split}_metrics_eval.json",
    )
    print(f"split={args.split} loss={result.loss:.4f} auc={result.auc:.4f} accuracy={result.accuracy:.4f}")


if __name__ == "__main__":
    main()
