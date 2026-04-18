from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.discovery import discover_volume_files
from src.utils.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fastMRI pipeline scaffold")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_yaml(args.config)

    data_root = Path(config["data"]["root"]).expanduser()
    if not data_root.exists():
        print(f"Configured data root does not exist: {data_root}")
        return

    records = discover_volume_files(data_root)
    counts = Counter(record.modality for record in records)
    print(f"Discovered {len(records)} volume files under {data_root}")
    for modality, count in sorted(counts.items()):
        print(f"  {modality}: {count}")


if __name__ == "__main__":
    main()
