from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.discovery import discover_volume_files
from src.reconstruction.pipeline import reconstruct_t2_rss
from src.utils.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fastMRI pipeline scaffold")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a synthetic reconstruction smoke test.",
    )
    return parser


def synthetic_header(encoded_y: int) -> str:
    return f"""
    <ismrmrdHeader xmlns=\"http://www.ismrm.org/ISMRMRD\">
      <encoding>
        <encodedSpace>
          <matrixSize>
            <x>64</x>
            <y>{encoded_y}</y>
            <z>1</z>
          </matrixSize>
        </encodedSpace>
        <reconSpace>
          <matrixSize>
            <x>64</x>
            <y>{encoded_y}</y>
            <z>1</z>
          </matrixSize>
        </reconSpace>
      </encoding>
    </ismrmrdHeader>
    """.strip()


def run_smoke_test() -> None:
    rng = np.random.default_rng(7)
    kspace = rng.standard_normal((2, 3, 4, 64, 48)) + 1j * rng.standard_normal((2, 3, 4, 64, 48))
    calib = rng.standard_normal((3, 4, 64, 24)) + 1j * rng.standard_normal((3, 4, 64, 24))

    image = reconstruct_t2_rss(
        kspace=kspace,
        calib_data=calib,
        hdr=synthetic_header(encoded_y=64),
        slice_idx=1,
        kernel_size=(5, 5),
    )
    print(f"Smoke test succeeded. Reconstructed image shape: {image.shape}")


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)

    if args.smoke_test:
        run_smoke_test()
        return

    data_root = Path(config["data"]["root"]).expanduser()
    if not data_root.exists():
        print(f"Configured data root does not exist: {data_root}")
        print("Use --smoke-test to validate the code path without the dataset.")
        return

    records = discover_volume_files(data_root)
    counts = Counter(record.modality for record in records)
    print(f"Discovered {len(records)} volume files under {data_root}")
    for modality, count in sorted(counts.items()):
        print(f"  {modality}: {count}")


if __name__ == "__main__":
    main()
