from __future__ import annotations

import argparse
from pathlib import Path

from t2_samples_preparation import T2SliceConfig, generate_t2_slices
LABEL_CSV_DEFAULT = Path("data/labels/t2_slice_level_labels.csv")
# DATA_ROOT_DEFAULT = Path("F:\\fastmri_prostate\\T2")
# OUTPUT_DIR_DEFAULT = Path("F:\\fastmri_prostate\\T2_Slices")
DATA_ROOT_DEFAULT = Path("data")
OUTPUT_DIR_DEFAULT = Path("data/output_complex")
OUTPUT_SIZE_DEFAULT = (320, 320)
COMPRESSED_COILS_DEFAULT = -1
KERNEL_SIZE_DEFAULT = (5, 5)
POSITIVE_THRESHOLD_DEFAULT = 2.0
LIMIT_DEFAULT = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate standalone T2 complex k-space slice tensors and a manifest.")
    parser.add_argument("--labels-csv", type=Path, default=LABEL_CSV_DEFAULT, help="T2 slice-level label CSV.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT, help="Root directory used with folder/fastmri_rawfile columns.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT, help="Directory for features, manifest.csv, and config JSON.")
    parser.add_argument("--output-size", type=int, nargs=2, default=OUTPUT_SIZE_DEFAULT, metavar=("ROWS", "COLS"))
    parser.add_argument("--compressed-coils", type=int, default=COMPRESSED_COILS_DEFAULT)
    parser.add_argument("--kernel-size", type=int, nargs=2, default=KERNEL_SIZE_DEFAULT, metavar=("KX", "KY"))
    parser.add_argument("--positive-threshold", type=float, default=POSITIVE_THRESHOLD_DEFAULT)
    parser.add_argument("--limit", type=int, default=LIMIT_DEFAULT, help="Optional first-N rows limit for smoke tests.")
    parser.add_argument("--keep-missing", action="store_true", help="Keep manifest rows even when raw files are missing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = T2SliceConfig(
        labels_csv=args.labels_csv.resolve(),
        data_root=args.data_root.resolve(),
        output_dir=args.output_dir.resolve(),
        output_size=tuple(args.output_size),
        compressed_coils=int(args.compressed_coils),
        kernel_size=tuple(args.kernel_size),
        positive_threshold=float(args.positive_threshold),
        drop_missing=not args.keep_missing,
        limit=args.limit,
    )
    manifest = generate_t2_slices(config)
    print(f"wrote {len(manifest)} rows to {config.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
