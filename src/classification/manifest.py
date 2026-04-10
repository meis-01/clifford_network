from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PAIRED_COLUMNS = {
    "fastmri_pt_id",
    "slice",
    "data_split",
    "label",
    "t2_path",
    "dwi_path",
}


def _validate_existing_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PAIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Paired manifest is missing required columns: {sorted(missing)}")
    return frame.copy()


def _build_modality_frame(frame: pd.DataFrame, root: Path, prefix: str, config: dict[str, Any]) -> pd.DataFrame:
    patient_col = config["patient_id_column"]
    slice_col = config["slice_column"]
    split_col = config["split_column"]
    label_col = config["label_column"]
    folder_col = config["folder_column"]
    file_col = config["file_column"]

    required = {patient_col, slice_col, split_col, label_col, folder_col, file_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{prefix} labels CSV is missing columns: {sorted(missing)}")

    subset = frame.loc[:, [patient_col, slice_col, split_col, label_col, folder_col, file_col]].copy()
    subset["label"] = (subset[label_col].astype(float) > float(config["positive_threshold"])) .astype(np.int64)
    subset[f"{prefix}_path"] = [
        str((root / Path(folder) / Path(filename)).resolve())
        for folder, filename in zip(subset[folder_col], subset[file_col], strict=False)
    ]
    subset = subset.rename(
        columns={
            patient_col: "fastmri_pt_id",
            slice_col: "slice",
            split_col: "data_split",
        }
    )
    return subset[["fastmri_pt_id", "slice", "data_split", "label", f"{prefix}_path"]]


def build_paired_manifest(config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    manifest_path = data_config["paired_manifest_csv"]
    if manifest_path is not None:
        paired = _validate_existing_manifest(pd.read_csv(manifest_path))
    else:
        if data_config["t2_labels_csv"] is None or data_config["dwi_labels_csv"] is None:
            raise ValueError("Either paired_manifest_csv or both modality label CSVs must be configured.")

        t2_frame = pd.read_csv(data_config["t2_labels_csv"])
        dwi_frame = pd.read_csv(data_config["dwi_labels_csv"])
        t2_manifest = _build_modality_frame(t2_frame, data_config["t2_root"], "t2", data_config)
        dwi_manifest = _build_modality_frame(dwi_frame, data_config["dwi_root"], "dwi", data_config)

        paired = t2_manifest.merge(
            dwi_manifest,
            on=["fastmri_pt_id", "slice", "data_split"],
            how="inner",
            suffixes=("_t2", "_dwi"),
        )
        paired["label_disagreement"] = paired["label_t2"] != paired["label_dwi"]
        if paired["label_disagreement"].any() and not data_config["allow_label_disagreement"]:
            disagreements = int(paired["label_disagreement"].sum())
            raise ValueError(f"Found {disagreements} T2/DWI label disagreements in the paired manifest.")

        paired["label"] = np.maximum(paired["label_t2"], paired["label_dwi"]).astype(np.int64)
        paired = paired.drop(columns=["label_t2", "label_dwi"])

    if data_config["drop_missing"]:
        paired = paired[
            paired["t2_path"].map(lambda value: Path(value).exists())
            & paired["dwi_path"].map(lambda value: Path(value).exists())
        ].copy()

    paired["slice_index"] = paired["slice"].astype(int) - 1
    paired["sample_id"] = paired["fastmri_pt_id"].astype(str) + "_slice_" + paired["slice"].astype(str)
    paired = paired.sort_values(["data_split", "fastmri_pt_id", "slice"]).reset_index(drop=True)
    return paired


def split_manifest(manifest: pd.DataFrame, config: dict[str, Any], split_name: str) -> pd.DataFrame:
    split_value = config["data"]["split_names"][split_name]
    return manifest.loc[manifest["data_split"] == split_value].reset_index(drop=True)