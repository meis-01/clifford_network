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
SUPPORTED_MODALITIES = {"t2", "dwi"}


def _configured_modalities(config: dict[str, Any]) -> list[str]:
    modalities = config.get("features", {}).get("modalities", ["t2", "dwi"])
    if isinstance(modalities, str):
        modalities = [modalities]
    normalized = [str(modality).lower() for modality in modalities]
    unsupported = set(normalized) - SUPPORTED_MODALITIES
    if unsupported:
        raise ValueError(f"Unsupported modalities: {sorted(unsupported)}")
    if not normalized:
        raise ValueError("At least one modality must be configured.")
    return normalized


def _required_manifest_columns(modalities: list[str]) -> set[str]:
    required = {"fastmri_pt_id", "slice", "data_split", "label"}
    required.update(f"{modality}_path" for modality in modalities)
    return required


def _validate_existing_manifest(frame: pd.DataFrame, modalities: list[str]) -> pd.DataFrame:
    missing = _required_manifest_columns(modalities) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
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
    modalities = _configured_modalities(config)
    manifest_path = data_config["paired_manifest_csv"]
    if manifest_path is not None:
        paired = _validate_existing_manifest(pd.read_csv(manifest_path), modalities)
    else:
        missing_label_csvs = [
            modality for modality in modalities if data_config[f"{modality}_labels_csv"] is None
        ]
        if missing_label_csvs:
            raise ValueError(
                "Either paired_manifest_csv or the selected modality label CSVs must be configured. "
                f"Missing: {missing_label_csvs}"
            )

        modality_manifests = {
            modality: _build_modality_frame(
                pd.read_csv(data_config[f"{modality}_labels_csv"]),
                data_config[f"{modality}_root"],
                modality,
                data_config,
            )
            for modality in modalities
        }

        if len(modalities) == 1:
            paired = modality_manifests[modalities[0]]
        else:
            paired = modality_manifests[modalities[0]]
            for modality in modalities[1:]:
                paired = paired.merge(
                    modality_manifests[modality],
                    on=["fastmri_pt_id", "slice", "data_split"],
                    how="inner",
                    suffixes=("", f"_{modality}"),
                )

            label_columns = [column for column in paired.columns if column == "label" or column.startswith("label_")]
            if len(label_columns) > 1:
                label_values = paired.loc[:, label_columns]
                paired["label_disagreement"] = label_values.nunique(axis=1) > 1
                if paired["label_disagreement"].any() and not data_config["allow_label_disagreement"]:
                    disagreements = int(paired["label_disagreement"].sum())
                    raise ValueError(f"Found {disagreements} label disagreements in the paired manifest.")

                paired["label"] = label_values.max(axis=1).astype(np.int64)
                paired = paired.drop(columns=[column for column in label_columns if column != "label"])

    if data_config["drop_missing"]:
        path_columns = [f"{modality}_path" for modality in modalities]
        existing = np.logical_and.reduce(
            [paired[path_column].map(lambda value: Path(value).exists()) for path_column in path_columns]
        )
        paired = paired[
            existing
        ].copy()

    paired["slice_index"] = paired["slice"].astype(int) - 1
    paired["sample_id"] = paired["fastmri_pt_id"].astype(str) + "_slice_" + paired["slice"].astype(str)
    paired = paired.sort_values(["data_split", "fastmri_pt_id", "slice"]).reset_index(drop=True)
    return paired


def split_manifest(manifest: pd.DataFrame, config: dict[str, Any], split_name: str) -> pd.DataFrame:
    split_value = config["data"]["split_names"][split_name]
    return manifest.loc[manifest["data_split"] == split_value].reset_index(drop=True)
