"""File IO helpers for experiment configuration and results.

The helpers here centralize YAML, JSON, and CSV record persistence while making
sure parent directories exist before outputs are written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return an empty dict for empty files."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def save_yaml(payload: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to YAML, creating parent directories first."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to formatted JSON, creating parent directories first."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_records(records: list[dict[str, Any]], path: str | Path) -> None:
    """Persist a list of record dictionaries as a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    """Read a CSV file if present, otherwise return an empty DataFrame."""
    input_path = Path(path)
    if not input_path.exists():
        return pd.DataFrame()
    return pd.read_csv(input_path)
