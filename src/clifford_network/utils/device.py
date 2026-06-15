"""Device selection helper.

This module converts config-level device requests into `torch.device` objects
and enforces CUDA availability when it is requested explicitly.
"""

from __future__ import annotations

import torch


def resolve_device(requested: str | None) -> torch.device:
    """Resolve `auto`, explicit CPU, or explicit CUDA device strings."""
    name = (requested or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)
