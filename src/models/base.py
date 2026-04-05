from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None


def torch_available() -> bool:
    return torch is not None and nn is not None


class BaseModel(nn.Module if nn is not None else object):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        if not torch_available():
            raise RuntimeError("PyTorch is required to instantiate models. Install requirements.txt first.")
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

    def forward(self, inputs: Any) -> Any:
        raise NotImplementedError
