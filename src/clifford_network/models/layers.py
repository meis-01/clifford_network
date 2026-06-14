from __future__ import annotations

import math

import torch
from torch import nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.complex64)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(max(1, self.in_features))
        with torch.no_grad():
            self.weight.real.normal_(0.0, std)
            self.weight.imag.normal_(0.0, std)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = values @ self.weight.transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output
