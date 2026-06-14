from __future__ import annotations

import torch

from clifford_network.initialization import initialize_tensor


def test_structured_preserve_initializes_complex_matrix() -> None:
    weight = torch.empty(8, 4, dtype=torch.complex64)
    initialize_tensor(weight, "structured_preserve")
    assert weight.shape == (8, 4)
    assert weight.dtype == torch.complex64
    assert torch.isfinite(weight.real).all()
    assert torch.isfinite(weight.imag).all()


def test_trabelsi_initializes_complex_matrix() -> None:
    weight = torch.empty(8, 4, dtype=torch.complex64)
    initialize_tensor(weight, "trabelsi")
    assert weight.abs().mean() > 0
