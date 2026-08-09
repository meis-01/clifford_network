from __future__ import annotations

import torch

from clifford_network.initialization import initialize_model, initialize_tensor


def test_structured_preserve_initializes_complex_matrix() -> None:
    weight = torch.empty(8, 4, dtype=torch.complex64)
    initialize_tensor(weight, "structured_preserve")
    assert weight.shape == (8, 4)
    assert weight.dtype == torch.complex64
    assert torch.isfinite(weight.real).all()
    assert torch.isfinite(weight.imag).all()


def test_structured_preserve_gain_one_recovers_original_initializer() -> None:
    original = torch.empty(8, 4, dtype=torch.complex64)
    explicit_gain_one = torch.empty_like(original)

    torch.manual_seed(7)
    initialize_tensor(original, "structured_preserve")
    torch.manual_seed(7)
    initialize_tensor(explicit_gain_one, "structured_preserve", gain=1.0)

    assert torch.equal(original, explicit_gain_one)


def test_structured_preserve_configurable_gain_scales_identity_path() -> None:
    weight = torch.empty(8, 4, dtype=torch.complex64)
    initialize_tensor(weight, "structured_preserve", alpha=0.0, gain=1.05)

    row_idx = torch.arange(8)
    col_idx = row_idx % 4
    assert torch.allclose(weight[row_idx, col_idx], torch.full((8,), 1.05 + 0.0j))
    weight[row_idx, col_idx] = 0.0
    assert torch.count_nonzero(weight) == 0


def test_structured_preserve_model_gain_reaches_maximum_at_last_layer() -> None:
    model = torch.nn.Module()
    model.weights = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.empty(4, 4, dtype=torch.complex64)) for _ in range(3)]
    )

    initialize_model(model, "structured_preserve", alpha=0.0, gain=1.2)

    expected_gains = [1.0, 1.1, 1.2]
    diagonal = torch.arange(4)
    for weight, expected_gain in zip(model.weights, expected_gains, strict=True):
        assert torch.allclose(weight[diagonal, diagonal], torch.full((4,), expected_gain + 0.0j))


def test_trabelsi_initializes_complex_matrix() -> None:
    weight = torch.empty(8, 4, dtype=torch.complex64)
    initialize_tensor(weight, "trabelsi")
    assert weight.abs().mean() > 0
