from __future__ import annotations

import unittest

import torch

from src.classification.model import ComplexValuedClassifier, build_classifier


class ModelVariantTests(unittest.TestCase):
    def test_real_classifier_forward_shape(self) -> None:
        model = build_classifier("real", in_channels=2, channels=(8, 16), dropout=0.1)
        batch = torch.randn(3, 2, 64, 64)
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (3,))

    def test_complex_classifier_forward_shape(self) -> None:
        model = build_classifier("complex", in_channels=2, channels=(8, 16), dropout=0.1)
        batch = torch.randn(3, 2, 64, 64, dtype=torch.complex64)
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertTrue(torch.is_floating_point(logits))

    def test_complex_classifier_uses_complex_parameters(self) -> None:
        model = ComplexValuedClassifier(in_channels=2, channels=(8, 16), dropout=0.1)
        self.assertTrue(any(parameter.is_complex() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()