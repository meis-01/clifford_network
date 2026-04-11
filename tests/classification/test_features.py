from __future__ import annotations

import unittest

from src.classification.features import get_input_channels


class FeatureConfigTests(unittest.TestCase):
    def test_real_representation_uses_two_channels(self) -> None:
        self.assertEqual(get_input_channels({"representation": "real"}), 2)

    def test_complex_representation_uses_two_complex_channels(self) -> None:
        self.assertEqual(get_input_channels({"representation": "complex"}), 2)

    def test_invalid_representation_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_input_channels({"representation": "polar"})


if __name__ == "__main__":
    unittest.main()