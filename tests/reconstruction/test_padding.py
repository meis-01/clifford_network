from __future__ import annotations

import unittest

import numpy as np

from src.reconstruction.padding import compute_symmetric_padding, pad_kspace_phase


class PaddingTests(unittest.TestCase):
    def test_compute_padding(self) -> None:
        self.assertEqual(compute_symmetric_padding(10, 14), (2, 2))
        self.assertEqual(compute_symmetric_padding(11, 14), (1, 2))

    def test_pad_kspace_phase(self) -> None:
        kspace = np.zeros((2, 8, 10), dtype=np.complex64)
        padded = pad_kspace_phase(kspace, 14)
        self.assertEqual(padded.shape, (2, 8, 14))


if __name__ == "__main__":
    unittest.main()
