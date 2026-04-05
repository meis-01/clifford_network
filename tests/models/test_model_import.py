from __future__ import annotations

import unittest

from src.models.base import torch_available


class ModelImportTests(unittest.TestCase):
    def test_torch_availability_flag_is_boolean(self) -> None:
        self.assertIsInstance(torch_available(), bool)


if __name__ == "__main__":
    unittest.main()
