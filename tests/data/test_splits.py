from __future__ import annotations

import unittest

from src.data.splits import volume_wise_split


class VolumeSplitTests(unittest.TestCase):
    def test_volume_split_is_deterministic_and_disjoint(self) -> None:
        split = volume_wise_split([f"vol_{idx}" for idx in range(10)], seed=3)
        self.assertEqual(set(split["train"]) & set(split["val"]), set())
        self.assertEqual(set(split["train"]) & set(split["test"]), set())
        self.assertEqual(set(split["val"]) & set(split["test"]), set())
        self.assertEqual(sum(len(part) for part in split.values()), 10)


if __name__ == "__main__":
    unittest.main()
