from __future__ import annotations

import unittest

import numpy as np

from src.reconstruction.pipeline import reconstruct_t2_rss


class SmokeIntegrationTests(unittest.TestCase):
    def test_reconstruction_pipeline_runs_on_synthetic_input(self) -> None:
        rng = np.random.default_rng(5)
        kspace = rng.standard_normal((1, 2, 3, 32, 24)) + 1j * rng.standard_normal((1, 2, 3, 32, 24))
        calib = rng.standard_normal((2, 3, 32, 16)) + 1j * rng.standard_normal((2, 3, 32, 16))
        header = """
        <ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
          <encoding>
            <encodedSpace>
              <matrixSize>
                <x>32</x>
                <y>32</y>
                <z>1</z>
              </matrixSize>
            </encodedSpace>
            <reconSpace>
              <matrixSize>
                <x>32</x>
                <y>32</y>
                <z>1</z>
              </matrixSize>
            </reconSpace>
          </encoding>
        </ismrmrdHeader>
        """.strip()

        image = reconstruct_t2_rss(kspace=kspace, calib_data=calib, hdr=header, slice_idx=0)
        self.assertEqual(image.shape, (32, 32))


if __name__ == "__main__":
    unittest.main()
