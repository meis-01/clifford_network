from __future__ import annotations

import unittest

import numpy as np

from src.reconstruction.pipeline import reconstruct_t2


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

        image = reconstruct_t2(kspace=kspace, calib_data=calib, hdr=header, slice_idx=0, kernel_size=(5, 5),return_type="rss" )
        self.assertEqual(image.shape, (32, 32))

    def test_reconstruction_pipeline_returns_reconstructed_kspace(self) -> None:
        rng = np.random.default_rng(9)
        kspace = np.zeros((2, 2, 3, 32, 24), dtype=np.complex128)
        calib = rng.standard_normal((2, 3, 32, 16)) + 1j * rng.standard_normal((2, 3, 32, 16))
        kspace[:, :, :, :, ::2] = rng.standard_normal((2, 2, 3, 32, 12)) + 1j * rng.standard_normal((2, 2, 3, 32, 12))
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

        reconstruction = reconstruct_t2(kspace=kspace, calib_data=calib, hdr=header)

        self.assertEqual(reconstruction["reconstruction_rss"].shape, (2, 32, 32))
        self.assertEqual(reconstruction["reconstruction_kspace"].shape, (2, 2, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
