from __future__ import annotations

import unittest

from src.reconstruction.header import parse_ismrmrd_header


HEADER = """
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <encoding>
    <encodedSpace>
      <matrixSize>
        <x>640</x>
        <y>512</y>
        <z>1</z>
      </matrixSize>
    </encodedSpace>
    <reconSpace>
      <matrixSize>
        <x>320</x>
        <y>320</y>
        <z>1</z>
      </matrixSize>
    </reconSpace>
  </encoding>
</ismrmrdHeader>
""".strip()


class HeaderParsingTests(unittest.TestCase):
    def test_namespace_safe_header_parsing(self) -> None:
        header = parse_ismrmrd_header(HEADER)
        self.assertEqual(header.encoded_matrix, (640, 512, 1))
        self.assertEqual(header.recon_matrix, (320, 320, 1))


if __name__ == "__main__":
    unittest.main()
