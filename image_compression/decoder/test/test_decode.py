import io
import unittest
from unittest.mock import patch

import numpy as np

import decode

class TestDecode(unittest.TestCase):
    @patch.dict("tables.HUFFMAN_TABLES", {
        "TEST_TABLE": {"codes": [0, 0, 0, 0, 0x0f, 0, 0, 0],
            "symbols": [0, 0, 0, 0, 0x51, 0, 0, 0]}})
    def test_decode_huffman(self):
        actual = decode.decode_huffman('1111', "TEST_TABLE")
        self.assertEqual(actual, "01010001")

    @patch.dict("tables.HUFFMAN_TABLES", {"test_table": {"codes": ['110']}})
    @patch("decode.decode_huffman")
    def test_interpret_normal_symbol(self, m_huffman_decode):
        m_huffman_decode.return_value = '01011101'
        actual = decode.interpret_symbol(
            decode.Bitstream(io.BytesIO(b'\xD5')), "test_table"
        )
        self.assertEqual(actual, (5, 13))
        m_huffman_decode.assert_called_once_with(
            "110", "test_table"
        )
        
    def test_unzigzag(self):
        original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        actual = decode.unzigzag(original_array)
        expected = np.array([
            [1, 2, 6],
            [3, 5, 7],
            [4, 8, 9]
        ])
        np.testing.assert_equal(actual, expected)

class TestBitstream(unittest.TestCase):
    def test_read(self):
        stream = decode.Bitstream(io.BytesIO(b'\x0b\x8d\x9c\xa6'))
        actual = stream.read(4)
        self.assertEqual(actual, '0000')
        actual = stream.read(2)
        self.assertEqual(actual, '10')
        actual = stream.read(5)
        self.assertEqual(actual, '11100')
        actual = stream.read()
        self.assertEqual(actual, '0')
        print(stream.read(100))
        
    def test_twos_complement(self):
        self.assertEqual(decode.Bitstream.twos_complement('1010'), 10)
        self.assertEqual(decode.Bitstream.twos_complement('01010'), -21)

if __name__ == '__main__':
    unittest.main()
