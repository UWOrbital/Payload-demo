import argparse
import io
import math
from typing import Generator

import cv2
import numpy as np
from PIL import Image
from scipy.fftpack import idct

from decode import tables

class Bitstream:
    def __init__(self, file: io.FileIO):
        self.file = file
        self.leftover = ''
        self._empty = False

    def read(self, n: int=1) -> str:
        leftover_size = len(self.leftover)
        if leftover_size >= n:
            output = self.leftover[:n]
            self.leftover = self.leftover[n:]
            return output
        required_bits = n - leftover_size
        raw = ''.join([
            format(byte, '08b')
            for byte in self.file.read(math.ceil(required_bits / 8.))
        ])
        output = self.leftover + raw[:required_bits]
        self.leftover = raw[required_bits:]
        if len(self.leftover) == 0:
            out = self.file.read(1)
            if len(out) == 0:
                self._empty = True
            else:
                self.leftover += format(out[0], '08b')
        return output
    
    def empty(self):
        return self._empty
    
    @staticmethod
    def twos_complement(s: str) -> int:
        if s[0] == '0':
            complement = ''.join(['1' if x == '0' else '0' for x in s])
            return -int(complement, 2)
        return int(s, 2)

def decode_huffman(huffman_encoded: str, table: str) -> str:
    """Map huffman encoded bits to binary.

    Args:
        huffman_encoded (str): bits searched for in huffman codes
        table (str): huffman table to search

    Returns:
        str: string representation of byte
    """
    code_idx = tables.HUFFMAN_TABLES[table]["codes"].index(huffman_encoded)
    symbol = tables.HUFFMAN_TABLES[table]["symbols"][code_idx]
    if '_DC_' in table:
        binary_repr = format(symbol, 'b')
    else:
        binary_repr = format(symbol, '08b')
    return binary_repr

def interpret_symbol(stream: Bitstream, table: str) -> "tuple[int, int]":
    huffman_encoded = ''
    while True:
        huffman_encoded += stream.read(1)
        if huffman_encoded in tables.HUFFMAN_TABLES[table]["codes"]:
            break
    binary_repr = decode_huffman(huffman_encoded, table)
    if '_DC_' in table:
        return Bitstream.twos_complement(binary_repr)
    return int(binary_repr[:4], 2), int(binary_repr[4:], 2)

class _ZigZagWalker:
    """Helper class to walk in zigzag pattern through array
    """
    direction = None
    target = None
    size = None
    def __init__(self, side_length):
        self.size = side_length
        self.direction = 0
        self.target = [0, 0]

    def _is_top_edge(self):
        return self.target[0] == 0 or self.target[1] == self.size - 1

    def _is_bottom_edge(self):
        return self.target[1] == 0 or self.target[0] == self.size - 1
    
    def _not_edge_step(self):
        self.target[0] -= 1 * self.direction
        self.target[1] += 1 * self.direction
    
    def _turn_down(self):
        self.direction = -1
        if self.target[1] == self.size - 1:
            self.target[0] += 1
            return
        self.target[1] += 1
        
    def _turn_up(self):
        self.direction = 1
        if self.target[0] == self.size - 1:
            self.target[1] += 1
            return
        self.target[0] += 1

    def _update_target(self):
        # target is on edge
        if self.direction == 0:
            if self._is_top_edge():
                self._turn_down()
                return
            self._turn_up()
            return
        # target is not on edge
        if self.direction == 1:
            self._not_edge_step()
            if self._is_top_edge():
                self.direction = 0
            return
        self._not_edge_step()
        if self._is_bottom_edge():
            self.direction = 0

    def get_iterable(self) -> Generator[int, None, None]:
        """Get generator that yields indices of square 2d array in zigzag order

        Yields:
            Generator[int, None, None]: zigzag generator
        """
        for _ in range(self.size ** 2):
            yield tuple(self.target)
            self._update_target()

def unzigzag(a: list) -> np.ndarray:
    """Given an array of elements, unzigzag them into square 2d array

    Args:
        a (list): 1d array of elements

    Returns:
        np.ndarray: 2d square array
    """
    size = int(math.sqrt(len(a)))
    result = np.empty([size] * 2, dtype=object)
    walker = _ZigZagWalker(int(math.sqrt(len(a)))).get_iterable()
    for _, x in enumerate(a):
        result[next(walker)] = x
    return result

def _read_channel(stream: Bitstream, table: str) -> "list[int]":
    data = []
    while True:
        zeros, data_length = interpret_symbol(stream, table)
        if data_length == 0:
            data.extend([0] * (63 - len(data)))
            break
        data.extend([0] * (zeros))
        data.append(Bitstream.twos_complement(stream.read(data_length)))
    return data

def quantize(channel, quant_table):
    return np.multiply(
        channel, np.reshape(tables.QUANTIZATION_TABLES[quant_table], (64)))

def decode_mcu(stream: Bitstream, last_dc) -> "np.ndarray, np.ndarray, np.ndarray":
    y = idct(quantize([
            last_dc[0] + interpret_symbol(stream, 'LUM_DC_TABLE'),
            *_read_channel(stream, 'LUM_AC_TABLE')], 'LUM'))
    cb = idct(quantize([
            last_dc[1] + interpret_symbol(stream, 'COL_DC_TABLE'),
            *_read_channel(stream, 'COL_AC_TABLE')], 'COL'))
    cr = idct(quantize([
            last_dc[2] + interpret_symbol(stream, 'COL_DC_TABLE'),
            *_read_channel(stream, 'COL_AC_TABLE')], 'COL'))
    return y, cr, cb

def decode(stream: Bitstream) -> np.ndarray:
    """Decode stream to 2d array of rgb pixels

    Args:
        stream (io.BytesIO): contains jpeg encoded bits

    Returns:
        np.ndarray: 2d array of rgb pixels
    """
    data = []
    last_dc = [0, 0, 0]
    while not stream.empty():
        mcu_zigzag = list(zip(*decode_mcu(stream, last_dc)))
        last_dc = mcu_zigzag[0]
        mcu = unzigzag(mcu_zigzag)
        data.append(mcu)
    side = math.sqrt(len(data))
    ycbcr_img = np.reshape(data, (8 * side, 8 * side, 3))
    rgb_img = cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2RGB)
    return rgb_img

def load_codes(stream: io.FileIO):
    bstream = Bitstream(stream)
    for _, table in tables.HUFFMAN_TABLES.items():
        table["codes"] = [
            bstream.read(length)
            for length, count in enumerate(table["offsets"])
            for i in range(count)
        ]
    assert bstream.read(1) == ''
    assert tables.HUFFMAN_TABLES[list(tables.HUFFMAN_TABLES.keys())[-1]]['codes'][-1] != ''

def decoder_arguments():
    parser = argparse.ArgumentParser(
        description="Decode UWOrbital Payload JPEG Compression"
    )
    parser.add_argument(
        "filename", type=str, help="name of binary file to decode"
    )
    parser.add_argument(
        "--codes_filename", "-c",
        type=str,
        default="codes.bin",
        help="name of binary file containing huffman codes"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = decoder_arguments()
    with open(args.codes_filename, 'rb') as codes_file:
        load_codes(codes_file)
    with open(args.filename, 'rb') as data_file:
        data_stream = Bitstream(data_file)
    img_arr = decode(data_stream)
    img = Image.fromarray(img_arr)
    img.show()
