import argparse
import io
import logging
import math
from typing import Generator

import cv2
import numpy as np
from scipy.fftpack import idct

from decoder import tables

logging.basicConfig(format="%(levelname)s:%(module)s: %(message)s", level=logging.DEBUG)

SYMBOL_SIZE = 8
MCU_SIDE_LENGTH = 8
MCU_SIZE = MCU_SIDE_LENGTH ** 2
AC_SIZE = MCU_SIZE - 1
CHANNELS = 3

class Bitstream:
    def __init__(self, file: io.FileIO):
        self.file = file
        self.leftover = ''
        self._empty = False

    def read(self, n: int=1) -> str:
        """Read `n` bits from stream

        Args:
            n (int, optional): number of bits to read. Defaults to 1.

        Returns:
            str: binary number read from stream
        """
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
    
    def empty(self) -> bool:
        """Returns true if stream is empty

        Returns:
            bool: stream empty
        """
        return self._empty
    
    @staticmethod
    def twos_complement(s: str) -> int:
        """Apply twos complement to binary number

        Args:
            s (str): original binary number

        Raises:
            ValueError: `s` is empty string

        Returns:
            int: twos complement of `s`
        """
        if len(s) == 0:
            raise ValueError("binary string must not be empty")
        if s == '0':
            return 0
        if s[0] == '0':
            complement = ''.join(['1' if x == '0' else '0' for x in s])
            return -int(complement, 2)
        return int(s, 2)

def get_symbol(code: str, table: str) -> str:
    """Translate code to symbol

    Args:
        code (str): code to translate
        table (str): table to find code in

    Returns:
        str: symbol
    """
    code_idx = tables.HUFFMAN_TABLES[table]["codes"].index(code)
    symbol = tables.HUFFMAN_TABLES[table]["symbols"][code_idx]
    return symbol

def _interpret_symbol(symbol: int) -> "int, int":
    """Parse zero run length and data length from symbol

    Args:
        symbol (int): symbol

    Returns:
        int, int: zero run length, data length
    """
    binary_repr = format(symbol, '08b')
    return int(binary_repr[:4], 2), int(binary_repr[4:], 2)

def read_symbol(stream: Bitstream, table: str) -> "int, int":
    """Read code from stream and interpret corresponding symbol

    Args:
        stream (Bitstream): stream to read from
        table (str): table to look up code in

    Returns:
        int, int: zero run length, next value length
    """
    code = ''
    while code not in tables.HUFFMAN_TABLES[table]["codes"]:
        code += stream.read(1)
    symbol = get_symbol(code, table)
    return _interpret_symbol(symbol)

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

    def get_iterable(self) -> "Generator[tuple[int], None, None]":
        """Get generator that yields indices of square 2d array in zigzag order

        Yields:
            Generator[tuple[int], None, None]: zigzag generator
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

def read_ac(stream: Bitstream, table: str) -> "list[int]":
    """Read ac values from stream

    Args:
        stream (Bitstream): stream to read from
        table (str): table to look up codes in

    Returns:
        list[int]: ac values read from stream
    """
    data = []
    while True:
        zeros, data_length = read_symbol(stream, table)
        data.extend([0] * (zeros)) # apply 0 run
        if data_length == 0:
            if zeros == 0:
                data.extend([0] * (AC_SIZE - len(data)))
                break
            continue
        value = Bitstream.twos_complement(stream.read(data_length))
        data.append(value)
    assert len(data) == AC_SIZE, f"len(data) = {len(data)} != {AC_SIZE}"
    return data

def read_dc(stream: Bitstream, table: str) -> int:
    """Read DC symbol from stream

    Args:
        stream (Bitstream): stream to read
        table (str): table to look up symbol in

    Raises:
        ValueError: DC symbol has data length of 0

    Returns:
        int: DC value
    """
    zeros, data_length = read_symbol(stream, table)
    assert zeros == 0
    if data_length == 0:
        raise ValueError("Data length from DC Symbol should be non 0")
    value = Bitstream.twos_complement(stream.read(data_length))
    return value

def quantize(channel: "list[int]", quant_table: str) -> np.array:
    """Quantize color channel

    Args:
        channel (list[int]): original color channel
        quant_table (str): table to quantize with

    Returns:
        np.array: quantized color channel
    """
    return np.multiply(
        channel, np.reshape(tables.QUANTIZATION_TABLES[quant_table], (MCU_SIZE)))

def decode_mcu(
    stream: Bitstream,
    last_dc: "list[int]") -> "np.ndarray, np.ndarray, np.ndarray":
    """decode MCU from stream

    Args:
        stream (Bitstream): stream to read from
        last_dc (list[int]): dc values of last mcu

    Returns:
        np.ndarray, np.ndarray, np.ndarray: y, cr, cb color channels
    """
    y_raw = [read_dc(stream, 'LUM_DC_TABLE'), *read_ac(stream, 'LUM_AC_TABLE')]
    logging.debug(f'y_raw: {y_raw}')
    cb_raw = [read_dc(stream, 'COL_DC_TABLE'), *read_ac(stream, 'COL_AC_TABLE')]
    logging.debug(f'cb_raw: {cb_raw}')
    cr_raw = [read_dc(stream, 'COL_DC_TABLE'), *read_ac(stream, 'COL_AC_TABLE')]
    logging.debug(f'cr_raw: {cr_raw}')

    y_raw[0] += last_dc[0]
    cb_raw[0] += last_dc[1]
    cr_raw[0] += last_dc[2]
    
    idct_type = 2
    idct_norm='ortho'
    y = idct(quantize(y_raw, 'LUM'), type=idct_type, norm=idct_norm)
    cb = idct(quantize(cb_raw, 'COL'), type=idct_type, norm=idct_norm)
    cr = idct(quantize(cr_raw, 'COL'), type=idct_type, norm=idct_norm)
    return y, cr, cb

def trim_domain(a: np.ndarray, domain: tuple=None) -> np.ndarray:
    """Reduce domain of array to given domain

    Args:
        a (np.ndarray): array
        domain (tuple, optional): target domain. Defaults to None.

    Returns:
        np.ndarray: array within domain
    """
    if domain is None:
        domain = (0, 255)
    cp = a.copy()
    cp[cp < domain[0]] = 0
    cp[cp > domain[1]] = domain[1]
    return cp

def decode(stream: Bitstream, mcu_count: int) -> np.ndarray:
    """Decode stream to 2d array of rgb pixels

    Args:
        stream (io.BytesIO): contains jpeg encoded bits
        mcu_count (int): number of mcu's to read

    Returns:
        np.ndarray: 2d array of rgb pixels
    """
    data = []
    last_dc = [0, 0, 0]
    for i in range(mcu_count):
        mcu_zigzag = list(zip(*decode_mcu(stream, last_dc)))
        last_dc = mcu_zigzag[0]
        mcu = unzigzag(mcu_zigzag)
        mcu = [
            [list(pixel) for pixel in row]
            for row in mcu
        ]
        data.append(mcu)
        logging.debug(f'mcu {i + 1} done')
    side = math.floor(math.sqrt(len(data)))
    ycrcb_img = np.reshape(data, (MCU_SIDE_LENGTH * side, MCU_SIDE_LENGTH * side, CHANNELS))
    
    rgb_img = cv2.cvtColor(np.float32(ycrcb_img), cv2.COLOR_YCrCb2RGB)
    return rgb_img

def load_codes(stream: io.FileIO):
    """Load codes from code bin file

    Args:
        stream (io.FileIO): stream to read codes from
    """
    bstream = Bitstream(stream)
    for _, table in tables.HUFFMAN_TABLES.items():
        table["codes"] = [
            bstream.read(length)
            for length, count in enumerate(table["offsets"])
            for i in range(count)
        ]
    assert bstream.read(1) == ''
    assert tables.HUFFMAN_TABLES[list(tables.HUFFMAN_TABLES.keys())[-1]]['codes'][-1] != ''

def decoder_arguments() -> argparse.Namespace:
    """Parse arguments for decoder

    Returns:
        argparse.namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Decode UWOrbital Payload JPEG Compression"
    )
    parser.add_argument(
        "filename", type=str, help="name of binary file to decode"
    )
    parser.add_argument(
        "mcu_count",
        type=int,
        default=1,
        help="number of mcus in image"
    )
    parser.add_argument(
        "--codes_filename", "-c",
        type=str,
        default="decoder/codes.bin",
        help="name of binary file containing huffman codes"
    )
    args = parser.parse_args()
    return args
