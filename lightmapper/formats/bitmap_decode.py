"""Decode H1A bitmap tags to RGBA, mip 0 of one bitmap.

Decoders use metadata from the tag reader's fields. decode_bitmap decodes only
plain 2D colour formats and refuses palettised bitmaps; decode_bitmap_alpha
also reads a palettised bump map's alpha. Swizzled, volume and cube bitmaps
are refused to avoid guessing their layout.
"""

import io
import struct

import numpy as np
from PIL import Image

from ..tagio.tagfile import Tag


FORMATS = {0: 'A8', 1: 'Y8', 2: 'AY8', 3: 'A8Y8', 6: 'R5G6B5', 8: 'A1R5G5B5', 9: 'A4R4G4B4',
           10: 'X8R8G8B8', 11: 'A8R8G8B8', 14: 'DXT1', 15: 'DXT3', 16: 'DXT5', 17: 'P8', 18: 'BC7'}
BYTES_PER_TEXEL = {'A8': 1, 'Y8': 1, 'AY8': 1, 'A8Y8': 2, 'R5G6B5': 2, 'A1R5G5B5': 2,
                   'A4R4G4B4': 2, 'X8R8G8B8': 4, 'A8R8G8B8': 4}
SUPPORTED = {0, 1, 2, 3, 6, 8, 9, 10, 11, 14, 15, 16, 18}
COMPRESSED = ('DXT1', 'DXT3', 'DXT5', 'BC7')
SWIZZLED = 0x8


class UnsupportedBitmap(ValueError):
    pass


def _dds(fourcc: bytes, width: int, height: int, data: bytes) -> np.ndarray:
    header = struct.pack('<4sIIIIIII44xII4sIIIIIIIIII', b'DDS ', 124, 0x1007, height, width, 0, 0, 0,
                         32, 4, fourcc, 0, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    if fourcc == b'DX10':
        # (BC7 has no legacy fourcc: the DX10 extension names DXGI_FORMAT_BC7_UNORM = 98)
        header += struct.pack('<IIIII', 98, 3, 0, 1, 0)
    return np.asarray(Image.open(io.BytesIO(header + data)).convert('RGBA'))


def _block_size(name: str, width: int, height: int) -> int:
    return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * (8 if name == 'DXT1' else 16)


def _decode(name: str, width: int, height: int, pixels: bytes) -> np.ndarray:
    if name in COMPRESSED:
        return _dds(b'DX10' if name == 'BC7' else name.encode(), width, height, pixels)
    if name in ('A8R8G8B8', 'X8R8G8B8'):
        # stored b, g, r, a
        out = np.frombuffer(pixels, np.uint8).reshape(height, width, 4)[:, :, [2, 1, 0, 3]].copy()
        if name == 'X8R8G8B8':
            out[:, :, 3] = 255
        return out
    if name in ('R5G6B5', 'A1R5G5B5', 'A4R4G4B4'):
        v = np.frombuffer(pixels, '<u2').reshape(height, width).astype(np.uint32)
        if name == 'R5G6B5':
            r, g, b = (v >> 11 & 31) * 255 // 31, (v >> 5 & 63) * 255 // 63, (v & 31) * 255 // 31
            a = np.full_like(r, 255)
        elif name == 'A1R5G5B5':
            a, r, g, b = ((v >> 15) * 255, (v >> 10 & 31) * 255 // 31,
                          (v >> 5 & 31) * 255 // 31, (v & 31) * 255 // 31)
        else:
            a, r, g, b = [(v >> s & 15) * 17 for s in (12, 8, 4, 0)]
        return np.dstack([r, g, b, a]).astype(np.uint8)
    v = np.frombuffer(pixels, np.uint8)
    if name == 'A8':
        return np.dstack([np.full((height, width), 255, np.uint8)] * 3 + [v.reshape(height, width)])
    if name == 'Y8':
        y = v.reshape(height, width)
        return np.dstack([y, y, y, np.full_like(y, 255)])
    if name == 'AY8':
        y = v.reshape(height, width)
        return np.dstack([y, y, y, y])
    pair = v.reshape(height, width, 2)
    return np.dstack([pair[:, :, 0]] * 3 + [pair[:, :, 1]])


def decode_bitmap(tag: Tag, index: int = 0) -> np.ndarray:
    bitmaps = tag.root.block('bitmaps')
    if not 0 <= index < len(bitmaps):
        raise UnsupportedBitmap(f'no bitmap {index}')
    bitmap = bitmaps[index]
    width, height = bitmap.get('width'), bitmap.get('height')
    code = bitmap.get('format')
    if code not in SUPPORTED:
        raise UnsupportedBitmap(f'format {code}')
    name = FORMATS[code]
    if bitmap.get('type') != 0 or bitmap.get('depth') > 1:
        raise UnsupportedBitmap('only 2D bitmaps are decoded')
    if bitmap.get('flags') & SWIZZLED:
        raise UnsupportedBitmap('swizzled bitmap')
    size = (_block_size(name, width, height) if name in COMPRESSED
            else BYTES_PER_TEXEL[name] * width * height)
    offset = bitmap.get('pixel_data_offset')
    pixels = tag.root.data('processed_pixel_data')[offset:offset + size]
    if len(pixels) < size or not width or not height:
        raise UnsupportedBitmap('pixel data is shorter than the bitmap')
    return _decode(name, width, height, pixels)


def decode_bitmap_alpha(tag: Tag, index: int = 0) -> np.ndarray:
    bitmaps = tag.root.block('bitmaps')
    if not 0 <= index < len(bitmaps):
        raise UnsupportedBitmap(f'no bitmap {index}')
    bitmap = bitmaps[index]
    if bitmap.get('format') != 17:
        return decode_bitmap(tag, index)[:, :, 3]
    width, height = bitmap.get('width'), bitmap.get('height')
    if bitmap.get('type') != 0 or bitmap.get('depth') > 1:
        raise UnsupportedBitmap('only 2D bitmaps are decoded')
    if bitmap.get('flags') & SWIZZLED:
        raise UnsupportedBitmap('swizzled bitmap')
    size = width * height
    offset = bitmap.get('pixel_data_offset')
    pixels = tag.root.data('processed_pixel_data')[offset:offset + size]
    if len(pixels) < size or not width or not height:
        raise UnsupportedBitmap('pixel data is shorter than the bitmap')
    # (the encoder writes 255 for alpha up to 0x80, bitmaps.c:895-898;
    # only palette entries 249-255 are transparent, bitmaps.c:320-353)
    indices = np.frombuffer(pixels, np.uint8).reshape(height, width)
    return np.where(indices >= 249, np.uint8(0), np.uint8(255))
