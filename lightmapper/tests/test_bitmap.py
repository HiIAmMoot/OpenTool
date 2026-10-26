"""Check serialized lightmap metadata independently of the bitmap decoder."""

import numpy as np
import pytest
import struct
import zlib
import logging

from lightmapper.tagio.tagfile import new_tag, read_tag, tag_bytes, write_tag


@pytest.mark.parametrize('template', [False, True])
def test_bitmap_fields(tmp_path, template):
    from lightmapper.formats.bitmap import write_bitmap

    original = new_tag('bitm') if template else None
    if original:
        original.root.set('processing.bump_height', 0.75)
        original.root.set('flags', 19)
        original.root.set('color_plate.color_plate_width', 258)
        original.root.set('color_plate.color_plate_height', 2526)
        original.root.set_data('color_plate.compressed_color_plate_data', b'old plate')
    before = tag_bytes(original) if original else None
    pages = [np.array([[0x1234, 0xabcd], [0, 0xffff]], dtype=np.uint16),
             np.full((4, 2), 0x5678, dtype=np.uint16)]
    result = write_bitmap(pages, original)
    write_tag(result, tmp_path / 'out.bitmap')
    root = read_tag(tmp_path / 'out.bitmap').root
    if original:
        assert root.get('processing.bump_height') == 0.75
        assert root.get('flags') == 19
        assert tag_bytes(original) == before
    else:
        assert (root.get('type'), root.get('encoding_format'), root.get('usage')) == (0, 3, 4)
    assert root.get('color_plate.color_plate_width') == 4
    assert root.get('color_plate.color_plate_height') == 12
    data = root.data('color_plate.compressed_color_plate_data')
    assert struct.unpack('>I', data[:4]) == (4 * 12 * 4,)
    plate = np.frombuffer(zlib.decompress(data[4:]), np.uint8).reshape(12, 4, 4)
    expected = np.full((12, 4, 4), [255, 0, 0, 0], np.uint8)
    expected[1:3, 1:3] = [[[164, 68, 16, 255], [106, 121, 172, 255]],
                          [[0, 0, 0, 255], [255, 255, 255, 255]]]
    expected[5:9, 1:3] = [197, 206, 82, 255]
    np.testing.assert_array_equal(plate, expected)
    assert root.data('processed_pixel_data') == bytes.fromhex('3412cdab0000ffff' + '7856' * 8)
    assert [(s.get('first_bitmap_index'), s.get('bitmap_count'))
            for s in root.block('sequences')] == [(0, 1), (1, 1), (-1, 0)]
    assert all(s.block('sprites') == [] for s in root.block('sequences'))
    assert len(root.block('bitmaps')) == 2
    for i, (width, height, offset) in enumerate([(2, 2, 0), (2, 4, 8)]):
        page = root.block('bitmaps')[i]
        for key, expected in {'signature': b'bitm', 'width': width, 'height': height,
                              'depth': 1, 'type': 0, 'format': 6, 'flags': 1,
                              'registration_point': (1, height // 2), 'mipmap_count': 0,
                              'pixel_data_offset': offset, 'pixel_data_size': 0,
                              'bitmap_tag_id': 0, 'pointer': 0, 'hardware_format': 0,
                              'base_address': 0}.items():
            assert page.get(key) == expected, key


def test_overflow_plate_does_not_refuse_pages(tmp_path, caplog):
    from lightmapper.formats.bitmap import write_bitmap

    pages = [np.full((1024, 8), 0x1234, np.uint16)] * 127
    with caplog.at_level(logging.WARNING, logger='lightmapper.formats.bitmap'):
        tag = write_bitmap(pages)
    write_tag(tag, tmp_path / 'overflow.bitmap')
    root = read_tag(tmp_path / 'overflow.bitmap').root
    assert root.get('color_plate.color_plate_width') == 0
    assert root.get('color_plate.color_plate_height') == 0
    assert root.data('color_plate.compressed_color_plate_data') == b''
    assert len(root.block('bitmaps')) == 127
    assert root.data('processed_pixel_data') == bytes.fromhex('3412') * (127 * 1024 * 8)
    assert len(caplog.records) == 1
    assert '10x130304' in caplog.text


def test_plate_width_overflow_is_also_optional(caplog):
    from lightmapper.formats.bitmap import _color_plate

    with caplog.at_level(logging.WARNING, logger='lightmapper.formats.bitmap'):
        result = _color_plate([np.zeros((1, 65534), np.uint16)])
    assert result == (0, 0, b'')
    assert len(caplog.records) == 1
    assert '65536x5' in caplog.text
