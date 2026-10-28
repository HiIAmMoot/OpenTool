"""Compare stock vertex bytes and independently checked colour quantisation."""

import struct

import numpy as np
import pytest

from lightmapper.formats import vertices as v


def test_compress_vector_known():
    result = v.compress_vector(np.array([[1.0, -1.0, 0.0], [0, 0, 0],
                                         [-1, 1, -1], [2, -2, 2], [-0.5, 0.5, -0.5]]))
    assert result.dtype == np.dtype("uint32")
    assert result.tolist() == [((0 << 11 | (-1024 & 0x7ff)) << 11 | 1023), 0,
                               0x80000000 | (1023 << 11) | 1024,
                               (511 << 22) | (1024 << 11) | 1023,
                               (768 << 22) | (511 << 11) | 1536]


def test_uncompressed_layout_and_compressed_order():
    raw = struct.pack("<14f", 1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 4, 5)
    raw += struct.pack("<5f", 0, 0, -1, -2, 2)
    env, lm = v.unpack_uncompressed(raw, 1, 1)
    assert env.dtype == v.ENV and env.itemsize == 56
    assert lm.dtype == v.LM and lm.itemsize == 20
    assert env["position"].tolist() == [[1, 2, 3]]
    assert env["normal"].tolist() == [[1, 0, 0]]
    assert env["binormal"].tolist() == [[0, 1, 0]]
    assert env["tangent"].tolist() == [[0, 0, 1]]
    assert env["texcoord"].tolist() == [[4, 5]]
    assert lm["incident_radiosity"].tolist() == [[0, 0, -1]]
    assert lm["texcoord"].tolist() == [[-2, 2]]
    assert v.pack_uncompressed(env, lm) == raw
    assert v.compress(env, lm) == (struct.pack("<3f3I2f", 1, 2, 3, 1023, 1023 << 11,
                                             511 << 22, 4, 5)
                                   + struct.pack("<Ihh", 512 << 22, -32768, 32767))


def test_lightmap_uv_floor():
    env = np.zeros(0, dtype=v.ENV)
    lm = np.zeros(3, dtype=v.LM)
    lm["texcoord"] = [[-0.5, 0.5], [-1, 1], [0, 0]]
    assert v.compress(env, lm) == (struct.pack("<Ihh", 0, -16384, 16383)
                                   + struct.pack("<Ihh", 0, -32768, 32767)
                                   + bytes(8))


def test_empty_buffers():
    env, lm = v.unpack_uncompressed(b"", 0, 0)
    assert env.shape == lm.shape == (0,)
    assert v.pack_uncompressed(env, lm) == v.compress(env, lm) == b""
    assert v.compress_vector(np.empty((0, 3))).shape == (0,)


def test_r5g6b5_extremes():
    assert v.encode_r5g6b5(np.ones((4, 4, 3)), True).tolist() == [[0xFFFF] * 4] * 4
    assert v.encode_r5g6b5(np.zeros((4, 4, 3)), True).tolist() == [[0] * 4] * 4


def test_dither_preserves_mean():
    rgb = np.full((64, 64, 3), 0.4321)
    decoded = v.decode_r5g6b5(v.encode_r5g6b5(rgb, True))
    assert abs(decoded[..., 0].mean() - 0.4321) < 0.5 / 31 / 4
    assert len(np.unique(decoded[..., 0])) == 2


def test_no_dither_rounds():
    encoded = v.encode_r5g6b5(np.full((1, 1, 3), 0.5), False)
    assert encoded.dtype == np.dtype("uint16")
    assert encoded.tolist() == [[0x8410]]
    assert v.decode_r5g6b5(encoded)[0, 0, 1] == 32 / 63


def test_bayer_order_and_rectangular_repeat():
    expected = np.array([[0x7BEF, 0x8410, 0x7BEF, 0x8410],
                         [0x8410, 0x7BEF, 0x8410, 0x7BEF],
                         [0x7BEF, 0x8410, 0x7BEF, 0x8410],
                         [0x8410, 0x7BEF, 0x8410, 0x7BEF]], dtype=np.uint16)
    np.testing.assert_array_equal(v.encode_r5g6b5(np.full((8, 12, 3), 0.5), True),
                                  np.tile(expected, (2, 3)))
    # a half-level only tests parity; these distinguish all sixteen thresholds
    red = np.arange(16, dtype=float) / 16 / 31
    rgb = np.zeros((16, 4, 4, 3))
    rgb[..., 0] = red[:, None, None]
    transitions = np.stack([v.encode_r5g6b5(image, True) != 0 for image in rgb]).sum(axis=0)
    np.testing.assert_array_equal(transitions, [[0, 8, 2, 10], [12, 4, 14, 6],
                                               [3, 11, 1, 9], [15, 7, 13, 5]])


def test_decode_channel_order():
    px = np.array([[0xF800, 0x07E0, 0x001F, 0xFFFF, 0]], dtype=np.uint16)
    np.testing.assert_array_equal(v.decode_r5g6b5(px),
                                  [[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]])


@pytest.mark.parametrize("count", [64000, 65535])
def test_stock_maximum_vertex_buffers(count):
    env = np.zeros(count, dtype=v.ENV)
    lm = np.zeros(count, dtype=v.LM)
    env["position"] = np.arange(count)[:, None] % 128
    env["normal"] = [0, 0, 1]
    lm["incident_radiosity"] = [0, 0, 1]
    lm["texcoord"] = [0.5, 1]
    raw = v.pack_uncompressed(env, lm)
    assert len(raw) == count * 76
    back_env, back_lm = v.unpack_uncompressed(raw, count, count)
    np.testing.assert_array_equal(back_env, env)
    np.testing.assert_array_equal(back_lm, lm)
    compressed = v.compress(env, lm)
    assert len(compressed) == count * 40
    env_bytes = np.frombuffer(compressed[:count * 32], dtype=np.dtype((np.void, 32)))
    assert all(bytes(env_bytes[i]) == struct.pack("<3f3I2f", i % 128, i % 128, i % 128,
                                                 511 << 22, 0, 0, 0, 0)
               for i in [0, count // 2, count - 1])
    assert compressed[count * 32:] == struct.pack("<Ihh", 511 << 22, 16383, 32767) * count


def test_tool_maximum_reference_page():
    rgb = np.full((256, 256, 3), 0.5, dtype=np.float32)
    encoded = v.encode_r5g6b5(rgb, True)
    assert encoded.shape == (256, 256)
    assert np.count_nonzero(encoded == 0x8410) == 32768
    assert np.count_nonzero(encoded == 0x7BEF) == 32768


def test_x8r8g8b8_channel_order_and_opaque_alpha():
    rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    encoded = v.encode_x8r8g8b8(rgb, False)
    assert encoded.dtype == np.dtype("uint32")
    assert encoded.tolist() == [[0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFF000000]]


def test_x8r8g8b8_dither_preserves_mean():
    rgb = np.full((64, 64, 3), 0.4321)
    red = (v.encode_x8r8g8b8(rgb, True) >> 16 & 255) / 255
    assert abs(red.mean() - 0.4321) < 0.5 / 255 / 4
    assert len(np.unique(red)) == 2
