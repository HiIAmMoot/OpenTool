"""Check bitmap decoding against hand-made blocks and colour plates, and the albedo choice per shader."""

import struct
import zlib

import numpy as np
import pytest

from conftest import HCEEK_TAGS as TAGS
from lightmapper.formats.bitmap_decode import UnsupportedBitmap, decode_bitmap
from lightmapper.formats.scenery import gather_occluders
from lightmapper.tagio.tagfile import new_element, new_tag, read_tag, write_tag
from lightmapper.textures import TextureSet, shader_texture


FORMATS = {'A8': 0, 'Y8': 1, 'AY8': 2, 'A8Y8': 3, 'R5G6B5': 6, 'A1R5G5B5': 8, 'A4R4G4B4': 9,
           'X8R8G8B8': 10, 'A8R8G8B8': 11, 'DXT1': 14, 'DXT3': 15, 'DXT5': 16, 'P8': 17, 'BC7': 18}


def child(parent, key):
    return new_element(next(f.layout for f in parent.layout.fields if f.key == key))


def bitmap_tag(width, height, fmt, pixels, kind=0, flags=0, depth=1):
    tag = new_tag('bitm')
    element = child(tag.root, 'bitmaps')
    element.set('width', width)
    element.set('height', height)
    element.set('depth', depth)
    element.set('type', kind)
    element.set('format', FORMATS[fmt])
    element.set('flags', flags)
    tag.root.block('bitmaps').append(element)
    tag.root.set_data('processed_pixel_data', pixels)
    return tag


def test_dxt1_block():
    # c0 red > c1 blue: four colours; index rows 0, 1, 2, 3 (two bits per texel, low bits first)
    opaque = struct.pack('<HHBBBB', 0xF800, 0x001F, 0x00, 0x55, 0xAA, 0xFF)
    image = decode_bitmap(bitmap_tag(4, 4, 'DXT1', opaque))
    assert image.shape == (4, 4, 4) and image.dtype == np.uint8
    assert image[0, 0].tolist() == [255, 0, 0, 255]
    assert image[1, 3].tolist() == [0, 0, 255, 255]
    assert np.abs(image[2, 1].astype(int) - [170, 0, 85, 255]).max() <= 1
    assert np.abs(image[3, 2].astype(int) - [85, 0, 170, 255]).max() <= 1
    # c0 <= c1: index 3 is transparent black
    punched = struct.pack('<HHBBBB', 0x001F, 0xF800, 0x00, 0x55, 0xAA, 0xFF)
    image = decode_bitmap(bitmap_tag(4, 4, 'DXT1', punched))
    assert image[3, 0, 3] == 0 and image[0, 0, 3] == 255 and image[2, 0, 3] == 255


def test_dxt3_block_keeps_explicit_alpha():
    # alpha nibbles 0..15 for texels row by row (low nibble first); both colours white
    alpha = bytes((2 * i) | (2 * i + 1) << 4 for i in range(8))
    block = alpha + struct.pack('<HHI', 0xFFFF, 0xFFFF, 0)
    image = decode_bitmap(bitmap_tag(4, 4, 'DXT3', block))
    assert image[:, :, 3].flatten().tolist() == [i * 17 for i in range(16)]
    assert (image[:, :, :3] == 255).all()


def test_bc7_mode_6_block():
    # mode 6, index 0 everywhere: endpoint 0 with its p-bit, r=127 g=0 b=0 a=127 p=1 -> (255, 1, 1, 255)
    bits = 1 << 6
    for position, value in ((7, 127), (14, 0), (21, 0), (28, 0), (35, 0), (42, 0), (49, 127), (56, 0), (63, 1)):
        bits |= value << position
    image = decode_bitmap(bitmap_tag(4, 4, 'BC7', bits.to_bytes(16, 'little')))
    assert (image == [255, 1, 1, 255]).all()


@pytest.mark.parametrize('fmt, texel, want', [
    ('A8', bytes([200]), [255, 255, 255, 200]),
    ('Y8', bytes([90]), [90, 90, 90, 255]),
    ('AY8', bytes([70]), [70, 70, 70, 70]),
    ('A8Y8', bytes([90, 30]), [90, 90, 90, 30]),
    ('R5G6B5', struct.pack('<H', 0xF800), [255, 0, 0, 255]),
    ('A1R5G5B5', struct.pack('<H', 0x83E0), [0, 255, 0, 255]),
    ('A4R4G4B4', struct.pack('<H', 0x800F), [0, 0, 255, 136]),
    ('X8R8G8B8', bytes([3, 2, 1, 0]), [1, 2, 3, 255]),
    ('A8R8G8B8', bytes([3, 2, 1, 9]), [1, 2, 3, 9]),
])
def test_uncompressed_formats_one_texel(fmt, texel, want):
    assert decode_bitmap(bitmap_tag(1, 1, fmt, texel))[0, 0].tolist() == want


def test_second_bitmap_of_a_tag_is_addressed_by_its_offset():
    tag = bitmap_tag(1, 1, 'A8R8G8B8', bytes([3, 2, 1, 9]) + bytes([6, 5, 4, 8]))
    second = child(tag.root, 'bitmaps')
    for key, value in (('width', 1), ('height', 1), ('depth', 1), ('format', 11), ('pixel_data_offset', 4)):
        second.set(key, value)
    tag.root.block('bitmaps').append(second)
    assert decode_bitmap(tag, 1)[0, 0].tolist() == [4, 5, 6, 8]
    assert decode_bitmap(tag, 0)[0, 0].tolist() == [1, 2, 3, 9]


@pytest.mark.parametrize('kwargs', [
    dict(fmt='P8'), dict(fmt='A8R8G8B8', flags=0x8), dict(fmt='A8R8G8B8', kind=1, depth=2),
    dict(fmt='A8R8G8B8', kind=2),
])
def test_unsupported_bitmaps_raise(kwargs):
    with pytest.raises(UnsupportedBitmap):
        decode_bitmap(bitmap_tag(1, 1, pixels=bytes(8), **kwargs))


def test_truncated_pixel_data_raises():
    with pytest.raises(UnsupportedBitmap):
        decode_bitmap(bitmap_tag(2, 2, 'A8R8G8B8', bytes(8)))


def plate_candidate(fmt):
    code = FORMATS[fmt]
    for path in sorted((TAGS / 'levels').rglob('*.bitmap'), key=lambda p: p.stat().st_size):
        if path.stat().st_size > 3_000_000:
            break
        tag = read_tag(path)
        bitmaps = tag.root.block('bitmaps')
        plate = tag.root.data('color_plate.compressed_color_plate_data')
        if len(bitmaps) != 1 or bitmaps[0].get('format') != code or len(plate) <= 4:
            continue
        if bitmaps[0].get('type') or bitmaps[0].get('flags') & 0x8:
            continue
        width, height = bitmaps[0].get('width'), bitmaps[0].get('height')
        if (tag.root.get('color_plate.color_plate_width'),
                tag.root.get('color_plate.color_plate_height')) != (width, height):
            continue
        raw = zlib.decompress(plate[4:])
        if len(raw) == width * height * 4:
            return tag, np.frombuffer(raw, np.uint8).reshape(height, width, 4)[:, :, [2, 1, 0, 3]]
    pytest.fail(f'no {fmt} bitmap with a colour plate under levels')


@pytest.mark.ek
@pytest.mark.parametrize('fmt', ['DXT1', 'DXT5', 'A8R8G8B8'])
def test_real_bitmap_matches_plate(fmt):
    tag, plate = plate_candidate(fmt)
    image = decode_bitmap(tag)
    assert image.shape == plate.shape
    error = np.abs(image[:, :, :3].astype(float) - plate[:, :, :3]).mean() / 255
    assert error <= (8 if fmt.startswith('DXT') else 1) / 255
    if (plate[:, :, 3] != 255).any():
        assert np.abs(image[:, :, 3].astype(float) - plate[:, :, 3]).mean() / 255 <= (8 if fmt.startswith('DXT') else 1) / 255


def write_tags(tmp_path):
    tags = tmp_path / 'tags'
    tags.mkdir(exist_ok=True)
    write_tag(bitmap_tag(1, 1, 'A8R8G8B8', bytes([3, 2, 1, 9])), tags / 'bm.bitmap')
    return tags


def shader_with_flags(group, flags_key, flags, tmp_path, **values):
    tags = write_tags(tmp_path)
    tag = new_tag(group)
    if flags_key:
        tag.root.set(flags_key, flags)
    for key, value in values.items():
        tag.root.set(key.replace('__', '.'), value)
    return tags, tag


def test_shader_model_cutout_default(tmp_path):
    for flags, cutout in ((0, True), (4, False)):
        tags, tag = shader_with_flags('soso', 'properties.flags', flags, tmp_path)
        tag.root.set_reference('maps.base_map', 'bitm', 'bm')
        write_tag(tag, tags / 'm.shader_model')
        found = shader_texture(tags, 'soso', 'm')
        assert (found.albedo, found.cutout, found.casts_shadow) == ('bm', cutout, True)


def test_shader_model_uv_scale_multiplies_and_zero_reads_as_one(tmp_path):
    tags, tag = shader_with_flags('soso', None, 0, tmp_path, maps__map_u_scale=4.0, maps__map_v_scale=0.0)
    tag.root.set_reference('maps.base_map', 'bitm', 'bm')
    write_tag(tag, tags / 'm.shader_model')
    assert shader_texture(tags, 'soso', 'm', (2.0, 3.0)).uv_scale == (8.0, 3.0)


def test_zero_scale_factors_read_as_one_in_model_and_shader(tmp_path):
    tags, tag = shader_with_flags('soso', None, 0, tmp_path, maps__map_u_scale=4.0, maps__map_v_scale=0.0)
    tag.root.set_reference('maps.base_map', 'bitm', 'bm')
    write_tag(tag, tags / 'm.shader_model')
    assert shader_texture(tags, 'soso', 'm', (0.0, 0.0)).uv_scale == (4.0, 1.0)
    assert shader_texture(tags, 'soso', 'm', (0.0, 2.0)).uv_scale == (4.0, 2.0)


def test_shader_texture_is_immutable():
    with pytest.raises(AttributeError):
        shader_texture('.', None, '').cutout = True


def test_shader_environment_alpha_tested_is_cutout(tmp_path):
    for flags, cutout in ((0, False), (1, True)):
        tags, tag = shader_with_flags('senv', 'properties.flags', flags, tmp_path)
        tag.root.set_reference('diffuse.base_map', 'bitm', 'bm')
        write_tag(tag, tags / 'e.shader_environment')
        found = shader_texture(tags, 'senv', 'e', (5.0, 5.0))
        assert (found.albedo, found.uv_scale, found.cutout, found.casts_shadow) == (
            'bm', (1.0, 1.0), cutout, True)


def test_glass_uses_background_tint_map_and_casts_no_shadow(tmp_path):
    tags, tag = shader_with_flags('sgla', None, 0, tmp_path)
    tag.root.set_reference('background_tint.background_tint_map', 'bitm', 'bm')
    write_tag(tag, tags / 'g.shader_transparent_glass')
    found = shader_texture(tags, 'sgla', 'g')
    assert (found.albedo, found.cutout, found.casts_shadow) == ('bm', False, False)
    write_tag(new_tag('sgla'), tags / 'g2.shader_transparent_glass')
    assert shader_texture(tags, 'sgla', 'g2').albedo is None


@pytest.mark.parametrize('group, name, map_block', [
    ('schi', 'shader_transparent_chicago', 'maps'),
    ('scex', 'shader_transparent_chicago_extended', '_4_stage_maps')])
def test_chicago_first_map_and_alpha_test(tmp_path, group, name, map_block):
    for flags, cutout in ((0, False), (1, True)):
        tags, tag = shader_with_flags(group, 'properties.flags', flags, tmp_path)
        stage = child(tag.root, map_block)
        stage.set_reference('parameters.map', 'bitm', 'bm')
        tag.root.block(map_block).append(stage)
        write_tag(tag, tags / f'c.{name}')
        found = shader_texture(tags, group, 'c')
        assert (found.albedo, found.cutout, found.casts_shadow) == ('bm', cutout, cutout)


@pytest.mark.parametrize('group', ['sotr', 'swat', 'smet', 'spla', '', None])
def test_other_groups_have_no_albedo(tmp_path, group):
    found = shader_texture(tmp_path, group, 'anything')
    assert (found.albedo, found.cutout, found.casts_shadow) == (None, False, False)


def test_texture_set_decodes_once_and_flags_unreadable(tmp_path, caplog):
    tags = write_tags(tmp_path)
    (tags / 'bad.bitmap').write_bytes(b'not a tag')
    textures = TextureSet(tags)
    assert textures.index('bm') == 0 and textures.index('bm') == 0
    assert textures.index('bad') == -1 and textures.index('missing') == -1
    assert 'bad' in caplog.text and 'missing' in caplog.text
    assert len(textures.arrays()) == 1 and textures.arrays()[0][0, 0].tolist() == [1, 2, 3, 9]


@pytest.mark.ek
def test_bloodgulch_scenery_textures():
    scenario = read_tag(TAGS / 'levels/test/bloodgulch/bloodgulch.scenario')
    occluders = gather_occluders(scenario, TAGS, 0)
    textures = TextureSet(TAGS)
    found = [shader_texture(TAGS, group, path, (u, v)) for group, path, u, v in occluders.shaders]
    assert found and any(f.albedo for f in found)
    assert any(f.cutout and f.casts_shadow for f in found)
    assert all(textures.index(f.albedo) >= 0 for f in found if f.albedo)
    assert all(f.uv_scale[0] > 0 and f.uv_scale[1] > 0 for f in found)
