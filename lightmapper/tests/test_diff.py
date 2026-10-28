"""Check that tool's and our lightmaps are paired surface by surface, each in its own layout."""

import struct

import numpy as np
import pytest

from lightmapper.diff import (Calibration, Samples, emitter_shares, paired_samples, pearson, report,
                              render_markdown)
from lightmapper.formats.bitmap import write_bitmap
from lightmapper.formats.vertices import ENV, LM, compress, encode_r5g6b5, pack_uncompressed
from lightmapper.tagio.tagfile import new_element, new_tag


TRIANGLE = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0]], np.float64)


def child(parent, key):
    return new_element(next(f.layout for f in parent.layout.fields if f.key == key))


def make_bsp(world, uv, vectors=None, shader='wall'):
    """One lightmapped page holding every triangle; surface s uses vertices 3s..3s+2."""
    count = len(world)
    bsp = new_tag('sbsp')
    page, final = child(bsp.root, 'lightmaps'), child(bsp.root, 'lightmaps')
    final.set('bitmap', -1)
    material = child(page, 'materials')
    material.set_reference('shader', 'senv', shader)
    env, lm = np.zeros(3 * count, ENV), np.zeros(3 * count, LM)
    env['position'] = np.asarray(world, np.float32).reshape(-1, 3)
    env['normal'] = [0, 0, 1]
    lm['texcoord'] = np.asarray(uv, np.float32).reshape(-1, 2)
    if vectors is not None:
        lm['incident_radiosity'] = np.asarray(vectors, np.float32).reshape(-1, 3)
    for prefix in ('rendered_vertices', 'lightmap_vertices'):
        material.set(prefix + '.vertex_count', 3 * count)
    material.set('surface_count', count)
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    material.set_data('compressed_vertices', compress(env, lm))
    page.set_block('materials', [material])
    surfaces = []
    for s in range(count):
        surface = child(bsp.root, 'surfaces')
        surface.body[:] = struct.pack('>3H', 3 * s, 3 * s + 1, 3 * s + 2)
        surfaces.append(surface)
    bsp.root.set_block('surfaces', surfaces)
    bsp.root.set_block('lightmaps', [page, final])
    return bsp


def base(i):
    return np.array([10 * i, 5 * i, 0.0])


def world_triangles(count=4):
    return np.array([TRIANGLE + base(i) for i in range(count)])


def colour(position):
    # red and blue are steps of the 5-bit grid per triangle, green a gradient on the 6-bit one
    x, y = position[..., 0], position[..., 1]
    red = (3 + 5 * np.floor(x / 10 + 1e-6)) / 31
    return np.stack([red, 0.1 + 0.01 * x + 0.005 * y, (20 - 4 * np.floor(x / 10 + 1e-6)) / 31], axis=-1)


def layout(blocks, size, scale):
    """Triangle j gets the 32-texel block `blocks[j]` (column, row) of a page of `size` (w, h)."""
    width, height = size
    uv, page = [], np.zeros((height, width, 3))
    for j, (column, row) in enumerate(blocks):
        origin = np.array([32 * column, 32 * row], np.float64)
        uv.append((origin + 4 + TRIANGLE[:, :2] * scale) / [width, height])
        for y in range(32):
            for x in range(32):
                local = (np.array([x, y]) + 0.5 - 4) / scale
                page[32 * row + y, 32 * column + x] = colour(base(j) + [local[0], local[1], 0])
    return np.array(uv), page


def pair(tool_scale=6, count=4, vectors_tool=None, vectors_ours=None, our_scale=5):
    world = world_triangles(count)
    tool_uv, tool_page = layout([(0, 0), (1, 0), (0, 1), (1, 1)][:count], (64, 64), tool_scale)
    our_blocks = [(0, 3), (0, 2), (0, 1), (0, 0)][:count]
    our_uv, our_page = layout(our_blocks, (32, 128), our_scale)
    # our surface n is input surface order[n]
    order = np.array([2, 0, 3, 1][:count]) if count == 4 else np.arange(count)
    tool_bsp = make_bsp(world, tool_uv, vectors_tool)
    ours_vectors = None if vectors_ours is None else np.asarray(vectors_ours)[order]
    our_bsp = make_bsp(world[order], our_uv[order], ours_vectors)
    bitmap = write_bitmap([encode_r5g6b5(tool_page, False)])
    return tool_bsp, bitmap, [our_page], our_bsp, order, world


def test_pairs_follow_each_layout():
    tool_bsp, bitmap, ours, our_bsp, order, world = pair()
    samples = paired_samples(tool_bsp, bitmap, ours, our_bsp, order)
    np.testing.assert_array_equal(samples.surface, [0, 1, 2, 3])
    np.testing.assert_allclose(samples.tool_rgb, samples.ours_rgb, atol=1 / 63, rtol=0)
    np.testing.assert_allclose(samples.ours_rgb, colour(world.mean(axis=1)), atol=1 / 63, rtol=0)
    np.testing.assert_allclose(samples.surface_area, 8)
    assert len(set(samples.tool_rgb[:, 0])) == 4
    assert samples.layout['tool']['pages'] == 1 and samples.layout['ours']['pages'] == 1


def test_vectors_pair_the_same_input_corner():
    vectors = np.arange(36, dtype=np.float32).reshape(4, 3, 3) / 40
    tool_bsp, bitmap, ours, our_bsp, order, _ = pair(vectors_tool=vectors, vectors_ours=vectors + 1)
    samples = paired_samples(tool_bsp, bitmap, ours, our_bsp, order)
    np.testing.assert_array_equal(samples.tool_vec, vectors.reshape(-1, 3))
    np.testing.assert_array_equal(samples.our_vec, vectors.reshape(-1, 3) + 1)


def test_surfaces_under_a_texel_are_dropped():
    # at scale 0.2 the tool's triangles cover 0.32 texel^2
    tool_bsp, bitmap, ours, our_bsp, order, _ = pair(tool_scale=0.2)
    samples = paired_samples(tool_bsp, bitmap, ours, our_bsp, order)
    assert len(samples.surface) == 0 and len(samples.tool_vec) == 0


def grid_pair(flip):
    """64 triangles on one smooth page, paired against the same page (vertically flipped when asked)."""
    uv, world = [], []
    for j in range(64):
        origin = np.array([16 * (j % 8), 16 * (j // 8)], np.float64)
        uv.append((origin + 2 + TRIANGLE[:, :2] * 3) / 128)
        world.append(TRIANGLE + base(j))
    u, v = np.meshgrid((np.arange(128) + 0.5) / 128, (np.arange(128) + 0.5) / 128)
    page = np.stack([v, 0.1 + 0.8 * v, 0.5 * u + 0.25 * v], axis=-1)
    bsp = make_bsp(np.array(world), np.array(uv))
    bitmap = write_bitmap([encode_r5g6b5(page, False)])
    return paired_samples(bsp, bitmap, [page[::-1] if flip else page], bsp, np.arange(64))


def test_vertical_flip_is_caught_by_the_report():
    assert report(grid_pair(False))['pearson_r'] > 0.99
    assert report(grid_pair(True))['pearson_r'] < 0.3
    assert pearson(np.ones((5, 3)), np.arange(15).reshape(5, 3)) == 0.0


def test_surfaces_under_a_texel_in_ours_only_are_dropped():
    tool_bsp, bitmap, ours, our_bsp, order, _ = pair(our_scale=0.2)
    samples = paired_samples(tool_bsp, bitmap, ours, our_bsp, order)
    assert len(samples.surface) == 0


def sample_set(tool, ours, material, vec_tool=None, vec_ours=None):
    n = len(tool)
    vec_tool = np.tile([0, 0, 1.0], (n, 1)) if vec_tool is None else vec_tool
    vec_ours = np.tile([0, 0, 1.0], (n, 1)) if vec_ours is None else vec_ours
    return Samples(np.asarray(tool, float), np.asarray(ours, float), np.ones(n), np.asarray(material, object),
                   vec_tool, vec_ours, vec_tool, vec_ours, np.arange(n), {side: {'pages': 1, 'utilisation': 0.5, 'texels_per_unit': 1.0} for side in ('tool', 'ours')})


def test_report_materials_and_acceptance():
    rng = np.random.default_rng(2)
    tool = rng.random((120, 3)) * 0.5
    ours = tool.copy()
    ours[60:] *= 1.5
    names = ['good'] * 60 + ['bad'] * 50 + ['rare'] * 10
    data = report(sample_set(tool, ours, names), Calibration(1, 1, 1, 1))
    assert set(data['materials']) == {'good', 'bad'}
    assert data['materials']['good']['ratio'] == pytest.approx(1)
    assert data['materials']['bad']['ratio'] == pytest.approx(1.5)
    assert data['acceptance']['materials'] is False
    assert data['median_abs_error'] == pytest.approx(np.median(np.abs(tool - ours)))
    assert data['vector']['median_angle_deg'] == pytest.approx(0)
    assert data['fit']['gamma'] == 1
    assert '| bad | 50 |' in render_markdown(data)


def test_report_clipped_fraction_needs_unclipped_values():
    s = sample_set(np.zeros((4, 3)), np.zeros((4, 3)), ['m'] * 4)
    assert report(s)['clipped_fraction'] is None
    s.ours_peak = np.array([0.5, 1.5, 2.0, 1.0])
    data = report(s)
    assert data['clipped_fraction'] == 0.5
    assert 'clipped' in render_markdown(data)


def test_report_vectors():
    tool = np.array([[0, 0, 1.0]] * 3)
    ours = np.array([[0, 0.5, 0.0]] * 3)
    ours[:, 2] = np.sqrt(0.75)
    data = report(sample_set(np.zeros((3, 3)), np.zeros((3, 3)), ['m'] * 3, tool, ours))
    assert data['vector']['median_angle_deg'] == pytest.approx(30)
    assert data['vector']['median_length_difference'] == pytest.approx(0)


def test_median_error_bar_is_0_09():
    tool = np.zeros((100, 3))
    for error, passes in ((0.085, True), (0.095, False)):
        data = report(sample_set(tool, tool + error, ['m'] * 100))
        assert data['acceptance']['median_abs_error'] is passes


def test_materials_next_to_emitters_are_excluded_and_listed():
    tool = np.full((120, 3), 0.4)
    ours = tool.copy()
    ours[60:] *= 1.5
    names = ['good'] * 60 + ['lamp'] * 60
    data = report(sample_set(tool, ours, names), excluded={'lamp': 0.6})
    assert data['acceptance']['materials'] is True
    assert data['excluded'] == {'lamp': 0.6}
    assert data['materials']['lamp']['ratio'] == pytest.approx(1.5)
    markdown = render_markdown(data)
    assert 'lamp' in markdown.split('## Excluded')[1] and 'emission' in markdown
    assert report(sample_set(tool, ours, names))['acceptance']['materials'] is False
    assert report(sample_set(tool, ours, names))['excluded'] == {}


def test_emitter_shares_use_the_luminance_of_the_emission_term():
    names = np.array(['lamp'] * 60 + ['wall'] * 60 + ['rare'] * 5, object)
    emit, total = np.zeros((125, 3)), np.ones((125, 3))
    emit[:60] = [0.5, 0.5, 0.5]
    emit[60:120] = [0.2, 0.2, 0.2]
    emit[120:] = 1
    shares = emitter_shares(names, emit, total)
    assert shares == {'lamp': pytest.approx(0.5)}


def test_bitmap_pages_decode_32bit():
    from lightmapper.diff import bitmap_pages
    from lightmapper.formats.vertices import encode_x8r8g8b8

    rgb = np.array([[[0.2, 0.4, 0.6], [1.0, 0.0, 0.5]]])
    pages = bitmap_pages(write_bitmap([encode_x8r8g8b8(rgb, False)]))
    np.testing.assert_allclose(pages[0], rgb, atol=0.5 / 255)
