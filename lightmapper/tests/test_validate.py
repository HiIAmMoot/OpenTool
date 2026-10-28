"""Exercise loader invariants on stock tags and isolated corruptions."""

import shutil
import struct

import numpy as np
import pytest

from conftest import HCEEK_TAGS
from lightmapper.formats.vertices import ENV, LM, compress, pack_uncompressed, unpack_uncompressed
from lightmapper.tagio.tagfile import new_element, new_tag, read_tag, tag_bytes


def copied_pair(tmp_path, stem):
    tags = []
    for extension in ('scenario_structure_bsp', 'bitmap'):
        source = HCEEK_TAGS / (stem + '.' + extension)
        destination = tmp_path / source.name
        shutil.copy2(source, destination)
        tags.append(read_tag(destination))
    return tuple(tags)


def validate(bsp, bitmap, strict_compression=False):
    from lightmapper.validate import validate_bsp
    return validate_bsp(bsp, bitmap, strict_compression=strict_compression)


@pytest.mark.ek
def test_retail_bloodgulch(tmp_path):
    assert validate(*copied_pair(tmp_path, 'levels/test/bloodgulch/bloodgulch')) == []


@pytest.mark.ek
def test_retail_bloodgulch_strict_compression(tmp_path):
    problems = validate(*copied_pair(tmp_path, 'levels/test/bloodgulch/bloodgulch'),
                        strict_compression=True)
    assert len(problems) == 38, problems
    compression = [p for p in problems if 'recompression' in p]
    assert len(compression) == 3, problems
    for problem, (page, material) in zip(compression, [(0, 0), (1, 1), (3, 0)]):
        assert f'lightmap {page}, material {material}:' in problem
        assert 'compressed' in problem and 'recompression' in problem
    types = [p.split(':', 1)[0] for p in problems if 'lightmap_vertices vertex type' in p]
    assert types == [f'lightmap {page}, material {material}'
                     for page, count in enumerate([2, 2, 1, 1, 1, 1, 1, 4, 5, 2, 2, 2, 2, 5, 4])
                     for material in range(count)]


def child(parent, key):
    return new_element(next(f.layout for f in parent.layout.fields if f.key == key))


def set_vertices(material, env, lm):
    material.set('rendered_vertices.vertex_count', len(env))
    material.set('lightmap_vertices.vertex_count', len(lm))
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    material.set_data('compressed_vertices', compress(env, lm))


@pytest.fixture
def small():
    bsp, bitmap = new_tag('sbsp'), new_tag('bitm')
    bsp.root.set_reference('lightmaps_bitmap', 'bitm', 'test_lightmap')
    page = child(bsp.root, 'lightmaps')
    final = child(bsp.root, 'lightmaps')
    final.set('bitmap', -1)
    material = child(page, 'materials')
    material.set_reference('shader', 'senv', 'test_shader')
    material.set('surface_count', 1)
    env, lm = np.zeros(3, dtype=ENV), np.zeros(3, dtype=LM)
    env['normal'][:, 2] = 1
    lm['incident_radiosity'][:, 2] = 1
    lm['texcoord'][:] = [(0, 0), (1, 0), (0, 1)]
    set_vertices(material, env, lm)
    page.set_block('materials', [material])
    bsp.root.set_block('lightmaps', [page, final])
    surface = child(bsp.root, 'surfaces')
    surface.body[:] = struct.pack('>3H', 0, 1, 2)
    bsp.root.set_block('surfaces', [surface])
    bit = child(bitmap.root, 'bitmaps')
    bit.set('width', 2)
    bit.set('height', 4)
    bit.set('format', 6)
    bit.set('depth', 1)
    bitmap.root.set_block('bitmaps', [bit])
    bitmap.root.set_block('sequences', [child(bitmap.root, 'sequences') for _ in range(2)])
    bitmap.root.set_data('processed_pixel_data', bytes(16))
    return bsp, bitmap


def first_material(bsp):
    return bsp.root.block('lightmaps')[0].block('materials')[0]


def test_valid_buffers_and_rectangular_page_are_not_modified(small):
    before = tuple(tag_bytes(tag) for tag in small)
    assert validate(*small, strict_compression=True) == []
    assert tuple(tag_bytes(tag) for tag in small) == before


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize(('field', 'value', 'rule'), [
    ('rendered_vertices.vertex_type', 1, 'vertex type'),
    ('rendered_vertices.vertex_count', 64001, '64000'),
    ('rendered_vertices.vertex_count', 0xffffffff, '64000'),
    ('lightmap_vertices.vertex_count', 65536, 'lightmap vertex count'),
    ('lightmap_vertices.vertex_count', 2, 'lightmap vertex count'),
    ('surface_count', 20001, '20000'),
    ('surface_count', -1, 'surface range'),
    ('surfaces', -1, 'surface range'),
    ('surfaces', 1, 'surface range'),
])
def test_material_fields_always_checked(small, strict, field, value, rule):
    bsp, bitmap = small
    first_material(bsp).set(field, value)
    problems = validate(bsp, bitmap, strict_compression=strict)
    assert any('lightmap 0, material 0:' in p and rule in p for p in problems), problems


@pytest.mark.parametrize('value', [1, 512])
def test_lightmap_buffer_type_is_strict_only(small, value):
    bsp, bitmap = small
    first_material(bsp).set('lightmap_vertices.vertex_type', value)
    assert validate(bsp, bitmap) == []
    problems = validate(bsp, bitmap, strict_compression=True)
    assert len(problems) == 1, problems
    assert 'lightmap 0, material 0:' in problems[0]
    assert 'lightmap_vertices vertex type' in problems[0]


@pytest.mark.parametrize(('group', 'path'), [(None, ''), ('senv', '')])
def test_null_shader(small, group, path):
    bsp, bitmap = small
    first_material(bsp).set_reference('shader', group, path)
    assert any('shader' in p for p in validate(bsp, bitmap))


@pytest.mark.parametrize('key', ['uncompressed_vertices', 'compressed_vertices'])
@pytest.mark.parametrize('delta', [-1, 1])
def test_buffer_sizes_always_checked(small, key, delta):
    bsp, bitmap = small
    material = first_material(bsp)
    raw = material.data(key)
    material.set_data(key, raw[:-1] if delta < 0 else raw + b'\0')
    assert any(key in p and 'size' in p for p in validate(bsp, bitmap))


def test_compression_equality_is_opt_in(small):
    bsp, bitmap = small
    material = first_material(bsp)
    raw = bytearray(material.data('compressed_vertices'))
    raw[0] ^= 1
    material.set_data('compressed_vertices', raw)
    assert validate(bsp, bitmap) == []
    problems = validate(bsp, bitmap, strict_compression=True)
    assert len(problems) == 1
    assert 'compressed' in problems[0] and 'recompression' in problems[0]


@pytest.mark.parametrize('field', ['normal', 'binormal', 'tangent', 'incident_radiosity'])
@pytest.mark.parametrize('value', [-1.01, 1.01, float('nan'), float('inf'), -float('inf')])
def test_direction_components(small, field, value):
    bsp, bitmap = small
    material = first_material(bsp)
    env, lm = unpack_uncompressed(material.data('uncompressed_vertices'), 3, 3)
    (lm if field == 'incident_radiosity' else env)[field][1, 1] = value
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    problems = validate(bsp, bitmap, strict_compression=True)
    assert any(field in p and 'finite' in p and '1.005' in p for p in problems), problems


@pytest.mark.parametrize('value', [-0.001, 1.001, float('nan'), float('inf')])
def test_lightmap_uv_range(small, value):
    bsp, bitmap = small
    material = first_material(bsp)
    env, lm = unpack_uncompressed(material.data('uncompressed_vertices'), 3, 3)
    lm['texcoord'][0, 0] = value
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    assert any('lightmap UV' in p for p in validate(bsp, bitmap, strict_compression=True))


def test_direction_tolerance_and_uv_boundaries(small):
    bsp, bitmap = small
    material = first_material(bsp)
    env, lm = unpack_uncompressed(material.data('uncompressed_vertices'), 3, 3)
    for field in ('normal', 'binormal', 'tangent'):
        env[field][0] = (-1.005, 0, 1.005)
    lm['incident_radiosity'][0] = (-1.005, 0, 1.005)
    set_vertices(material, env, lm)
    assert validate(bsp, bitmap, strict_compression=True) == []


@pytest.mark.parametrize('case', ['empty', 'too_many', 'last_bitmap', 'page_bitmap', 'materials'])
def test_lightmap_structure(small, case):
    bsp, bitmap = small
    pages = bsp.root.block('lightmaps')
    if case == 'empty':
        bsp.root.set_block('lightmaps', [])
        rule = 'last element'
    elif case == 'too_many':
        bsp.root.set_block('lightmaps', [pages[0]] * 128 + [pages[-1]])
        rule = '128'
    elif case == 'last_bitmap':
        pages[-1].set('bitmap', 0)
        rule = 'last element'
    elif case == 'page_bitmap':
        pages[0].set('bitmap', -1)
        rule = 'bitmap index'
    else:
        pages[-1].set_block('materials', [first_material(bsp)] * 2049)
        rule = '2048'
    assert any(rule in p for p in validate(bsp, bitmap))


def test_final_material_has_no_lightmap_vertices(small):
    bsp, bitmap = small
    page, final = bsp.root.block('lightmaps')
    final.set_block('materials', page.block('materials'))
    page.set_block('materials', [])
    assert any('lightmap 1, material 0:' in p and 'lightmap vertex count' in p
               for p in validate(bsp, bitmap))


def test_non_lightmapped_only_bsp(small):
    bsp, _ = small
    page, final = bsp.root.block('lightmaps')
    material = first_material(bsp)
    env, _ = unpack_uncompressed(material.data('uncompressed_vertices'), 3, 3)
    set_vertices(material, env, np.zeros(0, dtype=LM))
    final.set_block('materials', page.block('materials'))
    bsp.root.set_block('lightmaps', [final])
    bsp.root.set_reference('lightmaps_bitmap', None, '')
    assert validate(bsp, None, strict_compression=True) == []


def test_lightmapped_bsp_requires_nonempty_bitmap_reference(small):
    bsp, bitmap = small
    bsp.root.set_reference('lightmaps_bitmap', 'bitm', '')
    problems = validate(bsp, bitmap)
    assert len(problems) == 1, problems
    assert 'lightmaps_bitmap' in problems[0] and 'empty' in problems[0]


@pytest.mark.parametrize('case', ['gap', 'overlap', 'descending', 'uncovered', 'out_of_bounds'])
def test_material_surface_partition(small, case):
    bsp, bitmap = small
    page = bsp.root.block('lightmaps')[0]
    material = first_material(bsp)
    bsp.root.set_block('surfaces', bsp.root.block('surfaces') * 3)
    second = material.copy()
    second.set('surfaces', 1)
    second.set('surface_count', 2)
    page.set_block('materials', [material, second])
    if case == 'gap':
        second.set('surfaces', 2)
        second.set('surface_count', 1)
    elif case == 'overlap':
        second.set('surfaces', 0)
    elif case == 'descending':
        material.set('surfaces', 1)
        second.set('surfaces', 0)
    elif case == 'uncovered':
        second.set('surface_count', 1)
    else:
        second.set('surface_count', 3)
    assert any('surface range' in p for p in validate(bsp, bitmap))


@pytest.mark.parametrize('kind', ['leaf', 'subcluster'])
@pytest.mark.parametrize('index', [-1, 1])
def test_visibility_surface_indices(small, kind, index):
    bsp, bitmap = small
    if kind == 'leaf':
        ref = child(bsp.root, 'leaf_surfaces')
        ref.set('surface', index)
        bsp.root.set_block('leaf_surfaces', [ref])
        rule = 'leaf surfaces'
    else:
        cluster = child(bsp.root, 'clusters')
        bsp.root.set_block('clusters', [cluster])
        parent = cluster
        if kind == 'subcluster':
            parent = child(cluster, 'subclusters')
            cluster.set_block('subclusters', [parent])
        ref = child(parent, 'surface_indices')
        ref.set('index', index)
        parent.set_block('surface_indices', [ref])
        rule = kind
    assert any(rule in p and 'surface index' in p for p in validate(bsp, bitmap))


def set_cluster_groups(bsp, values):
    cluster = child(bsp.root, 'clusters')
    refs = []
    for value in values:
        ref = child(cluster, 'surface_indices')
        ref.set('index', value)
        refs.append(ref)
    cluster.set_block('surface_indices', refs)
    bsp.root.set_block('clusters', [cluster])


@pytest.mark.parametrize('values', [[], [0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]])
def test_cluster_grouped_surface_lists(small, values):
    bsp, bitmap = small
    set_cluster_groups(bsp, values)
    assert validate(bsp, bitmap) == []


@pytest.mark.parametrize(('values', 'rule'), [
    ([2, 0, 1, 0], 'lightmap index'),
    ([-1, 0, 1, 0], 'lightmap index'),
    ([0, 1, 1, 0], 'material index'),
    ([0, -1, 1, 0], 'material index'),
    ([0, 0, 2, 0], 'count'),
    ([0, 0, -1, 0], 'count'),
    ([0, 0], 'header'),
    ([0, 0, 1, 1], 'material surface range'),
])
def test_invalid_cluster_group_reports_one_rule(small, values, rule):
    bsp, bitmap = small
    set_cluster_groups(bsp, values)
    problems = validate(bsp, bitmap)
    assert len(problems) == 1, problems
    assert all(word in problems[0] for word in ('cluster 0', 'group 0', rule)), problems


def test_cluster_group_surface_must_belong_to_selected_material(small):
    bsp, bitmap = small
    page = bsp.root.block('lightmaps')[0]
    second = first_material(bsp).copy()
    second.set('surfaces', 1)
    page.set_block('materials', [first_material(bsp), second])
    bsp.root.set_block('surfaces', bsp.root.block('surfaces') * 2)
    set_cluster_groups(bsp, [0, 1, 1, 1, 0, 0, 1, 1])
    problems = validate(bsp, bitmap)
    assert len(problems) == 1, problems
    assert all(word in problems[0] for word in ('cluster 0', 'group 1', 'material surface range'))


@pytest.mark.parametrize('group', [None, 'senv'])
def test_lightmaps_bitmap_group(small, group):
    bsp, bitmap = small
    bsp.root.set_reference('lightmaps_bitmap', group, 'test_lightmap')
    assert any('lightmaps_bitmap' in p and 'bitm' in p for p in validate(bsp, bitmap))


@pytest.mark.parametrize(('field', 'value', 'rule'), [
    ('format', 0, 'format 6'),
    ('width', 0, 'power-of-two'),
    ('width', 3, 'power-of-two'),
    ('height', 0, 'power-of-two'),
    ('height', 3, 'power-of-two'),
    ('mipmap_count', 1, 'mipmap'),
    ('pixel_data_offset', 1, 'offset'),
    ('type', 1, 'type 0'),
    ('type', 2, 'type 0'),
    ('depth', 0, 'depth 1'),
    ('depth', 2, 'depth 1'),
])
def test_bitmap_page_fields(small, field, value, rule):
    bsp, bitmap = small
    bitmap.root.block('bitmaps')[0].set(field, value)
    assert any('lightmap 0, material all:' in p and rule in p for p in validate(bsp, bitmap))


def test_bitmap_second_offset_must_follow_first(small):
    bsp, bitmap = small
    bitmap.root.set_block('bitmaps', bitmap.root.block('bitmaps') * 2)
    bitmap.root.set_block('sequences', [child(bitmap.root, 'sequences') for _ in range(3)])
    bitmap.root.set_data('processed_pixel_data', bytes(32))
    assert any('lightmap 1, material all:' in p and 'offset' in p for p in validate(bsp, bitmap))


@pytest.mark.parametrize('case', ['pixels_short', 'pixels_long', 'sequences'])
def test_bitmap_totals(small, case):
    bsp, bitmap = small
    if case == 'sequences':
        bitmap.root.set_block('sequences', [])
        rule = 'sequences'
    else:
        bitmap.root.set_data('processed_pixel_data', bytes(15 if case == 'pixels_short' else 17))
        rule = 'pixel data'
    assert any(rule in p for p in validate(bsp, bitmap))


def test_stock_caps_and_unsigned_triangle_indices(small):
    bsp, bitmap = small
    page, final = bsp.root.block('lightmaps')
    template = first_material(bsp)
    env, lm = np.zeros(64000, dtype=ENV), np.zeros(64000, dtype=LM)
    set_vertices(template, env, lm)
    surface = bsp.root.block('surfaces')[0]
    surface.body[:] = struct.pack('>3H', 0, 1, 63999)
    bsp.root.set_block('surfaces', [surface] * 131072)
    materials = []
    for start in range(0, 131072, 20000):
        material = template.copy()
        material.set('surfaces', start)
        material.set('surface_count', min(20000, 131072 - start))
        materials.append(material)
    page.set_block('materials', materials)
    pages = [page]
    for i in range(1, 127):
        empty = child(bsp.root, 'lightmaps')
        empty.set('bitmap', i)
        pages.append(empty)
    bsp.root.set_block('lightmaps', pages + [final])
    bitmaps = []
    for i in range(127):
        bit = bitmap.root.block('bitmaps')[0].copy()
        bit.set('pixel_data_offset', i * 16)
        bitmaps.append(bit)
    bitmap.root.set_block('bitmaps', bitmaps)
    bitmap.root.set_block('sequences', [child(bitmap.root, 'sequences') for _ in range(128)])
    bitmap.root.set_data('processed_pixel_data', bytes(127 * 16))
    assert validate(bsp, bitmap, strict_compression=True) == []
    last = pages[-1]
    last.set_block('materials', [child(last, 'materials') for _ in range(2048)])
    for material in last.block('materials'):
        material.set_reference('shader', 'senv', 'empty_shader')
        material.set('surfaces', 131072)
    assert validate(bsp, bitmap) == []


def test_32bit_page_is_valid(small):
    bsp, bitmap = small
    page = bitmap.root.block('bitmaps')[0]
    page.set('format', 10)
    width, height = page.get('width'), page.get('height')
    bitmap.root.set_data('processed_pixel_data', bytes(width * height * 4))
    assert validate(bsp, bitmap) == []
