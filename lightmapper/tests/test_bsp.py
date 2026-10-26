"""Check BSP welding and layout remapping against independent surface positions."""

import importlib
import importlib.util
import struct

import numpy as np
import pytest

from lightmapper.formats.vertices import ENV, LM, unpack_uncompressed
from lightmapper.tagio.tagfile import new_element, new_tag, read_tag, write_tag, tag_bytes


def api():
    assert importlib.util.find_spec('lightmapper.formats.bsp') is not None, 'BSP geometry API is missing'
    return importlib.import_module('lightmapper.formats.bsp')


def child(parent, key):
    return new_element(next(f.layout for f in parent.layout.fields if f.key == key))


def ids(material):
    start = material.get('surfaces')
    return np.arange(start, start + material.get('surface_count'), dtype=np.int32)


def buffers(material):
    return unpack_uncompressed(material.data('uncompressed_vertices'),
                               material.get('rendered_vertices.vertex_count'),
                               material.get('lightmap_vertices.vertex_count'))


def triangles(root):
    return np.array([struct.unpack('>3H', s.body) for s in root.block('surfaces')], dtype=np.uint16)


def positions(root):
    result = np.empty((len(root.block('surfaces')), 3, 3), dtype=np.float32)
    tris = triangles(root)
    for page in root.block('lightmaps'):
        for material in page.block('materials'):
            env, _ = buffers(material)
            result[ids(material)] = env['position'][tris[ids(material)]]
    return result


def references(root):
    result = [[s.get('surface') for s in root.block('leaf_surfaces')]]
    for cluster in root.block('clusters'):
        result.append([s.get('index') for s in cluster.block('surface_indices')])
        result.extend([s.get('index') for s in sub.block('surface_indices')]
                      for sub in cluster.block('subclusters'))
    return result


def synthetic():
    bsp = new_tag('sbsp')
    root = bsp.root
    root.set_block('surfaces', [child(root, 'surfaces') for _ in range(5)])
    p = child(root, 'lightmaps')
    p.set('bitmap', -1)
    root.set_block('lightmaps', [p])
    clusters = [child(root, 'clusters') for _ in range(4)]
    for c in (3, 1):
        sub = child(clusters[c], 'subclusters')
        s = child(sub, 'surface_indices')
        s.set('index', 0)
        sub.set_block('surface_indices', [s])
        clusters[c].set_block('subclusters', [sub])
    root.set_block('clusters', clusters)
    leaves, refs = [], []
    for c, surfaces in [(0, [0]), (3, [1]), (1, [1, 2]), (-1, [2, 3])]:
        leaf = child(root, 'leaves')
        leaf.set('cluster', c)
        leaf.set('surface_references', len(refs))
        leaf.set('surface_reference_count', len(surfaces))
        for index in surfaces:
            ref = child(root, 'leaf_surfaces')
            ref.set('surface', index)
            refs.append(ref)
        leaves.append(leaf)
    root.set_block('leaves', leaves)
    root.set_block('leaf_surfaces', refs)
    return bsp


def test_cluster_tie_takes_lowest():
    g = api().read_geometry(synthetic())
    np.testing.assert_array_equal(g.surface_cluster, [1, 1, 1, -32768, -32768])
    assert g.surface_cluster.dtype == np.int16


def test_last_element_requires_no_bitmap():
    bsp = synthetic()
    bsp.root.block('lightmaps')[-1].set('bitmap', 0)
    with pytest.raises(ValueError, match='^last lightmaps element has a bitmap$'):
        api().read_geometry(bsp)


def test_weld_grid_at_largest_material_size():
    module = api()
    bsp = new_tag('sbsp')
    root = bsp.root
    page = child(root, 'lightmaps')
    material = child(page, 'materials')
    env = np.zeros(60000, dtype=ENV)
    env['position'][:, 0] = np.arange(60000) % 256
    env['position'][:, 1] = np.arange(60000) // 256
    material.set_reference('shader', 'senv', r'test\grid')
    material.set('surface_count', 20000)
    material.set('rendered_vertices.vertex_count', 60000)
    material.set_data('uncompressed_vertices', env.tobytes())
    page.set_block('materials', [material])
    passthrough = child(root, 'lightmaps')
    passthrough.set('bitmap', -1)
    root.set_block('lightmaps', [page, passthrough])
    surfaces = []
    for triangle in np.arange(60000).reshape(-1, 3):
        surface = child(root, 'surfaces')
        surface.body[:] = struct.pack('>3H', *triangle)
        surfaces.append(surface)
    root.set_block('surfaces', surfaces)
    g = module.read_geometry(bsp)
    assert len(g.groups[0].env) == 60000
    np.testing.assert_array_equal(g.groups[0].env['position'][g.groups[0].tris],
                                  env['position'].reshape(20000, 3, 3))


@pytest.mark.parametrize('cluster_indices', [[], [0, 0, 1, 0], [0]])
def test_layout_refuses_cluster_lists_or_remaps_empty_lists(cluster_indices):
    module = api()
    bsp = synthetic()
    root = bsp.root
    page = child(root, 'lightmaps')
    materials = []
    env = np.zeros(6, dtype=ENV)
    env['position'][:3] = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    env[3:] = env[:3]
    env['texcoord'][3] = (0.5, 0.25)
    for start in (0, 2):
        m = child(page, 'materials')
        m.set_reference('shader', 'senv', r'test\seam')
        m.set('surfaces', start)
        m.set('surface_count', 2)
        m.set('rendered_vertices.vertex_count', 6)
        m.set('rendered_vertices.vertex_type', 1)
        m.set('lightmap_vertices.vertex_type', 3)
        m.set('ambient_color', (9., 8., 7.))
        m.set_data('uncompressed_vertices', env.tobytes())
        materials.append(m)
    page.set_block('materials', materials)
    passthrough = root.block('lightmaps')[0]
    material = materials[0].copy()
    material.set('surfaces', 4)
    material.set('surface_count', 1)
    passthrough.set_block('materials', [material])
    root.set_block('lightmaps', [page, passthrough])
    for s, tri in zip(root.block('surfaces'), [(0, 1, 2), (3, 4, 5)] * 2 + [(0, 1, 2)]):
        s.body[:] = struct.pack('>3H', *tri)
    cluster = root.block('clusters')[3]
    refs = []
    for index in cluster_indices:
        ref = child(cluster, 'surface_indices')
        ref.set('index', index)
        refs.append(ref)
    cluster.set_block('surface_indices', refs)
    before = positions(root)
    old_refs = references(root)
    geometry = module.read_geometry(bsp)
    assert len(geometry.groups) == 1
    group = geometry.groups[0]
    assert len(group.env) == 4
    np.testing.assert_array_equal(group.surface_ids, [0, 1, 2, 3])
    order = np.array([3, 2, 1, 0])
    lm = np.zeros(4, dtype=LM)
    out = module.OutMaterial(0, group.env, lm, group.tris[order].astype(np.uint16), group.surface_ids[order])
    if cluster_indices:
        before_bytes = tag_bytes(bsp)
        with pytest.raises(ValueError, match='cluster 3'):
            module.apply_layout(bsp, geometry, [module.OutPage(8, 8, [out])], 'new')
        assert tag_bytes(bsp) == before_bytes
        return
    mapping = module.apply_layout(bsp, geometry, [module.OutPage(8, 8, [out])], 'new')
    np.testing.assert_array_equal(mapping, [3, 2, 1, 0, 4])
    assert all(not c.block('surface_indices') for c in root.block('clusters'))
    assert [ref.get('surface') for ref in root.block('leaf_surfaces')] == [3, 2, 2, 1, 1, 0]
    for c in (1, 3):
        assert root.block('clusters')[c].block('subclusters')[0].block('surface_indices')[0].get('index') == 3
    after = positions(root)
    for old, new in zip(old_refs, references(root)):
        np.testing.assert_array_equal(after[np.array(new, dtype=int)], before[np.array(old, dtype=int)])
    written = root.block('lightmaps')[0].block('materials')[0]
    assert written.get('rendered_vertices.vertex_count') == written.get('lightmap_vertices.vertex_count') == 4
    assert written.get('rendered_vertices.vertex_type') == written.get('lightmap_vertices.vertex_type') == 0
    for field, value in module.LIGHTMAPPED_LIGHTING.items():
        np.testing.assert_array_equal(written.get(field), value)
    assert written.data('uncompressed_vertices') == group.env.tobytes() + lm.tobytes()
    assert len(written.data('compressed_vertices')) == 4 * 40


@pytest.mark.parametrize('kind', ['pages', 'materials', 'vertices', 'triangles', 'bsp', 'passthrough'])
def test_layout_refuses_stock_limit_overflow(kind):
    module = api()
    bsp = synthetic()
    geometry = module.read_geometry(bsp)
    env = np.zeros(64001 if kind == 'vertices' else 3, dtype=ENV)
    out = module.OutMaterial(0, env, np.zeros(len(env), dtype=LM),
                             np.zeros((20001 if kind == 'triangles' else 1, 3), dtype=np.uint16),
                             np.array([0], dtype=np.int32))
    pages = [module.OutPage(8, 8, [out] * (2049 if kind == 'materials' else 1))]
    if kind == 'pages':
        pages *= 128
    if kind == 'bsp':
        bsp.root.set_block('surfaces', [bsp.root.block('surfaces')[0]] * 131073)
    if kind == 'passthrough':
        material = child(geometry.passthrough, 'materials')
        material.set('rendered_vertices.vertex_count', 64001)
        material.set('surface_count', 5)
        geometry.passthrough.set_block('materials', [material])
        pages = []
    before = tag_bytes(bsp)
    with pytest.raises(ValueError, match='(exceeds|stock)'):
        module.apply_layout(bsp, geometry, pages, 'new')
    assert tag_bytes(bsp) == before


def two_material_bsp():
    bsp = new_tag('sbsp')
    root = bsp.root
    root.set_block('surfaces', [child(root, 'surfaces') for _ in range(2)])
    page = child(root, 'lightmaps')
    materials = []
    for i in range(2):
        material = child(page, 'materials')
        material.set_reference('shader', 'sgla', r'test\glass')
        material.set('surfaces', i)
        material.set('surface_count', 1)
        material.set('rendered_vertices.vertex_count', 3)
        material.set_data('uncompressed_vertices', np.zeros(3, dtype=ENV).tobytes())
        materials.append(material)
    page.set_block('materials', materials)
    passthrough = child(root, 'lightmaps')
    passthrough.set('bitmap', -1)
    root.set_block('lightmaps', [page, passthrough])
    return bsp


def test_merged_chart_material_keeps_surface_provenance(tmp_path):
    from lightmapper.charts import _finish, ChartSettings
    module = api()
    bsp = two_material_bsp()
    for i, m in enumerate(bsp.root.block('lightmaps')[0].block('materials')):
        env = np.zeros(3, ENV)
        env['position'] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) + [2 * i, 0, 0]
        env['normal'] = [0, 0, 1]
        m.set_data('uncompressed_vertices', env.tobytes())
        bsp.root.block('surfaces')[i].body[:] = struct.pack('>3H', 0, 1, 2)
    refs = [child(bsp.root, 'leaf_surfaces') for _ in range(2)]
    for i, ref in enumerate(refs):
        ref.set('surface', i)
    bsp.root.set_block('leaf_surfaces', refs)
    old_positions = positions(bsp.root)
    geometry = module.read_geometry(bsp)
    group = geometry.groups[0]
    pieces = []
    for row in (1, 0):
        used = np.unique(group.tris[row])
        pieces.append((0, group, used, np.array([[0, 0], [1, 0], [0, 1]]) + [2 * row, 0],
                       np.searchsorted(used, group.tris[[row]]), group.surface_ids[[row]]))
    pages = _finish([pieces], [np.array([3, 1])], ChartSettings(padding=0))
    assert len(pages[0].materials) == 1
    mapping = module.apply_layout(bsp, geometry, pages, 'new')
    np.testing.assert_array_equal(mapping, [1, 0])
    np.testing.assert_array_equal([r.get('surface') for r in bsp.root.block('leaf_surfaces')], [1, 0])
    np.testing.assert_array_equal(positions(bsp.root), old_positions[[1, 0]])
    path = tmp_path / 'merged.scenario_structure_bsp'
    write_tag(bsp, path)
    loaded = read_tag(path)
    np.testing.assert_array_equal(positions(loaded.root), old_positions[[1, 0]])
    written = loaded.root.block('lightmaps')[0].block('materials')
    assert len(written) == 1
    assert written[0].get('surface_count') == 2
    assert written[0].get('rendered_vertices.vertex_count') == 6


@pytest.mark.parametrize('difference,flags', [('plane', 1), ('centroid', 1), ('both', 1), ('both', 0)])
def test_coplanar_materials_keep_own_plane_and_centroid(difference, flags):
    module = api()
    bsp = two_material_bsp()
    materials = bsp.root.block('lightmaps')[0].block('materials')
    planes = [(0., 0., 1., 2.), (0., 1., 0., 3.) if difference != 'centroid' else (0., 0., 1., 2.)]
    centroids = [(1., 2., 3.), (4., 5., 6.) if difference != 'plane' else (1., 2., 3.)]
    for i, material in enumerate(materials):
        material.set('flags', flags)
        material.set('plane', planes[i])
        material.set('centroid', centroids[i])
    geometry = module.read_geometry(bsp)
    assert len(geometry.groups) == (2 if flags else 1)
    for i, group in enumerate(geometry.groups):
        expected = ('sgla', r'test\glass', 0, flags, 0)
        if flags:
            expected += (planes[i], centroids[i])
        assert group.key == expected
    outputs = [module.OutMaterial(i, g.env, np.zeros(len(g.env), dtype=LM),
                                  g.tris.astype(np.uint16), g.surface_ids)
               for i, g in enumerate(geometry.groups)]
    module.apply_layout(bsp, geometry, [module.OutPage(8, 8, outputs)], 'new')
    for i, material in enumerate(bsp.root.block('lightmaps')[0].block('materials')):
        assert material.get('plane') == planes[i]
        assert material.get('centroid') == centroids[i]


@pytest.mark.parametrize('invalid', ['triangle', 'negative_group', 'group'])
def test_invalid_layout_indices_refused_without_mutation(invalid):
    module = api()
    bsp = two_material_bsp()
    geometry = module.read_geometry(bsp)
    env = np.zeros(3, dtype=ENV)
    outputs = [module.OutMaterial(0, env, np.zeros(3, dtype=LM),
                                  np.array([[0, 1, 2]], dtype=np.uint16), np.array([i], dtype=np.int32))
               for i in range(2)]
    if invalid == 'triangle':
        outputs[1].tris[0, 2] = 3
    else:
        outputs[1].group = -1 if invalid == 'negative_group' else len(geometry.groups)
    before = tag_bytes(bsp)
    with pytest.raises(ValueError, match='(triangle|group)'):
        module.apply_layout(bsp, geometry, [module.OutPage(8, 8, outputs)], 'new')
    assert tag_bytes(bsp) == before


def test_empty_lightmaps_refused():
    with pytest.raises(ValueError, match='lightmaps.*empty'):
        api().read_geometry(new_tag('sbsp'))
