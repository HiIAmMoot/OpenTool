import logging
import time
import tracemalloc

import numpy as np
import pytest

from lightmapper.charts import CapError, ChartSettings, chart_and_pack, next_pow2
from lightmapper.formats.bsp import SourceGroup
from lightmapper.formats.vertices import ENV
from lightmapper.gather import SIMPLE_PARAMETERIZATION, ShaderRadiosity


def radiosity(flags=0, detail_level=1):
    # level 1 is the global density, so the older tests keep their expected areas
    return ShaderRadiosity(flags, detail_level, 0.0, np.zeros(3), np.zeros(3))


def make_group(positions, tris, normals=None, uvs=None, name='shader', sid0=0):
    env = np.zeros(len(positions), ENV)
    env['position'] = positions
    env['normal'] = [0, 0, 1] if normals is None else normals
    if uvs is not None:
        env['texcoord'] = uvs
    tris = np.asarray(tris, dtype=np.int32)
    return SourceGroup(('senv', name, 0, 0, -1), None, env, tris, np.arange(sid0, sid0 + len(tris)))


def grid(n, size=1.0, z=0.0):
    pos = [[x * size, y * size, z] for y in range(n + 1) for x in range(n + 1)]
    tris = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            tris += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return np.array(pos, np.float32), np.array(tris)


def scattered_quads(count):
    pos, tris = [], []
    for q in range(count):
        o = q * 10.0
        base = len(pos)
        pos += [[o, 0, 0], [o + 4, 0, 0], [o + 4, 4, 0], [o, 4, 0]]
        tris += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.array(pos, np.float32), np.array(tris)


def texel_triangles(page, material):
    uv = material.lm['texcoord'] * np.array([page.width, page.height])
    return uv[material.tris]


def rasterise(page):
    covered = np.zeros((page.height, page.width), int)
    for material in page.materials:
        for a, b, c in texel_triangles(page, material):
            low = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
            high = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
            xs, ys = np.meshgrid(np.arange(low[0], high[0]) + 0.5, np.arange(low[1], high[1]) + 0.5)
            d1 = (xs - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (ys - b[1])
            d2 = (xs - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (ys - c[1])
            d3 = (xs - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (ys - a[1])
            inside = ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
            covered[(ys[inside] - 0.5).astype(int), (xs[inside] - 0.5).astype(int)] += 1
    return covered


def area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def test_next_pow2():
    assert [next_pow2(n) for n in (1, 2, 3, 8, 9, 1000)] == [1, 2, 4, 8, 16, 1024]


def test_density_and_pow2():
    pos, tris = grid(10)
    pages = chart_and_pack([make_group(pos, tris)], [radiosity()], ChartSettings(texels_per_unit=2.0))
    total = sum(area(*t) for p in pages for m in p.materials for t in texel_triangles(p, m))
    assert total == pytest.approx(400, rel=0.05)
    for page in pages:
        for side in (page.width, page.height):
            assert side >= 8 and side & (side - 1) == 0


def test_uvs_inside_unit_square_and_charts_disjoint():
    pos, tris = scattered_quads(12)
    g1 = make_group(pos, tris, sid0=0)
    pos2, tris2 = grid(6, 3.0, z=5.0)
    g2 = make_group(pos2, tris2, sid0=100)
    pages = chart_and_pack([g1, g2], [radiosity(), radiosity()], ChartSettings())
    for page in pages:
        for material in page.materials:
            assert np.all(material.lm['texcoord'] >= 0) and np.all(material.lm['texcoord'] <= 1)
        assert rasterise(page).max() <= 1


def test_seam_duplicates_env():
    pos = np.array([[x, y, z] for x in (0, 4) for y in (0, 4) for z in (0, 4)], np.float32)
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    group = make_group(pos, tris, normals=pos - 2)
    pages = chart_and_pack([group], [radiosity()], ChartSettings())
    env = np.concatenate([m.env for p in pages for m in p.materials])
    assert {tuple(p) for p in pos} <= {tuple(p) for p in env['position']}
    assert len(env) > 8
    assert sorted(np.concatenate([m.surface_ids for p in pages for m in p.materials])) == list(range(12))


def test_material_split_under_caps():
    pos = [[i, y, 0] for i in range(501) for y in (0, 1)]
    tris = []
    for i in range(500):
        a = 2 * i
        tris += [[a, a + 2, a + 3], [a, a + 3, a + 1]]
    group = make_group(np.array(pos, np.float32), tris)
    s = ChartSettings(max_vertices=100)
    pages = chart_and_pack([group], [radiosity()], s)
    materials = [m for p in pages for m in p.materials]
    assert len(materials) > 1
    assert all(len(m.env) <= 100 for m in materials)
    assert sum(len(m.tris) for m in materials) == 1000
    assert sorted(np.concatenate([m.surface_ids for m in materials])) == list(range(1000))
    for m in materials:
        assert m.tris.max() < len(m.env) and len(m.env) == len(m.lm)


def test_too_many_pages_refuses():
    pos, tris = scattered_quads(60)
    s = ChartSettings(page_max=32, max_pages=2)
    with pytest.raises(CapError, match=r'max_pages \(2\)'):
        chart_and_pack([make_group(pos, tris)], [radiosity()], s)


def test_degenerate_triangles():
    pos, tris = grid(4)
    pos = np.vstack([pos, [[20, 0, 0], [21, 0, 0], [22, 0, 0]]]).astype(np.float32)
    tris = np.vstack([tris, [[25, 26, 27], [0, 0, 1]]])
    group = make_group(pos, tris)
    pages = chart_and_pack([group], [radiosity()], ChartSettings())
    ids = np.concatenate([m.surface_ids for p in pages for m in p.materials])
    assert sorted(ids) == list(range(len(tris)))
    for page in pages:
        for m in page.materials:
            assert np.all(np.isfinite(m.lm['texcoord']))


def quad_uv_group(uvs, name='simple'):
    pos = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], np.float32)
    return make_group(pos, [[0, 1, 2], [0, 2, 3]], uvs=uvs, name=name)


def test_simple_param_overlap_falls_back(caplog):
    pos = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [10, 0, 0], [14, 0, 0], [10, 4, 0]], np.float32)
    uvs = [[0, 0], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]]
    group = make_group(pos, [[0, 1, 2], [3, 4, 5]], uvs=uvs, name='overlapper')
    with caplog.at_level(logging.WARNING, logger='lightmapper.charts'):
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    assert 'overlapper' in caplog.text and 'overlap' in caplog.text
    assert sum(len(m.tris) for p in pages for m in p.materials) == 2
    for page in pages:
        assert rasterise(page).max() <= 1


def test_simple_param_keeps_layout():
    uvs = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.7], [0.1, 0.7]]
    group = quad_uv_group(uvs)
    pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    material = pages[0].materials[0]
    source = material.env['texcoord'].astype(np.float64)
    target = material.lm['texcoord'].astype(np.float64) * [pages[0].width, pages[0].height]
    design = np.hstack([source, np.ones((len(source), 1))])
    fit, *_ = np.linalg.lstsq(design, target, rcond=None)
    assert np.abs(design @ fit - target).max() < 1e-3
    # xatlas rounds a uv chart's extent up to whole texels, so the two scales differ by under a texel
    linear = fit[:2]
    assert linear[:, 0] @ linear[:, 1] == pytest.approx(0, abs=1e-3)
    assert np.linalg.norm(linear[:, 0]) == pytest.approx(np.linalg.norm(linear[:, 1]), rel=0.05)


def test_mixed_groups_share_a_page():
    pos, tris = grid(3, 2.0)
    ordinary = make_group(pos, tris, name='ordinary')
    simple = quad_uv_group([[0, 0], [1, 0], [1, 1], [0, 1]])
    pages = chart_and_pack([simple, ordinary], [radiosity(SIMPLE_PARAMETERIZATION), radiosity()], ChartSettings())
    kinds = [{m.group for m in p.materials} for p in pages]
    assert len(pages) == 1
    assert kinds == [{0, 1}]


def test_large_chart_split_keeps_density_on_one_page():
    pos, tris = grid(100, size=3)
    pages = chart_and_pack([make_group(pos, tris)], [radiosity()], ChartSettings(texels_per_unit=1))
    assert len(pages) == 1
    assert sum(p.width * p.height for p in pages) <= 512 * 512
    assert len(pages[0].materials) == 1
    assert len(pages[0].materials[0].env) > len(pos)
    areas = np.array([area(*t) for p in pages for m in p.materials for t in texel_triangles(p, m)])
    assert np.median(np.abs(areas / 4.5 - 1)) <= 0.01
    assert abs(areas.sum() / 90000 - 1) <= 0.02
    assert sorted(np.concatenate([m.surface_ids for p in pages for m in p.materials])) == list(range(20000))


def test_rectangle_repack_preserves_each_triangle_area():
    from lightmapper.charts import _finish, _packing_areas, _pack_rectangles, _run_atlas
    pos, tris = scattered_quads(12)
    group = make_group(pos, tris)
    s = ChartSettings()
    atlas_pages, extent = _run_atlas([(0, group, None, 1.0)], False, s)
    before = _finish(atlas_pages, extent, s)
    after = _pack_rectangles([piece for page in atlas_pages for piece in page], s)
    np.testing.assert_allclose(_packing_areas(after), _packing_areas(before), rtol=1e-6)
    for page in after:
        assert rasterise(page).max() <= 1


def test_chart_split_at_stock_bsp_limit():
    from lightmapper.charts import _bounded_charts
    pos, tris = grid(256)
    rows = list(_bounded_charts(pos[:, :2], tris, np.arange(131072)))
    np.testing.assert_array_equal(np.sort(np.concatenate(rows)), np.arange(131072))
    assert len(rows) > 1
    assert all(np.ptp(pos[tris[r], :2].reshape(-1, 2), axis=0).max() <= 120 for r in rows)


def test_split_density_against_unsplit_layout_with_group_scales():
    from lightmapper.charts import _density_change, _piece_areas, _run_atlas, _packing_areas
    pos, tris = grid(100, size=3)
    groups = [make_group(pos, tris), make_group(pos + [0, 0, 10], tris, sid0=20000)]
    shaders = [radiosity(detail_level=1), radiosity(detail_level=2)]
    s = ChartSettings(texels_per_unit=1)
    before = _piece_areas([piece for page in _run_atlas(
        [(0, groups[0], None, 1.), (1, groups[1], None, 0.6)], False, s, limit_charts=False)[0]
        for piece in page])
    pages = chart_and_pack(groups, shaders, s)
    after = _packing_areas(pages)
    median, total = _density_change(before, after)
    assert median <= 0.01
    assert total <= 0.02
    assert s.texels_per_unit == 1
    assert [s.density_scale(shader) for shader in shaders] == [1., 0.6]
    assert texel_area(pages, 1) / texel_area(pages, 0) == pytest.approx(0.36, rel=0.02)


def test_density_comparison_at_stock_bsp_limit():
    from lightmapper.charts import _density_change
    reference = np.ones(131072)
    assert _density_change(reference, reference * 0.995) == pytest.approx((0.005, 0.005))
    median, total = _density_change(reference, reference * 0.97)
    assert median > 0.01 and total > 0.02


def test_extent_on_power_of_two():
    pos, tris = grid(29, 1.0)
    s = ChartSettings(texels_per_unit=2.0, padding=4)
    pages = chart_and_pack([make_group(pos, tris)], [radiosity()], s)
    assert len(pages) == 1
    page = pages[0]
    texels = np.concatenate([texel_triangles(page, m).reshape(-1, 2) for m in page.materials])
    assert texels.max(axis=0).max() > 60  # shifted past 62 by the low-edge margin
    assert (page.width, page.height) == (128, 128)
    assert np.all(page.width - texels[:, 0] >= s.padding)
    assert np.all(page.height - texels[:, 1] >= s.padding)
    assert np.all(texels >= s.padding)


def test_oversized_chart_refuses_instead_of_losing_density():
    pos, tris = grid(20, 1.0)
    s = ChartSettings(page_max=16, padding=4)
    with pytest.raises(CapError, match='page_max'):
        chart_and_pack([make_group(pos, tris)], [radiosity()], s)


@pytest.mark.parametrize('factors,accepted', [([0.998] * 10, True), ([0.99] * 10, False),
                                            ([1.] * 9 + [0.5], False)])
def test_density_selection_checks_median_and_total(monkeypatch, factors, accepted):
    from lightmapper import charts
    pos, tris = scattered_quads(10)
    group = make_group(pos, tris)
    original = charts._run_atlas

    def candidate(entries, uv_mode, s, **kwargs):
        pages, extent = original(entries, uv_mode, s, **kwargs)
        if kwargs.get('limit_charts', True):
            changed = []
            for page in pages:
                pieces = []
                for piece in page:
                    factor = factors[int(piece[5][0]) // 2]
                    uv = piece[3]
                    moved = uv[0] + (uv - uv[0]) * factor
                    pieces.append((*piece[:3], moved, *piece[4:]))
                changed.append(pieces)
            pages = changed
        return pages, extent

    monkeypatch.setattr(charts, '_run_atlas', candidate)
    s = ChartSettings(padding=1, texels_per_unit=2)
    pages = chart_and_pack([group], [radiosity()], s)
    expected = 640 * factors[0] ** 2 if accepted else 640
    assert texel_area(pages, 0) == pytest.approx(expected, abs=0.01)
    assert s.texels_per_unit == 2


def test_unsplittable_oversized_triangle_refuses_instead_of_shrinking():
    group = make_group([[0, 0, 0], [100, 0, 0], [0, 100, 0]], [[0, 1, 2]])
    with pytest.raises(CapError, match='page_max'):
        chart_and_pack([group], [radiosity()], ChartSettings(page_max=16, padding=1))


def quad_group(origin, size, uv_min, uv_size, name, sid0):
    x, y = origin
    pos = [[x, y, 0], [x + size, y, 0], [x + size, y + size, 0], [x, y + size, 0]]
    u, v = uv_min
    uvs = [[u, v], [u + uv_size, v], [u + uv_size, v + uv_size], [u, v + uv_size]]
    return pos, uvs


def test_touching_islands_stay_disjoint_and_affine():
    # A and B touch in texture space with different world/uv ratios; C is apart with a third ratio
    parts = [quad_group((0, 0), 4, (0, 0), 1, 'a', 0), quad_group((10, 0), 2, (1, 0), 1, 'b', 2),
             quad_group((20, 0), 8, (5, 5), 0.5, 'c', 4)]
    pos = np.array([p for part in parts for p in part[0]], np.float32)
    uvs = np.array([u for part in parts for u in part[1]], np.float32)
    tris = [[4 * q + k for k in tri] for q in range(3) for tri in ([0, 1, 2], [0, 2, 3])]
    group = make_group(pos, tris, uvs=uvs)
    pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    for page in pages:
        assert rasterise(page).max() <= 1
    for q in range(3):
        rows = []
        for page in pages:
            for m in page.materials:
                for tri, sid in zip(m.tris, m.surface_ids):
                    if sid // 2 == q:
                        rows.append((m.env['texcoord'][tri], m.lm['texcoord'][tri] * [page.width, page.height]))
        source = np.concatenate([r[0] for r in rows]).astype(np.float64)
        target = np.concatenate([r[1] for r in rows]).astype(np.float64)
        design = np.hstack([source, np.ones((len(source), 1))])
        fit, *_ = np.linalg.lstsq(design, target, rcond=None)
        assert np.abs(design @ fit - target).max() < 1e-2


def test_huge_tiling_uvs_are_bounded():
    pos, uvs = quad_group((0, 0), 4, (0, 0), 50, 'tile', 0)
    group = make_group(np.array(pos, np.float32), [[0, 1, 2], [0, 2, 3]], uvs=np.array(uvs, np.float32))
    tracemalloc.start()
    started = time.time()
    pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    elapsed = time.time() - started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert elapsed < 5 and peak < 100e6
    assert sum(len(m.tris) for p in pages for m in p.materials) == 2


def test_zero_uv_area_falls_back(caplog):
    pos = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], np.float32)
    uvs = [[0, 0], [1, 0], [2, 0], [0, 1]]
    group = make_group(pos, [[0, 1, 2], [0, 2, 3]], uvs=uvs, name='flat')
    with caplog.at_level(logging.WARNING, logger='lightmapper.charts'):
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    assert 'flat' in caplog.text and 'no texture area' in caplog.text
    for page in pages:
        for m in page.materials:
            for tri in texel_triangles(page, m):
                assert area(*tri) > 0.5


def test_chart_settings_validation():
    for bad in (dict(page_max=100), dict(page_min=6), dict(page_min=64, page_max=32)):
        with pytest.raises(ValueError):
            ChartSettings(**bad)


def test_materials_per_page_cap_refuses():
    pos, tris = grid(2)
    groups = [make_group(pos, tris, sid0=i * 8) for i in range(3)]
    s = ChartSettings(max_materials_per_page=2, max_pages=1)
    with pytest.raises(CapError, match=r'max_materials_per_page \(2\)'):
        chart_and_pack(groups, [radiosity()] * 3, s)


def test_same_group_charts_aggregate_before_material_page_cap():
    from lightmapper.charts import _density_checked_pieces, _pack_rectangles
    pos, tris = scattered_quads(5)
    s = ChartSettings(padding=1, page_max=128, max_materials_per_page=2, max_pages=3)
    pieces = _density_checked_pieces([(0, make_group(pos, tris), None, 1.)], False, s)
    pages = _pack_rectangles(pieces, s)
    assert len(pages) == 1
    assert len(pages[0].materials) == 1
    assert len(pages[0].materials[0].env) == 20
    assert sorted(pages[0].materials[0].surface_ids) == list(range(10))
    assert texel_area(pages, 0) == pytest.approx(320, abs=0.01)
    for page in pages:
        assert rasterise(page).max() <= 1


@pytest.mark.parametrize('limits, counts', [({'max_vertices': 7}, [7, 3]),
                                          ({'max_surfaces': 3}, [7, 3])])
def test_group_merge_splits_greedily_at_actual_caps(limits, counts):
    from lightmapper.charts import _pack_rectangles
    pieces = rectangle_pieces([(8, 8)] * 2)
    pieces = [(0, *p[1:]) for p in pieces]
    pages = _pack_rectangles(pieces, ChartSettings(padding=0, page_max=64, **limits))
    assert len(pages) == 1
    assert [len(m.env) for m in pages[0].materials] == counts
    assert [len(m.tris) for m in pages[0].materials] == [3, 1]
    np.testing.assert_array_equal(np.concatenate([m.surface_ids for m in pages[0].materials]), [0, 1, 2, 3])


def test_full_material_budget_still_accepts_existing_group():
    from lightmapper.charts import _pack_rectangles
    pieces = rectangle_pieces([(8, 8)] * 3)
    pieces[2] = (0, *pieces[2][1:])
    pages = _pack_rectangles(pieces, ChartSettings(padding=0, page_max=64, max_materials_per_page=2))
    assert len(pages) == 1
    assert [m.group for m in pages[0].materials] == [0, 1]
    assert [len(m.tris) for m in pages[0].materials] == [4, 2]


def test_group_cap_splits_count_toward_material_budget():
    from lightmapper.charts import _pack_rectangles
    pieces = [(0, *p[1:]) for p in rectangle_pieces([(8, 8)] * 3)]
    pages = _pack_rectangles(pieces, ChartSettings(padding=0, page_max=64,
                                                 max_vertices=4, max_materials_per_page=2))
    assert len(pages) == 2
    assert sorted(len(p.materials) for p in pages) == [1, 2]


def rectangle_pieces(sizes):
    pieces = []
    for i, (w, h) in enumerate(sizes):
        group = make_group([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]],
                           [[0, 1, 2], [0, 2, 3]], sid0=2 * i)
        pieces.append((i, group, np.arange(4), group.env['position'][:, :2].copy(),
                       group.tris, group.surface_ids))
    return pieces


def test_page_count_wins_even_when_multiple_pages_save_area():
    from lightmapper.charts import _pack_rectangles
    pages = _pack_rectangles(rectangle_pieces([(24, 16)] * 5),
                             ChartSettings(padding=0, page_max=64))
    # two pages can spend only 2560 texels; the one-page solution costs 4096
    assert [(p.width, p.height) for p in pages] == [(64, 64)]
    assert rasterise(pages[0]).max() <= 1


def test_page_count_is_minimal_when_one_page_cannot_hold_the_area():
    from lightmapper.charts import _pack_rectangles
    sizes = [(16, 16), (8, 16), (4, 16), (8, 8), (16, 4), (4, 16),
             (16, 4), (12, 12), (8, 8), (24, 12), (4, 12), (24, 12)]
    # 1536 rectangle texels exceed one 32x32 page; a two-page arrangement exists
    assert sum(w * h for w, h in sizes) == 1536
    pages = _pack_rectangles(rectangle_pieces(sizes), ChartSettings(padding=0, page_max=32))
    assert len(pages) == 2
    assert sum(p.width * p.height for p in pages) <= 2048
    for page in pages:
        assert rasterise(page).max() <= 1


def test_material_budget_forces_split_despite_one_page_geometry_fit():
    from lightmapper.charts import _pack_rectangles
    pieces = rectangle_pieces([(8, 8)] * 5)
    together = _pack_rectangles(pieces, ChartSettings(padding=0, page_max=64))
    split = _pack_rectangles(pieces, ChartSettings(padding=0, page_max=64, max_materials_per_page=3))
    assert len(together) == 1
    assert len(split) == 2
    assert sorted(len(p.materials) for p in split) == [2, 3]


def test_least_area_breaks_equal_page_count_ties():
    from lightmapper.charts import _pack_rectangles
    sizes = [(36, 20), (8, 40), (36, 4), (36, 28), (16, 28), (8, 16),
             (36, 36), (32, 8), (28, 16), (36, 36), (20, 16), (24, 20)]
    # 6864 chart texels rule out 4096; a 128x128 candidate is valid but wasteful
    assert sum(w * h for w, h in sizes) == 6864
    pages = _pack_rectangles(rectangle_pieces(sizes), ChartSettings(padding=0, page_max=128))
    assert len(pages) == 1
    assert pages[0].width * pages[0].height == 8192
    assert rasterise(pages[0]).max() <= 1


def test_spilled_page_is_tight_after_filling_earlier_gaps():
    from lightmapper.charts import _pack_rectangles
    pages = _pack_rectangles(rectangle_pieces([(24, 24), (16, 20)] + [(8, 8)] * 5),
                             ChartSettings(padding=0, page_max=32))
    assert len(pages) == 2
    assert sorted(page.width * page.height for page in pages) == [512, 1024]
    assert sum(texel_area(pages, group) for group in range(7)) == pytest.approx(1216)
    for page in pages:
        assert rasterise(page).max() <= 1
        extent = np.concatenate([m.lm['texcoord'] * [page.width, page.height] for m in page.materials]).max(axis=0)
        assert (page.width, page.height) == tuple(max(8, next_pow2(int(np.ceil(e)))) for e in extent)


def test_filling_earlier_shelves_avoids_an_unnecessary_spill():
    from lightmapper.charts import _pack_rectangles
    pages = _pack_rectangles(rectangle_pieces([(24, 24), (16, 8)] + [(8, 8)] * 3),
                             ChartSettings(padding=0, page_max=32))
    assert [(page.width, page.height) for page in pages] == [(32, 32)]
    assert rasterise(pages[0]).max() <= 1


def test_indexed_shelf_packing_at_stock_bsp_chart_count():
    from lightmapper.charts import _shelf_pack
    count = 131072
    piece = rectangle_pieces([(1, 1)])[0]
    rectangles = [(piece, piece[3], (1, 1))] * count
    s = ChartSettings(padding=0)
    pages, extents = _shelf_pack(rectangles, range(count), 128, 16, False, s)
    assert len(pages) == 64
    assert all(len(page) == 2048 for page in pages)
    np.testing.assert_array_equal(extents, np.tile([128, 16], (64, 1)))


def test_material_split_by_surface_cap():
    pos = [[i, y, 0] for i in range(251) for y in (0, 1)]
    tris = []
    for i in range(250):
        a = 2 * i
        tris += [[a, a + 2, a + 3], [a, a + 3, a + 1]]
    group = make_group(np.array(pos, np.float32), tris)
    pages = chart_and_pack([group], [radiosity()], ChartSettings(max_surfaces=100))
    materials = [m for p in pages for m in p.materials]
    assert len(materials) > 1 and all(len(m.tris) <= 100 for m in materials)
    assert sorted(np.concatenate([m.surface_ids for m in materials])) == list(range(500))


def texel_area(pages, group):
    return sum(area(*t) for p in pages for m in p.materials if m.group == group
               for t in texel_triangles(p, m))


def two_quads(*levels, size=10.0):
    groups = []
    for k, _ in enumerate(levels):
        pos, tris = grid(1, size, z=float(k))
        groups.append(make_group(pos, tris, name=f'shader{k}', sid0=2 * k))
    return groups, [radiosity(detail_level=level) for level in levels]


def test_texel_density_follows_the_detail_level():
    groups, shaders = two_quads(0, 1)
    pages = chart_and_pack(groups, shaders, ChartSettings(texels_per_unit=2.0))
    high, medium = texel_area(pages, 0), texel_area(pages, 1)
    assert medium == pytest.approx(100 * 2.0 ** 2, rel=0.05)
    assert high / medium == pytest.approx(2.667 ** 2, rel=0.08)


def test_every_detail_level_has_its_own_multiplier():
    groups, shaders = two_quads(2, 1, 0)
    pages = chart_and_pack(groups, shaders, ChartSettings(texels_per_unit=1.0))
    low = texel_area(pages, 0)
    assert low / texel_area(pages, 1) == pytest.approx(0.6 ** 2, rel=0.08)
    assert texel_area(pages, 2) / texel_area(pages, 1) == pytest.approx(2.667 ** 2, rel=0.08)


def test_detail_level_scale_is_a_setting():
    groups, shaders = two_quads(0, 1)
    pages = chart_and_pack(groups, shaders, ChartSettings(texels_per_unit=2.0, detail_level_scale=(1.0, 1.0, 1.0, 1.0)))
    assert texel_area(pages, 0) == pytest.approx(texel_area(pages, 1), rel=0.05)


def test_turd_level_still_gets_a_chart():
    groups, shaders = two_quads(3, 1)
    pages = chart_and_pack(groups, shaders, ChartSettings(texels_per_unit=2.0))
    assert sum(len(m.tris) for p in pages for m in p.materials if m.group == 0) == 2
    for page in pages:
        for material in (m for m in page.materials if m.group == 0):
            texels = material.lm['texcoord'] * [page.width, page.height]
            assert (texels.max(axis=0) - texels.min(axis=0) >= 1).all()
        assert rasterise(page).max() <= 1


def test_a_small_turd_level_triangle_is_not_scaled_into_xatlass_zero_area_cut():
    # 0.01 units across is 5e-5 square units; times 0.03 squared it falls under xatlas's area epsilon
    pos = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], np.float32)
    group = make_group(pos, [[0, 1, 2]], name='small')
    pages = chart_and_pack([group], [radiosity(detail_level=3)], ChartSettings(texels_per_unit=100.0))
    material = pages[0].materials[0]
    texels = material.lm['texcoord'] * [pages[0].width, pages[0].height]
    assert np.ptp(texels, axis=0).min() > 0


def test_simple_param_density_follows_the_detail_level():
    def run(level):
        # a big quad, since xatlas rounds a uv chart's extent up to whole texels
        pos = np.array([[0, 0, 0], [40, 0, 0], [40, 40, 0], [0, 40, 0]], np.float32)
        group = make_group(pos, [[0, 1, 2], [0, 2, 3]], uvs=[[0.1, 0.2], [0.9, 0.2], [0.9, 0.7], [0.1, 0.7]])
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION, level)], ChartSettings(texels_per_unit=1.0))
        return texel_area(pages, 0)
    assert run(0) / run(1) == pytest.approx(2.667 ** 2, rel=0.12)


def test_small_simple_param_level_3_island_gets_a_real_chart():
    group = make_group([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], [[0, 1, 2]],
                       uvs=[[0, 0], [1, 0], [0, 1]])
    pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION, 3)],
                           ChartSettings(texels_per_unit=100.0))
    assert sum(len(m.tris) for p in pages for m in p.materials) == 1
    # ignored uv faces retain tiny input uvs; a packed chart spans at least one texel
    assert texel_area(pages, 0) >= 0.5


def test_simple_param_no_shrink_at_stock_material_surface_limit():
    pos, tris = grid(100, size=0.01)
    group = make_group(pos, tris, uvs=pos[:, :2])
    pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION, 3)],
                           ChartSettings(texels_per_unit=100.0))
    ids = np.concatenate([m.surface_ids for p in pages for m in p.materials])
    np.testing.assert_array_equal(np.sort(ids), np.arange(20000))
    assert texel_area(pages, 0) == pytest.approx(9, rel=0.4)


@pytest.mark.parametrize('bad', [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (1.0, -1.0, 1.0, 1.0)])
def test_detail_level_scale_validation(bad):
    with pytest.raises(ValueError, match='detail_level_scale'):
        ChartSettings(detail_level_scale=bad)


def test_detail_level_scale_defaults_to_tools_ratios():
    assert ChartSettings().detail_level_scale == (2.667, 1.0, 0.6, 0.03)


def test_pack_search_reduces_wasted_area_without_changing_density():
    pos, tris = scattered_quads(8)
    group = make_group(pos, tris)
    s = ChartSettings(page_max=128, padding=1)
    pages = chart_and_pack([group], [radiosity()], s)
    # the old single 64x64 packing spent 4096 texels on 512 square texels of charts
    assert sum(p.width * p.height for p in pages) == 2048
    assert len(pages) == 1
    assert texel_area(pages, 0) == pytest.approx(512, abs=1e-3)
    again = chart_and_pack([group], [radiosity()], s)
    assert [(p.width, p.height, len(p.materials)) for p in pages] == [
        (p.width, p.height, len(p.materials)) for p in again]
    for p, q in zip(pages, again):
        for m, n in zip(p.materials, q.materials):
            np.testing.assert_array_equal(m.lm, n.lm)


def test_pack_search_keeps_a_filled_page_and_rejects_chart_shrinking():
    pos, tris = grid(1, size=28)
    pages = chart_and_pack([make_group(pos, tris)], [radiosity()],
                           ChartSettings(page_max=64, texels_per_unit=2, padding=1))
    assert [(p.width, p.height) for p in pages] == [(64, 64)]
    assert texel_area(pages, 0) == pytest.approx(3136, abs=0.1)


def test_pack_search_respects_page_budget():
    pos, tris = scattered_quads(8)
    pages = chart_and_pack([make_group(pos, tris)], [radiosity()],
                           ChartSettings(page_max=128, padding=1, max_pages=1))
    assert len(pages) == 1
    assert sum(p.width * p.height for p in pages) == 2048
    assert texel_area(pages, 0) == pytest.approx(512, abs=1e-3)


def test_pack_search_at_stock_material_surface_limit():
    pos, tris = grid(100)
    pages = chart_and_pack([make_group(pos, tris)], [radiosity()],
                           ChartSettings(page_max=256, texels_per_unit=1, padding=1))
    assert [(p.width, p.height) for p in pages] == [(128, 128)]
    np.testing.assert_array_equal(np.sort(np.concatenate([m.surface_ids for p in pages for m in p.materials])),
                                  np.arange(20000))
    assert texel_area(pages, 0) == pytest.approx(10000, rel=0.01)


def test_packing_area_comparison_at_stock_bsp_limit():
    from lightmapper.charts import _packing_areas
    from lightmapper.formats.bsp import OutMaterial, OutPage
    from lightmapper.formats.vertices import LM

    materials, expected = [], []
    for group, start in enumerate(range(0, 131072, 20000)):
        count = min(20000, 131072 - start)
        lm = np.zeros(3, LM)
        lm['texcoord'] = [[0, 0], [(group + 1) / 16, 0], [0, 0.125]]
        materials.append(OutMaterial(group, np.zeros(3, ENV), lm,
            np.tile([0, 1, 2], (count, 1)), np.arange(count)[::-1]))
        expected.append(np.full(count, (group + 1) / 2))
    actual = _packing_areas([OutPage(8, 8, materials[::-1])])
    np.testing.assert_array_equal(actual, np.concatenate(expected))


def fold(hard, degrees=20.0, n=4):
    # two n x n planar grids meeting at x = n; the second tilts up by `degrees`
    tilt = np.radians(degrees)
    left = np.array([[x, y, 0.0] for y in range(n + 1) for x in range(n + 1)], np.float32)
    right = np.array([[n + x * np.cos(tilt), y, x * np.sin(tilt)] for y in range(n + 1) for x in range(n + 1)], np.float32)
    flat, flat_tris = grid(n)
    up_normal = np.array([-np.sin(tilt), 0, np.cos(tilt)])
    if hard:
        pos = np.concatenate([left, right])
        tris = np.concatenate([flat_tris, flat_tris + len(left)])
        normals = np.concatenate([np.tile([0, 0, 1.0], (len(left), 1)), np.tile(up_normal, (len(right), 1))])
        return make_group(pos, tris, normals=normals)
    # smooth: the shared edge is one vertex per row, with the average normal
    shared = {(n, y): y * (n + 1) + n for y in range(n + 1)}
    right_ids = np.array([shared[(n, y)] if x == 0 else len(left) + y * (n + 1) + x
                          for y in range(n + 1) for x in range(n + 1)])
    pos = np.concatenate([left, right])
    tris = np.concatenate([flat_tris, right_ids[flat_tris]])
    average = (np.array([0, 0, 1.0]) + up_normal)
    average /= np.linalg.norm(average)
    normals = np.concatenate([np.tile([0, 0, 1.0], (len(left), 1)), np.tile(up_normal, (len(right), 1))])
    normals[list(shared.values())] = average
    return make_group(pos, tris, normals=normals)


def chart_count(pages, group=None):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    total = 0
    for page in pages:
        for material in page.materials:
            if group is not None and material.group != group:
                continue
            # (vertices sharing a uv are one: the input repeats a crease's corners with distinct normals)
            _, ids = np.unique(np.round(material.lm['texcoord'] * [page.width, page.height], 3), axis=0, return_inverse=True)
            tris = ids.reshape(-1)[material.tris.astype(int)]
            count = ids.max() + 1
            edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]]])
            graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
            total += connected_components(graph, directed=False)[0]
    return total


def test_hard_crease_splits_into_charts_by_default():
    pages = chart_and_pack([fold(hard=True)], [radiosity()], ChartSettings())
    assert chart_count(pages) == 2


def test_smooth_fold_stays_one_chart():
    pages = chart_and_pack([fold(hard=False)], [radiosity()], ChartSettings())
    assert chart_count(pages) == 1


def test_hard_crease_may_merge_when_seams_are_not_respected():
    pages = chart_and_pack([fold(hard=True)], [radiosity()], ChartSettings(respect_normal_seams=False))
    assert chart_count(pages) == 1


def no_uv_area_fold(hard, name):
    # all-equal uvs have no texture area, so a simple-parameterization group falls back to normal charting
    group = fold(hard)
    group.env['texcoord'] = 0
    group.key = ('senv', name, 0, 0, -1)
    return group


def test_simple_parameterization_fallback_stays_joined_across_a_hard_crease():
    simple = no_uv_area_fold(True, 'rock')
    plain = no_uv_area_fold(True, 'steel')
    pages = chart_and_pack([simple, plain], [radiosity(SIMPLE_PARAMETERIZATION), radiosity()], ChartSettings())
    assert chart_count(pages, group=0) == 1
    assert chart_count(pages, group=1) == 2
