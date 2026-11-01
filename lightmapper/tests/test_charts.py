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
    # (both triangles share their uv values, so they are one island that overlaps itself)
    assert 'overlapper' in caplog.text and '1 island(s) split where they overlap themselves, 0 island(s) charted normally (none)' in caplog.text
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
    s = ChartSettings(page_max=1024)
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


def test_oversized_chart_still_hits_the_page_count_cap():
    pos, tris = grid(20, 1.0)
    s = ChartSettings(page_max=16, padding=4, max_pages=32)
    with pytest.raises(CapError, match=r'max_pages \(32\)'):
        chart_and_pack([make_group(pos, tris)], [radiosity()], s)


def test_oversized_grid_now_packs_at_16():
    # bisected into parts that fit a 16 texel page, instead of refused for exceeding it
    pos, tris = grid(20, 1.0)
    pages = chart_and_pack([make_group(pos, tris)], [radiosity()], ChartSettings(page_max=16, padding=4))
    assert all(max(p.width, p.height) <= 16 for p in pages)
    areas = np.array([area(*t) for p in pages for m in p.materials for t in texel_triangles(p, m)])
    assert len(areas) == 800
    np.testing.assert_allclose(areas, 2.0, rtol=1e-5)


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
    # (a triangle cannot be cut, so it is refused by name rather than shrunk)
    from lightmapper.charts import TriangleTooWide
    group = make_group([[0, 0, 0], [100, 0, 0], [0, 100, 0]], [[0, 1, 2]])
    with pytest.raises(TriangleTooWide) as caught:
        chart_and_pack([group], [radiosity()], ChartSettings(page_max=16, padding=1))
    assert caught.value.shader == 'shader'


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


def cells_piece(cells, cell=10, offset=(0, 0), name='shader'):
    # one quad of two triangles per (x, y) cell, each with its own vertices, so only uv values join neighbours
    positions, tris = [], []
    for x, y in cells:
        base = len(positions)
        positions += [[x, y, 0], [x + 1, y, 0], [x + 1, y + 1, 0], [x, y + 1, 0]]
        tris += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return piece_from_group(make_group(np.array(positions, np.float32), tris, name=name), cell, offset)


def piece_from_group(group, cell, offset=(0, 0)):
    return (0, group, np.arange(len(group.env)), group.env['position'][:, :2].astype(np.float64) * cell + offset,
            group.tris, group.surface_ids)


def piece_from_grid(n, cell, offset=(0, 0), name='shader'):
    return piece_from_group(make_group(*grid(n), name=name), cell, offset)


def test_components_follow_shared_uv_values():
    from lightmapper.charts import _components
    quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11]])
    uv = np.concatenate([quad, quad + [1, 0], quad + [10, 10]])
    found = _components(uv, tris, np.arange(6))
    assert [list(part) for part in found] == [[0, 1, 2, 3], [4, 5]]
    # the second quad could sit at the first one's positions, but its uvs touch nothing
    uv = np.concatenate([quad, quad + [5, 0]])
    assert [list(part) for part in _components(uv, tris[:4], np.arange(4))] == [[0, 1], [2, 3]]


def test_bisection_keeps_uvs_and_areas():
    from lightmapper.charts import _bisect_piece, _padded_size
    piece = cells_piece([(x, 0) for x in range(40)])
    parts = _bisect_piece(piece, 128, 4)
    assert len(parts) > 1
    assert all(_padded_size(p[3], 4).max() <= 128 for p in parts)
    assert sum(len(p[4]) for p in parts) == len(piece[4])
    rows = np.concatenate([p[5] for p in parts])
    corners = np.concatenate([p[3][p[4]] for p in parts])[np.argsort(rows)]
    source = piece[3][piece[4]]
    np.testing.assert_array_equal(corners, source)
    for p in parts:
        np.testing.assert_array_equal(p[3], piece[3][p[2]])

    def areas(c):
        return 0.5 * np.abs((c[:, 1, 0] - c[:, 0, 0]) * (c[:, 2, 1] - c[:, 0, 1])
                            - (c[:, 1, 1] - c[:, 0, 1]) * (c[:, 2, 0] - c[:, 0, 0]))
    np.testing.assert_allclose(areas(corners), areas(source), atol=1e-9)


def test_bisection_parts_are_uv_connected():
    from lightmapper.charts import _bisect_piece, _components
    # a U of 10-texel cells, 5 wide and 14 tall: a bar along y=0 and arms up x=0 and x=4.
    # The median cut on y leaves the bar and the arms' lower rows below it and, above it, the two arms
    # joined only through the bar; each arm alone is under 12 cells tall, so the recursion would stop there
    cells = [(x, 0) for x in range(5)] + [(x, y) for x in (0, 4) for y in range(1, 14)]
    parts = _bisect_piece(cells_piece(cells), 128, 4)
    assert len(parts) > 1
    for part in parts:
        assert len(_components(part[3], part[4], np.arange(len(part[4])))) == 1


def test_pieces_that_fit_are_untouched():
    from lightmapper.charts import _split_oversized
    piece = cells_piece([(0, 0), (1, 0)])
    parts, split, made = _split_oversized([piece], 128, 4)
    assert len(parts) == 1 and parts[0] is piece
    assert (split, made) == (0, 0)


def test_oversized_pieces_are_counted():
    from lightmapper.charts import _split_oversized
    fits, wide = cells_piece([(0, 0)]), cells_piece([(x, 0) for x in range(40)])
    parts, split, made = _split_oversized([fits, wide], 128, 4)
    assert parts[0] is fits
    assert (split, made) == (1, len(parts) - 1)


def test_widest_triangle_names_its_shader():
    from lightmapper.charts import _widest_triangle
    narrow = cells_piece([(0, 0)], name='narrow')
    wide = piece_from_group(make_group(np.array([[0, 0, 0], [600, 0, 0], [0, 5, 0]], np.float32), [[0, 1, 2]], name='wide'), 1)
    assert _widest_triangle([narrow, wide], 4) == (608, 'wide')
    assert _widest_triangle([], 4) == (0, '')


def test_bisection_with_a_degenerate_triangle_terminates():
    from lightmapper.charts import _bisect_piece, _padded_size
    index, group, xref, uv, tris, sid = cells_piece([(x, 0) for x in range(40)], offset=(100, 100))
    n = len(uv)
    piece = (index, group, np.r_[xref, np.arange(n, n + 3)], np.concatenate([uv, np.zeros((3, 2))]),
             np.concatenate([tris, [[n, n + 1, n + 2]]]).astype(np.int32), np.r_[sid, len(sid)])
    start = time.perf_counter()
    parts = _bisect_piece(piece, 64, 4)
    assert time.perf_counter() - start < 10
    assert all(_padded_size(p[3], 4).max() <= 64 for p in parts)
    assert sum(len(p[4]) for p in parts) == len(piece[4])


def test_bisection_of_a_large_island_is_fast():
    from lightmapper.charts import _bisect_piece, _padded_size
    piece = piece_from_grid(64, 8)
    start = time.perf_counter()
    parts = _bisect_piece(piece, 32, 4)
    assert time.perf_counter() - start < 20
    assert all(_padded_size(p[3], 4).max() <= 32 for p in parts)
    assert sum(len(p[4]) for p in parts) == len(piece[4])


def quad_texels(cells, cell=10):
    texels, tris = [], []
    for x, y in cells:
        base = len(texels)
        texels += [[x * cell, y * cell], [(x + 1) * cell, y * cell], [(x + 1) * cell, (y + 1) * cell], [x * cell, (y + 1) * cell]]
        tris += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.array(texels, float), np.array(tris)


def test_mirrored_island_splits_into_two_pieces():
    from lightmapper.charts import _components, _island_overlaps, _split_self_overlap
    # the right half folds back over the left: quad 2 lies on quad 1 and quad 3 on quad 0
    texels, tris = quad_texels([(0, 0), (1, 0), (2, 0), (3, 0)])
    texels[:, 0] = np.where(texels[:, 0] > 20, 40 - texels[:, 0], texels[:, 0])
    found = _split_self_overlap(texels, tris)
    assert len(found) == 2
    assert sorted(np.concatenate(found)) == list(range(len(tris)))
    for rows in found:
        assert not _island_overlaps(texels, np.unique(tris[rows]), tris[rows])
        assert len(_components(texels, tris, rows)) == 1


def test_self_overlap_split_keeps_non_colliding_neighbours_together():
    from lightmapper.charts import _split_self_overlap
    # a ring of eight cells round a 3x3 block, then one triangle laid over the first cell's second triangle
    ring = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    texels, tris = quad_texels(ring)
    n = len(texels)
    texels = np.concatenate([texels, [[0, 0], [10, 10], [0, 10]]])
    tris = np.concatenate([tris, [[n, n + 1, n + 2]]])
    found = _split_self_overlap(texels, tris)
    assert sorted(len(rows) for rows in found) == [1, 16]


def island_fit_residual(pages, low, high):
    # worst deviation of the vertices whose world x lies in [low, high) from one affine map of their texture uvs to the page texels
    source, target = [], []
    for page in pages:
        for m in page.materials:
            pick = (m.env['position'][:, 0] >= low) & (m.env['position'][:, 0] < high)
            source.append(m.env['texcoord'][pick].astype(np.float64))
            target.append(m.lm['texcoord'][pick].astype(np.float64) * [page.width, page.height])
    source, target = np.concatenate(source), np.concatenate(target)
    design = np.hstack([source, np.ones((len(source), 1))])
    fit, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.abs(design @ fit - target).max()


def two_strips_group(name):
    # three islands: a quad at texture u 0-1, two world-apart strips whose texture uvs coincide (one island
    # that overlaps itself) at u 10-11, and a quad at u 20-21
    pos, uvs, tris = [], [], []

    def quad(x0, y0, w, h, u0, v0):
        base = len(pos)
        pos.extend([[x0, y0, 0], [x0 + w, y0, 0], [x0 + w, y0 + h, 0], [x0, y0 + h, 0]])
        uvs.extend([[u0, v0], [u0 + 1, v0], [u0 + 1, v0 + 1], [u0, v0 + 1]])
        tris.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    quad(0, 0, 4, 4, 0, 0)
    quad(20, 0, 4, 4, 10, 0)
    quad(20, 10, 4, 4, 10, 0)
    quad(40, 0, 4, 4, 20, 0)
    return make_group(np.array(pos, np.float32), tris, uvs=uvs, name=name)


def test_only_the_self_overlapping_island_is_split(caplog):
    group = two_strips_group('threeway')
    with caplog.at_level(logging.WARNING, logger='lightmapper.charts'):
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    assert island_fit_residual(pages, 0, 5) < 1e-3
    assert island_fit_residual(pages, 40, 50) < 1e-3
    assert sorted(np.concatenate([m.surface_ids for p in pages for m in p.materials])) == list(range(8))
    warnings = [r.getMessage() for r in caplog.records if r.name == 'lightmapper.charts' and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and 'threeway' in warnings[0] and '1 island(s) split' in warnings[0]
    for page in pages:
        assert rasterise(page).max() <= 1


def test_overlapping_islands_keep_their_stitching(caplog):
    # two islands whose texture uvs overlap each other, with no uv value in common, are two islands
    pos = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [10, 0, 0], [14, 0, 0], [14, 4, 0], [10, 4, 0]], np.float32)
    uvs = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]
    group = make_group(pos, [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], uvs=uvs, name='nested')
    with caplog.at_level(logging.WARNING, logger='lightmapper.charts'):
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    assert caplog.text == ''
    assert island_fit_residual(pages, 0, 5) < 1e-3
    assert island_fit_residual(pages, 5, 20) < 1e-3


def test_split_worse_than_a_rechart_is_recharted(caplog):
    # twelve triangles of one flat strip, every one with the same three uvs: the split makes twelve pieces
    pos, uvs, tris = [], [], []
    for x in range(6):
        for corners in ([[x, 0], [x + 1, 0], [x + 1, 1]], [[x, 0], [x + 1, 1], [x, 1]]):
            tris.append([len(pos), len(pos) + 1, len(pos) + 2])
            pos.extend([[cx, cy, 0] for cx, cy in corners])
            uvs.extend([[0, 0], [1, 0], [0, 1]])
    good = len(pos)
    pos.extend([[20, 0, 0], [24, 0, 0], [24, 4, 0], [20, 4, 0]])
    uvs.extend([[10, 0], [11, 0], [11, 1], [10, 1]])
    tris.extend([[good, good + 1, good + 2], [good, good + 2, good + 3]])
    group = make_group(np.array(pos, np.float32), tris, uvs=uvs, name='stack')
    with caplog.at_level(logging.WARNING, logger='lightmapper.charts'):
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    assert '1 island(s) charted normally (it overlaps itself)' in caplog.text
    assert '0 island(s) split' in caplog.text
    assert island_fit_residual(pages, 15, 30) < 1e-3
    assert sorted(np.concatenate([m.surface_ids for p in pages for m in p.materials])) == list(range(14))


def test_zero_texture_area_island_is_recharted_alone(caplog):
    pos = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [10, 0, 0], [14, 0, 0], [14, 4, 0], [10, 4, 0]], np.float32)
    uvs = [[0, 0], [1, 0], [2, 0], [0, 1], [10, 0], [11, 0], [11, 1], [10, 1]]
    group = make_group(pos, [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], uvs=uvs, name='flat')
    with caplog.at_level(logging.WARNING, logger='lightmapper.charts'):
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    assert '0 island(s) split' in caplog.text and '1 island(s) charted normally (no texture area)' in caplog.text
    assert island_fit_residual(pages, 5, 20) < 1e-3
    for page in pages:
        for m in page.materials:
            for tri in texel_triangles(page, m):
                assert area(*tri) > 0.5
    assert sorted(np.concatenate([m.surface_ids for p in pages for m in p.materials])) == [0, 1, 2, 3]


def test_page_max_auto_resolves_to_4096():
    s = ChartSettings()
    assert s.page_max == 'auto' and s.resolved_page_max == 4096
    assert ChartSettings(page_max=512).resolved_page_max == 512


@pytest.mark.parametrize('bad', ['Auto', 'big', '512', 100, 0])
def test_page_max_refuses_other_strings(bad):
    with pytest.raises(ValueError, match='page_max'):
        ChartSettings(page_max=bad)


def island_group(sizes, tpu, name='shader'):
    # separate quads, each `size` world units on a side with its own texture uvs: an island `size * tpu` texels wide
    pos, uvs, tris = [], [], []
    for i, size in enumerate(sizes):
        base = len(pos)
        pos.extend([[i * 100, 0, 0], [i * 100 + size, 0, 0], [i * 100 + size, size, 0], [i * 100, size, 0]])
        uvs.extend([[i * 2, 0], [i * 2 + 1, 0], [i * 2 + 1, 1], [i * 2, 1]])
        tris.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    return make_group(np.array(pos, np.float32), tris, uvs=uvs, name=name)


def test_bake_side_is_the_smallest_that_fits():
    group = island_group([4] * 6, 50)
    for page_max in ('auto', 1024):
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)],
                               ChartSettings(texels_per_unit=50, page_max=page_max))
        assert len(pages) == 2
        assert all(max(p.width, p.height) <= 512 for p in pages)


def test_one_triangle_forces_the_side_up():
    pos = np.array([[0, 0, 0], [12, 0, 0], [0, 12, 0]], np.float32)
    triangle = make_group(pos, [[0, 1, 2]], uvs=[[0, 0], [1, 0], [0, 1]], name='wide')
    pages = chart_and_pack([island_group([4] * 6, 50), triangle], [radiosity(SIMPLE_PARAMETERIZATION)] * 2,
                           ChartSettings(texels_per_unit=50))
    assert all(max(p.width, p.height) <= 1024 for p in pages)
    assert any(max(p.width, p.height) == 1024 for p in pages)


def test_triangle_wider_than_max_resolution_names_the_shader():
    from lightmapper.charts import TriangleTooWide
    # (599 texels of triangle, and xatlas starts a uv chart half a texel in: 600 wide, 608 with padding)
    pos = np.array([[0, 0, 0], [11.98, 0, 0], [0, 11.98, 0]], np.float32)
    triangle = make_group(pos, [[0, 1, 2]], uvs=[[0, 0], [1, 0], [0, 1]], name='wide')
    with pytest.raises(TriangleTooWide) as caught:
        chart_and_pack([island_group([4] * 6, 50), triangle], [radiosity(SIMPLE_PARAMETERIZATION)] * 2,
                       ChartSettings(texels_per_unit=50, page_max=512))
    error = caught.value
    assert (error.shader, error.width, error.page_max) == ('wide', 608, 512)
    assert 'wide' in str(error) and '608' in str(error) and '512' in str(error)
    assert isinstance(error, CapError)


def big_island():
    pos, tris = grid(14, 1.0)
    return make_group(pos, tris, uvs=pos[:, :2] / 14, name='big')


def test_oversized_island_is_bisected_not_refused():
    pages = chart_and_pack([big_island()], [radiosity(SIMPLE_PARAMETERIZATION)],
                           ChartSettings(texels_per_unit=50, page_max='auto'))
    assert all(max(p.width, p.height) <= 512 for p in pages)
    areas = np.array([area(*t) for p in pages for m in p.materials for t in texel_triangles(p, m)])
    assert len(areas) == 392
    np.testing.assert_allclose(areas, 1250.0, rtol=1e-5)


def test_fixed_256_never_exceeds_256():
    pages = chart_and_pack([big_island()], [radiosity(SIMPLE_PARAMETERIZATION)],
                           ChartSettings(texels_per_unit=50, page_max=256))
    assert all(max(p.width, p.height) <= 256 for p in pages)
    assert sum(len(m.tris) for p in pages for m in p.materials) == 392


def test_charting_does_not_depend_on_page_max(monkeypatch):
    from lightmapper import charts
    seen = []
    real = charts.xatlas_c.PackSettings

    def recorder(**kwargs):
        seen.append(kwargs['resolution'])
        return real(**kwargs)
    monkeypatch.setattr(charts.xatlas_c, 'PackSettings', recorder)
    pos, tris = grid(8, 1.0)
    group = make_group(pos, tris)
    results = {}
    for page_max in (256, 'auto'):
        seen.clear()
        pages = chart_and_pack([group], [radiosity()], ChartSettings(page_max=page_max))
        assert {r for r in seen if r} == {4096 - 2 * 4}
        results[page_max] = sorted(area(*t) for p in pages for m in p.materials for t in texel_triangles(p, m))
    np.testing.assert_allclose(results[256], results['auto'], rtol=1e-5)


def test_island_with_no_world_area_is_recharted_alone(caplog):
    # the second island has texture area but its three positions are on a line
    pos = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [10, 0, 0], [12, 0, 0], [14, 0, 0]], np.float32)
    uvs = [[0, 0], [1, 0], [1, 1], [0, 1], [10, 0], [11, 0], [10, 1]]
    group = make_group(pos, [[0, 1, 2], [0, 2, 3], [4, 5, 6]], uvs=uvs, name='sliver')
    with caplog.at_level(logging.WARNING, logger='lightmapper.charts'):
        pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)], ChartSettings())
    assert '0 island(s) split' in caplog.text and '1 island(s) charted normally (no texture or world area)' in caplog.text
    assert island_fit_residual(pages, 0, 5) < 1e-3
    assert sorted(np.concatenate([m.surface_ids for p in pages for m in p.materials])) == [0, 1, 2]


def rect_group(sizes, name='shader', sid0=0):
    # separate quads w x h texels at texels_per_unit 1, each with its own texture uvs (uniform scale, so no stretch)
    pos, uvs, tris = [], [], []
    for i, (w, h) in enumerate(sizes):
        base = len(pos)
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            pos.append([i * 1000 + x, y, 0])
            uvs.append([i * 2 + x / 1000, y / 1000])
        tris.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    return make_group(np.array(pos, np.float32), tris, uvs=uvs, name=name, sid0=sid0)


def nest_settings(**options):
    return ChartSettings(texels_per_unit=1.0, page_max='auto', **options)


def simple(count=1):
    return [radiosity(SIMPLE_PARAMETERIZATION)] * count


def scattered_islands():
    sizes = np.random.default_rng(1).uniform(6, 60, (30, 2))
    return rect_group([tuple(size) for size in sizes])


def total_area(pages):
    return sum(p.width * p.height for p in pages)


def same_layout(a, b):
    assert [(p.width, p.height) for p in a] == [(p.width, p.height) for p in b]
    for page_a, page_b in zip(a, b):
        assert len(page_a.materials) == len(page_b.materials)
        for m, n in zip(page_a.materials, page_b.materials):
            np.testing.assert_array_equal(m.lm, n.lm)
            np.testing.assert_array_equal(m.tris, n.tris)
            np.testing.assert_array_equal(m.surface_ids, n.surface_ids)


class NestSpy:
    def __init__(self, monkeypatch):
        from lightmapper import charts
        self.calls = []
        real = charts._nest_pack

        def spy(parts, s):
            result = real(parts, s)
            self.calls.append((parts, s, result))
            return result
        monkeypatch.setattr(charts, '_nest_pack', spy)


def corners_by_surface(pages):
    found = {}
    for page in pages:
        for m in page.materials:
            texels = m.lm['texcoord'].astype(np.float64) * [page.width, page.height]
            for tri, sid in zip(m.tris, m.surface_ids):
                found[int(sid)] = texels[tri]
    return found


def test_hollow_chart_holds_a_smaller_chart():
    # a frame island (100 texels out, a 60 texel hole) and a 20 texel island, as one group
    pos, uvs, tris = [], [], []
    edges = [0, 20, 80, 100]
    for y in edges:
        for x in edges:
            pos.append([x, y, 0])
            uvs.append([x / 1000, y / 1000])
    for cy in range(3):
        for cx in range(3):
            if (cx, cy) != (1, 1):
                a = cy * 4 + cx
                tris.extend([[a, a + 1, a + 5], [a, a + 5, a + 4]])
    base = len(pos)
    for x, y in ((0, 0), (20, 0), (20, 20), (0, 20)):
        pos.append([2000 + x, y, 0])
        uvs.append([2 + x / 1000, y / 1000])
    tris.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    group = make_group(np.array(pos, np.float32), tris, uvs=uvs)
    nested = chart_and_pack([group], simple(), nest_settings(), nest=True)
    rects = chart_and_pack([group], simple(), nest_settings(), nest=False)
    assert [(p.width, p.height) for p in nested] == [(128, 128)]
    assert total_area(rects) > total_area(nested)
    corners = corners_by_surface(nested)
    frame = np.concatenate([corners[i] for i in range(16)])
    small = np.concatenate([corners[16], corners[17]])
    assert np.all(small.min(axis=0) > frame.min(axis=0)) and np.all(small.max(axis=0) < frame.max(axis=0))


def nesting_wins_group():
    return rect_group([(37.3, 21.7)] * 6 + [(15.2, 44.6)] * 4)


def test_nested_charts_keep_the_full_gap(monkeypatch):
    from lightmapper.charts import _charts_separated
    s = nest_settings()
    spy = NestSpy(monkeypatch)
    pages = chart_and_pack([nesting_wins_group()], simple(), s, nest=True)
    assert spy.calls[0][2] is not None and pages is spy.calls[0][2]
    assert _charts_separated(pages, s)
    for page in pages:
        assert rasterise(page).max() <= 1


def test_rectangle_layout_passes_the_gap_check():
    from lightmapper.charts import _charts_separated, _pack_rectangles
    s = ChartSettings(page_max=256)
    # (integer extents put every edge on a texel boundary)
    pieces = [piece_from_grid(2, 10), piece_from_grid(1, 7), piece_from_grid(3, 9), piece_from_grid(4, 5)]
    assert _charts_separated(_pack_rectangles(pieces, s), s)


def test_charts_closer_than_the_gap_are_not_separated():
    from lightmapper.charts import _charts_separated
    s = nest_settings()
    pages = chart_and_pack([rect_group([(20, 20), (20, 20)])], simple(), s)
    assert _charts_separated(pages, s)
    page = pages[0]
    material = page.materials[0]
    labels = connected_components_of(material)
    texels = material.lm['texcoord'].astype(np.float64) * [page.width, page.height]
    centres = [texels[labels == k].mean(axis=0) for k in (0, 1)]
    axis = int(np.argmax(np.abs(centres[1] - centres[0])))
    moved = material.lm.copy()
    step = np.sign(centres[0][axis] - centres[1][axis]) / [page.width, page.height][axis]
    moved['texcoord'][labels == 1, axis] += step
    squeezed = [type(page)(page.width, page.height, [type(material)(material.group, material.env, moved,
                                                                    material.tris, material.surface_ids)])]
    assert not _charts_separated(squeezed, s)


def connected_components_of(material):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(material.env)
    edges = np.concatenate([material.tris[:, [0, 1]], material.tris[:, [1, 2]]])
    return connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)), directed=False)[1]


def test_nested_pack_keeps_texel_areas(monkeypatch):
    from lightmapper.charts import _piece_areas, _packing_areas
    spy = NestSpy(monkeypatch)
    group = rect_group([(37.3, 21.7)] * 6 + [(15.2, 44.6)] * 4)
    pages = chart_and_pack([group], simple(), nest_settings(), nest=True)
    parts, _, result = spy.calls[0]
    assert result is not None and pages is result
    np.testing.assert_allclose(_packing_areas(pages), _piece_areas(parts), rtol=1e-5)


def test_nested_pack_keeps_the_areas_of_parts_with_fractional_extents():
    # (charts xatlas made already have whole-texel extents, but a bisected part's cut edge does not, and xatlas
    # stretches such a chart to whole texels when it packs it again)
    from lightmapper.charts import _nest_pack, _packing_areas, _piece_areas
    s = ChartSettings(page_max=256)
    parts = [piece_from_grid(2, 18.65), piece_from_grid(3, 7.3), piece_from_grid(1, 33.45)]
    pages = _nest_pack(parts, s)
    assert pages is not None
    np.testing.assert_allclose(_packing_areas(pages), _piece_areas(parts), rtol=1e-5)


def test_nested_pack_only_translates_or_swaps(monkeypatch):
    spy = NestSpy(monkeypatch)
    pages = chart_and_pack([rect_group([(37.3, 21.7)] * 6 + [(15.2, 44.6)] * 4)], simple(), nest_settings(), nest=True)
    parts, s, result = spy.calls[0]
    assert pages is result
    new = corners_by_surface(pages)
    for part in parts:
        old = part[3][part[4]]
        sids = [int(i) for i in part[5]]
        moved = np.array([new[i] - s.padding for i in sids])
        shift = moved - old
        swapped_shift = moved - old[:, :, ::-1]
        assert min(np.ptp(shift.reshape(-1, 2), axis=0).max(), np.ptp(swapped_shift.reshape(-1, 2), axis=0).max()) < 1e-3


def test_a_turned_chart_is_read_back_swapped(monkeypatch):
    spy = NestSpy(monkeypatch)
    # (found by search: thin charts of both orientations, where xatlas turns some to fit them in)
    group = rect_group([(87, 17), (72, 12), (8, 84), (22, 88), (66, 6), (53, 8), (6, 95), (9, 89), (17, 105), (8, 90),
                        (50, 14), (99, 13), (78, 16)])
    pages = chart_and_pack([group], simple(), nest_settings(), nest=True)
    parts, s, result = spy.calls[0]
    assert result is not None and pages is result
    new = corners_by_surface(pages)
    turned = 0
    for part in parts:
        old = part[3][part[4]]
        moved = np.array([new[int(i)] - s.padding for i in part[5]])
        straight = np.ptp((moved - old).reshape(-1, 2), axis=0).max()
        swapped = np.ptp((moved - old[:, :, ::-1]).reshape(-1, 2), axis=0).max()
        assert min(straight, swapped) < 1e-3
        if swapped < straight:
            turned += 1
            np.testing.assert_allclose(_tri_area(moved), _tri_area(old), rtol=1e-5)
    assert turned > 0


def _tri_area(corners):
    ab, ac = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def test_nest_with_a_degenerate_triangle_keeps_every_triangle(monkeypatch):
    from lightmapper.charts import _charts_separated, _packing_areas
    pos, tris = grid(4)
    pos = np.vstack([pos, [[20, 0, 0], [21, 0, 0], [22, 0, 0]]]).astype(np.float32)
    tris = np.vstack([tris, [[25, 26, 27], [0, 0, 1]]])
    group = make_group(pos, tris)
    s = ChartSettings(page_max='auto')
    spy = NestSpy(monkeypatch)
    nested = chart_and_pack([group], [radiosity()], s, nest=True)
    assert spy.calls[0][2] is not None and nested is spy.calls[0][2]
    rects = chart_and_pack([group], [radiosity()], s)
    assert sorted(np.concatenate([m.surface_ids for p in nested for m in p.materials])) == list(range(len(tris)))
    np.testing.assert_allclose(_packing_areas(nested), _packing_areas(rects), atol=1e-6)
    assert _charts_separated(nested, s)


def test_nesting_is_deterministic(monkeypatch):
    s = nest_settings()
    spy = NestSpy(monkeypatch)
    first = chart_and_pack([nesting_wins_group()], simple(), s, nest=True)
    second = chart_and_pack([nesting_wins_group()], simple(), s, nest=True)
    assert first is spy.calls[0][2] and second is spy.calls[1][2] and first is not None
    same_layout(first, second)


def test_nesting_is_never_worse_than_rectangles():
    s = nest_settings()
    nested = chart_and_pack([scattered_islands()], simple(), s, nest=True)
    rects = chart_and_pack([scattered_islands()], simple(), s, nest=False)
    assert (len(nested), total_area(nested)) <= (len(rects), total_area(rects))


def test_a_nested_layout_without_the_gap_falls_back_to_rectangles(monkeypatch):
    from lightmapper import charts
    s = nest_settings()
    group = rect_group([(20, 20), (20, 20)])
    rects = chart_and_pack([group], simple(), s)

    def squeezed(parts, settings):
        page = rects[0]
        material = page.materials[0]
        labels = connected_components_of(material)
        texels = material.lm['texcoord'].astype(np.float64) * [page.width, page.height]
        centres = [texels[labels == k].mean(axis=0) for k in (0, 1)]
        axis = int(np.argmax(np.abs(centres[1] - centres[0])))
        moved = material.lm.copy()
        moved['texcoord'][labels == 1, axis] += np.sign(centres[0][axis] - centres[1][axis]) / [page.width, page.height][axis]
        return [type(page)(page.width, page.height, [type(material)(material.group, material.env, moved,
                                                                    material.tris, material.surface_ids)])]
    monkeypatch.setattr(charts, '_nest_pack', squeezed)
    same_layout(chart_and_pack([group], simple(), s, nest=True), rects)


def test_a_nested_layout_over_the_material_cap_falls_back(monkeypatch):
    from lightmapper.charts import _check_caps
    s = nest_settings(max_materials_per_page=2)
    groups = [rect_group([(20, 20)], name=f'g{i}', sid0=2 * i) for i in range(3)]
    spy = NestSpy(monkeypatch)
    nested = chart_and_pack(groups, simple(3), s, nest=True)
    rects = chart_and_pack(groups, simple(3), s)
    assert spy.calls[0][2] is not None and len(spy.calls[0][2]) == 1
    _check_caps(nested, s)
    same_layout(nested, rects)


def test_the_estimate_never_nests(monkeypatch):
    from lightmapper import charts

    def refuse(parts, settings):
        raise AssertionError('the estimate must not nest')
    monkeypatch.setattr(charts, '_nest_pack', refuse)
    assert chart_and_pack([scattered_islands()], simple(), nest_settings(), nest=False)
