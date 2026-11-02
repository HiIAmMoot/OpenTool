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
    pages = chart_and_pack([make_group(pos, tris)], [radiosity()], ChartSettings(texels_per_unit=1, page_max=512))
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


def test_page_max_auto_is_capped_at_4096():
    s = ChartSettings()
    assert s.page_max == 'auto' and s.page_cap == 4096
    assert ChartSettings(page_max=512).page_cap == 512


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
    pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)],
                           ChartSettings(texels_per_unit=50, page_max=1024))
    assert len(pages) == 2
    assert all(max(p.width, p.height) <= 512 for p in pages)
    # (auto's Max resolution is the biggest island's size, 208 texels with padding, so 256: one island to a page)
    pages = chart_and_pack([group], [radiosity(SIMPLE_PARAMETERIZATION)],
                           ChartSettings(texels_per_unit=50, page_max='auto'))
    assert pages.page_max == 256 and all(max(p.width, p.height) <= 256 for p in pages)


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
    # (700 texels and padding fit a 1024 page whole, so auto settles there and the island is not cut)
    assert pages.page_max == 1024 and all(max(p.width, p.height) <= 1024 for p in pages)
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

        def spy(parts, s, *margin):
            result = real(parts, s, *margin)
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
    chart_and_pack([nesting_wins_group()], simple(), s, nest=True)
    units, _, pages = spy.calls[0]
    assert pages is not None
    assert _charts_separated(pages, s, units)
    for page in pages:
        assert rasterise(page).max() <= 1


def test_rectangle_layout_passes_the_gap_check():
    from lightmapper.charts import _charts_separated, _pack_rectangles
    s = ChartSettings(page_max=256)
    # (integer extents put every edge on a texel boundary)
    pieces = [piece_from_grid(2, 10), piece_from_grid(1, 7), piece_from_grid(3, 9), piece_from_grid(4, 5)]
    pieces = [(i, *piece[1:]) for i, piece in enumerate(pieces)]
    assert _charts_separated(_pack_rectangles(pieces, s), s, pieces)


def test_charts_closer_than_the_gap_are_not_separated():
    from lightmapper.charts import _charts_separated
    s = nest_settings()
    pages = chart_and_pack([rect_group([(20, 20), (20, 20)])], simple(), s)
    page = pages[0]
    material = page.materials[0]
    labels = connected_components_of(material)
    quads = [(material.group, None, None, None, None, material.surface_ids[labels[material.tris[:, 0]] == k])
             for k in (0, 1)]
    assert _charts_separated(pages, s, quads)
    texels = material.lm['texcoord'].astype(np.float64) * [page.width, page.height]
    centres = [texels[labels == k].mean(axis=0) for k in (0, 1)]
    axis = int(np.argmax(np.abs(centres[1] - centres[0])))
    moved = material.lm.copy()
    step = np.sign(centres[0][axis] - centres[1][axis]) / [page.width, page.height][axis]
    moved['texcoord'][labels == 1, axis] += step
    squeezed = [type(page)(page.width, page.height, [type(material)(material.group, material.env, moved,
                                                                    material.tris, material.surface_ids)])]
    assert not _charts_separated(squeezed, s, quads)


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
    chart_and_pack([group], simple(), nest_settings(), nest=True)
    parts, _, pages = spy.calls[0]
    assert pages is not None
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
    chart_and_pack([rect_group([(37.3, 21.7)] * 6 + [(15.2, 44.6)] * 4)], simple(), nest_settings(), nest=True)
    parts, s, pages = spy.calls[0]
    assert pages is not None
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
    chart_and_pack([group], simple(), nest_settings(), nest=True)
    parts, s, pages = spy.calls[0]
    assert pages is not None
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
    chart_and_pack([group], [radiosity()], s, nest=True)
    nested = spy.calls[0][2]
    assert nested is not None
    rects = chart_and_pack([group], [radiosity()], s)
    assert sorted(np.concatenate([m.surface_ids for p in nested for m in p.materials])) == list(range(len(tris)))
    np.testing.assert_allclose(_packing_areas(nested), _packing_areas(rects), atol=1e-6)
    assert _charts_separated(nested, s, spy.calls[0][0])


def test_nesting_is_deterministic(monkeypatch):
    s = nest_settings()
    spy = NestSpy(monkeypatch)
    chart_and_pack([nesting_wins_group()], simple(), s, nest=True)
    chart_and_pack([nesting_wins_group()], simple(), s, nest=True)
    first, second = spy.calls[0][2], spy.calls[-1][2]
    assert first is not None and len(spy.calls) == 2
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

    def squeezed(parts, settings, *margin):
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


def test_pack_rectangles_accepts_an_auto_page_max():
    from lightmapper.charts import _pack_rectangles
    pages = _pack_rectangles(rectangle_pieces([(8, 8)] * 2), ChartSettings(padding=0, page_max='auto'))
    assert len(pages) == 1


def test_a_chart_over_the_auto_page_max_is_refused_not_a_type_error():
    from lightmapper.charts import AUTO_PAGE_MAX, _pack_rectangles
    with pytest.raises(CapError, match='exceeds page_max'):
        _pack_rectangles(rectangle_pieces([(AUTO_PAGE_MAX + 1, 8)]), ChartSettings(padding=0, page_max='auto'))


def test_nest_pack_accepts_an_auto_page_max():
    from lightmapper.charts import _nest_pack
    pages = _nest_pack([piece_from_grid(2, 18.65), piece_from_grid(3, 7.3)], ChartSettings(page_max='auto'))
    assert pages is not None


def loose_piece(index, quads):
    # one piece of quads with uvs (x, y, w, h); every triangle has its own vertices, as a part of a chart whose
    # source vertices differ across an edge (a normal seam), so its triangles only touch by uv value
    pos, uvs, tris = [], [], []
    for x, y, w, h in quads:
        for corners in (((x, y), (x + w, y), (x + w, y + h)), ((x, y), (x + w, y + h), (x, y + h))):
            tris.append([len(pos), len(pos) + 1, len(pos) + 2])
            for u, v in corners:
                pos.append([u, v, 0])
                uvs.append([u, v])
    group = make_group(np.array(pos, np.float32), tris)
    return (index, group, np.arange(len(pos)), np.array(uvs, np.float64), group.tris, group.surface_ids)


def frame_piece(index, outer, hole):
    edges = [0, (outer - hole) / 2, (outer + hole) / 2, outer]
    pos = [[x, y, 0] for y in edges for x in edges]
    tris = []
    for cy in range(3):
        for cx in range(3):
            if (cx, cy) != (1, 1):
                a = cy * 4 + cx
                tris += [[a, a + 1, a + 5], [a, a + 5, a + 4]]
    group = make_group(np.array(pos, np.float32), tris)
    return (index, group, np.arange(16), group.env['position'][:, :2].astype(np.float64), group.tris, group.surface_ids)


def quad_piece(index, w, h):
    group = make_group(np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], np.float32), [[0, 1, 2], [0, 2, 3]])
    return (index, group, np.arange(4), group.env['position'][:, :2].astype(np.float64), group.tris, group.surface_ids)


def test_triangles_of_one_chart_that_share_no_vertex_still_nest():
    # (a chart whose triangles meet only by uv value is still one chart: its own texels may touch)
    from lightmapper.charts import _charts_separated, _nested_or, _pack_rectangles
    s = ChartSettings(page_max=128)
    parts = [frame_piece(0, 100, 60), loose_piece(1, [(0, 0, 14, 14), (14, 0, 14, 14)])]
    rects = _pack_rectangles(parts, s)
    pages = _nested_or(rects, parts, s)
    assert pages is not rects and total_area(pages) < total_area(rects)
    assert _charts_separated(pages, s, parts)


def test_a_part_of_separate_uv_islands_nests_island_by_island():
    # (a bounded or bisected part can be several uv islands; xatlas places each island as its own chart)
    from lightmapper.charts import _charts_separated, _nest_units, _nested_or, _pack_rectangles
    s = ChartSettings(page_max=128)
    parts = [frame_piece(0, 100, 60), loose_piece(1, [(0, 0, 14, 14), (36, 0, 14, 14)])]
    rects = _pack_rectangles(parts, s)
    pages = _nested_or(rects, parts, s)
    assert pages is not rects and total_area(pages) < total_area(rects)
    assert _charts_separated(pages, s, _nest_units(parts))
    np.testing.assert_allclose(_packing_areas_of(pages), _piece_areas_of(parts), rtol=1e-5)


def _packing_areas_of(pages):
    from lightmapper.charts import _packing_areas
    return _packing_areas(pages)


def _piece_areas_of(parts):
    from lightmapper.charts import _piece_areas
    return _piece_areas(parts)


def test_separation_tells_charts_apart_by_their_triangles_not_their_vertices():
    from lightmapper.charts import _charts_separated, _pack_rectangles
    s = ChartSettings(page_max=128)
    parts = [loose_piece(0, [(0, 0, 10, 10), (10, 0, 10, 10)]), loose_piece(1, [(0, 0, 10, 10)])]
    pages = _pack_rectangles(parts, s)
    assert _charts_separated(pages, s, parts)
    # the same triangles, claimed as one chart each: the two halves of the first piece now touch
    split = [(0, None, None, None, None, parts[0][5][:2]), (0, None, None, None, None, parts[0][5][2:]), parts[1]]
    assert not _charts_separated(pages, s, split)


def test_separation_refuses_triangles_it_cannot_assign():
    from lightmapper.charts import _charts_separated, _pack_rectangles
    s = ChartSettings(page_max=128)
    parts = [quad_piece(0, 10, 10), quad_piece(1, 10, 10)]
    pages = _pack_rectangles(parts, s)
    assert not _charts_separated(pages, s, parts[:1])
    # (two charts claiming one triangle cannot be told apart either)
    assert not _charts_separated(pages, s, parts + [parts[0]])


def stretched_nest_parts():
    # (found by search: frames and quads of fractional extents, which xatlas stretches to whole texels before it
    # places a chart; at xatlas's own padding our unstretched copies end 7 texels apart)
    spec = [('f', 81.76, 46.78), ('f', 43.01, 31.97), ('f', 81.75, 57.03), ('f', 103.72, 69.71), ('q', 5.85, 17.72),
            ('q', 7.27, 12.44), ('q', 11.72, 12.0), ('q', 11.03, 17.94), ('q', 10.53, 14.57), ('q', 10.41, 17.97),
            ('q', 17.68, 8.9), ('q', 7.83, 15.93), ('q', 14.99, 13.37), ('q', 9.13, 8.85), ('q', 4.97, 12.38),
            ('q', 4.24, 7.12), ('q', 12.84, 3.22), ('q', 15.52, 4.09), ('q', 10.86, 11.2), ('q', 6.41, 17.34),
            ('q', 8.11, 10.57)]
    return [frame_piece(i, a, b) if kind == 'f' else quad_piece(i, a, b) for i, (kind, a, b) in enumerate(spec)]


def test_nesting_widens_the_xatlas_margin_when_the_stretch_closes_the_gap():
    from lightmapper.charts import _charts_separated, _nest_pack, _nested_or, _pack_rectangles
    s = ChartSettings(page_max=128)
    parts = stretched_nest_parts()
    assert not _charts_separated(_nest_pack(parts, s), s, parts)
    rects = _pack_rectangles(parts, s)
    pages = _nested_or(rects, parts, s)
    assert pages is not rects and total_area(pages) < total_area(rects)
    assert _charts_separated(pages, s, parts)
    np.testing.assert_allclose(_packing_areas_of(pages), _piece_areas_of(parts), rtol=1e-5)


def test_the_wider_margin_covers_the_stretch():
    # xatlas places a chart stretched by under a texel; one more texel of its padding on each chart keeps our
    # unstretched copies 2 x padding apart
    from lightmapper.charts import _charts_separated, _nest_pack
    s = ChartSettings(page_max=256)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        parts = [frame_piece(0, rng.uniform(40, 60), rng.uniform(20, 30))]
        parts += [quad_piece(i, *rng.uniform(3, 18, 2)) for i in range(1, 12)]
        pages = _nest_pack(parts, s, s.padding + 1)
        assert pages is not None and _charts_separated(pages, s, parts)


# ---------- auto Max resolution

def auto_wall(texels, density=1.0, name='wall'):
    # one simple-parameterization island of `texels` texels on a side at the density
    size = texels / density
    pos, tris = grid(2, size / 2)
    return make_group(pos, tris, uvs=pos[:, :2] / size, name=name)


def simple_shader():
    return [radiosity(SIMPLE_PARAMETERIZATION)]


def test_auto_is_the_smallest_side_the_biggest_chart_needs():
    # 500 texels and 8 of padding fit 512, and no written page is bigger
    pages = chart_and_pack([auto_wall(500)], simple_shader(), ChartSettings(texels_per_unit=1, page_max='auto'))
    assert pages.page_max == 512
    assert all(max(p.width, p.height) <= 512 for p in pages)


def test_auto_never_goes_below_256():
    pages = chart_and_pack([auto_wall(20)], simple_shader(), ChartSettings(texels_per_unit=1, page_max='auto'))
    assert pages.page_max == 256


def test_a_denser_setting_raises_auto():
    sizes = [chart_and_pack([auto_wall(500)], simple_shader(), ChartSettings(texels_per_unit=d, page_max='auto')).page_max
             for d in (1, 2)]
    assert sizes == [512, 1024]


def test_auto_cuts_what_is_over_4096_then_fits_the_rest():
    # 5000 texels of wall cannot be one page; the halves (about 2500, 2508 with padding) need 4096
    pages = chart_and_pack([auto_wall(5000)], simple_shader(), ChartSettings(texels_per_unit=1, page_max='auto'))
    assert pages.page_max == 4096
    assert all(max(p.width, p.height) <= 4096 for p in pages)
    # 8000 texels cut into pieces of about 2000 each, which fit 2048
    pages = chart_and_pack([auto_wall(8000)], simple_shader(), ChartSettings(texels_per_unit=1, page_max='auto'))
    assert pages.page_max <= 4096


def test_auto_rises_when_the_page_count_cap_fails_at_the_chart_size():
    # four 200-texel walls fit one 512 page, but not one page of the 256 side the chart alone would allow
    groups = [auto_wall(200, name=f'w{i}') for i in range(4)]
    pages = chart_and_pack(groups, simple_shader() * 4, ChartSettings(texels_per_unit=1, page_max='auto', max_pages=1))
    assert pages.page_max == 512 and len(pages) == 1


def test_fixed_values_stay_hard_caps():
    pages = chart_and_pack([auto_wall(500)], simple_shader(), ChartSettings(texels_per_unit=1, page_max=1024))
    assert pages.page_max == 1024
    with pytest.raises(CapError):
        chart_and_pack([auto_wall(2000, name='a'), auto_wall(2000, name='b')], simple_shader() * 2,
                       ChartSettings(texels_per_unit=1, page_max=256, max_pages=1))


def test_the_charts_cache_keeps_the_resolved_page_max(tmp_path):
    from lightmapper.charts import load_charts, save_charts
    pages = chart_and_pack([auto_wall(500)], simple_shader(), ChartSettings(texels_per_unit=1, page_max='auto'))
    save_charts(tmp_path / 'c.npz', pages, 'key')
    assert load_charts(tmp_path / 'c.npz', 'key').page_max == 512


def test_the_estimate_and_the_bake_agree_on_auto(monkeypatch):
    groups = [auto_wall(500, name='a'), auto_wall(300, name='b')]
    s = ChartSettings(texels_per_unit=1, page_max='auto')
    estimate = chart_and_pack(groups, simple_shader() * 2, s, nest=False)
    baked = chart_and_pack(groups, simple_shader() * 2, s, nest=True)
    assert estimate.page_max == baked.page_max == 512
    assert max(max(p.width, p.height) for p in baked) <= 512


def test_a_tie_keeps_the_rectangle_layout(monkeypatch, caplog):
    from lightmapper import charts
    s = ChartSettings(page_max=128)
    parts = [quad_piece(0, 20, 20), quad_piece(1, 20, 20)]
    rects = charts._pack_rectangles(parts, s)
    # (a separated layout of the same pages and texels, as a nest of these charts would be)
    monkeypatch.setattr(charts, '_nest_pack', lambda units, settings, *margin: charts._pack_rectangles(units, settings))
    with caplog.at_level(logging.INFO, logger='lightmapper.charts'):
        assert charts._nested_or(rects, parts, s) is rects
    assert 'ties with the rectangle layout' in caplog.text


def test_islands_of_one_part_too_close_fall_back(monkeypatch, caplog):
    # (xatlas places a part's islands apart, so they need the full gap between them like any two charts)
    from dataclasses import replace
    from lightmapper import charts
    s = ChartSettings(page_max=128)
    parts = [frame_piece(0, 100, 60), loose_piece(1, [(0, 0, 14, 14), (36, 0, 14, 14)])]
    rects = charts._pack_rectangles(parts, s)
    frame, left, right = charts._nest_units(parts)
    placed = [frame, (*left[:3], left[3] - left[3].min(axis=0) + 30, *left[4:]),
              (*right[:3], right[3] - right[3].min(axis=0) + [47, 30], *right[4:])]
    squeezed = charts._finish([placed], np.array([[100.0, 100.0]]), replace(s, page_max=128))
    assert total_area(squeezed) < total_area(rects)
    monkeypatch.setattr(charts, '_nest_pack', lambda units, settings, *margin: squeezed)
    with caplog.at_level(logging.INFO, logger='lightmapper.charts'):
        assert charts._nested_or(rects, parts, s) is rects
    assert 'charts too close' in caplog.text


def area_change_bound_ratio(pages, parts):
    # the largest |change of a triangle's doubled texel area| over ulp(page side) x its perimeter
    old = {(p[0], int(sid)): p[3][tri].astype(np.float64) for p in parts for tri, sid in zip(p[4], p[5])}
    worst = 0.0
    for page in pages:
        ulp = float(np.spacing(np.float32(max(page.width, page.height))))
        for m in page.materials:
            texels = m.lm['texcoord'].astype(np.float64) * [page.width, page.height]
            for tri, sid in zip(m.tris, m.surface_ids):
                before = old[(m.group, int(sid))]
                perimeter = sum(np.hypot(*(before[k] - before[(k + 1) % 3])) for k in range(3))
                change = abs(_tri_area(texels[tri][None])[0] - _tri_area(before[None])[0])
                worst = max(worst, change / (ulp * perimeter))
    return worst


def test_texel_areas_stay_within_the_float32_rounding_of_a_large_page():
    # texcoords are stored as float32 fractions of the page, so a triangle's area can only be kept to the rounding
    # of a coordinate near the page side; real levels measured up to 0.3 of this bound, for both layouts
    from lightmapper.charts import _nested_or, _pack_rectangles
    s = ChartSettings(page_max=4096)
    rng = np.random.default_rng(3)
    parts = [frame_piece(0, 4070.37, 3000.61)]
    parts += [quad_piece(i, *rng.uniform(0.3, 40, 2)) for i in range(1, 40)]
    rects = _pack_rectangles(parts, s)
    pages = _nested_or(rects, parts, s)
    assert pages is not rects and len(pages) < len(rects)
    assert max(p.width for p in pages) == 4096
    for layout in (rects, pages):
        assert area_change_bound_ratio(layout, parts) <= 1.0


def test_islands_split_on_the_values_xatlas_gets():
    # (far from the part's origin float32 joins two corners 1e-3 apart; in the island's own frame, where xatlas
    # gets it, they are apart again, so they are two islands)
    from lightmapper.charts import _components, _nest_input, _nest_units
    far = 1e6
    quad = [(0, 0, 4, 4)]
    near = loose_piece(1, [(far, far, 4, 4), (far + 4.001, far, 4, 4)])
    origin = loose_piece(1, quad)
    uv = np.vstack([origin[3], near[3]])
    tris = np.vstack([origin[4], near[4] + len(origin[3])])
    part = (1, None, np.arange(len(uv)), uv, tris, np.arange(len(tris)))
    assert len(_components(_nest_input(part), tris, np.arange(len(tris)))) == 2
    units = _nest_units([part])
    assert len(units) == 3
    for unit in units:
        assert len(_components(_nest_input(unit), unit[4], np.arange(len(unit[4])))) == 1
    assert sorted(int(i) for unit in units for i in unit[5]) == list(range(len(tris)))


def test_an_island_of_only_zero_area_triangles_does_not_block_nesting():
    # (xatlas does not place a zero-area island; it rides along at the first chart, and holds no texel there)
    from lightmapper.charts import _nested_or, _pack_rectangles
    s = ChartSettings(page_max=128)
    pos = [[0, 0, 0], [14, 0, 0], [14, 14, 0], [0, 14, 0], [60, 60, 0], [70, 60, 0], [80, 60, 0]]
    group = make_group(np.array(pos, np.float32), [[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    part = (1, group, np.arange(7), np.array(pos, np.float64)[:, :2], group.tris, group.surface_ids)
    parts = [frame_piece(0, 100, 60), part]
    rects = _pack_rectangles(parts, s)
    pages = _nested_or(rects, parts, s)
    assert pages is not rects and total_area(pages) < total_area(rects)
    assert sorted(int(i) for p in pages for m in p.materials for i in m.surface_ids) == sorted(list(range(16)) + [0, 1, 2])


def test_auto_never_cuts_a_chart_that_needs_the_full_shape():
    # (a 700 x 200 chart of 50-texel cells needs 1024 x 512; a page a step down would cut it)
    pos, uvs, tris = [], [], []
    for y in range(5):
        for x in range(15):
            pos.append([50 * x, 50 * y, 0])
            uvs.append([50 * x / 1000, 50 * y / 1000])
    for y in range(4):
        for x in range(14):
            a = 15 * y + x
            tris += [[a, a + 1, a + 16], [a, a + 16, a + 15]]
    group = make_group(np.array(pos, np.float32), tris, uvs=uvs)
    pages = chart_and_pack([group], simple(), nest_settings())
    corners = corners_by_surface(pages)
    where = {int(sid): k for k, p in enumerate(pages) for m in p.materials for sid in m.surface_ids}
    assert len(set(where.values())) == 1
    quad = np.concatenate([corners[i] for i in range(len(tris))])
    assert sorted(np.round(np.ptp(quad, axis=0))) == [200, 700]


def written_count(pages, shape):
    from lightmapper.charts import written_pages
    return written_pages(pages, shape, nest_settings())


def test_auto_steps_the_shape_up_until_ten_pages_or_fewer():
    # 75 charts of 68 padded texels: three to a 256 x 128 page would be 25 pages, nine to a 256 x 256 page are 9
    pages = chart_and_pack([rect_group([(60, 60)] * 75)], simple(), nest_settings())
    assert pages.page_shape == (256, 256)
    assert written_count(pages, (256, 128)) == 25
    assert written_count(pages, pages.page_shape) <= 10


def test_auto_within_the_page_target_keeps_the_shape_the_biggest_chart_needs():
    pages = chart_and_pack([rect_group([(60, 60)] * 9)], simple(), nest_settings())
    assert pages.page_shape == (256, 128)


def test_auto_at_the_cap_may_stay_above_ten_pages(monkeypatch):
    from lightmapper import charts
    monkeypatch.setattr(charts, 'AUTO_PAGE_MAX', 512)
    monkeypatch.setattr(charts, 'AUTO_PAGE_MIN', 256)
    pages = chart_and_pack([rect_group([(400, 400)] * 12)], simple(), nest_settings())
    assert pages.page_shape == (512, 512) and written_count(pages, (512, 512)) == 12


def test_the_estimate_and_the_bake_choose_the_same_shape():
    group = rect_group([(60, 60)] * 75)
    assert chart_and_pack([group], simple(), nest_settings(), nest=True).page_shape == \
        chart_and_pack([group], simple(), nest_settings(), nest=False).page_shape


def frames_and_quads(count, outer, hole, quad):
    # count hollow frames and as many quads, each quad small enough to nest in a frame
    pos, uvs, tris = [], [], []
    for i in range(count):
        edges = [0, (outer - hole) / 2, (outer + hole) / 2, outer]
        base = len(pos)
        for y in edges:
            for x in edges:
                pos.append([i * 1000 + x, y, 0])
                uvs.append([i * 2 + x / 1000, y / 1000])
        for cy in range(3):
            for cx in range(3):
                if (cx, cy) != (1, 1):
                    a = base + cy * 4 + cx
                    tris += [[a, a + 1, a + 5], [a, a + 5, a + 4]]
        base = len(pos)
        for x, y in ((0, 0), (quad, 0), (quad, quad), (0, quad)):
            pos.append([i * 1000 + 500 + x, y, 0])
            uvs.append([i * 2 + 1 + x / 1000, y / 1000])
        tris += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return make_group(np.array(pos, np.float32), tris, uvs=uvs)


def test_the_bake_counts_the_nested_layout_and_can_settle_smaller_than_the_estimate():
    # (found by search: 20 frames with a quad each; as rectangles 256 x 128 pages hold too few, nested they hold 10)
    group = frames_and_quads(20, 100, 70, 40)
    bake = chart_and_pack([group], simple(), nest_settings(), nest=True)
    estimate = chart_and_pack([group], simple(), nest_settings(), nest=False)
    assert bake.page_shape == (256, 128) and estimate.page_shape == (256, 256)
    assert bake.estimate_shape == estimate.page_shape
    assert written_count(bake, bake.page_shape) <= 10


def test_a_fixed_max_refuses_before_the_bake_when_the_write_needs_too_many_pages(monkeypatch):
    # (the bake pages fit, but the write would need more pages than a bitmap holds: that must fail before the bake)
    from lightmapper import charts
    shapes = []

    def too_many(pages, shape, s, blocks=None):
        shapes.append(shape)
        return s.max_pages + 1
    monkeypatch.setattr(charts, 'written_pages', too_many)
    with pytest.raises(CapError) as caught:
        chart_and_pack([rect_group([(60, 60)] * 3)], simple(), ChartSettings(texels_per_unit=1.0, page_max=256))
    assert shapes == [(256, 256)] * len(shapes) and len(shapes) >= 1 and caught.value.page_max == 256


def test_auto_at_the_cap_refuses_when_the_write_needs_too_many_pages(monkeypatch):
    from lightmapper import charts
    monkeypatch.setattr(charts, 'AUTO_PAGE_MAX', 256)
    monkeypatch.setattr(charts, 'AUTO_PAGE_MIN', 256)
    monkeypatch.setattr(charts, 'written_pages', lambda pages, shape, s, blocks=None: s.max_pages + 1)
    with pytest.raises(CapError) as caught:
        chart_and_pack([rect_group([(60, 60)] * 4)], simple(), nest_settings())
    assert caught.value.page_max == 'auto'


def test_a_nested_block_at_the_cap_does_not_step_past_it(monkeypatch):
    from lightmapper import charts
    monkeypatch.setattr(charts, 'AUTO_PAGE_MAX', 256)
    monkeypatch.setattr(charts, '_blocks_fit', lambda pages, shape, s: False)
    monkeypatch.setattr(charts, '_nested_or', lambda rects, parts, s: list(rects))
    pages = chart_and_pack([rect_group([(60, 60)] * 4)], simple(), nest_settings(), nest=True)
    assert pages.page_shape == (256, 256)
