"""A face whose plane turns away from a sun still takes it from its lit corners, as tool's vertex pass interpolates it."""

import dataclasses
import json

import numpy as np
import pytest

from lightmapper.blender_job import run_blender, write_job
from lightmapper.settings import Settings, load_settings
from lightmapper.terminator import sun_term
from test_blender_bake import UP, build, floor, quad, sun


# a reach no test geometry comes near, which leaves the whole triangle interpolated as before
WIDE = 1e9
LIGHT = np.float64([0.8, 0, 0.6])
STRENGTH = 3.0
# the cosine every corner sees: all corner normals are straight up
COSINE = 0.6


def terrace(occluders=(), seam=0.0):
    """Floor, a wall hanging from its edge whose plane faces away from the sun, and a lower floor."""
    a, b = 1 / 3 + seam, 2 / 3 + seam
    high = floor(0, 4, 0, 4, lm=((0, 0), (a, 0), (a, 1), (0, 1)))
    # wound so its plane normal is -x, away from the sun, while its smooth normals point up
    wall = quad([(0, 0, 0), (0, 4, 0), (0, 4, -4), (0, 0, -4)], (0, 0, 1),
                lm=((a, 0), (a, 1), (b, 1), (b, 0)))
    low = floor(-4, 0, 0, 4, z=-4, lm=((b, 0), (1, 0), (1, 1), (b, 1)))
    scene = build([dict(quad=high, casts=False), dict(quad=wall, casts=False), dict(quad=low, casts=False)],
                  occluders)
    positions, inverse = np.unique(scene.positions, axis=0, return_inverse=True)
    return dataclasses.replace(
        scene, query_pos=positions, query_normal=np.tile(np.float32(UP), (len(positions), 1)),
        query_class=np.zeros(len(positions), np.int16), query_inverse=inverse.ravel().astype(np.int32))


def bake_terrace(tmp_path, scene):
    s = Settings(samples=32, bounces=1, w_sun=1.0, w_emit=1.0, w_amb=1.0, albedo_scale=1.0,
                 albedo_fallback=0.5, vertex_rays=1)
    write_job(tmp_path, scene, [(24, 24)], [sun(LIGHT, strength=STRENGTH)], {}, s, 'all')
    result = run_blender(tmp_path, s)
    return result, json.loads((tmp_path / 'job.json').read_text())['suns']


@pytest.fixture(scope='module')
def lit_terrace(tmp_path_factory):
    scene = terrace()
    return (scene, *bake_terrace(tmp_path_factory.mktemp('terrace'), scene))


@pytest.mark.blender
def test_wall_facing_away_takes_a_gradient_that_meets_its_lit_neighbours(lit_terrace):
    scene, result, suns = lit_terrace
    page = result.pages[0]
    assert page[3:21, 9:15].max() < 1e-3, 'the wall is unlit in the bake, the premise of the test'
    term = sun_term(scene, suns, result.vertex_sun_lit, 0, 24, 24, WIDE)
    wall = term[3:21, 9:15]
    expected = STRENGTH / np.pi * COSINE
    # the wall's value at its top edge meets the floor's baked texels beside it, and the lower floor at its foot
    np.testing.assert_allclose(wall[:, 0:1, 0], page[3:21, 3:4, 0], rtol=0.1)
    np.testing.assert_allclose(wall[:, -1:, 0], page[3:21, 18:19, 0], rtol=0.1)
    assert wall[..., 0].min() > 0.9 * expected


@pytest.mark.blender
def test_term_has_the_units_of_a_plane_facing_cycles_texel(lit_terrace):
    scene, result, suns = lit_terrace
    term = sun_term(scene, suns, result.vertex_sun_lit, 0, 24, 24, WIDE)
    baked = result.pages[0][3:21, 2:6, 0].mean()
    assert term[3:21, 9:15, 0].mean() == pytest.approx(baked, rel=0.05)
    # (a plane-facing texel takes no term: the bake has it)
    assert term[3:21, 2:6].max() == 0


@pytest.mark.blender
def test_seams_inside_a_texel_mix_the_baked_and_the_added_light_like_the_bake_does(tmp_path):
    # the seams sit a third of a texel past the texel boundaries (24 texels, 72 sub-texels across)
    scene = terrace(seam=1 / 72)
    result, suns = bake_terrace(tmp_path, scene)
    total = result.pages[0] + sun_term(scene, suns, result.vertex_sun_lit, 0, 24, 24, WIDE)
    # the texels holding a seam take part of their light from each side, and the sum is the one value all faces share
    np.testing.assert_allclose(total[3:21, 2:22, 0], STRENGTH / np.pi * COSINE, rtol=0.05)


@pytest.mark.blender
def test_wall_whose_corners_are_all_shadowed_stays_unlit(tmp_path):
    roof = quad([(-10, -5, 10), (60, -5, 10), (60, 10, 10), (-10, 10, 10)], (0, 0, -1))
    scene = terrace(occluders=[dict(quad=roof)])
    result, suns = bake_terrace(tmp_path, scene)
    assert not result.vertex_sun_lit.any()
    assert sun_term(scene, suns, result.vertex_sun_lit, 0, 24, 24, WIDE).max() == 0


def test_a_class_the_sun_cannot_light_gets_nothing():
    scene = terrace()
    suns = [dict(direction=LIGHT.tolist(), color=[1, 1, 1], strength=STRENGTH, classes=[0])]
    lit = np.ones((len(scene.query_pos), 1), bool)
    assert sun_term(scene, suns, lit, 0, 24, 24, WIDE).max() > 0
    scene.tri_class[:] = 1
    assert sun_term(scene, suns, lit, 0, 24, 24, WIDE).max() == 0


def test_corner_that_is_not_lit_pulls_the_wall_toward_dark():
    scene = terrace()
    suns = [dict(direction=LIGHT.tolist(), color=[1, 1, 1], strength=STRENGTH, classes=[0])]
    lit = np.ones((len(scene.query_pos), 1), bool)
    lit[np.all(scene.query_pos == [0, 0, -4], axis=1) | np.all(scene.query_pos == [0, 4, -4], axis=1)] = False
    wall = sun_term(scene, suns, lit, 0, 24, 24, WIDE)[3:21, 9:15, 0]
    assert wall[:, 0].mean() > 2 * wall[:, -1].mean() > 0


def lone_wall():
    wall = quad([(0, 0, 0), (0, 4, 0), (0, 4, -4), (0, 0, -4)], (0, 0, 1), lm=((0, 0), (1, 0), (1, 1), (0, 1)))
    scene = build([dict(quad=wall, casts=False)], ())
    positions, inverse = np.unique(scene.positions, axis=0, return_inverse=True)
    return dataclasses.replace(
        scene, query_pos=positions, query_normal=np.tile(np.float32(UP), (len(positions), 1)),
        query_class=np.zeros(len(positions), np.int16), query_inverse=inverse.ravel().astype(np.int32))


def sun_of(scene):
    return ([dict(direction=LIGHT.tolist(), color=[1, 1, 1], strength=STRENGTH, classes=[0])],
            np.ones((len(scene.query_pos), 1), bool))


def flat_scene():
    scene = build([dict(quad=floor(0, 4, 0, 4), ignore_normals=True)])
    scene.query_inverse = np.arange(4)
    suns = [dict(direction=[0, 0, -1], color=[0.5, 1, 0.25], strength=np.pi * 2, classes=[0])]
    return scene, suns, np.ones((4, 1, 3), np.float32)


def test_flat_sun_term_lights_every_reached_corner_at_full_strength():
    from lightmapper.terminator import flat_sun_term
    scene, suns, throughput = flat_scene()
    np.testing.assert_allclose(flat_sun_term(scene, suns, throughput, 0, 2, 2),
                               np.tile([1, 2, 0.5], (2, 2, 1)), rtol=1e-6)
    assert flat_sun_term(scene, suns, throughput, 1, 2, 2).max() == 0
    scene.tri_class[:] = 1
    assert flat_sun_term(scene, suns, throughput, 0, 2, 2).max() == 0


def test_flat_sun_term_interpolates_an_unlit_corner():
    from lightmapper.terminator import flat_sun_term
    scene, suns, throughput = flat_scene()
    throughput[0] = 0
    want = np.array([[35 / 108, 0.75], [0.75, 89 / 108]])
    np.testing.assert_allclose(flat_sun_term(scene, suns, throughput, 0, 2, 2),
                               want[:, :, None] * [1, 2, 0.5], rtol=1e-6)


@pytest.mark.parametrize('tint', [(0.5, 0.25, 1), (0, 0, 0)])
def test_flat_sun_term_takes_the_throughputs_colour(tint):
    from lightmapper.terminator import flat_sun_term
    scene, suns, throughput = flat_scene()
    throughput[:] = tint
    np.testing.assert_allclose(flat_sun_term(scene, suns, throughput, 0, 2, 2),
                               np.tile(np.array([1, 2, 0.5]) * tint, (2, 2, 1)), rtol=1e-6)


def test_flat_sun_term_leaves_unflagged_triangles_alone():
    from lightmapper.terminator import flat_sun_term
    scene, suns, throughput = flat_scene()
    scene.tri_ignore_normals[:] = False
    assert flat_sun_term(scene, suns, throughput, 0, 2, 2).max() == 0


def test_flagged_faces_make_no_edge_probes():
    from lightmapper.terminator import edge_queries, prepare_boundaries
    scene = terrace()
    suns, _ = sun_of(scene)
    assert len(edge_queries(scene, suns, 1)['edge_pos']) > 0
    scene.tri_ignore_normals = np.array([False, False, True, True, False, False])
    assert len(edge_queries(scene, suns, 1)['edge_pos']) == 0
    assert not prepare_boundaries(scene, suns, 1).away.any()


def test_flat_sun_term_at_the_stock_triangle_limit():
    from lightmapper.terminator import flagged_fraction, flat_sun_term
    scene, suns, _ = flat_scene()
    size = 256
    x, y = np.meshgrid(np.arange(size + 1), np.arange(size + 1))
    scene.positions = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]).astype(np.float32)
    scene.lm_uv = scene.positions[:, :2] / size
    a = (y[:-1, :-1] * (size + 1) + x[:-1, :-1]).ravel()
    scene.tris = np.stack([np.column_stack([a, a + 1, a + size + 2]),
                           np.column_stack([a, a + size + 2, a + size + 1])], axis=1).reshape(-1, 3)
    assert len(scene.tris) == 131072
    scene.tri_page = np.zeros(len(scene.tris), np.int16)
    scene.tri_class = np.zeros(len(scene.tris), np.int16)
    scene.tri_ignore_normals = np.ones(len(scene.tris), bool)
    scene.query_inverse = np.arange(len(scene.positions))
    throughput = np.tile(np.float32([0.5, 0.25, 1]), (len(scene.positions), 1, 1))
    np.testing.assert_allclose(flat_sun_term(scene, suns, throughput, 0, size, size),
                               np.full((size, size, 3), 0.5), rtol=1e-6)
    np.testing.assert_array_equal(flagged_fraction(scene, 0, size, size), 1)


def test_wall_between_lit_floors_is_lit_only_within_reach_of_them():
    scene = terrace()
    suns, lit = sun_of(scene)
    size, reach = 192, 1.0
    term = sun_term(scene, suns, lit, 0, size, size, reach)[:, :, 0]
    full = STRENGTH / np.pi * COSINE
    # the wall's depth axis runs along u, from column 64 (its top edge) to 128 (its foot) at 0.0625 world units a column; row 168 is
    # 3.5 up from its foot, where its first triangle still has the top edge as its nearest lit boundary
    row = 168
    assert term[row, 64] == pytest.approx(full, rel=0.05)
    assert term[row, 72] == pytest.approx(full / 2, rel=0.1)
    assert term[row, 80:112].max() == 0


def test_wall_with_no_lit_neighbour_and_no_lit_corner_gets_nothing():
    scene = lone_wall()
    suns, _ = sun_of(scene)
    lit = np.zeros((len(scene.query_pos), 1), bool)
    assert sun_term(scene, suns, lit, 0, 24, 24, WIDE).max() == 0


def test_a_huge_reach_interpolates_the_whole_wall():
    scene = terrace()
    suns, lit = sun_of(scene)
    term = sun_term(scene, suns, lit, 0, 192, 192, WIDE)[:, :, 0]
    np.testing.assert_allclose(term[24:168, 70:122], STRENGTH / np.pi * COSINE, rtol=1e-4)


def test_reach_defaults_to_tools_mesh_floor_and_round_trips(tmp_path):
    assert Settings().reach == 0.225
    path = tmp_path / 'settings.json'
    path.write_text('{"reach": 0.5}')
    assert json.loads(load_settings(path).to_json())['reach'] == 0.5


@pytest.mark.parametrize('reach', [0, -0.1, float('nan')])
def test_reach_must_be_positive(reach):
    with pytest.raises(ValueError, match='reach'):
        Settings(reach=reach)


def test_sampled_boundary_preparation_reuses_its_plane_classification(monkeypatch):
    import lightmapper.terminator as terminator
    scene = terrace()
    suns, _ = sun_of(scene)
    edge = terminator.edge_queries(scene, suns, 0.225)
    classify = terminator._planes
    calls = []

    def counted(*args):
        calls.append(1)
        return classify(*args)

    monkeypatch.setattr(terminator, '_planes', counted)
    result = terminator.prepare_boundaries(scene, suns, 0.225, np.ones(edge['edge_active'].shape, bool))
    assert result.away.any() and result.segments is not None
    assert len(calls) == 1


def test_a_page_of_away_and_facing_faces_gives_the_same_term_in_small_chunks(monkeypatch):
    import lightmapper.terminator as terminator
    scene = terrace()
    suns, lit = sun_of(scene)
    whole = sun_term(scene, suns, lit, 0, 192, 192, 1.0)
    monkeypatch.setattr(terminator, 'CHUNK', 1000)
    chunked = sun_term(scene, suns, lit, 0, 192, 192, 1.0)
    assert whole[:, 70:120].max() > 0 and (whole[:, :60] == 0).all()
    np.testing.assert_array_equal(chunked, whole)


@pytest.mark.blender
def test_shadowed_neighbour_middle_does_not_seed_the_wall(tmp_path):
    # y is unchanged along LIGHT: this roof shadows only the middle of both shared edges.
    roof = floor(-10, 20, 1, 3, z=2)
    scene = terrace(occluders=[dict(quad=roof)])
    result, suns = bake_terrace(tmp_path, scene)
    assert result.vertex_sun_lit.all(), 'corners stay lit; only the edge interior is shadowed'
    flags = getattr(result, 'edge_sun_lit', None)
    term = sun_term(scene, suns, result.vertex_sun_lit, 0, 192, 192, 0.225,
                    **({'edge_sun_lit': flags} if flags is not None else {}))
    assert term[80:112, 64:66].max() == 0, 'a shadowed facing neighbour must not seed the edge middle'
    assert term[16:24, 64, 0].min() > 0.4, 'the genuinely lit end must still seed the wall'


def test_partly_lit_edge_requires_two_consecutive_lit_points():
    import lightmapper.terminator as terminator
    scene = terrace()
    suns, lit = sun_of(scene)
    make = getattr(terminator, 'edge_queries', None)
    assert callable(make), 'edge queries must sample facing sides before their visibility can seed the wall'
    edge = make(scene, suns, 0.225)
    visible = ((edge['edge_pos'][:, 1] < 1) | (edge['edge_pos'][:, 1] > 3))[:, None]
    term = sun_term(scene, suns, lit, 0, 192, 192, 0.225, edge_sun_lit=visible)
    assert term[80:112, 64:66].max() == 0
    assert term[16:24, 64, 0].min() > 0.4
    # Isolated lit points are not segments (corners still seed their original small discs).
    visible[:] = False
    visible[np.abs(edge['edge_pos'][:, 1] - 2).argmin()] = True
    isolated = sun_term(scene, suns, lit, 0, 192, 192, 0.225, edge_sun_lit=visible)
    assert isolated[80:112, 64:66].max() == 0


def test_clear_sampled_edges_match_16h_and_missing_flags_keep_16h():
    import lightmapper.terminator as terminator
    scene = terrace()
    suns, lit = sun_of(scene)
    make = getattr(terminator, 'edge_queries', None)
    assert callable(make), 'sampled boundary support is not implemented'
    edge = make(scene, suns, 0.225)
    old = sun_term(scene, suns, lit, 0, 192, 192, 0.225)
    explicit_old = sun_term(scene, suns, lit, 0, 192, 192, 0.225, edge_sun_lit=None)
    new = sun_term(scene, suns, lit, 0, 192, 192, 0.225,
                   edge_sun_lit=np.ones((len(edge['edge_pos']), 1), bool))
    np.testing.assert_array_equal(explicit_old, old)
    np.testing.assert_allclose(new, old, rtol=0.05, atol=1e-6)


def test_edge_queries_include_endpoints_inset_and_interpolate_facing_normals():
    import lightmapper.terminator as terminator
    scene = terrace()
    suns, _ = sun_of(scene)
    make = getattr(terminator, 'edge_queries', None)
    assert callable(make), 'facing-side edge queries are missing'
    edge = make(scene, suns, 0.225)
    pos = edge['edge_pos']
    assert len(pos) == 38, 'two four-unit edges, 18 segments + both endpoints each'
    for z, sign in [(0, 1), (-4, -1)]:
        side = pos[pos[:, 2] == z]
        np.testing.assert_allclose(side[:, 0], sign * 0.025, atol=1e-6)
        np.testing.assert_allclose(side[:, 1], np.linspace(0, 4, 19) * 0.99375 + (0.025 if z == 0 else 0), atol=1e-6)
        assert np.diff(side[:, 1]).max() <= 0.225
    assert pos.dtype == np.float32
    np.testing.assert_array_equal(edge['edge_normal'], np.tile(UP, (38, 1)))
    assert (edge['edge_class'] == 0).all()
    scene.tri_class[:] = 1
    assert len(make(scene, suns, 0.225)['edge_pos']) == 0


def test_position_shared_edges_are_not_resampled_for_duplicate_faces():
    from lightmapper.terminator import edge_queries
    scene = terrace()
    suns, _ = sun_of(scene)
    repeated = dataclasses.replace(scene, tris=np.tile(scene.tris, (3333, 1)),
                                   tri_class=np.tile(scene.tri_class, 3333))
    assert len(repeated.tris) == 19998
    assert len(edge_queries(repeated, suns, 0.225)['edge_pos']) == 38


def test_sampled_boundary_distances_are_identical_in_small_chunks(monkeypatch):
    import lightmapper.terminator as terminator
    scene = terrace()
    suns, lit = sun_of(scene)
    edge = terminator.edge_queries(scene, suns, 0.225)
    flags = (edge['edge_pos'][:, 1] < 2)[:, None]
    whole = sun_term(scene, suns, lit, 0, 192, 192, 0.225, edge_sun_lit=flags)
    monkeypatch.setattr(terminator, 'CHUNK', 101)
    small = sun_term(scene, suns, lit, 0, 192, 192, 0.225, edge_sun_lit=flags)
    np.testing.assert_array_equal(small, whole)


@pytest.mark.parametrize('triangles', [20000, 131072])
def test_grid_snapped_edge_queries_at_stock_limits(triangles):
    from lightmapper.terminator import _segment_distance, edge_queries, prepare_boundaries
    source = terrace()
    copies = triangles // 6
    shift = np.zeros((copies, 1, 3), np.float32)
    shift[:, 0, 0] = (np.arange(copies) % 148) * 8
    shift[:, 0, 1] = (np.arange(copies) // 148) * 8
    positions = (source.positions[None] + shift).reshape(-1, 3)
    tris = (source.tris[None] + np.arange(copies)[:, None, None] * len(source.positions)).reshape(-1, 3)
    # The remainder is facing-only, so it must add no boundary probes.
    tris = np.concatenate([tris, np.tile(tris[:1], (triangles % 6, 1))])
    scene = dataclasses.replace(source, positions=positions, normals=np.tile(source.normals, (copies, 1)),
                                tris=tris, tri_class=np.zeros(triangles, np.int16))
    suns, _ = sun_of(source)
    suns *= 16
    edge = edge_queries(scene, suns, 0.225)
    assert len(scene.tris) == triangles
    assert edge['edge_pos'].shape == (copies * 38, 3)
    assert edge['edge_active'].shape == (copies * 38, 16)
    assert edge['edge_active'].all()
    if triangles == 131072:
        boundary = prepare_boundaries(scene, suns, 0.225, np.ones_like(edge['edge_active']))
        around = scene.positions[scene.tris[[2]]]
        point = around.mean(axis=1)
        assert len(boundary.segments) == 16
        for table in boundary.segments:
            distance = _segment_distance(point, around, np.array([2]), table)
            np.testing.assert_allclose(distance, [4 / 3], atol=1e-6)


def test_edge_normals_and_sun_classes_belong_to_the_facing_side():
    from lightmapper.terminator import edge_queries
    scene = terrace()
    # Duplicated wall vertices must not replace the floor's endpoint normals.
    scene.normals[:4] = [[0, 0, 1], [0, 0, 1], [0.6, 0, 0.8], [0.6, 0, 0.8]]
    scene.tri_class[:2] = 7
    suns, _ = sun_of(scene)
    suns += [dict(suns[0], classes=[0, 7])]
    edge = edge_queries(scene, suns, 0.225)
    top = edge['edge_pos'][:, 2] == 0
    assert (edge['edge_class'][top] == 7).all()
    assert not edge['edge_active'][top, 0].any()
    assert edge['edge_active'][top, 1].all()
    np.testing.assert_allclose(edge['edge_normal'][top][[0, 9, 18]],
                               [[0.00375, 0, 0.99875], [0.301875, 0, 0.899375], [0.6, 0, 0.8]], atol=1e-6)


def narrow_ridge(width):
    positions = np.float32([[0, 0, 0], [0, 4, 0], [width, 2, 0],
                            [0, 0, 0], [0, 4, 0], [0, 2, -4]])
    return dataclasses.replace(
        terrace(), positions=positions, normals=np.tile(np.float32(UP), (6, 1)),
        tris=np.int32([[0, 1, 2], [3, 4, 5]]), tri_class=np.int16([0, 0]), tri_page=np.int16([0, 0]),
        lm_uv=np.float32([[0, 0], [0, 1], [1 / 3, 0.5], [1 / 3, 0], [1 / 3, 1], [1, 0.5]]),
        query_pos=positions, query_normal=np.tile(np.float32(UP), (6, 1)), query_inverse=np.arange(6))


def test_narrow_facing_probes_are_convex_combinations_including_normals():
    from lightmapper.terminator import edge_queries
    scene = narrow_ridge(0.01)
    scene.normals[2] = [0.6, 0, 0.8]
    suns, _ = sun_of(scene)
    edge = edge_queries(scene, suns, 0.225)
    p = edge['edge_pos']
    assert len(p) == 19
    opposite_weight = p[:, 0] / 0.01
    b_weight = (p[:, 1] - 2 * opposite_weight) / 4
    a_weight = 1 - opposite_weight - b_weight
    assert (a_weight >= -1e-6).all() and (b_weight >= -1e-6).all() and (opposite_weight >= 0).all()
    np.testing.assert_allclose(opposite_weight, 0.25, atol=1e-6)
    np.testing.assert_allclose(p[:, 1], np.linspace(0.5, 3.5, 19), atol=1e-6)
    np.testing.assert_allclose(edge['edge_normal'], np.tile([0.15, 0, 0.95], (19, 1)), atol=1e-6)


def test_degenerate_facing_side_has_no_probes_and_seeds_no_edge():
    from lightmapper.terminator import edge_queries
    scene = narrow_ridge(0)
    suns, lit = sun_of(scene)
    edge = edge_queries(scene, suns, 0.225)
    assert edge['edge_pos'].shape == (0, 3)
    term = sun_term(scene, suns, lit, 0, 192, 192, 0.225, edge_sun_lit=np.empty((0, 1), bool))
    assert term[80:112, 64:67].max() == 0
    assert term.max() > 0, 'degenerate neighbours must not disable the lit corners'


def test_prepared_boundaries_reused_across_pages_without_resampling(monkeypatch):
    import lightmapper.terminator as terminator
    scene = terrace()
    scene.tri_page[3:] = 1
    suns, lit = sun_of(scene)
    edge = terminator.edge_queries(scene, suns, 0.225)
    flags = (edge['edge_pos'][:, 1] < 2)[:, None]
    expected = [sun_term(scene, suns, lit, p, 96, 96, 0.225, edge_sun_lit=flags) for p in range(2)]
    prepare = getattr(terminator, 'prepare_boundaries', None)
    assert callable(prepare), 'boundary tables must be prepared once for every page to reuse'
    boundary = prepare(scene, suns, 0.225, flags)

    def no_resampling(*args, **kwargs):
        raise AssertionError('resampled an already prepared boundary')

    monkeypatch.setattr(terminator, 'edge_queries', no_resampling)
    monkeypatch.setattr(terminator, '_edge_keys', no_resampling)
    for page in range(2):
        actual = sun_term(scene, suns, lit, page, 96, 96, 0.225, boundaries=boundary)
        np.testing.assert_array_equal(actual, expected[page])
def test_texel_queries_cover_each_flagged_subtexel():
    from lightmapper.raster import texel_queries
    from test_blender_bake import build, full_quad
    scene = build([full_quad(ignore_normals=True)])
    query = texel_queries(scene, 0, 2, 2)
    assert query['texel_xy'].shape == (36, 2)
    assert len(np.unique(query['texel_xy'], axis=0)) == 36
    np.testing.assert_allclose(query['texel_pos'][:, :2], (query['texel_xy'] + .5) * (4 / 6), atol=1e-6)
    np.testing.assert_allclose(query['texel_normal'], [[0, 0, 1]] * 36)
    scene.tri_ignore_normals[:] = False
    assert len(texel_queries(scene, 0, 2, 2)['texel_pos']) == 0
