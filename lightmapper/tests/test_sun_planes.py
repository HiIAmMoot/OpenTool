"""Smooth normals cannot put direct sun onto the solid side of a receiver plane."""

import numpy as np
import pytest

from test_subtexels import bake_module
from test_blender_bake import bake, floor, quad, sun


SMOOTH = np.float32([-0.2, 0, np.sqrt(0.96)])
LIGHT = np.float64([0.8, 0, 0.6])


def away_quad():
    return quad([(0, 0, 0), (0, 0, 4), (0, 4, 4), (0, 4, 0)], SMOOTH,
                lm=((0, 0), (0.5, 0), (0.5, 1), (0, 1)))


@pytest.mark.blender
def test_plane_away_page_has_no_sun_and_facing_page_is_unchanged(tmp_path):
    facing = list(floor(10, 14, 0, 4, lm=((0.5, 0), (1, 0), (1, 1), (0.5, 1))))
    facing[3][:] = SMOOTH
    receiver = dict(quad=tuple(facing), casts=False)
    both = bake(tmp_path / 'both', [dict(quad=away_quad(), casts=False), receiver], [sun(LIGHT)])
    control = bake(tmp_path / 'control', [receiver], [sun(LIGHT)])
    assert both.pages[0][2:14, 1:7].max() == 0
    assert both.pages[0][2:14, 9:15].mean() > 0.1
    np.testing.assert_allclose(both.pages[0][2:14, 9:15], control.pages[0][2:14, 9:15], rtol=0.02)


def query_scene(reverse=False):
    # seam copies 0 and 3 share query 0; query 1 uses only the away face
    positions = np.float32([[0, 0, 0], [0, 0, 4], [0, 4, 0],
                            [0, 0, 0], [4, 0, 0], [0, 4, 0]])
    tris = np.int32([[0, 1, 2], [3, 4, 5]])
    if reverse:
        tris = tris[:, ::-1]
    return dict(positions=positions, normals=np.tile(SMOOTH, (6, 1)), tris=tris,
                query_pos=positions[[0, 1]], query_normal=np.tile(SMOOTH, (2, 1)),
                query_class=np.int16([0, 0]), query_inverse=np.int32([0, 1, 1, 0, 0, 0]))


@pytest.mark.parametrize('reverse', [False, True])
def test_vertex_sun_requires_any_adjacent_plane_facing_light(bake_module, reverse):
    vp = bake_module.VertexPass.__new__(bake_module.VertexPass)
    vp.scene = query_scene(reverse)
    vp.job = dict(vertex_rays=1, suns=[dict(direction=LIGHT.tolist(), color=[1, 1, 1], strength=3., classes=[0])])
    # only the ray backend is replaced: every ray misses, leaving the real direct-sun loop observable
    vp.cast = lambda *args: (-1, None)
    vp.shade = lambda tri, *args: np.zeros(len(tri))
    vp.ambient = np.zeros((2, 3))
    vector, energy, _ = vp.run()
    assert energy[1] == 0
    np.testing.assert_array_equal(vector[1], [0, 0, 0])
    assert energy[0] == pytest.approx(3 * (SMOOTH @ LIGHT), rel=1e-6)


def test_sun_query_relation_at_stock_surface_cap(bake_module):
    assert hasattr(bake_module, 'sun_facing_queries'), 'query-to-face sun reduction is missing'
    count = 131072
    scene = query_scene()
    scene['positions'] = np.tile(scene['positions'][:3], (count, 1))
    scene['normals'] = np.tile(SMOOTH, (3 * count, 1))
    scene['tris'] = np.arange(count * 3).reshape(-1, 3)
    scene['query_inverse'] = np.arange(count * 3)
    scene['query_pos'] = scene['positions']
    suns = [dict(direction=LIGHT.tolist()), dict(direction=[-1, 0, 0])]
    visible = bake_module.sun_facing_queries(scene, suns)
    assert visible.shape == (count * 3, 2)
    assert not visible[:, 0].any()
    assert visible[:, 1].all()


class Light:
    def __init__(self, diameter, pitch=0.9):
        self.values = {'radiosity.direction': (0.3, pitch), 'radiosity.diameter': diameter,
                       'radiosity.color': (1, 1, 1), 'radiosity.power': 2.0}

    def get(self, key):
        return self.values[key]


def grid_job_suns(light, n=4):
    from lightmapper.gather import sky_samples
    return [dict(direction=s.direction.tolist(), centre=s.centre.tolist(), classes=[0])
            for s in sky_samples(light, (0,), n)]


def test_grid_samples_of_one_sky_light_share_one_collection(bake_module):
    # a terminator face: the light's centre is just above its plane, some grid samples fall below
    suns = grid_job_suns(Light(0.05, pitch=0.01))
    scene = dict(positions=np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), normals=np.tile(np.float32([0, 0, 1]), (3, 1)),
                 tris=np.int32([[0, 1, 2]]))
    assert any(s['direction'][2] < 0 for s in suns) and any(s['direction'][2] > 0 for s in suns)
    facing = bake_module.sun_facing_planes(scene, suns)
    assert facing.all()
    groups = bake_module.link_groups(suns, [(dict(sun_facing=facing[0]), 0, 0)])
    assert list(groups.values()) == [list(range(16))]


def test_more_than_the_supported_light_link_sets_is_an_error(bake_module):
    def distinct(count):
        suns = [dict(direction=[0, 0, 1], classes=[0]) for _ in range(count)]
        # receiver k is lit only by sun k, so every sun has its own set
        return suns, [(dict(sun_facing=[n == k for n in range(count)]), 0, 0) for k in range(count)]

    suns, receivers = distinct(63)
    assert len(bake_module.link_groups(suns, receivers)) == 63
    suns, receivers = distinct(64)
    with pytest.raises(RuntimeError, match=r'64 .*63'):
        bake_module.link_groups(suns, receivers)


def test_job_carries_the_sky_light_centre(tmp_path):
    from lightmapper.blender_job import write_job
    from lightmapper.gather import sky_samples
    from lightmapper.settings import Settings
    from test_blender_job import make_scene
    import json

    write_job(tmp_path, make_scene(), [(16, 8)], sky_samples(Light(0.01), (0,), 2), {}, Settings())
    job = json.loads((tmp_path / 'job.json').read_text())
    assert len(job['suns']) == 4 and len({tuple(s['centre']) for s in job['suns']}) == 1
    assert len({tuple(s['direction']) for s in job['suns']}) == 4


def test_receivers_are_not_split_by_suns_that_cannot_light_their_class(bake_module):
    assert hasattr(bake_module, 'receiver_sun_bits'), 'class-masked sun bits are missing'
    # triangle 0 faces up, triangle 1 down; both are class 0, which the second sun (up, class 1 only) never lights
    scene = dict(positions=np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]]),
                 normals=np.float32([[0, 0, 1]] * 3 + [[0, 0, -1]] * 3),
                 tris=np.int32([[0, 1, 2], [3, 4, 5]]), tri_class=np.int16([0, 0]))
    suns = [dict(direction=[1, 0, 0], classes=[0]), dict(direction=[0, 0, 1], classes=[1])]
    raw = bake_module.sun_facing_planes(scene, suns)
    assert raw[0, 1] and not raw[1, 1]
    bits = bake_module.receiver_sun_bits(scene, suns)
    np.testing.assert_array_equal(bits[0], bits[1])
    assert not bits[:, 1].any()
    # for a class a sun can light, the bit is the plane test itself, so the lit sets do not change
    scene['tri_class'] = np.int16([1, 1])
    np.testing.assert_array_equal(bake_module.receiver_sun_bits(scene, suns)[:, 1], raw[:, 1])


def test_terminator_planes_complement_bake_planes_after_float32_exchange(bake_module):
    from types import SimpleNamespace
    from lightmapper.terminator import prepare_boundaries

    rng = np.random.default_rng(1618)
    positions = rng.normal(size=(3000, 3))
    positions[1500:] += 1e8
    normals = rng.normal(size=positions.shape)
    tris = np.arange(3000).reshape(-1, 3)
    classes = rng.integers(-1, 3, len(tris))
    suns = [dict(direction=d.tolist(), classes=[-1, 0, 2]) for d in rng.normal(size=(5, 3))]
    # the npz exchange hands bake.py float32 positions and normals, even for a float64 caller
    exchange = dict(positions=positions.astype(np.float32), normals=normals.astype(np.float32),
                    tris=tris, tri_class=classes)
    scene = SimpleNamespace(positions=positions, normals=normals, tris=tris, tri_class=classes,
                            tri_ignore_normals=None)
    away = prepare_boundaries(scene, suns, 0.225).away
    facing = bake_module.sun_facing_planes(exchange, suns)
    bits = bake_module.receiver_sun_bits(exchange, suns)
    eligible = np.tile(np.isin(classes, [-1, 0, 2])[:, None], (1, 5))
    np.testing.assert_array_equal(away, ~facing & eligible)
    np.testing.assert_array_equal(away | bits, eligible)
    assert not (away & bits).any()


def test_gather_rays_start_one_epsilon_along_their_own_direction(bake_module):
    vp = bake_module.VertexPass.__new__(bake_module.VertexPass)
    vp.scene = dict(query_pos=np.float32([[2, 3, 4]]), query_normal=np.float32([[0, 1, 0]]),
                    query_class=np.int16([0]))
    vp.job = dict(vertex_rays=4, suns=[])
    vp.ambient = np.zeros((2, 3))
    calls = []

    def cast(origin, direction):
        calls.append((np.array(origin), np.array(direction)))
        return -1, None

    vp.cast = cast
    vp.shade = lambda tri, *args: np.zeros(len(tri))
    vp.run()
    assert len(calls) == 4
    for origin, direction in calls:
        np.testing.assert_allclose(origin - [2, 3, 4], 0.001 * direction, rtol=0, atol=1e-14)
