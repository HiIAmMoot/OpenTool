"""The per-vertex incident radiosity pass in real Blender: the pass is driven directly (vertex_driver.py) on tiny
scenes with chosen page images, plus one end-to-end bake that checks out.npz carries it."""

import dataclasses
import json
import os
import subprocess

import numpy as np
import pytest

from lightmapper.blender_job import BAKE_SCRIPT, run_blender, write_job
from lightmapper.settings import Settings
from conftest import LIGHTMAPPER_BLENDER
from test_blender_bake import UP, build, floor, full_quad, quad, sun, wall_x


DRIVER = os.path.join(os.path.dirname(__file__), 'vertex_driver.py')
# built at import, before conftest's neutral calibration applies, so the unit weights are explicit
S = Settings(samples=8, bounces=1, albedo_scale=1.0, albedo_fallback=0.5, vertex_rays=64,
             w_sun=1.0, w_emit=1.0, w_amb=1.0, gamma=1.0, radiosity_mode='coherence', radiosity_scale=1.0)
FULL_UV = [(0, 0), (1, 0), (1, 1), (0, 1)]


def wall_facing_minus_x(x=3.0, emission=(0, 0, 0), lm=FULL_UV):
    corners = [(x, 0, 0), (x, 4, 0), (x, 4, 4), (x, 0, 4)]
    return dict(quad=quad(corners[::-1], (-1, 0, 0), lm[::-1]), emission=emission)


def run_driver(tmp_path, receivers, suns=(), occluders=(), ambient=None, textures=None, query=((2, 2, 0),),
               normal=(0, 0, 1), cls=0, pages=None, probes=None, rays=64, normals=None):
    scene = dataclasses.replace(
        build(receivers, occluders), query_pos=np.float32(query),
        query_normal=np.float32(normals if normals is not None else [normal] * len(query)), query_class=np.int16([cls] * len(query)))
    pages = pages or [np.zeros((16, 16, 3), np.float32)]
    settings = dataclasses.replace(S, vertex_rays=rays)
    write_job(tmp_path, scene, [(p.shape[1], p.shape[0]) for p in pages], list(suns), ambient or {}, settings,
              'all', textures=textures)
    np.savez(tmp_path / 'pages.npz', **{f'page_{i}': p for i, p in enumerate(pages)})
    if probes is not None:
        (tmp_path / 'probe.json').write_text(json.dumps(probes))
    env = dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent))
    done = subprocess.run([str(LIGHTMAPPER_BLENDER), '-b', '--factory-startup', '--python-exit-code', '1',
                           '--python', DRIVER, '--', str(tmp_path)], capture_output=True, text=True, env=env,
                          stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    return np.load(tmp_path / ('probe.npz' if probes is not None else 'vp.npz'))


@pytest.mark.blender
def test_sun_only_vector_points_at_the_sun(tmp_path):
    out = run_driver(tmp_path, [full_quad()], [sun(UP, strength=3.0)])
    v, e = out['vector'][0], out['irradiance'][0]
    assert e == pytest.approx(3.0, rel=1e-3)
    assert np.linalg.norm(v / e) > 0.98
    np.testing.assert_allclose(v / np.linalg.norm(v), [0, 0, 1], atol=1e-6)


@pytest.mark.blender
@pytest.mark.parametrize('edge_normal,lit', [(UP, True), ((-1, 0, 0.1), False), ((0, 0, 0), False)])
def test_edge_queries_cast_only_positive_cosine_sun_rays_not_hemispheres(tmp_path, edge_normal, lit):
    high = list(floor(0, 4, 0, 4))
    high[3] = np.tile(np.float32(edge_normal), (4, 1))
    wall = quad([(0, 0, 0), (0, 4, 0), (0, 4, -4), (0, 0, -4)], UP)
    out = run_driver(tmp_path, [dict(quad=high, casts=False), dict(quad=wall, casts=False)],
                     [sun((0.8, 0, 0.6))], normal=(0, 0, 0))
    assert out['edge_sun_lit'].shape == (19, 1)
    assert (out['edge_sun_lit'] == lit).all()
    assert int(out['rays']) == (19 if lit else 0)
    np.testing.assert_array_equal(out['vector'], [[0, 0, 0]])
    np.testing.assert_array_equal(out['irradiance'], [0])


@pytest.mark.blender
def test_emissive_wall_pulls_the_vector_towards_it(tmp_path):
    out = run_driver(tmp_path, [full_quad(), wall_facing_minus_x(emission=(4, 4, 4))], query=((1, 2, 0),))
    assert out['vector'][0][0] > 0 and out['irradiance'][0] > 0


@pytest.mark.blender
def test_emit_only_occluder_is_seen_by_the_rays_that_gather_light(tmp_path):
    def gathered(up, name):
        glow = dict(quad=floor(0, 4, 0, 4, z=1.0, up=up), casts=False, emission=(4, 4, 4))
        return run_driver(tmp_path / name, [full_quad()], occluders=[glow])['irradiance'][0]
    assert gathered(False, 'front') > 0
    assert gathered(True, 'back') == 0


@pytest.mark.blender
def test_emit_only_front_has_no_fallback_diffuse_ambient(tmp_path):
    glow = dict(quad=floor(-100, 100, -100, 100, z=1, up=False), casts=False, emission=(4, 4, 4))
    out = run_driver(tmp_path, [full_quad()], occluders=[glow], ambient={'0': [0.2] * 3})
    assert out['irradiance'][0] == pytest.approx(np.pi * (4 + 0.2), rel=1e-3)


@pytest.mark.blender
def test_full_shadow_without_other_light_is_dark(tmp_path):
    cover = dict(quad=floor(-1, 5, -1, 5, z=1.0))
    out = run_driver(tmp_path, [full_quad()], [sun(UP)], occluders=[cover])
    assert out['irradiance'][0] == 0 and not out['vector'][0].any()


@pytest.mark.blender
def test_back_face_of_a_wall_is_not_light(tmp_path):
    # wall_x faces +x, so the query at x < 3 sees its back
    back = dict(quad=wall_x(3, 0, 4, 0, 4), emission=(4, 4, 4))
    out = run_driver(tmp_path, [full_quad(), back], query=((1, 2, 0),))
    assert out['irradiance'][0] == 0


@pytest.mark.blender
def test_results_are_identical_across_runs(tmp_path):
    scene = ([full_quad(), wall_facing_minus_x(emission=(4, 4, 4))], [sun((1, 0, 2))])
    query = ((1, 2, 0), (2, 1, 0), (3.5, 3, 0))
    first = run_driver(tmp_path / 'a', *scene, query=query)
    second = run_driver(tmp_path / 'b', *scene, query=query)
    assert np.array_equal(first['vector'], second['vector'])
    assert np.array_equal(first['irradiance'], second['irradiance'])
    assert first['irradiance'].min() > 0


@pytest.mark.blender
def test_hit_uv_barycentric(tmp_path):
    # the page's red is 16 u - 0.5 and its green 16 v - 0.5 at the hit; the wall maps u = y/4, v = z/4
    page = np.zeros((16, 16, 3), np.float32)
    page[:, :, 0] = np.arange(16)[None, :]
    page[:, :, 1] = np.arange(16)[:, None]
    wall = wall_facing_minus_x(lm=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
    probes = [([1, 1.3, 2.6], [1, 0, 0], 0), ([1, 2.7, 1.1], [1, 0, 0], 0)]
    out = run_driver(tmp_path, [full_quad(), wall], pages=[page], probes=probes)
    for (origin, _, _), uv, radiance in zip(probes, out['uv'], out['radiance']):
        u, v = 0.25 + origin[1] / 8, 0.25 + origin[2] / 8
        assert uv == pytest.approx([u, v], abs=1e-5)
        red, green = 16 * u - 0.5, 16 * v - 0.5
        assert radiance == pytest.approx(0.5 * (0.2126 * red + 0.7152 * green), rel=1e-4)


def plane_with_alpha(alpha):
    texels = np.full((4, 4, 4), 255, np.uint8)
    texels[:, :, 3] = alpha
    return texels


@pytest.mark.blender
def test_cutout_ray_passes(tmp_path):
    hole = plane_with_alpha(255)
    hole[1:3, 1:3, 3] = 0
    screen = dict(quad=floor(0, 4, 0, 4, z=1.0), texture=0, cutout=True)
    open_hole = run_driver(tmp_path / 'hole', [full_quad()], [sun(UP)], occluders=[screen], textures=[hole])
    filled = run_driver(tmp_path / 'filled', [full_quad()], [sun(UP)], occluders=[screen],
                        textures=[plane_with_alpha(255)])
    assert open_hole['irradiance'][0] > 0
    assert filled['irradiance'][0] == 0


@pytest.mark.blender
def test_glass_casts_no_shadow_in_the_pass(tmp_path):
    glass = dict(quad=floor(-1, 5, -1, 5, z=1.0), casts=False)
    out = run_driver(tmp_path, [full_quad()], [sun(UP, strength=3.0)], occluders=[glass])
    assert out['irradiance'][0] == pytest.approx(3.0, rel=1e-3)


@pytest.mark.blender
def test_miss_and_occluder_use_ambient_of_the_query_class(tmp_path):
    ambient = {'0': [0.2, 0.2, 0.2]}
    sky = run_driver(tmp_path / 'sky', [full_quad()], ambient=ambient)
    # an enclosed query gets the flat ambient plus the occluder, lit by the ambient times its albedo (fallback 0.5)
    roof = dict(quad=floor(-50, 50, -50, 50, z=1.0))
    roofed = run_driver(tmp_path / 'roof', [full_quad()], ambient=ambient, occluders=[roof])
    assert sky['irradiance'][0] == pytest.approx(np.pi * 0.2, rel=1e-3)
    assert roofed['irradiance'][0] == pytest.approx(np.pi * 0.2 * 1.5, rel=1e-3)


@pytest.mark.blender
def test_pass_reaches_out_npz(tmp_path):
    scene = dataclasses.replace(build([full_quad()]), query_pos=np.float32([[2, 2, 0]]))
    write_job(tmp_path, scene, [(16, 16)], [sun(UP, strength=3.0)], {}, S, 'all')
    result = run_blender(tmp_path, S)
    assert result.vertex_irradiance[0] == pytest.approx(3.0, rel=1e-3)
    assert result.vertex_vector[0] == pytest.approx([0, 0, 3.0], abs=1e-2)


def roof(cls, emission=(0, 0, 0), z=1.0):
    # faces down; 100 wide, so every cosine-weighted ray from below hits it
    corners = [(-50, -50, z), (50, -50, z), (50, 50, z), (-50, 50, z)][::-1]
    return dict(quad=quad(corners, (0, 0, -1), FULL_UV[::-1]), cls=cls, emission=emission)


AMBIENT = {'0': [0.2, 0.2, 0.2], '-1': [0.4, 0.4, 0.4]}


@pytest.mark.blender
def test_indoor_receiver_hit_takes_the_indoor_ambient(tmp_path):
    out = run_driver(tmp_path, [roof(-1)], ambient=AMBIENT, query=((0, 0, 0),), cls=0)
    assert out['irradiance'][0] == pytest.approx(np.pi * (0.5 * 0.4 + 0.2), rel=1e-3)


@pytest.mark.blender
def test_indoor_query_sees_occluder_in_indoor_ambient(tmp_path):
    cover = dict(quad=floor(-50, 50, -50, 50, z=1.0))
    out = run_driver(tmp_path, [full_quad()], ambient=AMBIENT, occluders=[cover], cls=-1)
    assert out['irradiance'][0] == pytest.approx(np.pi * (0.5 * 0.4 + 0.4), rel=1e-3)


@pytest.mark.blender
def test_outdoor_ambient_is_kept_when_indoor_is_present(tmp_path):
    out = run_driver(tmp_path, [full_quad()], ambient=AMBIENT, cls=0)
    assert out['irradiance'][0] == pytest.approx(np.pi * 0.2, rel=1e-3)


@pytest.mark.blender
def test_open_indoor_query_gets_the_indoor_ambient_flat(tmp_path):
    out = run_driver(tmp_path, [full_quad()], ambient=AMBIENT, cls=-1)
    assert out['irradiance'][0] == pytest.approx(np.pi * 0.4, rel=1e-3)
    assert out['vector'][0] == pytest.approx([0, 0, np.pi * 0.4], abs=1e-3)


@pytest.mark.blender
def test_sun_not_reaching_the_class_adds_nothing(tmp_path):
    out = run_driver(tmp_path, [full_quad()], [sun(UP, classes=(1,))], cls=0)
    assert out['irradiance'][0] == 0 and not out['vector'][0].any()


@pytest.mark.blender
def test_zero_normal_query_casts_nothing(tmp_path):
    out = run_driver(tmp_path, [full_quad()], [sun(UP, strength=3.0)], ambient=AMBIENT,
                     query=((1, 1, 0), (2, 2, 0)), normals=[(0, 0, 0), (0, 0, 1)])
    assert np.isfinite(out['vector']).all() and np.isfinite(out['irradiance']).all()
    assert out['irradiance'][0] == 0 and not out['vector'][0].any()
    assert out['irradiance'][1] > 3.0


@pytest.mark.blender
def test_emission_is_counted_once(tmp_path):
    # the roof is lit from below by a sun the query's class does not get, so only the roof's own page reaches the query
    emission = 4.0
    scene = dataclasses.replace(
        build([roof(0, (emission,) * 3)]), query_pos=np.float32([[0, 0, 0]]), query_normal=np.float32([[0, 0, 1]]),
        query_class=np.int16([1]))
    write_job(tmp_path, scene, [(16, 16)], [sun((0, 0, -1), strength=3.0)], {}, S, 'all')
    result = run_blender(tmp_path, S)
    summed = result.pages[0].mean()
    diffuse = summed - emission
    assert diffuse > 0.5
    assert result.vertex_irradiance[0] == pytest.approx(np.pi * (0.5 * diffuse + emission), rel=0.02)


@pytest.mark.blender
def test_hit_albedo_is_read_raw_not_linearised(tmp_path):
    # the wall's page holds 1, so the query's irradiance is proportional to the wall texel's albedo
    def run(value, name):
        wall = dict(wall_facing_minus_x(), texture=0)
        texture = np.uint8([[[value] * 3 + [255]] * 2] * 2)
        out = run_driver(tmp_path / name, [wall], query=((1, 2, 0),), textures=[texture],
                         pages=[np.ones((16, 16, 3), np.float32)])
        return out['irradiance'][0]
    assert run(128, 'grey') / run(255, 'white') == pytest.approx(128 / 255, rel=0.02)


@pytest.mark.blender
@pytest.mark.parametrize('tint,passed', [((0, 0, 0), 0.0), ((1, 1, 1), 1.0), ((0.5, 0.5, 0.5), 0.5)])
def test_sun_shadow_ray_takes_the_pane_tint(tmp_path, tint, passed):
    glass = dict(quad=floor(-1, 5, -1, 5, z=1.0), casts=False, tint=tint)
    out = run_driver(tmp_path, [full_quad(), glass], [sun(UP, strength=3.0)])
    assert out['irradiance'][0] == pytest.approx(3.0 * passed, abs=1e-3)


@pytest.mark.blender
def test_sun_ray_starts_in_front_of_a_slightly_overhanging_face(tmp_path):
    # the face's plane passes through the query and leans 3 degrees past vertical, so a start straight up the
    # normal is behind it; a start along the sun is in front, where the corner is lit
    m = np.float64([1, 0, -0.054])
    m /= np.linalg.norm(m)
    t2 = np.float64([0.054, 0, 1])
    t2 /= np.linalg.norm(t2)
    origin = np.float64([2, 2, 0])
    corners = [origin + a * np.float64([0, 1, 0]) + b * t2 for a, b in ((-2, 0), (2, 0), (2, 2), (-2, 2))]
    slab = dict(quad=quad(corners, tuple(m)))
    out = run_driver(tmp_path, [full_quad()], [sun((1, 0, 1), strength=3.0)], occluders=[slab])
    assert out['irradiance'][0] == pytest.approx(3.0 * 2 ** -0.5, rel=1e-3)


@pytest.mark.blender
def test_open_flat_query_gets_the_vector_tools_formula_predicts(tmp_path):
    # bloodgulch's roof: sun luminance 0.8865 x power 1.8 at cosine 0.6428, outdoor ambient 0.168 along the normal
    toward = np.float64([0.7544, 0.1330, 0.6428])
    toward /= np.linalg.norm(toward)
    cosine, w_sun, w_amb = toward[2], 0.8865 * 1.8 * toward[2], 0.168
    out = run_driver(tmp_path, [full_quad()], [sun(toward, strength=np.pi * 0.8865 * 1.8)],
                     ambient={'0': [w_amb] * 3})
    v, e = out['vector'][0].astype(np.float64), out['irradiance'][0]
    expected = w_sun * toward + w_amb * UP
    assert e == pytest.approx(np.pi * (w_sun + w_amb), rel=1e-3)
    assert v == pytest.approx(np.pi * expected, rel=1e-3)
    length = min(np.linalg.norm(v) / e, 1)
    assert length == pytest.approx(0.956, abs=2e-3)


@pytest.mark.blender
def test_occluder_beside_an_open_query_changes_nothing_unless_it_bounces(tmp_path):
    # a black wall blocks no sun and gives no light, so the sky it hides must not change the vector
    black = np.uint8([[[0, 0, 0, 255]] * 2] * 2)
    wall = dict(quad=wall_facing_minus_x(x=2.5)['quad'], texture=0)
    kwargs = dict(suns=[sun((0, 0, 1), strength=3.0)], ambient={'0': [0.2, 0.2, 0.2]}, textures=[black],
                  query=((1, 2, 0),))
    bare = run_driver(tmp_path / 'bare', [full_quad()], **kwargs)
    walled = run_driver(tmp_path / 'walled', [full_quad()], occluders=[wall], **kwargs)
    assert walled['vector'][0] == pytest.approx(bare['vector'][0], rel=1e-4)
    assert walled['irradiance'][0] == pytest.approx(bare['irradiance'][0], rel=1e-4)
