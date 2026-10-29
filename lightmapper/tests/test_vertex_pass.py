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
               normal=(0, 0, 1), cls=0, pages=None, probes=None, rays=64, normals=None, seed=0, inverse=None,
                  sun_facing=False, albedo_fallback=0.5, ignore_normals=None, texel=None, texel_samples=None,
                  fast_path=False, mode='tool', bounces=1, texel_pages=None, texel_normals=None, texel_xy=None):
    scene = dataclasses.replace(
        build(receivers, occluders), query_pos=np.float32(query), query_inverse=inverse,
        query_normal=np.float32(normals if normals is not None else [normal] * len(query)), query_class=np.int16([cls] * len(query)))
    pages = pages or [np.zeros((16, 16, 3), np.float32)]
    if ignore_normals is not None:
        scene = dataclasses.replace(scene, query_ignore_normals=np.asarray(ignore_normals, bool))
    if texel is not None:
        scene = dataclasses.replace(scene, texel_pos=np.float32(texel),
                                    texel_normal=np.float32([[0, 0, 1]] * len(texel)),
                                    texel_class=np.zeros(len(texel), np.int16),
                                    texel_page=np.zeros(len(texel), np.int16),
                                    texel_xy=(np.indices((3, 3))[::-1].reshape(2, -1).T.astype(np.int32)
                                              if len(texel) == 9 else np.zeros((len(texel), 2), np.int32)))
    if texel_pages is not None:
        scene = dataclasses.replace(scene, texel_page=np.int16(texel_pages),
                                    texel_xy=np.tile(np.indices((3, 3))[::-1].reshape(2, -1).T, (2, 1)).astype(np.int32))
    if texel_normals is not None:
        scene = dataclasses.replace(scene, texel_normal=np.float32(texel_normals))
    if texel_xy is not None:
        scene = dataclasses.replace(scene, texel_xy=np.int32(texel_xy))
    settings = dataclasses.replace(S, vertex_rays=rays, seed=0 if seed is None else seed,
                                   albedo_fallback=albedo_fallback, ignore_normals_mode=mode, bounces=bounces)
    if texel is not None:
        settings = dataclasses.replace(settings, ignore_normals_mode='enhanced', samples=9 * rays)
    if texel_samples is not None:
        settings = dataclasses.replace(settings, texel_samples=texel_samples)
    write_job(tmp_path, scene, [(p.shape[1], p.shape[0]) for p in pages], list(suns), ambient or {}, settings,
              'all', textures=textures)
    if seed is None:
        job_path = tmp_path / 'job.json'
        job = json.loads(job_path.read_text())
        del job['seed']
        job_path.write_text(json.dumps(job, indent=2))
    np.savez(tmp_path / 'pages.npz', **{f'page_{i}': p for i, p in enumerate(pages)})
    if sun_facing:
        (tmp_path / 'sun_facing').write_text('')
    if probes is not None:
        (tmp_path / 'probe.json').write_text(json.dumps(probes))
    if fast_path:
        (tmp_path / 'fast_path').write_text('')
    env = dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent))
    done = subprocess.run([str(LIGHTMAPPER_BLENDER), '-b', '--factory-startup', '--python-exit-code', '1',
                           '--python', DRIVER, '--', str(tmp_path)], capture_output=True, text=True, env=env,
                          stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    return np.load(tmp_path / ('sun_facing.npz' if sun_facing else 'probe.npz' if probes is not None else 'vp.npz'))


# one texel shared by a flagged sheet (its two left sub-texel columns) and an ordinary one (the right column)
FLAGGED_THIRDS = [(0, 0), (2 / 3, 0), (2 / 3, 1), (0, 1)]
ORDINARY_THIRD = [(2 / 3, 0), (1, 0), (1, 1), (2 / 3, 1)]
FLAGGED_XY = [(x, y) for y in range(3) for x in range(2)]
WHOLE_XY = [(x, y) for y in range(3) for x in range(3)]


def mixed_scene(kind):
    """A third flagged receiver (page 1, nine samples) above sheets that share page 0's one texel six to three."""
    if kind in ('mixed_dark_flagged', 'mixed_bright_flagged'):
        bright = kind == 'mixed_bright_flagged'
        receivers = [dict(quad=floor(-1000, 1000, -1000, 1000, lm=FLAGGED_THIRDS), ignore_normals=True),
                     dict(quad=floor(990, 991, 990, 991, z=-5, up=False, lm=ORDINARY_THIRD)),
                     dict(quad=floor(-.1, .1, -.1, .1, z=1), ignore_normals=True, page=1)]
        return dict(receivers=receivers, occluders=[dict(quad=floor(-1, 1, -1, 1, z=2))],
                    suns=[sun()] if bright else [], query=((0, 0, 0),), normal=(0, 0, 0), rays=1024,
                    texel=[(5, 0, 0)] * 6 + [(0, 0, 1)] * 9, texel_pages=[0] * 6 + [1] * 9,
                    texel_xy=FLAGGED_XY + WHOLE_XY,
                    pages=[np.full((1, 1, 3), 0. if bright else 1 / 3, np.float32), np.zeros((1, 1, 3), np.float32)])
    if kind in ('mixed_delta', 'mixed_ordinary'):
        # the flagged sheet takes a stage-zero delta from the ordinary floor below; the receiver sees only the
        # ordinary sheet's top
        receivers = [dict(quad=floor(-1000, 1000, -1000, 1000, z=-1), page=2),
                     dict(quad=floor(-1000, 1000, -1000, 1000, lm=FLAGGED_THIRDS), ignore_normals=True),
                     dict(quad=floor(-1000, 1000, -1000, 1000, z=2, lm=ORDINARY_THIRD)),
                     dict(quad=floor(-.1, .1, -.1, .1, z=3), ignore_normals=True, page=1)]
        return dict(receivers=receivers, query=((0, 0, 0),), normal=(0, 0, 0), rays=1024,
                    bounces=2 if kind == 'mixed_delta' else 1,
                    texel=[(0, 0, 0)] * 6 + [(0, 0, 3)] * 9, texel_pages=[0] * 6 + [1] * 9,
                    texel_xy=FLAGGED_XY + WHOLE_XY,
                    pages=[np.full((1, 1, 3), 0. if kind == 'mixed_delta' else 1 / 3, np.float32),
                           np.zeros((1, 1, 3), np.float32), np.ones((1, 1, 3), np.float32)])
    raise ValueError(kind)


def assert_mixed(kind, rgb):
    receiver = rgb[6:]
    if kind == 'mixed_dark_flagged':
        # the ordinary third's light is not the dark flagged sheet's
        np.testing.assert_array_equal(receiver, 0)
    elif kind == 'mixed_bright_flagged':
        # half the sphere sees the flagged sheet's own 3 / pi at albedo .5, not its 6 / 9 share of the texel
        np.testing.assert_allclose(receiver, 3 / np.pi, rtol=.05)
    elif kind == 'mixed_delta':
        assert rgb[:6].min() > .5
        np.testing.assert_array_equal(receiver, 0)
    else:
        # the ordinary third's own light (1), not a third of it
        np.testing.assert_allclose(receiver, 1, rtol=.05)


@pytest.mark.blender
@pytest.mark.parametrize('kind', ['mixed_dark_flagged', 'mixed_bright_flagged', 'mixed_delta', 'mixed_ordinary'])
def test_reflected_stages_read_each_surface_its_own_light_in_a_mixed_texel(tmp_path, kind):
    out = run_driver(tmp_path, **mixed_scene(kind))
    assert_mixed(kind, out['texel_rgb'])


def texel_scene(kind):
    if kind in ('emission_budget', 'ordinary_budget'):
        source = dict(quad=floor(-1000, 1000, -1000, 1000, z=-1), page=1,
                      emission=(1, 1, 1) if kind == 'emission_budget' else (0, 0, 0))
        return dict(receivers=[dict(quad=floor(-1000, 1000, -1000, 1000), ignore_normals=True), source],
                    query=((0, 0, 0),), normal=(0, 0, 0), rays=1024, bounces=3,
                    texel=((0, 0, 0),) * 9,
                    pages=[np.zeros((1, 1, 3), np.float32),
                           np.full((1, 1, 3), 1. if kind == 'ordinary_budget' else 0., np.float32)])
    if kind == 'flagged_bounce':
        panel = quad([(2, -1, .1), (2, -1, 1.5), (2, 1, 1.5), (2, 1, .1)], (-1, 0, 0))
        return dict(receivers=[dict(quad=floor(-1, 1, -1, 1), ignore_normals=True),
                               dict(quad=panel, ignore_normals=True, page=1)],
                    occluders=[dict(quad=floor(1.5, 2.5, -1, 1, z=2))], suns=[sun()],
                    query=((0, 0, 0),), normal=(0, 0, 0), rays=2048, albedo_fallback=.5,
                    texel=((0, 0, 0),) * 9 + ((2, 0, .5),) * 9, texel_pages=[0] * 9 + [1] * 9,
                    texel_normals=[(0, 0, 1)] * 9 + [(-1, 0, 0)] * 9,
                    pages=[np.zeros((1, 1, 3), np.float32)] * 2)
    if kind == 'edge':
        uv = [(0, 0), (1/3, 0), (1/3, 1/3), (0, 1/3)]
        return dict(receivers=[dict(quad=floor(-1, 1, -1, 1, lm=uv), ignore_normals=True),
                               dict(quad=floor(-.3, .3, -.3, .3, z=1, up=False), page=1, emission=(1, 1, 1))],
                    query=((0, 0, 0),), normal=(0, 0, 0), texel=((0, 0, 0),), rays=1, texel_samples=512,
                    pages=[np.zeros((1, 1, 3), np.float32), np.zeros((1, 1, 3), np.float32)])
    if kind == 'grazing':
        # a thin band just above the horizon: a symmetric wall would give the cosine hemisphere the same answer by luck
        panel = dict(quad=quad([(1, -1000, 0), (1, -1000, .2),
                               (1, 1000, .2), (1, 1000, 0)], (-1, 0, 0)),
                     emission=(1, 2, 3))
        return dict(receivers=[panel], query=((0, 0, 0),), texel=((0, 0, 0),), rays=4096)
    if kind == 'behind':
        return dict(receivers=[dict(quad=floor(-1000, 1000, -1000, 1000, z=-1), emission=(1, 2, 3))],
                    query=((0, 0, 0),), texel=((0, 0, 0),), rays=1024)
    if kind == 'pane':
        return dict(receivers=[full_quad()],
                    occluders=[dict(quad=floor(-100, 100, -100, 100, z=-1),
                                    casts=False, tint=(.5, .25, 1))],
                    suns=[sun((0, 0, -1))], query=((2, 2, 0),), texel=((2, 2, 0),))
    if kind in ('vertex', 'vertex_back'):
        return dict(receivers=[dict(quad=floor(-1000, 1000, -1000, 1000, z=1, up=kind == 'vertex_back'),
                                     ignore_normals=True, emission=(7, 7, 7) if kind == 'vertex_back' else (0, 0, 0))],
                    suns=[sun((0, 0, 1))], query=((0, 0, 0),), texel=((0, 0, 1),) * 9,
                    pages=[np.zeros((1, 1, 3), np.float32)], rays=1024, albedo_fallback=1)
    raise ValueError(kind)


@pytest.mark.blender
@pytest.mark.parametrize('kind,want', [('emission_budget', 2.), ('ordinary_budget', 1.)])
def test_flagged_stages_consume_original_sources_only_once(tmp_path, kind, want):
    out = run_driver(tmp_path, **texel_scene(kind))
    np.testing.assert_allclose(out['texel_rgb'], want, rtol=.05)


@pytest.mark.blender
@pytest.mark.parametrize('bounces', [0, 1, 2])
def test_shadowed_flagged_panel_receives_the_sunlit_flagged_panels_bounce(tmp_path, bounces):
    out = run_driver(tmp_path, **texel_scene('flagged_bounce'), bounces=bounces)
    assert_flagged_bounce(out['texel_rgb'], bounces)


def assert_flagged_bounce(rgb, bounces):
    if bounces == 0:
        np.testing.assert_allclose(rgb[:9], 3 / np.pi, rtol=1e-6)
        np.testing.assert_array_equal(rgb[9:], 0)
    elif bounces == 1:
        np.testing.assert_allclose(rgb[:9], 3 / np.pi, rtol=1e-6)
        assert rgb[9:].min() > .01
    else:
        assert rgb[:9].min() > 3 / np.pi + .001
        assert .01 < rgb[9:].mean() < .2


@pytest.mark.blender
@pytest.mark.parametrize('mode', ['tool', 'enhanced'])
def test_ordinary_cpu_queries_skip_sphere_rgb_and_throughput(tmp_path, mode):
    out = run_driver(tmp_path, [full_quad(emission=(1, 2, 3))], fast_path=True,
                     query=((2, 2, 1),), normal=(0, 0, -1), pages=[np.ones((16, 16, 3), np.float32)], mode=mode)
    assert out['irradiance'][0] > 0
    assert 'vertex_sun_throughput' not in out.files


@pytest.mark.blender
def test_enhanced_edge_gather_uses_the_texel_budget_under_a_small_emitter(tmp_path):
    out = run_driver(tmp_path, **texel_scene('edge'))
    assert int(out['rays']) == 512
    np.testing.assert_allclose(out['texel_rgb'], .105, rtol=.3)


@pytest.mark.blender
def test_texel_gather_counts_a_grazing_emitter_without_cosine(tmp_path):
    out = run_driver(tmp_path, **texel_scene('grazing'))
    # the band covers atan(.2) / (2 pi) of the sphere; the cosine-weighted hemisphere would count under a third of that
    np.testing.assert_allclose(out['texel_rgb'], 4 * np.arctan(.2) / (2 * np.pi) * np.float32([[1, 2, 3]]), rtol=.05)


@pytest.mark.blender
def test_texel_gather_takes_light_from_behind(tmp_path):
    out = run_driver(tmp_path, **texel_scene('behind'))
    np.testing.assert_allclose(out['texel_rgb'], [[2, 4, 6]], rtol=.05)


@pytest.mark.blender
def test_texel_gather_takes_the_sun_from_behind_through_a_pane(tmp_path):
    out = run_driver(tmp_path, **texel_scene('pane'))
    np.testing.assert_allclose(out['texel_rgb'], np.float32([[.5, .25, 1]]) * 3 / np.pi, rtol=1e-6)


@pytest.mark.blender
@pytest.mark.parametrize('kind', ['vertex', 'vertex_back'])
def test_vertex_queries_see_the_gathered_flagged_texels(tmp_path, kind):
    out = run_driver(tmp_path, **texel_scene(kind))
    np.testing.assert_allclose(out['texel_rgb'], [[3 / np.pi] * 3] * 9, rtol=1e-6)
    assert out['irradiance'][0] == pytest.approx(3, rel=.05)


@pytest.mark.blender
def test_shade_rgb_luminance_is_todays_shade(tmp_path):
    out = run_driver(tmp_path, [full_quad(emission=(5, 6, 7))],
                     pages=[np.tile(np.float32([2, 3, 4]), (16, 16, 1))],
                     ambient={0: (.1, .2, .3)}, probes=[((2, 2, 1), (0, 0, -1), 0)])
    np.testing.assert_allclose(out['radiance_rgb'], [[6.05, 7.6, 9.15]], rtol=1e-6)
    np.testing.assert_allclose(out['radiance'], [7.38238], rtol=1e-6)


def twin_scene(reverse=False, cutout=False):
    corners = np.float32([[-100, -100, 1], [100, -100, 1], [0, 100, 1]])
    receivers = [dict(quad=(corners, np.float32([[0.25, 0.5]] * 3), np.zeros((3, 2), np.float32),
                                  np.float32([[0, 0, 1]] * 3))),
                 dict(quad=(corners[[1, 0, 2]], np.float32([[0.75, 0.5]] * 3), np.zeros((3, 2), np.float32),
                                  np.float32([[0, 0, -1]] * 3)))]
    if cutout:
        receivers = [dict(r, texture=0, cutout=True) for r in receivers]
    return dict(receivers=receivers[::-1] if reverse else receivers,
                pages=[np.float32([[[2, 2, 2], [7, 7, 7]]])],
                textures=[np.full((1, 1, 4), 255, np.uint8)] if cutout else [],
                query=((0, 0, 0), (0, 0, 2)), normals=((0, 0, 1), (0, 0, -1)),
                rays=1, seed=0, albedo_fallback=1.0)


def assert_twin_light(irradiance, vector, rays):
    expected = np.pi * np.array([7, 2])
    angle = 2 * np.pi * 0.618034
    directions = np.sqrt(0.5) * np.array([[0, 1, 1], [-np.sin(angle), -np.cos(angle), -1]])
    np.testing.assert_allclose(irradiance, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(vector, expected[:, None] * directions, rtol=1e-6, atol=1e-6)
    assert int(rays) == 2


@pytest.mark.blender
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('cutout', [False, True])
def test_twin_faces_light_both_sides_on_cpu(tmp_path, reverse, cutout):
    out = run_driver(tmp_path, **twin_scene(reverse, cutout))
    assert_twin_light(out['irradiance'], out['vector'], out['rays'])


def unpaired_back_face_scene():
    scene = twin_scene()
    front, back = scene['receivers']
    p, lm, tex, n = back['quad']
    behind = dict(quad=(p + np.float32([0, 0, 1]), lm, tex, n))
    return dict(scene, receivers=[front, behind], query=((0, 0, 0),), normals=((0, 0, 1),))


@pytest.mark.blender
def test_unpaired_back_face_still_blocks(tmp_path):
    out = run_driver(tmp_path, **unpaired_back_face_scene())
    np.testing.assert_array_equal(out['irradiance'], [0])
    np.testing.assert_array_equal(out['vector'], [[0, 0, 0]])


def asymmetric_twin_scene(reverse=False):
    scene = twin_scene(cutout=True)
    upper, lower = scene['receivers']
    p, lm, _, n = upper['quad']
    upper = dict(upper, quad=(p, lm, np.float32([[0.75, 0.5]] * 3), n))
    p, _, _, n = lower['quad']
    uv = np.float32([[1, 0.5], [0, 0.5], [0.5, 0.5]])
    lower = dict(lower, quad=(p, uv, uv, n))
    texture = np.uint8([[[255, 255, 255, 255], [255, 255, 255, 0]]])
    return dict(scene, receivers=[lower, upper] if reverse else [upper, lower], textures=[texture],
                query=((-50, -50, 0), (-50, -50, 2)), suns=[sun(UP), sun(-UP)])


def assert_asymmetric_twin(irradiance, sun_lit):
    # at x=-50 the lower twin's UV is .25 (opaque, page value 2); the upper twin is a hole
    np.testing.assert_allclose(irradiance, [2 * np.pi, 3], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(sun_lit, [[False, False], [False, True]])


@pytest.mark.blender
@pytest.mark.parametrize('reverse', [False, True])
def test_twin_uses_facing_alpha_and_corner_weights_before_stopping(tmp_path, reverse):
    out = run_driver(tmp_path, **asymmetric_twin_scene(reverse))
    assert_asymmetric_twin(out['irradiance'], out['sun_lit'])


@pytest.mark.blender
def test_seed_zero_is_todays_pass(tmp_path):
    scene = [full_quad(), wall_facing_minus_x(emission=(4, 4, 4))]
    a = run_driver(tmp_path / 'a', scene, query=((1, 2, 0),))
    b = run_driver(tmp_path / 'b', scene, query=((1, 2, 0),), seed=None)
    assert np.array_equal(a['vector'], b['vector']) and np.array_equal(a['irradiance'], b['irradiance'])


@pytest.mark.blender
def test_another_seed_turns_the_rays_but_keeps_the_estimate(tmp_path):
    scene = [full_quad(), wall_facing_minus_x(emission=(4, 4, 4))]
    s0 = run_driver(tmp_path / 's0', scene, query=((1, 2, 0),), rays=1024, seed=0)
    s1 = run_driver(tmp_path / 's1', scene, query=((1, 2, 0),), rays=1024, seed=1)
    assert not np.array_equal(s0['vector'], s1['vector'])
    assert s1['irradiance'][0] == pytest.approx(s0['irradiance'][0], rel=0.05)


@pytest.mark.blender
def test_sun_only_vector_points_at_the_sun(tmp_path):
    out = run_driver(tmp_path, [full_quad()], [sun(UP, strength=3.0)])
    v, e = out['vector'][0], out['irradiance'][0]
    assert e == pytest.approx(3.0, rel=1e-3)
    assert np.linalg.norm(v / e) > 0.98
    np.testing.assert_allclose(v / np.linalg.norm(v), [0, 0, 1], atol=1e-6)


def ignore_normals_sun_scene(behind=False):
    receivers = [full_quad(casts=False), dict(quad=floor(10, 14, 0, 4), casts=False)]
    scene = build(receivers)
    direction = (0, np.sqrt(3) / 2, -0.5 if behind else 0.5)
    return dict(receivers=receivers, suns=[sun(direction, strength=3.0)], query=scene.positions,
                normals=scene.normals, inverse=np.arange(8), ignore_normals=[True] * 4 + [False] * 4)


def sun_throughput_scene():
    return dict(receivers=[full_quad(casts=False)], suns=[sun(-UP)],
                occluders=[dict(quad=floor(0, 4, 0, 4, z=-1), casts=False, tint=(0.5, 0.25, 1)),
                           dict(quad=floor(10, 14, 0, 4, z=-1), casts=False, tint=(0, 0, 0)),
                           dict(quad=floor(20, 24, 0, 4, z=-1))],
                query=((2, 2, 0), (12, 2, 0), (22, 2, 0), (32, 2, 0)),
                ignore_normals=[True] * 4)


@pytest.mark.blender
def test_sun_throughput_carries_the_panes_colour(tmp_path):
    out = run_driver(tmp_path, **sun_throughput_scene())
    np.testing.assert_array_equal(out['sun_lit'], [[True], [True], [False], [True]])
    np.testing.assert_allclose(out['vertex_sun_throughput'][:, 0],
                               [[0.5, 0.25, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]], rtol=1e-6)


@pytest.mark.blender
def test_ignore_normals_query_takes_a_sun_from_behind_at_full_strength(tmp_path):
    case = ignore_normals_sun_scene(behind=True)
    out = run_driver(tmp_path, **case)
    np.testing.assert_allclose(out['irradiance'][:4], 3.0, rtol=1e-6)
    np.testing.assert_allclose(out['vector'][0], 3.0 * case['suns'][0].direction, rtol=1e-6)
    np.testing.assert_array_equal(out['irradiance'][4:], 0)
    assert out['sun_lit'][:4].all() and not out['sun_lit'][4:].any()


@pytest.mark.blender
def test_ignore_normals_query_counts_a_sun_at_60_degrees_fully(tmp_path):
    case = ignore_normals_sun_scene()
    out = run_driver(tmp_path, **case)
    np.testing.assert_allclose(out['irradiance'][:4], 3.0, rtol=1e-6)
    np.testing.assert_allclose(out['vector'][0], 3.0 * case['suns'][0].direction, rtol=1e-6)
    np.testing.assert_allclose(out['irradiance'][4:], 1.5, rtol=1e-6)


def ignore_normals_emitter_scene():
    return dict(receivers=[dict(quad=floor(-200, 200, -200, 200), emission=(4, 4, 4))],
                query=((0, 0, 2), (0, 0, 2)), ignore_normals=[True, False], rays=1024)


@pytest.mark.blender
def test_ignore_normals_gather_sees_an_emitter_behind_the_query(tmp_path):
    out = run_driver(tmp_path, **ignore_normals_emitter_scene())
    assert out['irradiance'][0] == pytest.approx(2 * np.pi * 4, rel=0.03)
    assert out['vector'][0, 2] == pytest.approx(-np.pi * 4, rel=0.03)
    assert out['irradiance'][1] == 0


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
