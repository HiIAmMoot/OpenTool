"""The vertex pass's numpy side against bake.py's CPU pass: formulas, copied constants, and the ray lists in real
Blender (vertex_driver.py)."""

import ast
import dataclasses
import json
import sys

import numpy as np
import pytest

from lightmapper import frozen, gpu, vertex_gpu
from lightmapper.blender_job import write_job
from lightmapper.sun_facing import sun_facing_queries
from lightmapper.vertex_gpu import (GOLDEN_FRACTION, GROUP_SIZE, SEED_SPIN, GpuPassError, edge_rays, gather_sums,
                                    hemisphere, load_and_run, query_frames, run_vertex_pass, sun_rays)
from conftest import LIGHTMAPPER_BLENDER
from test_blender_bake import CORNERS_UV, UP, build, floor, full_quad, quad, sun
from test_vertex_pass import (AMBIENT, FULL_UV, S, assert_asymmetric_twin, assert_twin_light, asymmetric_twin_scene,
                              roof, run_driver, twin_scene, ignore_normals_sun_scene, ignore_normals_emitter_scene,
                              unpaired_back_face_scene, wall_facing_minus_x, sun_throughput_scene, texel_scene,
                              assert_flagged_bounce, mixed_scene, assert_mixed, assert_emission_relay)


def test_hemisphere_and_frames_are_the_cpu_pass_formulas():
    local, phi, r = hemisphere(4)
    assert phi.tolist() == pytest.approx([0, np.pi, np.pi / 2, 3 * np.pi / 2])
    assert local[:, 2] == pytest.approx(np.sqrt(1 - (np.arange(4) + 0.5) / 4))
    usable, n, t, b, spin = query_frames(np.float32([[0, 0, 2], [0, 0, 0], [np.nan, 0, 1]]), seed=3)
    assert usable.tolist() == [True, False, False]
    assert spin[0] == pytest.approx(2 * np.pi * ((3 * SEED_SPIN) % 1))
    assert spin[1] == pytest.approx(2 * np.pi * ((GOLDEN_FRACTION + 3 * SEED_SPIN) % 1))

    count = 8
    local, phi, r = hemisphere(count)
    assert r == pytest.approx(np.sqrt((np.arange(count) + 0.5) / count))
    np.testing.assert_allclose(local[:, :2], np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1), atol=1e-15)
    # bake.py picks x as the helper axis unless the normal lies along it, then y; the frame is cross(n, axis) and
    # cross(n, tangent), so the two cases are written out by hand
    _, n, t, b, _ = query_frames(np.float32([[0, 0, 3], [2, 0, 0]]), seed=0)
    np.testing.assert_allclose(n, [[0, 0, 1], [1, 0, 0]], atol=1e-15)
    np.testing.assert_allclose(t, [[0, 1, 0], [0, 0, 1]], atol=1e-15)
    np.testing.assert_allclose(b, [[-1, 0, 0], [0, -1, 0]], atol=1e-15)


def test_gather_sums_equal_the_per_ray_sum():
    rng = np.random.default_rng(7)
    rays = 64
    lum = rng.random((5, rays))
    normals = rng.normal(size=(5, 3))
    _, n, t, b, spin = query_frames(normals, seed=2)
    local, phi, r = hemisphere(rays)
    angle = phi[None, :] + spin[:, None]
    dirs = (r[None, :, None] * np.cos(angle)[:, :, None] * t[:, None, :]
            + r[None, :, None] * np.sin(angle)[:, :, None] * b[:, None, :]
            + local[None, :, 2, None] * n[:, None, :])
    vector, irradiance = gather_sums(lum, phi, r, local[:, 2], n, t, b, spin)
    np.testing.assert_allclose(vector, np.pi / rays * (lum[:, :, None] * dirs).sum(1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(irradiance, np.pi / rays * lum.sum(1), rtol=0, atol=1e-12)


def test_flagged_subtexels_reach_one_texel_and_match_the_cpu_copy():
    from lightmapper import raster
    source = vertex_gpu.SHADERS.parent / 'blender' / 'bake.py'
    tree = ast.parse(source.read_text())
    wanted = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name == 'flagged_subtexels')
              or (isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'FLAGGED_REACH')]
    namespace = {'np': np}
    exec(compile(ast.Module(wanted, []), str(source), 'exec'), namespace)
    assert namespace['FLAGGED_REACH'] == raster.FLAGGED_REACH == 3
    rng = np.random.default_rng(3)
    cells = rng.choice(30 * 24, 12, replace=False)
    xy = np.stack([cells % 24, cells // 24], axis=1)
    rgb = rng.random((12, 3)).astype(np.float32)
    np.testing.assert_array_equal(raster.flagged_subtexels((30, 24), xy, rgb),
                                  namespace['flagged_subtexels']((30, 24), xy, rgb))
    one = raster.flagged_subtexels((1, 9), np.int32([[0, 0]]), np.float32([[1, 2, 3]]))
    np.testing.assert_array_equal(one[0, :, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0])


def test_sphere_is_the_cpu_pass_formula():
    count = 1024
    local, phi, r = vertex_gpu.sphere(count)
    np.testing.assert_allclose(local[:, 2], 1 - 2 * (np.arange(count) + 0.5) / count, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(local, axis=1), 1, atol=1e-15)
    assert abs(local[:, 2].mean()) <= 1 / count
    source = vertex_gpu.SHADERS.parent / 'blender' / 'bake.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VertexPass')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'sphere')
    namespace = {'np': np, 'radical_inverse': vertex_gpu.radical_inverse}
    exec(compile(ast.Module([method], []), str(source), 'exec'), namespace)
    theirs = namespace['sphere'](None, count)
    for ours, cpu in zip((local, phi, r), theirs):
        np.testing.assert_array_equal(ours, cpu)
    _, n, t, b, spin = query_frames([[0, 0, 1]], seed=0)
    vector, irradiance = gather_sums(np.ones((1, count)), phi, r, local[:, 2], n, t, b, spin, 4 * np.pi)
    np.testing.assert_allclose(irradiance, [4 * np.pi], rtol=1e-12)
    assert abs(vector[0, 2]) < 1e-12


def test_ambient_table_is_indexed_by_class_plus_one():
    job = {'ambient': {'-1': [1, 2, 3], '0': [4, 5, 6]}}
    table = vertex_gpu.ambient_table(job, np.int16([0, 2]), np.int16([1, -1]))
    assert table.shape == (4, 3)
    assert table[0].tolist() == [1, 2, 3] and table[1].tolist() == [4, 5, 6]
    assert table[2].tolist() == [0, 0, 0] and table[3].tolist() == [0, 0, 0]


def test_copied_constants_equal_bake_py():
    source = (vertex_gpu.__file__.rsplit('vertex_gpu.py', 1)[0] + 'blender/bake.py')
    tree = ast.parse(open(source, encoding='utf-8').read())
    names = ('RAY_LENGTH', 'RAY_EPSILON', 'PASS_LIMIT', 'LUMA', 'GOLDEN_FRACTION', 'SEED_SPIN')
    namespace = {'np': np}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in names:
            exec(compile(ast.Module([node], []), source, 'exec'), namespace)
    from_bake = {name: np.asarray(namespace[name]).tolist() for name in names}
    copied = {name: np.asarray(getattr(vertex_gpu, name)).tolist() for name in names}
    assert set(from_bake) == set(names)
    assert copied == from_bake


@pytest.mark.blender
def test_sun_facing_copy_matches_bake(tmp_path):
    receivers = [dict(quad=floor(0, 4, 0, 4)), dict(quad=floor(0, 4, 5, 9, up=False)),
                 dict(quad=quad([(10, 0, 0), (10, 4, 0), (10, 4, 4), (10, 0, 4)], (1, 0, 0))),
                 dict(quad=quad([(12, 0, 0), (12, 0, 4), (12, 4, 4), (12, 4, 0)], (-1, 0, 0)))]
    suns = [sun(UP), sun((0, 0, -1)), sun((1, 0.2, 0.1)),
            dataclasses.replace(sun((0.3, 0.2, 1)), centre=np.float64([0.3, 0.2, 1]) / np.linalg.norm([0.3, 0.2, 1]))]
    scene = build(receivers)
    out = run_driver(tmp_path, receivers, suns, query=scene.positions, normals=scene.normals,
                     inverse=np.arange(len(scene.positions)), sun_facing=True)
    theirs = out['facing']
    with np.load(tmp_path / 'scene.npz') as data:
        arrays = {key: data[key] for key in data.files}
    ours = sun_facing_queries(arrays, json.loads((tmp_path / 'job.json').read_text())['suns'])
    assert theirs.any() and not theirs.all()
    assert np.array_equal(ours, theirs)


@pytest.mark.blender
def test_sun_and_edge_ray_lists_are_the_rays_the_cpu_pass_casts(tmp_path):
    high = list(floor(0, 4, 0, 4))
    high[3] = np.tile(np.float32(UP), (4, 1))
    wall = quad([(0, 0, 0), (0, 4, 0), (0, 4, -4), (0, 0, -4)], UP)
    suns = [sun((0.8, 0, 0.6)), sun(UP, classes=(1,)), sun((0, 0, -1))]
    cpu = run_driver(tmp_path, [dict(quad=high, casts=False), dict(quad=wall, casts=False)], suns, normal=UP)
    with np.load(tmp_path / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    job = json.loads((tmp_path / 'job.json').read_text())
    usable, n, *_ = query_frames(scene['query_normal'], job['seed'])
    query, sun_index, origin, direction, cosine = sun_rays(scene, job, usable, n)
    edge, edge_sun, edge_origin, edge_direction = edge_rays(scene, job)
    assert len(edge) > 0 and query.tolist() == [0] and sun_index.tolist() == [0]
    assert cosine[0] == pytest.approx(0.6)
    count = len(query) + len(edge) + int(usable.sum()) * job['vertex_rays']
    assert count == int(cpu['rays'])
    assert cpu['sun_lit'][query, sun_index].tolist() == [True]
    assert cpu['edge_sun_lit'].shape == (len(scene['edge_pos']), len(suns))
    assert not cpu['edge_sun_lit'][:, 1].any()


# ---------- the GPU pass against the CPU pass, one scene at a time

VENDORS = {'nvidia': 0x10DE, 'amd': 0x1002}
RAMP = (np.arange(16)[None, :, None] * 0.25 + np.arange(16)[:, None, None] * 0.1
        + np.arange(3) * 0.05).astype(np.float32)


@pytest.fixture(params=[pytest.param('nvidia', marks=pytest.mark.gpu_nvidia),
                        pytest.param('amd', marks=pytest.mark.gpu_amd)])
def vendor(request):
    return request.param


def adapter_for(vendor):
    adapters, _ = gpu.list_adapters()
    return next(a for a in adapters if a.vendor_id == VENDORS[vendor] and a.ray_query)


def cpu_and_gpu(tmp_path, vendor, **kwargs):
    cpu = run_driver(tmp_path, **kwargs)
    with np.load(tmp_path / 'pages.npz') as data:
        pages = [data[f'page_{i}'] for i in range(len(data.files))]
    return cpu, load_and_run(tmp_path, pages, adapter_for(vendor))


@pytest.mark.blender
@pytest.mark.parametrize('mode', ['tool', 'enhanced'])
def test_ordinary_gpu_queries_omit_unused_throughput(tmp_path, vendor, mode):
    cpu, result = cpu_and_gpu(tmp_path, vendor, receivers=[full_quad()], suns=[sun()], mode=mode)
    assert result.irradiance[0] > 0
    assert result.sun_throughput is None
    assert 'vertex_sun_throughput' not in cpu.files


@pytest.mark.blender
@pytest.mark.parametrize('bounces', [0, 1, 2])
def test_flagged_stage_budget_matches_cpu(tmp_path, vendor, bounces):
    cpu, result = cpu_and_gpu(tmp_path, vendor, **texel_scene('flagged_bounce'), bounces=bounces)
    assert_flagged_bounce(result.texel_rgb, bounces)
    np.testing.assert_allclose(result.texel_rgb, cpu['texel_rgb'], rtol=1e-5, atol=1e-6)


@pytest.mark.blender
@pytest.mark.parametrize('bounces', [0, 1])
def test_direct_emission_takes_no_reflected_stage_on_gpu(tmp_path, vendor, bounces):
    cpu, result = cpu_and_gpu(tmp_path, vendor, **texel_scene('emission_relay'), bounces=bounces)
    assert_emission_relay(result.texel_rgb, bounces)
    np.testing.assert_allclose(result.texel_rgb, cpu['texel_rgb'], rtol=1e-5, atol=1e-6)


@pytest.mark.blender
@pytest.mark.parametrize('kind', ['mixed_dark_flagged', 'mixed_bright_flagged', 'mixed_delta', 'mixed_ordinary'])
def test_mixed_texel_sources_match_cpu(tmp_path, vendor, kind):
    cpu, result = cpu_and_gpu(tmp_path, vendor, **mixed_scene(kind))
    assert_mixed(kind, result.texel_rgb)
    np.testing.assert_allclose(result.texel_rgb, cpu['texel_rgb'], rtol=1e-5, atol=1e-6)


@pytest.mark.blender
@pytest.mark.parametrize('kind', ['grazing', 'behind', 'pane', 'vertex', 'vertex_back', 'edge',
                                'emission_budget', 'ordinary_budget'])
def test_gpu_matches_cpu_on_texel_scenes(tmp_path, vendor, kind):
    cpu, result = cpu_and_gpu(tmp_path, vendor, **texel_scene(kind))
    np.testing.assert_allclose(result.texel_rgb, cpu['texel_rgb'], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.irradiance, cpu['irradiance'], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.vector, cpu['vector'], rtol=1e-5, atol=1e-6)
    if kind == 'vertex_back':
        assert result.irradiance[0] == pytest.approx(3, rel=.05)
    if kind == 'edge':
        assert result.rays == 2 * 512
        np.testing.assert_allclose(result.texel_rgb, .105, rtol=.3)
    if kind in ('emission_budget', 'ordinary_budget'):
        np.testing.assert_allclose(result.texel_rgb, 2. if kind == 'emission_budget' else 1., rtol=.05)


@pytest.mark.blender
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('cutout', [False, True])
def test_twin_faces_light_both_sides_on_cpu_and_gpu(tmp_path, vendor, reverse, cutout):
    cpu, result = cpu_and_gpu(tmp_path, vendor, **twin_scene(reverse, cutout))
    assert_twin_light(cpu['irradiance'], cpu['vector'], cpu['rays'])
    assert_twin_light(result.irradiance, result.vector, result.rays)


@pytest.mark.blender
def test_unpaired_back_face_still_blocks(tmp_path, vendor):
    cpu, result = cpu_and_gpu(tmp_path, vendor, **unpaired_back_face_scene())
    np.testing.assert_array_equal(cpu['irradiance'], [0])
    np.testing.assert_array_equal(result.irradiance, [0])
    np.testing.assert_array_equal(result.vector, [[0, 0, 0]])


@pytest.mark.blender
@pytest.mark.parametrize('reverse', [False, True])
def test_twin_uses_facing_alpha_and_corner_weights_before_stopping(tmp_path, vendor, reverse):
    cpu, result = cpu_and_gpu(tmp_path, vendor, **asymmetric_twin_scene(reverse))
    assert_asymmetric_twin(cpu['irradiance'], cpu['sun_lit'])
    assert_asymmetric_twin(result.irradiance, result.sun_lit)


def alpha_texture(alpha, holes=False):
    texels = np.full((4, 4, 4), 255, np.uint8)
    texels[:, :, 3] = alpha
    if holes:
        texels[1:3, 1:3, 3] = 0
    return texels


def negative_uv_wall():
    corners = [(3, 0, 0), (3, 4, 0), (3, 4, 4), (3, 0, 4)]
    tex = [(-1.3, -1.3), (-0.2, -1.3), (-0.2, -0.2), (-1.3, -0.2)]
    return dict(quad=quad(corners[::-1], (-1, 0, 0), FULL_UV[::-1], tex[::-1]), texture=0)


def overhanging_slab():
    m = np.float64([1, 0, -0.054])
    m /= np.linalg.norm(m)
    t2 = np.float64([0.054, 0, 1])
    t2 /= np.linalg.norm(t2)
    origin = np.float64([2, 2, 0])
    corners = [origin + a * np.float64([0, 1, 0]) + b * t2 for a, b in ((-2, 0), (2, 0), (2, 2), (-2, 2))]
    return dict(quad=quad(corners, tuple(m)))


def edge_receivers():
    high = list(floor(0, 4, 0, 4))
    high[3] = np.tile(np.float32(UP), (4, 1))
    wall = quad([(0, 0, 0), (0, 4, 0), (0, 4, -4), (0, 0, -4)], UP)
    return [dict(quad=high, casts=False), dict(quad=wall, casts=False)]


COVER = dict(quad=floor(-50, 50, -50, 50, z=1.0))
GREY = np.uint8([[[128, 128, 128, 255]] * 2] * 2)
RANDOM_TEXELS = np.random.default_rng(3).integers(0, 256, (4, 4, 4)).astype(np.uint8)
CASES = {   # name -> run_driver kwargs, the scenes of test_vertex_pass.py
    'sun_throughput': sun_throughput_scene(),
    'ignore_normals_sun_behind': ignore_normals_sun_scene(True),
    'ignore_normals_sun_60': ignore_normals_sun_scene(),
    'ignore_normals_emitter_behind': ignore_normals_emitter_scene(),
    'sun_only': dict(receivers=[full_quad()], suns=[sun(UP, strength=3.0)]),
    'emissive_wall': dict(receivers=[full_quad(), wall_facing_minus_x(emission=(4, 4, 4))], query=((1, 2, 0),),
                          pages=[RAMP]),
    'emit_only_front': dict(receivers=[full_quad()], occluders=[
        dict(quad=floor(0, 4, 0, 4, z=1.0, up=False), casts=False, emission=(4, 4, 4))]),
    'emit_only_back': dict(receivers=[full_quad()], occluders=[
        dict(quad=floor(0, 4, 0, 4, z=1.0, up=True), casts=False, emission=(4, 4, 4))]),
    'cutout_hole': dict(receivers=[full_quad()], suns=[sun(UP)], textures=[alpha_texture(255, holes=True)],
                        occluders=[dict(quad=floor(0, 4, 0, 4, z=1.0), texture=0, cutout=True)]),
    'cutout_filled': dict(receivers=[full_quad()], suns=[sun(UP)], textures=[alpha_texture(255)],
                          occluders=[dict(quad=floor(0, 4, 0, 4, z=1.0), texture=0, cutout=True)]),
    'glass_tint_0': dict(receivers=[full_quad(), dict(quad=floor(-1, 5, -1, 5, z=1.0), casts=False, tint=(0, 0, 0))],
                         suns=[sun(UP, strength=3.0)]),
    'glass_tint_half': dict(receivers=[full_quad(), dict(quad=floor(-1, 5, -1, 5, z=1.0), casts=False,
                                                         tint=(0.5, 0.25, 0.75))], suns=[sun(UP, strength=3.0)]),
    'indoor_receiver': dict(receivers=[roof(-1)], ambient=AMBIENT, query=((0, 0, 0),), cls=0),
    'indoor_occluder': dict(receivers=[full_quad()], ambient=AMBIENT, occluders=[COVER], cls=-1),
    'zero_normal': dict(receivers=[full_quad()], suns=[sun(UP, strength=3.0)], ambient=AMBIENT,
                        query=((1, 1, 0), (2, 2, 0)), normals=[(0, 0, 0), (0, 0, 1)]),
    'overhanging_slab': dict(receivers=[full_quad()], suns=[sun((1, 0, 1), strength=3.0)],
                             occluders=[overhanging_slab()]),
    'grey_albedo': dict(receivers=[dict(wall_facing_minus_x(), texture=0)], query=((1, 2, 0),), textures=[GREY],
                        pages=[np.ones((16, 16, 3), np.float32)]),
    'edge_queries': dict(receivers=edge_receivers(), suns=[sun((0.8, 0, 0.6))], normal=(0, 0, 0)),
    'seed_1': dict(receivers=[full_quad(), wall_facing_minus_x(emission=(4, 4, 4))], query=((1, 2, 0), (2, 1, 0)),
                   seed=1, rays=256, pages=[RAMP], suns=[sun((1, 0.3, 2.6))]),
    'texture_negative_uv': dict(receivers=[full_quad(), negative_uv_wall()], query=((1, 2, 0),),
                                textures=[RANDOM_TEXELS], pages=[RAMP]),
}


@pytest.mark.blender
@pytest.mark.parametrize('case', CASES)
def test_gpu_matches_cpu_on_the_vertex_pass_scenes(tmp_path, vendor, case):
    cpu, result = cpu_and_gpu(tmp_path, vendor, **CASES[case])
    assert result.rays == int(cpu['rays'])
    np.testing.assert_allclose(result.irradiance, cpu['irradiance'], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(result.vector, cpu['vector'], rtol=1e-4, atol=1e-5)
    assert np.array_equal(result.sun_lit, cpu['sun_lit']) and np.array_equal(result.edge_sun_lit, cpu['edge_sun_lit'])
    if 'vertex_sun_throughput' in cpu.files:
        np.testing.assert_allclose(result.sun_throughput, cpu['vertex_sun_throughput'], rtol=1e-6, atol=1e-6)
    else:
        assert result.sun_throughput is None
    if case == 'sun_throughput':
        np.testing.assert_allclose(result.sun_throughput[:, 0],
                                   [[0.5, 0.25, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]], rtol=1e-6)


@pytest.mark.blender
def test_a_chunk_smaller_than_one_dispatch_gives_the_same_irradiance(tmp_path, vendor, monkeypatch):
    case = CASES['seed_1']
    cpu, whole = cpu_and_gpu(tmp_path, vendor, **case)
    assert len(whole.irradiance) >= 2
    with np.load(tmp_path / 'pages.npz') as data:
        pages = [data[f'page_{i}'] for i in range(len(data.files))]
    # one query's rays per chunk
    monkeypatch.setattr(vertex_gpu, 'MAX_LUMINANCES_PER_CHUNK', case['rays'])
    split = load_and_run(tmp_path, pages, adapter_for(vendor))
    np.testing.assert_allclose(split.irradiance, whole.irradiance, rtol=1e-6)
    np.testing.assert_allclose(split.vector, whole.vector, rtol=1e-6, atol=1e-7)
    assert split.rays == whole.rays


@pytest.mark.blender
@pytest.mark.parametrize('panes,lit', [(8, False), (9, True)])
def test_pass_limit_stops_after_nine_hits(tmp_path, vendor, panes, lit):
    # panes clear (tint 1, not casting) stacked under a solid roof, sun straight up: eight passed panes leave the
    # roof as the ninth hit (shadow); nine use up the loop and the ray counts as escaped, on both passes
    stack = [dict(quad=floor(-1, 5, -1, 5, z=0.1 * (k + 1)), casts=False) for k in range(panes)]
    roof_slab = dict(quad=floor(-1, 5, -1, 5, z=2.0))
    cpu, result = cpu_and_gpu(tmp_path, vendor, receivers=[full_quad()], occluders=stack + [roof_slab],
                              suns=[sun(UP, strength=3.0)])
    assert (result.irradiance[0] > 0) == lit == (cpu['irradiance'][0] > 0)


def textured_room():
    south = quad([(0, 0, 0), (0, 0, 4), (8, 0, 4), (8, 0, 0)], (0, 1, 0))
    north = quad([(8, 8, 0), (8, 8, 4), (0, 8, 4), (0, 8, 0)], (0, -1, 0))
    west = quad([(0, 0, 0), (0, 8, 0), (0, 8, 4), (0, 0, 4)], (1, 0, 0))
    east = quad([(8, 0, 0), (8, 8, 0), (8, 8, 4), (8, 0, 4)][::-1], (-1, 0, 0), CORNERS_UV[::-1])
    receivers = [dict(quad=floor(0, 8, 0, 8), texture=0, page=0),
                 # the ceiling stops short of x 3 to 5: the window the suns come through
                 dict(quad=floor(0, 3, 0, 8, z=4, up=False), texture=0, page=0),
                 dict(quad=floor(5, 8, 0, 8, z=4, up=False), texture=0, page=0)]
    receivers += [dict(quad=w, texture=1, page=1) for w in (south, north, west, east)]
    occluders = [dict(quad=floor(3, 5, 3, 5, z=2, up=False), casts=False, emission=(4, 4, 4)),
                 dict(quad=floor(0.85, 4.15, 0.85, 4.15, z=3), texture=2, cutout=True)]
    grid = np.linspace(1, 7, 6)
    floor_q = [(x, y, 0.0) for x in grid for y in grid]
    wall_q = ([(0.0, y, z) for y in np.linspace(1, 7, 7) for z in (1.0, 3.0)]
              + [(x, 0.0, z) for x in np.linspace(1, 7, 7) for z in (1.0, 3.0)])
    normals = [(0, 0, 1)] * len(floor_q) + [(1, 0, 0)] * 14 + [(0, 1, 0)] * 14
    rng = np.random.default_rng(11)
    textures = [rng.integers(0, 256, (8, 8, 4)).astype(np.uint8) for _ in range(2)]
    for texture in textures:
        texture[:, :, 3] = 255
    textures.append(alpha_texture(255, holes=True))
    pages = [(rng.random((16, 16, 3)) * 2).astype(np.float32), (rng.random((8, 32, 3)) * 2).astype(np.float32)]
    return dict(receivers=receivers, occluders=occluders, query=floor_q + wall_q, normals=normals, textures=textures,
                pages=pages, suns=[sun((0.3, 0, 1), strength=3.0), sun((-0.2, 0.3, 1), strength=2.0)],
                ambient={'0': [0.2, 0.25, 0.3]}, rays=256, seed=0)


@pytest.mark.blender
def test_gpu_matches_cpu_on_a_textured_room(tmp_path, vendor):
    # a closed 8x8x4 box of textured receivers on two pages with a non-constant page image, an emissive and a
    # cut-out occluder, two suns through a window, 64 queries on the floor and walls, 256 rays, seed 0
    scene = textured_room()
    assert len(scene['query']) == 64
    cpu, result = cpu_and_gpu(tmp_path, vendor, **scene)
    rel = np.abs(result.irradiance - cpu['irradiance']) / np.maximum(cpu['irradiance'], 1e-6)
    assert cpu['irradiance'].min() > 0
    assert np.median(rel) <= 1e-4 and rel.max() <= 1e-2
    assert result.rays == int(cpu['rays']) and np.array_equal(result.sun_lit, cpu['sun_lit'])
    assert result.sun_lit.any()


def write_bare_scene(work, queries=True, triangles=True):
    scene = build([full_quad()])
    if not triangles:
        scene = dataclasses.replace(
            scene, positions=scene.positions[:0], normals=scene.normals[:0], lm_uv=scene.lm_uv[:0],
            tex_uv=scene.tex_uv[:0], tris=scene.tris[:0], tri_page=scene.tri_page[:0], tri_class=scene.tri_class[:0],
            tri_casts_shadow=scene.tri_casts_shadow[:0], tri_emission=scene.tri_emission[:0],
            tri_texture=scene.tri_texture[:0], tri_cutout=scene.tri_cutout[:0], tri_tint=scene.tri_tint[:0])
    if not queries:
        scene = dataclasses.replace(scene, query_pos=scene.query_pos[:0], query_normal=scene.query_normal[:0],
                                    query_class=scene.query_class[:0])
    write_job(work, scene, [(16, 16), (8, 8)], [sun(UP, strength=3.0)], {'0': [0.2, 0.2, 0.2]},
              dataclasses.replace(S, vertex_rays=64), 'all')
    return [np.zeros((16, 16, 3), np.float32), np.ones((8, 8, 3), np.float32)]


@pytest.mark.blender
def test_gpu_pass_with_empty_parts(tmp_path, vendor, monkeypatch):
    # no occluders, no textures, no edge points, a second page with no receivers; then no triangles at all (both
    # kernels skipped, _dispatch never called); then no queries at all
    adapter = adapter_for(vendor)
    cpu, result = cpu_and_gpu(tmp_path / 'parts', vendor, receivers=[full_quad()], suns=[sun(UP, strength=3.0)],
                              ambient={'0': [0.2] * 3}, pages=[RAMP, np.ones((8, 8, 3), np.float32)])
    assert result.rays == int(cpu['rays']) and np.allclose(result.irradiance, cpu['irradiance'], rtol=1e-4, atol=1e-6)
    cpu, result = cpu_and_gpu(tmp_path / 'no_suns', vendor, receivers=[full_quad()])
    assert result.rays == int(cpu['rays']) and np.allclose(result.irradiance, cpu['irradiance'], rtol=1e-4, atol=1e-6)

    with monkeypatch.context() as patch:
        patch.setattr(vertex_gpu, '_dispatch', lambda *args, **kwargs: pytest.fail('a kernel ran without triangles'))
        pages = write_bare_scene(tmp_path / 'no_triangles', triangles=False)
        bare = load_and_run(tmp_path / 'no_triangles', pages, adapter)
    assert bare.rays == 64 + 1 and bare.sun_lit.tolist() == [[True]]
    np.testing.assert_allclose(bare.irradiance, [3.0 + np.pi * 0.2], rtol=1e-6)
    np.testing.assert_allclose(bare.vector, [[0, 0, 3.0 + np.pi * 0.2]], rtol=1e-6)

    pages = write_bare_scene(tmp_path / 'no_queries', queries=False)
    empty = load_and_run(tmp_path / 'no_queries', pages, adapter)
    assert empty.vector.shape == (0, 3) and empty.irradiance.shape == (0,) and empty.rays == 0
    assert empty.sun_lit.shape == (0, 1)


def scene_folder(tmp_path, queries, suns):
    scene = build([full_quad()])
    positions = [[1 + 2 * (k % 7) / 7, 1 + (k % 5) * 0.4, 0] for k in range(queries)]
    scene = dataclasses.replace(scene, query_pos=np.float32(positions), query_normal=np.float32([[0, 0, 1]] * queries),
                                query_class=np.int16([0] * queries))
    write_job(tmp_path, scene, [(16, 16)], suns, {}, dataclasses.replace(S, vertex_rays=64), 'all')
    with np.load(tmp_path / 'scene.npz') as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((tmp_path / 'job.json').read_text())


@pytest.fixture
def device(vendor):
    opened = gpu.open_device(adapter_for(vendor))
    yield opened
    opened.close()


def test_dispatches_stay_under_the_ray_budget(monkeypatch, tmp_path, vendor, device):
    monkeypatch.setattr(vertex_gpu, 'RAYS_PER_DISPATCH', 4096)
    queries = 3 * 4096 // 64 + 1
    suns = [sun((0.01 * k, 0.02, 1.0)) for k in range(64)]
    arrays, job = scene_folder(tmp_path, queries, suns)
    arrays['query_ignore_normals'] = np.arange(queries) % 2 == 0
    seen = []
    real = vertex_gpu._dispatch

    def recording(kernel, count, **kwargs):
        seen.append(count)
        return real(kernel, count, **kwargs)
    monkeypatch.setattr(vertex_gpu, '_dispatch', recording)
    result = run_vertex_pass(arrays, job, [], [np.zeros((16, 16, 3), np.float32)], device)
    cap = min(4096, device.info.limits.max_compute_dispatch_thread_groups.x * GROUP_SIZE)
    assert vertex_gpu.dispatch_cap(device) == cap
    assert max(seen) <= cap and len(seen) >= 8
    assert result.rays == queries * 64 + queries * 64


def test_slangpy_failure_becomes_a_gpu_pass_error(monkeypatch, tmp_path, vendor, device):
    arrays, job = scene_folder(tmp_path, 2, [sun(UP)])

    def refuse(*args, **kwargs):
        raise RuntimeError('out of memory')
    monkeypatch.setattr(vertex_gpu, '_create_buffer', refuse)
    with pytest.raises(GpuPassError, match='upload: out of memory'):
        run_vertex_pass(arrays, job, [], [np.zeros((16, 16, 3), np.float32)], device)


def test_chunks_are_reduced_as_they_land(monkeypatch, tmp_path, vendor, device):
    queries = 70
    arrays, job = scene_folder(tmp_path, queries, [sun((0.2, 0.1, 1.0))])
    pages = [np.zeros((16, 16, 3), np.float32)]
    whole = run_vertex_pass(arrays, job, [], pages, device)
    monkeypatch.setattr(vertex_gpu, 'RAYS_PER_DISPATCH', 4096)
    held = []
    real = vertex_gpu.gather_sums

    def recording(lum, *args):
        held.append(lum.shape)
        return real(lum, *args)
    monkeypatch.setattr(vertex_gpu, 'gather_sums', recording)
    chunked = run_vertex_pass(arrays, job, [], pages, device)
    # no ray matrix larger than one dispatch's queries is ever held
    assert len(held) >= 2 and max(rows for rows, _ in held) <= 4096 // 64
    assert np.array_equal(chunked.vector, whole.vector) and np.array_equal(chunked.irradiance, whole.irradiance)
    assert np.array_equal(chunked.sun_lit, whole.sun_lit) and chunked.rays == whole.rays


def test_more_rays_than_one_dispatch_holds_is_refused(monkeypatch, tmp_path, vendor, device):
    arrays, job = scene_folder(tmp_path, 2, [sun(UP)])
    monkeypatch.setattr(vertex_gpu, 'RAYS_PER_DISPATCH', 32)
    monkeypatch.setattr(vertex_gpu, '_dispatch', lambda *args, **kwargs: pytest.fail('dispatched over the cap'))
    with pytest.raises(GpuPassError, match='gather: vertex_rays 64 exceeds the dispatch cap 32'):
        run_vertex_pass(arrays, job, [], [np.zeros((16, 16, 3), np.float32)], device)


def test_chunk_luminances_are_summed_in_float64(monkeypatch, tmp_path, vendor, device):
    scene = dataclasses.replace(
        build([full_quad(), wall_facing_minus_x(emission=(4, 4, 4))]),
        query_pos=np.float32([[1 + 0.1 * k, 1 + 0.07 * k, 0] for k in range(20)]),
        query_normal=np.float32([[0, 0, 1]] * 20), query_class=np.int16([0] * 20))
    write_job(tmp_path, scene, [(16, 16)], [], {'0': [0.2, 0.3, 0.1]}, dataclasses.replace(S, vertex_rays=256), 'all')
    with np.load(tmp_path / 'scene.npz') as data:
        arrays = {key: data[key] for key in data.files}
    job = json.loads((tmp_path / 'job.json').read_text())
    real = vertex_gpu.gather_sums
    seen = []

    def checking(lum, *args):
        got = real(lum, *args)
        # luminances that vary per ray, summed as float64 whatever precision the kernel read back in
        want = real(lum.astype(np.float64), *args)
        assert lum.std() > 0 and lum.any()
        seen.append((got, want))
        return got
    monkeypatch.setattr(vertex_gpu, 'gather_sums', checking)
    monkeypatch.setattr(vertex_gpu, 'RAYS_PER_DISPATCH', 4096)
    run_vertex_pass(arrays, job, [], [RAMP], device)
    assert len(seen) >= 2
    for got, want in seen:
        np.testing.assert_allclose(got[0], want[0], rtol=1e-13, atol=0)
        np.testing.assert_allclose(got[1], want[1], rtol=1e-13, atol=0)


def test_a_dying_gpu_child_becomes_a_gpu_pass_error(tmp_path, monkeypatch):
    from test_pipeline import NV
    monkeypatch.setattr(vertex_gpu, '_child_command', lambda work, adapter:
                        [sys.executable, '-c', 'import sys; print("driver fault", file=sys.stderr); sys.exit(3)'],
                        raising=False)
    with pytest.raises(GpuPassError, match=r'process died \(exit 3\):.*driver fault'):
        vertex_gpu.run_in_child(tmp_path, [np.zeros((4, 4, 3), np.float32)], NV, 60)


@pytest.mark.parametrize('code', [
    'pass',
    'from pathlib import Path; Path("gpu_out.npz").write_bytes(b"short")',
    'import numpy as np; np.savez("gpu_out.npz", vector=np.zeros((1,3)))',
])
def test_missing_or_short_gpu_output_is_an_error(tmp_path, monkeypatch, code):
    from test_pipeline import NV
    monkeypatch.setattr(vertex_gpu, '_child_command', lambda work, adapter: [sys.executable, '-c', code], raising=False)
    with pytest.raises(GpuPassError, match=r'process died \(exit 0\)'):
        vertex_gpu.run_in_child(tmp_path, [], NV, 60)


def test_gpu_child_timeout_discards_stale_output(tmp_path, monkeypatch):
    from test_pipeline import NV
    (tmp_path / 'gpu_out.npz').write_bytes(b'stale')
    monkeypatch.setattr(vertex_gpu, '_child_command', lambda work, adapter:
                        [sys.executable, '-c', 'import time; time.sleep(60)'], raising=False)
    with pytest.raises(GpuPassError, match='gather: timed out'):
        vertex_gpu.run_in_child(tmp_path, [], NV, .2)
    assert not (tmp_path / 'gpu_out.npz').exists()


def test_child_exchange_at_the_stock_query_limit(tmp_path, monkeypatch):
    from test_pipeline import NV
    # 131072 independent triangles can supply three distinct queries each
    q = 393216
    np.savez(tmp_path / 'scene.npz', query_pos=np.zeros((q, 3), np.float32), edge_pos=np.zeros((0, 3)))
    (tmp_path / 'job.json').write_text(json.dumps({'scene': 'scene.npz', 'suns': [{}, {}]}))
    code = ('import numpy as np; '
            f'q={q}; '
            'np.savez("gpu_out.npz", vector=np.full((q,3),.25,np.float32), '
            'irradiance=np.ones(q,np.float32), sun_lit=np.ones((q,2),bool), '
            'vertex_sun_throughput=np.full((q,2,3),.125,np.float32), '
            'edge_sun_lit=np.zeros((0,2),bool), rays=np.int64(q*4096))')
    monkeypatch.setattr(vertex_gpu, '_child_command', lambda work, adapter: [sys.executable, '-c', code], raising=False)
    result = vertex_gpu.run_in_child(tmp_path, [RAMP], NV, 60)
    assert result.vector.shape == (393216, 3) and result.rays == 1610612736
    np.testing.assert_array_equal(result.vector, .25)
    assert result.sun_throughput.shape == (393216, 2, 3)
    np.testing.assert_array_equal(result.sun_throughput, .125)
    with np.load(tmp_path / 'diffuse.npz') as data:
        np.testing.assert_array_equal(data['diffuse_0'], RAMP)


@pytest.mark.blender
def test_bake_with_the_gpu_pass_matches_the_cpu_pass(tmp_path, vendor):
    from lightmapper.blender_job import run_blender
    room = textured_room()
    scene = dataclasses.replace(build(room['receivers'], room['occluders']),
        query_pos=np.float32(room['query']), query_normal=np.float32(room['normals']),
        query_class=np.zeros(64, np.int16))
    s = dataclasses.replace(S, samples=8, vertex_rays=256)
    write_job(tmp_path, scene, [(16, 16), (32, 8)], room['suns'], room['ambient'], s,
              textures=room['textures'])
    first = run_blender(tmp_path, s)
    result = vertex_gpu.run_in_child(tmp_path, first.diffuse_pages, adapter_for(vendor), 120)
    rel = np.abs(result.irradiance - first.vertex_irradiance) / np.maximum(first.vertex_irradiance, 1e-6)
    assert np.median(rel) <= 1e-4 and rel.max() <= 1e-2
    assert result.rays == first.vertex_rays_cast
    np.testing.assert_array_equal(result.sun_lit, first.vertex_sun_lit)
    assert result.sun_throughput is None and first.vertex_sun_throughput is None
    direct = load_and_run(tmp_path, first.diffuse_pages, adapter_for(vendor))
    np.testing.assert_array_equal(result.edge_sun_lit, direct.edge_sun_lit)


def test_consistently_short_gpu_arrays_are_refused(tmp_path, monkeypatch):
    from test_pipeline import NV
    write_bare_scene(tmp_path)
    with np.load(tmp_path / 'scene.npz') as data:
        edges = len(data['edge_pos'])
    # complete keys and internally consistent shapes, but the job has one query, not zero
    code = ('import numpy as np; np.savez("gpu_out.npz", vector=np.zeros((0,3),np.float32), '
            'irradiance=np.zeros(0,np.float32), sun_lit=np.zeros((0,1),bool), '
            'vertex_sun_throughput=np.zeros((0,1,3),np.float32), '
            f'edge_sun_lit=np.zeros(({edges},1),bool), rays=np.int64(0))')
    monkeypatch.setattr(vertex_gpu, '_child_command', lambda work, adapter: [sys.executable, '-c', code])
    with pytest.raises(GpuPassError, match='process died'):
        vertex_gpu.run_in_child(tmp_path, [], NV, 60)


def test_gpu_child_starts_without_a_window(tmp_path, monkeypatch):
    from test_pipeline import NV
    work = tmp_path.resolve()
    seen = {}
    real = vertex_gpu.subprocess.run

    def spy(args, **kwargs):
        seen.update(kwargs, args=args)
        return real(args, **kwargs)

    monkeypatch.setattr(vertex_gpu.subprocess, 'run', spy)
    with pytest.raises(GpuPassError):
        vertex_gpu.run_in_child(work, [], NV, 60)
    assert seen['creationflags'] == frozen.NO_WINDOW and seen['args'][-2:] == [str(work), json.dumps(dataclasses.asdict(NV))]
    assert seen['cwd'] == work and 'env' not in seen


@pytest.mark.gpu_nvidia
def test_gpu_child_runs_in_a_folder_with_spaces_and_accents(tmp_path):
    work = tmp_path / 'a b é'
    pages = write_bare_scene(work)
    with np.load(work / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    result = vertex_gpu.run_in_child(work, pages, adapter_for('nvidia'), 120)
    assert result.vector.shape == (len(scene['query_pos']), 3)
