import ast
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import numba
import pytest

from lightmapper import entry, frozen, vertex_cpu, vertex_gpu
from lightmapper.formats.lights import SceneLight, ramp
from lightmapper.blender_job import BAKE_SCRIPT, write_job
from test_blender_bake import UP, build, floor, full_quad, quad, sun, wall_x
from test_vertex_gpu import CASES, edge_receivers, textured_room, write_bare_scene
from conftest import LIGHTMAPPER_BLENDER
from test_vertex_pass import (DRIVER, S, assert_asymmetric_twin, assert_twin_light, asymmetric_twin_scene, roof,
                              run_driver, twin_scene, unpaired_back_face_scene, wall_facing_minus_x,
                              texel_scene, mixed_scene, assert_flagged_bounce, assert_emission_relay, assert_mixed)


def run_probe(out):
    done = subprocess.run(
        frozen.child_command(frozen.CPU_VERTEX_FLAG, '--probe', str(out)),
        capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(out.read_text())


def test_cpu_vertex_flag_dispatches_the_child(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(tmp_path / 'cache'))
    probe = run_probe(tmp_path / 'probe.json')
    assert probe['threading_layer'] in ('workqueue', 'omp', 'tbb')
    assert probe['threads'] >= 1
    assert probe['cache_dir'] == str(tmp_path / 'cache')
    assert probe['seconds'] > 0


def test_second_probe_loads_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(tmp_path / 'cache'))
    first = run_probe(tmp_path / 'first.json')
    second = run_probe(tmp_path / 'second.json')
    assert first['cache_misses'] >= 1
    assert second['cache_hits'] >= 1
    assert second['cache_misses'] == 0


def test_cache_dir_defaults_under_local_app_data(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    assert frozen.numba_cache_dir() == tmp_path / 'OpenTool' / 'numba-cache'


def test_cache_dir_falls_back_to_home(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', '')
    monkeypatch.setattr(frozen.Path, 'home', lambda: tmp_path)
    assert frozen.numba_cache_dir() == tmp_path / 'OpenTool' / 'numba-cache'


def test_a_cache_dir_the_user_set_is_kept(tmp_path, monkeypatch):
    from lightmapper import vertex_cpu
    cache = str(tmp_path / 'user-cache')
    monkeypatch.setenv('NUMBA_CACHE_DIR', cache)
    seen = {}

    def child_main(argv):
        seen.update(cache=os.environ['NUMBA_CACHE_DIR'], argv=argv)
        return 17

    monkeypatch.setattr(vertex_cpu, 'child_main', child_main)
    assert entry.cli_entry([frozen.CPU_VERTEX_FLAG, '--probe', 'out.json']) == 17
    assert seen == {'cache': cache, 'argv': ['--probe', 'out.json']}


def test_entry_sets_cache_before_importing_vertex_cpu(tmp_path, monkeypatch):
    monkeypatch.delenv('NUMBA_CACHE_DIR', raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    code = (
        'from lightmapper import entry; '
        'entry.cli_entry(["--cpu-vertex-pass", "--probe", ' + repr(str(tmp_path / 'probe.json')) + '])')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    probe = json.loads((tmp_path / 'probe.json').read_text())
    assert probe['cache_dir'] == str(tmp_path / 'OpenTool' / 'numba-cache')
    assert list((tmp_path / 'OpenTool' / 'numba-cache').rglob('*.nbc'))


def test_spec_collects_vertex_cpu_as_source():
    text = (frozen.bundle_dir() / 'packaging' / 'opentool.spec').read_text()
    assert "module_collection_mode={'lightmapper.vertex_cpu': 'py'}" in text


def soup(rng, n, offset=0.0):
    corners = rng.uniform(-5, 5, (n, 3, 3)).astype(np.float32)
    shared = corners[100:200].copy()
    shared[:, 2] = rng.uniform(-5, 5, (100, 3))
    return np.concatenate((corners, corners[:100], shared)) + np.float32(offset)


def random_rays(rng, n, offset=0.0):
    origins = rng.uniform(-5, 5, (n, 3)).astype(np.float32) + np.float32(offset)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return origins, dirs.astype(np.float32)


def compare_brute_force(corners, origins, dirs, limits=None):
    if limits is None:
        limits = np.full(len(origins), vertex_cpu.RAY_LENGTH, np.float32)
    bvh = vertex_cpu.build_bvh(corners)
    tri, t = vertex_cpu.closest_hits(bvh, corners, origins, dirs, limits)
    ref_tri, ref_t = vertex_cpu.brute_force_hits(corners, origins, dirs, limits)
    assert tri.dtype == np.int64 and t.dtype == np.float32
    np.testing.assert_array_equal(tri, ref_tri)
    np.testing.assert_array_equal(t, ref_t)
    return tri, t


def test_bvh_closest_hit_matches_brute_force():
    rng = np.random.default_rng(5)
    corners = soup(rng, 2000)
    origins, dirs = random_rays(rng, 20000)
    tri, _ = compare_brute_force(corners, origins, dirs)
    assert (tri >= 0).mean() > 0.3


def test_coincident_pair_resolves_to_the_lower_index():
    corners = np.zeros((8, 3, 3), np.float32)
    corners[:, :, 2] = 2
    corners[3] = corners[7] = [[-1, -1, 2], [1, -1, 2], [0, 1, 2]]
    # the later child enters at the hit distance and holds the lower-index twin
    bvh = vertex_cpu.Bvh(
        np.float32([[-1, -1, 2], [-1, -1, 2], [-1, -1, 2]]),
        np.float32([[1, 1, 2], [1, 1, 2], [1, 1, 2]]),
        np.int32([1, 0, 4]), np.int32([0, 4, 4]), np.int32([7, 0, 1, 2, 3, 4, 5, 6]))
    tri, t = vertex_cpu.closest_hits(bvh, corners, np.zeros((1, 3), np.float32),
                                      np.float32([[0, 0, 1]]), np.float32([10]))
    np.testing.assert_array_equal(tri, [3])
    np.testing.assert_array_equal(t, [2])


def test_bvh_matches_brute_force_far_from_the_origin():
    rng = np.random.default_rng(5)
    offset = [4000, -4000, 300]
    corners = soup(rng, 2000, offset)
    origins, dirs = random_rays(rng, 20000, offset)
    tri, _ = compare_brute_force(corners, origins, dirs)
    assert (tri >= 0).mean() > 0.3


def test_ray_along_a_shared_edge_hits_one_of_the_two():
    corners = np.float32([[[0, 0, 2], [1, 1, 2], [0, 1, 2]],
                          [[1, 1, 2], [0, 0, 2], [1, 0, 2]]])
    points = np.linspace(0, 1, 1000, dtype=np.float32)
    origins = np.column_stack((points, points, np.zeros(1000, np.float32)))
    dirs = np.tile(np.float32([0, 0, 1]), (1000, 1))
    tri, t = compare_brute_force(corners, origins, dirs)
    assert np.all((tri == 0) | (tri == 1))
    np.testing.assert_array_equal(t, np.full(1000, 2, np.float32))


def test_both_faces_are_hit():
    corners = np.float32([[[-1, -1, 2], [1, -1, 2], [0, 1, 2]]])
    tri, t = compare_brute_force(corners, np.float32([[0, 0, 0], [0, 0, 4]]),
                                np.float32([[0, 0, 1], [0, 0, -1]]))
    np.testing.assert_array_equal(tri, [0, 0])
    np.testing.assert_array_equal(t, [2, 2])


def test_hit_must_be_closer_than_the_limit():
    corners = np.float32([[[-1, -1, 2], [1, -1, 2], [0, 1, 2]]])
    tri, t = compare_brute_force(corners, np.zeros((4, 3), np.float32),
                                np.tile(np.float32([0, 0, 1]), (4, 1)),
                                np.float32([2, 2.0001, 0, -1]))
    np.testing.assert_array_equal(tri, [-1, 0, -1, -1])
    np.testing.assert_array_equal(t, [np.inf, 2, np.inf, np.inf])


def test_degenerate_triangle_is_never_hit():
    corners = np.float32([[[0, 0, 2], [1, 0, 2], [2, 0, 2]]])
    tri, t = compare_brute_force(corners, np.float32([[1, 0, 0]]), np.float32([[0, 0, 1]]))
    np.testing.assert_array_equal(tri, [-1])
    np.testing.assert_array_equal(t, [np.inf])


def test_empty_bvh_misses():
    corners = np.empty((0, 3, 3), np.float32)
    origins, dirs = random_rays(np.random.default_rng(2), 10)
    tri, t = compare_brute_force(corners, origins, dirs)
    np.testing.assert_array_equal(tri, np.full(10, -1))
    assert np.isinf(t).all()


def test_bvh_grid_snapped_twenty_thousand_triangles():
    x = np.arange(20000, dtype=np.float32) * 2
    corners = np.zeros((20000, 3, 3), np.float32)
    corners[:, :, 0] = x[:, None]
    corners[:, 1, 0] += 1
    corners[:, 2, 1] = 1
    bvh = vertex_cpu.build_bvh(corners)
    assert bvh.lo.dtype == bvh.hi.dtype == np.float32
    assert bvh.first.dtype == bvh.count.dtype == bvh.order.dtype == np.int32
    np.testing.assert_array_equal(np.sort(bvh.order), np.arange(20000))
    todo = [(0, 0)]
    covered = []
    while todo:
        node, depth = todo.pop()
        assert depth <= 15
        start, count = bvh.first[node], bvh.count[node]
        if count:
            triangles = bvh.order[start:start + count]
            covered.extend(triangles)
            assert np.all(corners[triangles] >= bvh.lo[node])
            assert np.all(corners[triangles] <= bvh.hi[node])
        else:
            for child in (start, start + 1):
                assert np.all(bvh.lo[child] >= bvh.lo[node])
                assert np.all(bvh.hi[child] <= bvh.hi[node])
                todo.append((child, depth + 1))
    np.testing.assert_array_equal(np.sort(covered), np.arange(20000))
    selected = np.arange(0, 20000, 100)
    origins = np.column_stack((x[selected] + np.float32(.25),
                               np.full(200, .25, np.float32), np.full(200, 1, np.float32)))
    tri, t = compare_brute_force(corners, origins, np.tile(np.float32([0, 0, -1]), (200, 1)))
    np.testing.assert_array_equal(tri, selected)
    np.testing.assert_array_equal(t, np.ones(200))


@pytest.mark.parametrize('direction', [[0, 0, 2], [0, 0, 0], [0, 0, 1e-20],
                                     [np.nan, 0, 1], [np.inf, 0, 1]])
def test_bvh_direction_normalization(direction):
    corners = np.float32([[[-1, -1, 2], [1, -1, 2], [0, 1, 2]]])
    tri, t = compare_brute_force(corners, np.zeros((1, 3), np.float32), np.float32([direction]))
    expected = direction == [0, 0, 2]
    np.testing.assert_array_equal(tri, [0 if expected else -1])
    np.testing.assert_array_equal(t, [2 if expected else np.inf])


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_degenerate_nonfinite_triangle_is_never_hit(bad):
    corners = np.float32([[-1, -1, 2], [1, -1, 2], [0, bad, 2]])
    assert np.isinf(vertex_cpu._ray_triangle(np.zeros(3, np.float32), np.float32([0, 0, 1]),
                                           *corners))


@pytest.mark.blender
def test_closest_hit_matches_blender_on_a_random_soup(tmp_path, blender_exe):
    rng = np.random.default_rng(5)
    corners = rng.uniform(-5, 5, (2000, 3, 3)).astype(np.float32)
    origins, dirs = random_rays(rng, 20000)
    limits = np.full(20000, vertex_cpu.RAY_LENGTH, np.float32)
    np.savez(tmp_path / 'soup.npz', corners=corners, origins=origins, dirs=dirs, limits=limits)
    driver = Path(__file__).with_name('bvh_driver.py')
    done = subprocess.run([str(blender_exe), '-b', '-t', '1', '--python-exit-code', '1',
                           '--python', str(driver), '--', str(tmp_path)],
                          capture_output=True, timeout=300)
    assert done.returncode == 0, (done.stdout + done.stderr)[-4000:]
    with np.load(tmp_path / 'hits.npz') as hits:
        ref_tri, ref_t = hits['index'], hits['distance']
    tri, t = vertex_cpu.closest_hits(vertex_cpu.build_bvh(corners), corners, origins, dirs, limits)
    mismatch = (tri != ref_tri) | ~np.isclose(t, ref_t, rtol=1e-6, atol=0)
    if mismatch.any():
        rays = np.flatnonzero(mismatch)
        details = [{'ray': int(i), 'origin': origins[i].tolist(), 'direction': dirs[i].tolist(),
                    'cpu': [int(tri[i]), float(t[i])], 'blender': [int(ref_tri[i]), float(ref_t[i])]}
                   for i in rays]
        (tmp_path / 'disagreements.json').write_text(json.dumps(details, indent=2))
        pytest.fail(f'{len(rays)} Blender disagreements: {details[:10]}; full rays: {tmp_path}')
    np.testing.assert_array_equal(tri, ref_tri)
    np.testing.assert_allclose(t, ref_t, rtol=1e-6, atol=0)


def test_copied_constants_equal_bake_py():
    source = Path(vertex_cpu.__file__).parent / 'blender' / 'bake.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = ('RAY_LENGTH', 'RAY_EPSILON', 'PASS_LIMIT', 'SKIP_LIMIT', 'LUMA', 'GOLDEN_FRACTION', 'SEED_SPIN')
    namespace = {'np': np}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in names:
            exec(compile(ast.Module([node], []), str(source), 'exec'), namespace)
    from_bake = {name: np.asarray(namespace[name]).tolist() for name in names}
    copied = {name: np.asarray(getattr(vertex_cpu, name)).tolist() for name in names}
    assert set(from_bake) == set(names)
    assert copied == from_bake


def load_cpu_scene(work, flagged_pages=None):
    with np.load(work / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    job = json.loads((work / 'job.json').read_text())
    with np.load(work / 'textures.npz') as data:
        textures = [data[f'tex_{k}'] for k in range(len(data.files))]
    with np.load(work / 'pages.npz') as data:
        pages = [data[f'page_{i}'] for i in range(len(data.files))]
    cs = vertex_cpu.cpu_scene(scene, job, textures, pages, flagged_pages)
    return cs, vertex_cpu.build_bvh(cs.corners32)


def blender_and_numba_probes(tmp_path, probes, **scene):
    blender = run_driver(tmp_path, probes=probes, **scene)
    cs, bvh = load_cpu_scene(tmp_path)
    origins, dirs, classes = zip(*probes)
    return blender, vertex_cpu.probe(cs, bvh, np.float64(origins), np.float64(dirs), np.int64(classes))


def random_probes(scene, count, seed=1):
    rng = np.random.default_rng(seed)
    query = scene.get('query', ((2, 2, 0),))
    dirs = rng.normal(size=(count, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    # origins start one RAY_EPSILON along the ray as in bake.py: a query on a shared edge would otherwise hit both
    # triangles at distance 0, which Blender settles by its tree order rather than by index
    return [[(np.float64(query[i % len(query)]) + vertex_cpu.RAY_EPSILON * dirs[i]).tolist(), dirs[i].tolist(),
             scene.get('cls', 0)] for i in range(count)]


PROBE_CASES = {'twin': twin_scene(False, False), 'twin_reverse': twin_scene(True, False),
               'twin_cutout': twin_scene(False, True), 'asymmetric_twin': asymmetric_twin_scene(),
               'unpaired_back_face': unpaired_back_face_scene(), 'textured_room': textured_room(),
               **{name: CASES[name] for name in ('cutout_hole', 'glass_tint_half', 'emit_only_front',
                                                 'emit_only_back', 'grey_albedo', 'texture_negative_uv',
                                                 'indoor_occluder')}}


@pytest.mark.blender
@pytest.mark.parametrize('case', PROBE_CASES)
def test_probes_match_blender(tmp_path, case):
    scene = PROBE_CASES[case]
    probes = random_probes(scene, 2000 if case == 'textured_room' else 500)
    blender, ours = blender_and_numba_probes(tmp_path, probes, **scene)
    # a ray gathering light passes clear panes and emitters seen from behind, so those two scenes only miss
    assert (blender['tri'] >= 0).any() or case in ('glass_tint_half', 'emit_only_back'), (
        f'{case}: zero probe hits; only clear panes and back-facing emitters should all miss')
    np.testing.assert_array_equal(ours['tri'], blender['tri'])
    np.testing.assert_allclose(ours['uv'], blender['uv'], rtol=1e-5, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(ours['radiance'], blender['radiance'], rtol=1e-4, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(ours['radiance_rgb'], blender['radiance_rgb'], rtol=1e-4, atol=1e-6, equal_nan=True)


@pytest.mark.blender
def test_hit_uv_barycentric(tmp_path):
    # the page's red is 16 u - 0.5 and its green 16 v - 0.5 at the hit; the wall maps u = y/4, v = z/4
    page = np.zeros((16, 16, 3), np.float32)
    page[:, :, 0] = np.arange(16)[None, :]
    page[:, :, 1] = np.arange(16)[:, None]
    wall = wall_facing_minus_x(lm=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
    probes = [([1, 1.3, 2.6], [1, 0, 0], 0), ([1, 2.7, 1.1], [1, 0, 0], 0)]
    _, ours = blender_and_numba_probes(tmp_path, probes, receivers=[full_quad(), wall], pages=[page])
    for (origin, _, _), uv, radiance in zip(probes, ours['uv'], ours['radiance']):
        u, v = 0.25 + origin[1] / 8, 0.25 + origin[2] / 8
        assert uv == pytest.approx([u, v], abs=1e-5)
        red, green = 16 * u - 0.5, 16 * v - 0.5
        assert radiance == pytest.approx(0.5 * (0.2126 * red + 0.7152 * green), rel=1e-4)


def cutout_quad_scene(tmp_path):
    scene = dataclasses.replace(
        build([full_quad()], [dict(quad=floor(0, 4, 0, 4, z=1.0), texture=0, cutout=True)]),
        query_pos=np.float32([[2, 2, 0]]), query_normal=np.float32([UP]), query_class=np.int16([0]))
    write_job(tmp_path, scene, [(16, 16)], [], {}, S, 'all', textures=[np.full((1, 1, 4), 255, np.uint8)])
    np.savez(tmp_path / 'pages.npz', page_0=np.zeros((16, 16, 3), np.float32))
    return load_cpu_scene(tmp_path)


@pytest.mark.parametrize('alpha,blocks', [(0.5, True), (0.4999, False)])
def test_alpha_exactly_half_blocks(tmp_path, alpha, blocks):
    cs, bvh = cutout_quad_scene(tmp_path)
    cs.texture_texels[:, 3] = alpha
    tri, _ = vertex_cpu._cast(cs.arrays(), (bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order),
                              np.float32([2, 2, 0.5]), np.float32([0, 0, 1]), False, np.ones(3))
    assert (tri >= 0) == blocks


def test_zero_area_triangle_uses_its_centre():
    a, b, c = np.float64([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    # the clamped weights of a finite point are never all zero, so only an unusable point reaches the centre
    for point in ([np.nan, 0, 0], [np.inf, 0, 0]):
        assert vertex_cpu._barycentric(a, b, c, np.float64(point)) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert vertex_cpu._barycentric(a, b, c, np.float64([1, 1, 0])) == (1.0, 0.0, 0.0)
    assert vertex_cpu._barycentric(
        *np.float64([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.25, 0.25, 0]])) == pytest.approx((0.5, 0.25, 0.25))


def saved_pages(work):
    with np.load(work / 'pages.npz') as data:
        return [data[f'page_{i}'] for i in range(len(data.files))]


def blender_and_numba(tmp_path, **kwargs):
    blender = run_driver(tmp_path, **kwargs)
    ours = vertex_cpu.load_and_run(tmp_path, saved_pages(tmp_path))
    metrics = {'rays': ours.rays}
    for field in ('irradiance', 'vector', 'texel_rgb'):
        if getattr(ours, field) is None:
            continue
        diff = np.abs(getattr(ours, field) - blender[field])
        rel = diff / np.maximum(np.abs(blender[field]), 1e-6)
        metrics[field] = {'max_abs': float(diff.max(initial=0)), 'max_rel': float(rel.max(initial=0)),
                          'median_rel': float(np.median(rel)) if rel.size else 0.0}
    metrics['sun_lit_equal'] = np.array_equal(ours.sun_lit, blender['sun_lit'])
    metrics['edge_sun_lit_equal'] = np.array_equal(ours.edge_sun_lit, blender['edge_sun_lit'])
    if ours.sun_throughput is not None and 'vertex_sun_throughput' in blender.files:
        diff = np.abs(ours.sun_throughput - blender['vertex_sun_throughput'])
        metrics['sun_throughput_max_abs'] = float(diff.max(initial=0))
    (tmp_path / 'parity.json').write_text(json.dumps(metrics, indent=2))
    return blender, ours


def assert_parity(blender, ours):
    assert ours.rays == int(blender['rays'])
    np.testing.assert_allclose(ours.irradiance, blender['irradiance'], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(ours.vector, blender['vector'], rtol=1e-4, atol=1e-5)
    assert np.array_equal(ours.sun_lit, blender['sun_lit']) and np.array_equal(ours.edge_sun_lit, blender['edge_sun_lit'])
    if 'vertex_sun_throughput' in blender.files:
        np.testing.assert_allclose(ours.sun_throughput, blender['vertex_sun_throughput'], rtol=1e-6, atol=1e-6)
    else:
        assert ours.sun_throughput is None


ORDINARY = [c for c in CASES if not c.startswith(('ignore_normals', 'sun_throughput'))]


@pytest.mark.blender
@pytest.mark.parametrize('case', ['sun_throughput', 'ignore_normals_sun_behind', 'ignore_normals_sun_60',
                                  'ignore_normals_emitter_behind'])
def test_numba_matches_blender_pass_on_the_flagged_scenes(tmp_path, case):
    blender, ours = blender_and_numba(tmp_path, **CASES[case])
    assert_parity(blender, ours)
    assert ours.sun_throughput.dtype == np.float32
    if case == 'sun_throughput':
        np.testing.assert_allclose(ours.sun_throughput[:, 0],
                                   [[0.5, 0.25, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]], rtol=1e-6)


@pytest.mark.blender
def test_ignore_normals_tool_mode_matches(tmp_path):
    scene = textured_room()
    scene['ignore_normals'] = [i % 3 == 0 for i in range(64)]
    blender, ours = blender_and_numba(tmp_path, **scene)
    rel = np.abs(ours.irradiance - blender['irradiance']) / np.maximum(blender['irradiance'], 1e-6)
    assert blender['irradiance'].min() > 0
    assert np.median(rel) <= 1e-4 and rel.max() <= 1e-2
    assert ours.rays == int(blender['rays']) and np.array_equal(ours.sun_lit, blender['sun_lit'])
    assert ours.sun_lit.any()
    np.testing.assert_allclose(ours.sun_throughput, blender['vertex_sun_throughput'], rtol=1e-6, atol=1e-6)
    assert ours.sun_throughput.dtype == np.float32


@pytest.mark.blender
@pytest.mark.parametrize('mode', ['tool', 'enhanced'])
def test_ordinary_queries_omit_unused_throughput(tmp_path, mode):
    blender, ours = blender_and_numba(tmp_path, receivers=[full_quad()], suns=[sun()], mode=mode)
    assert_parity(blender, ours)
    assert ours.irradiance[0] > 0
    assert ours.sun_throughput is None
    assert 'vertex_sun_throughput' not in blender.files


@pytest.mark.blender
@pytest.mark.parametrize('case', ORDINARY)
def test_numba_matches_blender_pass_on_the_vertex_pass_scenes(tmp_path, case):
    assert_parity(*blender_and_numba(tmp_path, **CASES[case]))


@pytest.mark.blender
def test_numba_matches_blender_pass_on_a_textured_room(tmp_path):
    blender, ours = blender_and_numba(tmp_path, **textured_room())
    rel = np.abs(ours.irradiance - blender['irradiance']) / np.maximum(blender['irradiance'], 1e-6)
    assert blender['irradiance'].min() > 0
    assert np.median(rel) <= 1e-4 and rel.max() <= 1e-2
    assert ours.rays == int(blender['rays']) and np.array_equal(ours.sun_lit, blender['sun_lit'])
    assert ours.sun_lit.any()


@pytest.mark.blender
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('cutout', [False, True])
def test_twin_faces_light_both_sides(tmp_path, reverse, cutout):
    blender, ours = blender_and_numba(tmp_path, **twin_scene(reverse, cutout))
    assert_parity(blender, ours)
    assert_twin_light(blender['irradiance'], blender['vector'], blender['rays'])
    assert_twin_light(ours.irradiance, ours.vector, ours.rays)


@pytest.mark.blender
def test_unpaired_back_face_still_blocks(tmp_path):
    blender, ours = blender_and_numba(tmp_path, **unpaired_back_face_scene())
    assert_parity(blender, ours)
    np.testing.assert_array_equal(blender['irradiance'], [0])
    np.testing.assert_array_equal(ours.irradiance, [0])
    np.testing.assert_array_equal(ours.vector, [[0, 0, 0]])


@pytest.mark.blender
@pytest.mark.parametrize('reverse', [False, True])
def test_twin_uses_facing_alpha_and_corner_weights_before_stopping(tmp_path, reverse):
    blender, ours = blender_and_numba(tmp_path, **asymmetric_twin_scene(reverse))
    assert_parity(blender, ours)
    assert_asymmetric_twin(blender['irradiance'], blender['sun_lit'])
    assert_asymmetric_twin(ours.irradiance, ours.sun_lit)


@pytest.mark.blender
@pytest.mark.parametrize('panes,lit', [(8, False), (9, True)])
def test_pass_limit_stops_after_nine_hits(tmp_path, panes, lit):
    stack = [dict(quad=floor(-1, 5, -1, 5, z=0.1 * (k + 1)), casts=False) for k in range(panes)]
    roof_slab = dict(quad=floor(-1, 5, -1, 5, z=2.0))
    blender, ours = blender_and_numba(tmp_path, receivers=[full_quad()], occluders=stack + [roof_slab],
                                     suns=[sun(UP, strength=3.0)])
    assert_parity(blender, ours)
    assert (ours.irradiance[0] > 0) == lit == (blender['irradiance'][0] > 0)


@pytest.mark.blender
def test_cutout_ray_passes(tmp_path):
    for case, lit in [('cutout_hole', True), ('cutout_filled', False)]:
        blender, ours = blender_and_numba(tmp_path / case, **CASES[case])
        assert_parity(blender, ours)
        assert (ours.irradiance[0] > 0) == lit


@pytest.mark.blender
@pytest.mark.parametrize('tint,passed', [((0, 0, 0), 0.0), ((1, 1, 1), 1.0), ((0.5, 0.5, 0.5), 0.5)])
def test_sun_shadow_ray_takes_the_pane_tint(tmp_path, tint, passed):
    glass = dict(quad=floor(-1, 5, -1, 5, z=1.0), casts=False, tint=tint)
    blender, ours = blender_and_numba(tmp_path, receivers=[full_quad(), glass], suns=[sun(UP, strength=3.0)])
    assert_parity(blender, ours)
    assert ours.irradiance[0] == pytest.approx(3.0 * passed, abs=1e-3)
    assert ours.sun_lit.tolist() == [[True]]


@pytest.mark.blender
def test_sun_ray_starts_in_front_of_a_slightly_overhanging_face(tmp_path):
    blender, ours = blender_and_numba(tmp_path, **CASES['overhanging_slab'])
    assert_parity(blender, ours)
    assert ours.irradiance[0] == pytest.approx(3.0 * 2 ** -0.5, rel=1e-3)


@pytest.mark.blender
def test_emit_only_occluder_is_seen_by_the_rays_that_gather_light(tmp_path):
    for up in (False, True):
        glow = dict(quad=floor(0, 4, 0, 4, z=1.0, up=up), casts=False, emission=(4, 4, 4))
        blender, ours = blender_and_numba(tmp_path / str(up), receivers=[full_quad()], occluders=[glow],
                                         suns=[sun(UP, strength=3.0)])
        assert_parity(blender, ours)
        assert ours.sun_lit.tolist() == [[True]]
        assert (ours.irradiance[0] > 3) == (not up)
        if up:
            assert ours.irradiance[0] == pytest.approx(3.0, rel=1e-3)


@pytest.mark.blender
def test_emission_is_counted_once(tmp_path):
    diffuse, emission = 2.0, 4.0
    blender, ours = blender_and_numba(tmp_path, receivers=[roof(0, (emission,) * 3)], query=((0, 0, 0),),
                                     cls=1, pages=[np.full((16, 16, 3), diffuse, np.float32)],
                                     suns=[sun((0, 0, -1), strength=3.0)])
    assert_parity(blender, ours)
    assert ours.irradiance[0] == pytest.approx(np.pi * (0.5 * diffuse + emission), rel=0.02)


@pytest.mark.blender
def test_zero_normal_query_casts_nothing(tmp_path):
    blender, ours = blender_and_numba(tmp_path, **CASES['zero_normal'])
    assert_parity(blender, ours)
    assert np.isfinite(ours.vector).all() and np.isfinite(ours.irradiance).all()
    assert ours.irradiance[0] == 0 and not ours.vector[0].any()
    assert ours.irradiance[1] > 3.0


@pytest.mark.blender
def test_no_occluders(tmp_path):
    blender, ours = blender_and_numba(tmp_path, receivers=[full_quad(), wall_facing_minus_x(emission=(4, 4, 4))],
                                     suns=[sun(UP, strength=3.0)])
    assert_parity(blender, ours)
    assert ours.irradiance[0] > 3 and ours.sun_lit.tolist() == [[True]]


@pytest.mark.blender
def test_no_queries(tmp_path):
    blender, ours = blender_and_numba(tmp_path, receivers=edge_receivers(), suns=[sun((0.8, 0, 0.6))],
                                     query=np.empty((0, 3)), normals=np.empty((0, 3)))
    assert_parity(blender, ours)
    assert ours.vector.shape == (0, 3) and ours.irradiance.shape == (0,) and ours.sun_lit.shape == (0, 1)
    assert ours.rays > 0 and ours.rays == ours.edge_sun_lit.size


@pytest.mark.blender
def test_no_suns(tmp_path):
    blender, ours = blender_and_numba(tmp_path, receivers=[full_quad()])
    assert_parity(blender, ours)
    assert ours.sun_lit.shape == (1, 0) and ours.rays == 64


@pytest.mark.blender
def test_seed_turns_the_rays(tmp_path):
    blender, one = blender_and_numba(tmp_path / 'one', **CASES['seed_1'])
    assert_parity(blender, one)
    blender, zero = blender_and_numba(tmp_path / 'zero', **dict(CASES['seed_1'], seed=0))
    assert_parity(blender, zero)
    assert not np.array_equal(one.vector, zero.vector)


def test_shadow_ray_stops_at_its_limit(tmp_path):
    scene = build([full_quad()], [dict(quad=floor(-1, 5, -1, 5, z=5))])
    write_job(tmp_path, scene, [(16, 16)], [], {}, S, 'all')
    np.savez(tmp_path / 'pages.npz', page_0=np.zeros((16, 16, 3), np.float32))
    cs, bvh = load_cpu_scene(tmp_path)
    carried, escaped = np.empty((2, 3)), np.empty(2, bool)
    vertex_cpu._shadow(cs.arrays(), (bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order),
                       np.float32([[-0.5, 2, 0], [-0.5, 2, 0]]), np.float32([[0, 0, 1]] * 2),
                       np.float64([4, 6]), carried, escaped)
    np.testing.assert_array_equal(escaped, [True, False])
    np.testing.assert_array_equal(carried, 1)


@pytest.mark.blender
def test_results_do_not_depend_on_thread_count(tmp_path):
    blender = run_driver(tmp_path, **textured_room())
    pages = saved_pages(tmp_path)
    default = numba.get_num_threads()
    assert default > 1
    try:
        numba.set_num_threads(1)
        one = vertex_cpu.load_and_run(tmp_path, pages)
        numba.set_num_threads(default)
        many = vertex_cpu.load_and_run(tmp_path, pages)
    finally:
        numba.set_num_threads(default)
    assert all(np.array_equal(getattr(one, f), getattr(many, f)) for f in
               ('vector', 'irradiance', 'sun_lit', 'edge_sun_lit')) and one.rays == many.rays
    assert_parity(blender, many)


@pytest.mark.blender
def test_a_small_chunk_gives_the_same_result(tmp_path, monkeypatch):
    blender, whole = blender_and_numba(tmp_path, **textured_room())
    assert_parity(blender, whole)
    monkeypatch.setattr(vertex_cpu, 'CHUNK_RAYS', 256)
    split = vertex_cpu.load_and_run(tmp_path, saved_pages(tmp_path))
    assert all(np.array_equal(getattr(split, f), getattr(whole, f)) for f in
               ('vector', 'irradiance', 'sun_lit', 'edge_sun_lit')) and split.rays == whole.rays


def test_tool_mode_returns_empty_texel_rgb(tmp_path):
    pages = write_bare_scene(tmp_path)
    result = vertex_cpu.load_and_run(tmp_path, pages)
    assert result.texel_rgb.shape == (0, 3) and result.texel_rgb.dtype == np.float32


@pytest.mark.parametrize('mode', ['enhanced', 'tool'])
def test_sun_and_edge_generation_is_bounded_and_bit_identical(tmp_path, monkeypatch, mode):
    pages = write_bare_scene(tmp_path, triangles=False)
    with np.load(tmp_path / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    count = 11
    scene.update(query_pos=np.zeros((count, 3), np.float32),
                 query_normal=np.tile(np.float32(UP), (count, 1)), query_class=np.zeros(count, np.int16),
                 query_ignore_normals=np.arange(count) % 2 == 0,
                 edge_pos=np.zeros((count, 3), np.float32), edge_normal=np.tile(np.float32(UP), (count, 1)),
                 edge_class=np.zeros(count, np.int16), edge_active=np.ones((count, 3), bool))
    if mode == 'enhanced':
        scene.update(texel_pos=scene['query_pos'], texel_normal=scene['query_normal'],
                     texel_class=scene['query_class'], texel_page=np.zeros(count, np.int16),
                     texel_xy=np.zeros((count, 2), np.int32))
    job = json.loads((tmp_path / 'job.json').read_text())
    job.update(ignore_normals_mode=mode, cpu_vertex_rays=1, cpu_samples=9, cpu_texel_samples=1,
               bounces=0, suns=[dict(direction=UP, classes=[0], color=[1., .7, .3], strength=s)
                               for s in (1., .1234567, .7654321)])
    def unbatched(scene, job, sc, bv, usable=None, n=None, edge=False):
        if edge:
            query, index, origins, dirs = vertex_cpu.edge_rays(scene, job)
            cosine = None
        else:
            query, index, origins, dirs, cosine = vertex_cpu.sun_rays(scene, job, usable, n)
        carried, escaped = np.empty((len(origins), 3)), np.empty(len(origins), bool)
        vertex_cpu._shadow(sc, bv, origins.astype(np.float32), dirs.astype(np.float32),
                           np.full(len(origins), vertex_cpu.RAY_LENGTH), carried, escaped)
        for k in range(len(job['suns'])):
            rows = np.flatnonzero(index == k)
            yield k, query[rows], dirs[rows], None if edge else cosine[rows], carried[rows], escaped[rows]

    with monkeypatch.context() as oracle:
        oracle.setattr(vertex_cpu, '_shadow_batches', unbatched)
        whole = vertex_cpu.run_vertex_pass(scene, job, [], pages)
    batches = {'sun': [], 'edge': [], 'shadow': []}

    def record(kind, fn):
        def call(*args):
            count = len(args[2]) if kind == 'shadow' else len(args[0]['query_pos' if kind == 'sun' else 'edge_pos'])
            batches[kind].append(count)
            assert count <= 4, f'{kind} generated {count} rays before batching'
            if kind != 'shadow':
                assert len(args[1]['suns']) == 1, 'batch before expanding query x sun'
            return fn(*args)
        return call

    monkeypatch.setattr(vertex_cpu, 'CHUNK_RAYS', 4)
    for kind, name in [('sun', 'sun_rays'), ('edge', 'edge_rays'), ('shadow', '_shadow')]:
        monkeypatch.setattr(vertex_cpu, name, record(kind, getattr(vertex_cpu, name)))
    split = vertex_cpu.run_vertex_pass(scene, job, [], pages)
    assert all(len(batches[kind]) > 3 for kind in batches)
    for field in ('vector', 'irradiance', 'sun_lit', 'edge_sun_lit', 'texel_rgb', 'sun_throughput'):
        want, got = getattr(whole, field), getattr(split, field)
        assert got is None if want is None else np.array_equal(got, want), field
    assert split.rays == whole.rays
    assert split.sun_lit.all() and split.edge_sun_lit.all()


@pytest.mark.parametrize('length', [0, 3])
def test_cpu_scene_rejects_mismatched_flagged_pages(tmp_path, length):
    pages = write_bare_scene(tmp_path)
    with np.load(tmp_path / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    job = json.loads((tmp_path / 'job.json').read_text())
    with pytest.raises(ValueError, match='flagged_pages.*pages'):
        vertex_cpu.cpu_scene(scene, job, [], pages, [None] * length)


def test_shadow_generation_at_the_largest_enhanced_page_size(monkeypatch):
    count = 1024 * 1024 * 9
    scene = dict(query_pos=np.broadcast_to(np.float32([0, 0, 0]), (count, 3)),
                 query_normal=np.broadcast_to(np.float32(UP), (count, 3)),
                 query_class=np.broadcast_to(np.int16(0), (count,)),
                 query_ignore_normals=np.broadcast_to(True, (count,)))
    job = dict(suns=[dict(direction=UP, classes=[0], color=[1, 1, 1], strength=1)] * 16)

    def clear(sc, bv, origins, dirs, limits, carried, escaped):
        carried[:] = 1
        escaped[:] = True

    monkeypatch.setattr(vertex_cpu, '_shadow', clear)
    usable = np.broadcast_to(True, (count,))
    total, batches = 0, 0
    for k, query, dirs, cosine, carried, escaped in vertex_cpu._shadow_batches(
            scene, job, (), (), usable, scene['query_normal']):
        assert len(query) <= vertex_cpu.CHUNK_RAYS
        assert len(query) == len(dirs) == len(carried) == len(escaped)
        total += len(query)
        batches += 1
        del query, dirs, cosine, carried, escaped
    assert total == 150994944 and batches == 80


@pytest.mark.blender
@pytest.mark.parametrize('kind', ['grazing', 'behind', 'pane', 'vertex', 'vertex_back', 'edge',
                                  'emission_budget', 'ordinary_budget'])
def test_numba_matches_blender_pass_on_texel_scenes(tmp_path, kind):
    blender, ours = blender_and_numba(tmp_path, **texel_scene(kind))
    np.testing.assert_allclose(ours.texel_rgb, blender['texel_rgb'], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ours.irradiance, blender['irradiance'], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ours.vector, blender['vector'], rtol=1e-5, atol=1e-6)
    assert ours.rays == int(blender['rays'])
    if kind == 'vertex_back':
        assert ours.irradiance[0] == pytest.approx(3, rel=.05)
    if kind == 'edge':
        assert ours.rays == 2 * 512
        np.testing.assert_allclose(ours.texel_rgb, .105, rtol=.3)
    if kind in ('emission_budget', 'ordinary_budget'):
        np.testing.assert_allclose(ours.texel_rgb, 2. if kind == 'emission_budget' else 1., rtol=.05)


@pytest.mark.blender
@pytest.mark.parametrize('bounces', [0, 1, 2])
def test_flagged_stage_budget_matches(tmp_path, bounces):
    blender, ours = blender_and_numba(tmp_path, **texel_scene('flagged_bounce'), bounces=bounces)
    assert_flagged_bounce(ours.texel_rgb, bounces)
    np.testing.assert_allclose(ours.texel_rgb, blender['texel_rgb'], rtol=1e-5, atol=1e-6)
    assert ours.rays == int(blender['rays'])


@pytest.mark.blender
@pytest.mark.parametrize('bounces', [0, 1])
def test_direct_emission_takes_no_reflected_stage(tmp_path, bounces):
    blender, ours = blender_and_numba(tmp_path, **texel_scene('emission_relay'), bounces=bounces)
    assert_emission_relay(ours.texel_rgb, bounces)
    np.testing.assert_allclose(ours.texel_rgb, blender['texel_rgb'], rtol=1e-5, atol=1e-6)
    assert ours.rays == int(blender['rays'])


@pytest.mark.blender
@pytest.mark.parametrize('kind', ['mixed_dark_flagged', 'mixed_bright_flagged', 'mixed_delta', 'mixed_ordinary'])
def test_mixed_texel_sources_match(tmp_path, kind):
    blender, ours = blender_and_numba(tmp_path, **mixed_scene(kind))
    assert_mixed(kind, ours.texel_rgb)
    np.testing.assert_allclose(ours.texel_rgb, blender['texel_rgb'], rtol=1e-5, atol=1e-6)
    assert ours.rays == int(blender['rays'])


@pytest.mark.blender
def test_ignore_normals_enhanced_mode_matches(tmp_path):
    scene = textured_room()
    scene['receivers'][0]['ignore_normals'] = True
    scene.update(mode='enhanced', ignore_normals=[True] * 36 + [False] * 28)
    blender, ours = blender_and_numba(tmp_path, **scene)
    np.testing.assert_allclose(ours.texel_rgb, blender['texel_rgb'], rtol=1e-5, atol=1e-6)
    rel = np.abs(ours.irradiance - blender['irradiance']) / np.maximum(blender['irradiance'], 1e-6)
    assert blender['irradiance'].min() > 0
    assert np.median(rel) <= 1e-4 and rel.max() <= 1e-2
    assert ours.rays == int(blender['rays']) and np.array_equal(ours.sun_lit, blender['sun_lit'])
    assert ours.sun_lit.any()
    assert ours.sun_throughput is None


def test_texel_gather_at_the_stock_query_limit_uses_cpu_budgets(tmp_path, monkeypatch):
    pages = write_bare_scene(tmp_path, triangles=False)
    with np.load(tmp_path / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    count = 393216
    scene.update(texel_pos=np.zeros((count, 3), np.float32),
                 texel_normal=np.tile(np.float32(UP), (count, 1)), texel_class=np.zeros(count, np.int16),
                 texel_page=np.zeros(count, np.int16), texel_xy=np.zeros((count, 2), np.int32),
                 query_pos=np.empty((0, 3), np.float32), query_normal=np.empty((0, 3), np.float32),
                 query_class=np.empty(0, np.int16), query_ignore_normals=np.empty(0, bool))
    scene['texel_normal'][-1] = 0
    job = json.loads((tmp_path / 'job.json').read_text())
    job.update(ignore_normals_mode='enhanced', cpu_samples=72, cpu_texel_samples=0,
               samples=144, texel_samples=512, bounces=0, suns=[])
    chunks = []
    gather = vertex_cpu._gather

    def recording(*args):
        chunks.append(args[2].shape[:2])
        return gather(*args)

    monkeypatch.setattr(vertex_cpu, '_gather', recording)
    result = vertex_cpu.run_vertex_pass(scene, job, [], pages)
    assert result.rays == (count - 1) * 8
    assert len(chunks) == 2 and sum(m for m, _ in chunks) == count - 1
    assert all(rays == 8 and m * rays <= vertex_cpu.CHUNK_RAYS for m, rays in chunks)
    assert result.texel_rgb.shape == (count, 3) and result.texel_rgb.dtype == np.float32
    np.testing.assert_array_equal(result.texel_rgb, 0)


@pytest.mark.blender
def test_texel_stages_are_chunk_and_thread_invariant(tmp_path, monkeypatch):
    blender, whole = blender_and_numba(tmp_path, **texel_scene('flagged_bounce'), bounces=2)
    monkeypatch.setattr(vertex_cpu, 'CHUNK_RAYS', 4096)
    default = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        split = vertex_cpu.load_and_run(tmp_path, saved_pages(tmp_path))
    finally:
        numba.set_num_threads(default)
    for field in ('texel_rgb', 'vector', 'irradiance', 'sun_lit', 'edge_sun_lit'):
        np.testing.assert_array_equal(getattr(split, field), getattr(whole, field))
    assert split.rays == whole.rays == int(blender['rays'])


@pytest.mark.parametrize('mixed', [False, True])
def test_pass_at_the_stock_query_limit(tmp_path, monkeypatch, mixed):
    pages = write_bare_scene(tmp_path, triangles=False)
    with np.load(tmp_path / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    count = 393216
    scene.update(query_pos=np.zeros((count, 3), np.float32),
                 query_normal=np.tile(np.float32(UP), (count, 1)), query_class=np.zeros(count, np.int16),
                 query_ignore_normals=(np.arange(count) % 2 == 0) if mixed else np.zeros(count, bool))
    job = json.loads((tmp_path / 'job.json').read_text())
    job.update(cpu_vertex_rays=8, suns=[], ignore_normals_mode='tool')
    job.pop('seed', None)
    chunks = []
    gather = vertex_cpu._gather

    def recording(*args):
        chunks.append(args[2].shape[:2])
        return gather(*args)

    monkeypatch.setattr(vertex_cpu, '_gather', recording)
    result = vertex_cpu.run_vertex_pass(scene, job, [], pages)
    assert result.rays == count * 8 and result.vector.shape == (count, 3)
    assert len(chunks) == 2 and sum(m for m, _ in chunks) == count
    assert max(m * rays for m, rays in chunks) <= vertex_cpu.CHUNK_RAYS
    np.testing.assert_allclose(result.irradiance, np.pi * .2, rtol=1e-6)
    np.testing.assert_array_equal(result.vector[:, :2], 0)
    np.testing.assert_array_equal(result.vector[:, 2], result.irradiance)
    if mixed:
        assert result.sun_throughput.shape == (count, 0, 3)
        assert result.sun_throughput.dtype == np.float32
    else:
        assert result.sun_throughput is None


def test_gather_rgb_switches_and_inactive_query(tmp_path):
    scene = build([roof(0, (1, 2, 3))])
    write_job(tmp_path, scene, [(16, 16)], [], {'0': [.2, .4, .6]}, S, 'all')
    np.savez(tmp_path / 'pages.npz', page_0=np.full((16, 16, 3), 2, np.float32))
    cs, bvh = load_cpu_scene(tmp_path)
    bv = (bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order)
    origins = np.zeros((2, 1, 3), np.float32)
    dirs = np.tile(np.float32([0, 0, 1]), (2, 1, 1))
    for ambient, pages, emission, want in [(True, True, True, [2.1, 3.2, 4.3]),
                                          (False, True, False, [1, 1, 1]),
                                          (True, False, False, [.1, .2, .3]),
                                          (False, False, True, [1, 2, 3])]:
        sums = np.full((2, 4), np.nan)
        vertex_cpu._gather(cs.arrays(), bv, origins, dirs, dirs.astype(np.float64), np.zeros(2, np.int64),
                           np.array([True, False]), True, ambient, pages, emission, sums)
        np.testing.assert_allclose(sums[0, :3], want, rtol=1e-6)
        np.testing.assert_array_equal(sums[1], 0)


def write_cpu_child_scene(work):
    scene = build([full_quad()], [roof(0)])
    write_job(work, scene, [], [sun(UP, strength=3.0)], {'0': [.2] * 3},
              dataclasses.replace(S, vertex_rays=64), 'all')
    return []


def assert_child_equal(work, pages, timeout):
    expected = vertex_cpu.load_and_run(work, pages)
    result, timing = vertex_cpu.run_in_child(work, pages, timeout)
    for field in ('vector', 'irradiance', 'sun_lit', 'edge_sun_lit', 'texel_rgb', 'sun_throughput'):
        want, got = getattr(expected, field), getattr(result, field)
        assert got is None if want is None else np.array_equal(got, want)
    assert result.rays == expected.rays
    assert set(timing) == {'compile', 'trace', 'threading_layer', 'threads'}
    assert timing['compile'] >= 0 and timing['trace'] >= 0
    assert timing['threading_layer'] in ('workqueue', 'omp', 'tbb') and timing['threads'] >= 1
    with np.load(work / 'cpu_out.npz', allow_pickle=False) as data:
        for field in ('rays', 'threads'):
            assert data[field].dtype == np.int64 and data[field].shape == ()
        for field in ('seconds_compile', 'seconds_trace'):
            assert data[field].dtype == np.float64 and data[field].shape == ()
        assert data['threading_layer'].dtype.kind == 'U'
    return result


@pytest.mark.blender
@pytest.mark.parametrize('case', ['seed_1', 'sun_throughput', 'enhanced'])
def test_cpu_child_matches_the_in_process_pass(tmp_path, case):
    scene = texel_scene('vertex') if case == 'enhanced' else CASES[case]
    blender, direct = blender_and_numba(tmp_path, **scene)
    assert_parity(blender, direct)
    assert_child_equal(tmp_path, saved_pages(tmp_path), 300)


def cpu_child_code(monkeypatch, code):
    monkeypatch.setattr(frozen, 'child_command', lambda *args: [sys.executable, '-c', code])


def test_a_dying_cpu_child_becomes_a_cpu_pass_error(tmp_path, monkeypatch):
    cpu_child_code(monkeypatch, 'import sys; print("driver fault", file=sys.stderr); sys.exit(3)')
    with pytest.raises(vertex_cpu.CpuPassError, match=r'vertex: process died \(exit 3\):.*driver fault'):
        vertex_cpu.run_in_child(tmp_path, [np.zeros((4, 4, 3), np.float32)], 60)


@pytest.mark.parametrize('code', [
    'pass',
    'from pathlib import Path; Path("cpu_out.npz").write_bytes(b"short")',
    'import numpy as np; np.savez("cpu_out.npz", vector=np.zeros((1,3)))',
])
def test_missing_or_short_cpu_output_is_an_error(tmp_path, monkeypatch, code):
    cpu_child_code(monkeypatch, code)
    with pytest.raises(vertex_cpu.CpuPassError, match=r'vertex: process died \(exit 0\)'):
        vertex_cpu.run_in_child(tmp_path, [], 60)


def test_child_timeout_discards_stale_output(tmp_path, monkeypatch):
    (tmp_path / 'cpu_out.npz').write_bytes(b'stale')
    cpu_child_code(monkeypatch, 'import time; time.sleep(60)')
    with pytest.raises(vertex_cpu.CpuPassError, match='vertex: timed out after 0.2 s'):
        vertex_cpu.run_in_child(tmp_path, [], .2)
    assert not (tmp_path / 'cpu_out.npz').exists()


def test_cpu_child_launch_failure_is_a_cpu_pass_error(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, 'child_command', lambda *args: [str(tmp_path / 'absent.exe')])
    with pytest.raises(vertex_cpu.CpuPassError, match=r'vertex: process died \(exit unavailable\):'):
        vertex_cpu.run_in_child(tmp_path, [], 60)


def test_cpu_child_starts_without_a_window(tmp_path, monkeypatch):
    real = subprocess.run
    seen = {}

    def spy(args, **kwargs):
        seen.update(kwargs, args=args)
        return real(args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', spy)
    with pytest.raises(vertex_cpu.CpuPassError):
        vertex_cpu.run_in_child(tmp_path, [], 60)
    assert seen['creationflags'] == frozen.NO_WINDOW
    assert seen['args'] == frozen.child_command(frozen.CPU_VERTEX_FLAG, str(tmp_path.resolve()))
    assert seen['cwd'] == tmp_path.resolve() and 'env' not in seen
    assert seen['stdin'] == subprocess.DEVNULL and seen['capture_output'] is True and seen['timeout'] == 60


def test_child_without_a_timeout_runs(tmp_path):
    assert_child_equal(tmp_path, write_cpu_child_scene(tmp_path), None)


def test_cpu_child_runs_in_a_folder_with_spaces_and_accents(tmp_path):
    work = tmp_path / 'my bakes é'
    assert_child_equal(work, write_cpu_child_scene(work), 300)


def test_unwritable_cache_dir_still_runs_the_pass(tmp_path, monkeypatch):
    cache = tmp_path / 'not_a_directory'
    cache.write_text('occupied')
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(cache))
    assert_child_equal(tmp_path, write_cpu_child_scene(tmp_path), 300)
    assert cache.read_text() == 'occupied'


@pytest.mark.parametrize('field', ['vector', 'irradiance', 'sun_lit', 'edge_sun_lit', 'texel_rgb',
                                  'missing_throughput', 'short_throughput', 'unexpected_throughput'])
def test_cpu_child_refuses_consistently_short_or_unexpected_arrays(tmp_path, monkeypatch, field):
    np.savez(tmp_path / 'scene.npz', query_pos=np.zeros((2, 3)), edge_pos=np.zeros((3, 3)),
             texel_pos=np.zeros((4, 3)), query_ignore_normals=np.array([True, False]))
    mode = 'tool' if 'throughput' in field and field != 'unexpected_throughput' else 'enhanced'
    (tmp_path / 'job.json').write_text(json.dumps({'scene': 'scene.npz', 'suns': [{}], 'ignore_normals_mode': mode}))
    arrays = dict(vector=np.zeros((2, 3)), irradiance=np.zeros(2), sun_lit=np.zeros((2, 1), bool),
                  edge_sun_lit=np.zeros((3, 1), bool), texel_rgb=np.zeros((4, 3)), rays=np.int64(0),
                  seconds_compile=0., seconds_trace=0., threading_layer='workqueue', threads=np.int64(1))
    if field == 'short_throughput':
        arrays['vertex_sun_throughput'] = np.zeros((1, 1, 3))
    elif field == 'unexpected_throughput':
        arrays['vertex_sun_throughput'] = np.zeros((2, 1, 3))
    elif field != 'missing_throughput':
        arrays[field] = arrays[field][:-1]

    def child(*args, **kwargs):
        np.savez(tmp_path / 'cpu_out.npz', **arrays)
        return subprocess.CompletedProcess(args[0], 0, b'', b'')

    monkeypatch.setattr(subprocess, 'run', child)
    with pytest.raises(vertex_cpu.CpuPassError, match=r'vertex: process died \(exit 0\).*\nshort vertex arrays'):
        vertex_cpu.run_in_child(tmp_path, [], 60)


def test_cpu_child_exchange_at_the_stock_query_limit(tmp_path, monkeypatch):
    q = 393216
    np.savez(tmp_path / 'scene.npz', query_pos=np.zeros((q, 3), np.float32), edge_pos=np.zeros((0, 3)),
             query_ignore_normals=np.ones(q, bool))
    (tmp_path / 'job.json').write_text(json.dumps({'scene': 'scene.npz', 'suns': [{}, {}]}))
    code = ('import numpy as np; '
            f'q={q}; '
            'np.savez("cpu_out.npz", vector=np.full((q,3),.25,np.float32), '
            'irradiance=np.ones(q,np.float32), sun_lit=np.ones((q,2),bool), '
            'vertex_sun_throughput=np.full((q,2,3),.125,np.float32), '
            'texel_rgb=np.zeros((0,3),np.float32), seconds_compile=np.float64(.5), '
            'seconds_trace=np.float64(1.5), threading_layer="workqueue", threads=np.int64(2), '
            'edge_sun_lit=np.zeros((0,2),bool), rays=np.int64(q*4096))')
    cpu_child_code(monkeypatch, code)
    pages = [np.ones((2, 2, 3), np.float32)]
    result, timing = vertex_cpu.run_in_child(tmp_path, pages, 60)
    assert result.vector.shape == (393216, 3) and result.rays == 1610612736
    np.testing.assert_array_equal(result.vector, .25)
    assert result.sun_throughput.shape == (393216, 2, 3)
    np.testing.assert_array_equal(result.sun_throughput, .125)
    assert timing == {'compile': .5, 'trace': 1.5, 'threading_layer': 'workqueue', 'threads': 2}
    with np.load(tmp_path / 'diffuse.npz') as data:
        np.testing.assert_array_equal(data['diffuse_0'], pages[0])


def test_child_warm_up_compiles_every_kernel_on_one_triangle(monkeypatch):
    # cached entry points contain their compiled callees without populating the callees' Python dispatchers
    kernels = [vertex_cpu._gather, vertex_cpu._shadow, vertex_cpu._probe_rays, vertex_cpu._closest_hits,
               vertex_cpu.brute_force_hits, vertex_cpu._probe_kernel]
    seen = []
    build = vertex_cpu.build_bvh

    def recording(corners):
        seen.append(corners.shape)
        return build(corners)

    monkeypatch.setattr(vertex_cpu, 'build_bvh', recording)
    vertex_cpu.warm_up()
    assert seen and all(shape == (1, 3, 3) for shape in seen)
    assert not [kernel.py_func.__name__ for kernel in kernels if not kernel.nopython_signatures]


# ---------- scenery lights (the lamp_case scenes are shared with the GPU and Blender passes' lamp tests)

PI = np.pi
FLOOR = floor(-20, 20, -20, 20)
# off-centre corners keep a ray through the origin from running along a quad's diagonal
ROOF = floor(-3, 4, -3.5, 3, z=1)


def lamp(position, rgb=(PI, PI, PI), kind=2, axis=(0, 0, -1), reach=1e4, cos_cutoff=-1.0, cos_falloff=-1.0, owner=-1):
    return SceneLight(np.float64(position), np.float64(axis), np.float64(rgb), kind, np.float32(reach),
                      np.float32(cos_cutoff), np.float32(cos_falloff), owner, 'lamp')


def lamp_case(name, **extra):
    """The scenes of the lamp tests, as the keywords of write_lamp_case: occluders are (quad, owner, extras) and
    queries (position, normal, ignore_normals)."""
    light = lamp((0, 0, 2))
    box = [floor(-.5, .7, -.4, .5, z=1.5), floor(-.5, .7, -.4, .5, z=2.5, up=False),
           quad([(-.5, -.4, 1.5), (.7, -.4, 1.5), (.7, -.4, 2.5), (-.5, -.4, 2.5)], (0, -1, 0)),
           quad([(-.5, .5, 1.5), (-.5, .5, 2.5), (.7, .5, 2.5), (.7, .5, 1.5)], (0, 1, 0)),
           wall_x(-.5, -.4, .5, 1.5, 2.5), wall_x(.7, -.4, .5, 1.5, 2.5)]
    cases = {
        'plane': dict(queries=[((x, 0, 0), UP, False) for x in (0, 1, 3)], lights=[light]),
        'wall': dict(occluders=[(ROOF, -1, {})], queries=[((0, 0, 0), UP, False)], lights=[light]),
        'glass': dict(occluders=[(ROOF, -1, dict(casts=False, tint=(1, 0, .5)))],
                      queries=[((0, 0, 0), UP, False)], lights=[light]),
        'cutout': dict(occluders=[(ROOF, -1, dict(cutout=True, texture=0))], queries=[((0, 0, 0), UP, False)],
                       lights=[light]),
        'own_box': dict(occluders=[(q, 4, {}) for q in box], queries=[((0, 0, 0), UP, False)],
                        lights=[lamp((0, 0, 2), owner=4)]),
        'flagged': dict(queries=[((0, 0, 0), UP, True), ((0, 0, 0), (0, 0, -1), True), ((0, 0, 0), (0, 0, -1), False)],
                        lights=[light]),
        'spot': dict(queries=[((np.tan(a), 0, 0), UP, False) for a in np.arccos([1.0, .9, .75, .55, .4])],
                     lights=[lamp((0, 0, 1), kind=3, cos_cutoff=.5, cos_falloff=.8)]),
        'light_on_a_wall': dict(occluders=[(wall_x(2, -3, 3, -1, 3.5), 3, {})], queries=[((0, 0, 0), UP, False)],
                                lights=[lamp((2.01, 0, 1), owner=3)]),
    }
    return {**cases[name], **extra}


def write_lamp_case(work, receivers=(FLOOR,), occluders=(), queries=(), lights=(), mode='tool', textures=None,
                    bounces=1, albedo_fallback=0.0, flagged_receivers=(), pages=1):
    """Writes the job and the pages, and returns the pages. The receivers listed by index are ignore-normals ones."""
    scene = build([dict(quad=r, ignore_normals=i in flagged_receivers, page=min(i, pages - 1))
                   for i, r in enumerate(receivers)],
                  [dict(quad=q, **extras) for q, _, extras in occluders])
    scene = dataclasses.replace(
        scene, occ_tri_owner=np.repeat(np.int32([owner for _, owner, _ in occluders]), 2),
        query_pos=np.float32([q[0] for q in queries]).reshape(-1, 3),
        query_normal=np.float32([q[1] for q in queries]).reshape(-1, 3), query_class=np.zeros(len(queries), np.int16),
        query_ignore_normals=np.bool_([q[2] for q in queries]))
    settings = dataclasses.replace(S, vertex_rays=1, albedo_fallback=albedo_fallback, ignore_normals_mode=mode,
                                   bounces=bounces, w_light=1.0, samples=9 * 256, texel_samples=256)
    sizes = [(16, 16)] * pages
    write_job(work, scene, sizes, [], {}, settings, 'all', textures=textures, lights=list(lights))
    result = [np.zeros((h, w, 3), np.float32) for w, h in sizes]
    np.savez(Path(work) / 'pages.npz', **{f'page_{i}': p for i, p in enumerate(result)})
    return result


def run_lamp_case(work, name, **extra):
    case = lamp_case(name, **extra)
    textures = case.pop('textures', None)
    pages = write_lamp_case(work, textures=textures, **case)
    return vertex_cpu.load_and_run(work, pages)


def test_point_light_on_a_plane_gives_cosine_over_distance_squared(tmp_path):
    result = run_lamp_case(tmp_path, 'plane')
    want = np.float64([PI * 2 / (4 + x ** 2) ** 1.5 for x in (0, 1, 3)])
    np.testing.assert_allclose(result.irradiance, want, rtol=1e-5)
    toward = np.float64([[0, 0, 2], [-1, 0, 2], [-3, 0, 2]])
    toward /= np.linalg.norm(toward, axis=1)[:, None]
    np.testing.assert_allclose(result.vector, want[:, None] * toward, rtol=1e-4, atol=1e-7)


def test_spot_light_lights_only_its_cone_with_tools_ramp(tmp_path):
    result = run_lamp_case(tmp_path, 'spot')
    cosines = np.float64([1.0, .9, .75, .55, .4])
    # a height of 1 makes 1 / d^2 the cosine squared
    want = PI * cosines ** 3 * np.float64([ramp(c, .5, .8) for c in cosines])
    np.testing.assert_allclose(result.irradiance, want, rtol=1e-4, atol=1e-7)
    assert result.irradiance[-1] == 0 and result.irradiance[-2] > 0


def test_flagged_receiver_takes_a_light_without_the_cosine_from_either_side(tmp_path):
    result = run_lamp_case(tmp_path, 'flagged')
    np.testing.assert_allclose(result.irradiance, [PI / 4, PI / 4, 0], atol=1e-6)


def test_point_light_is_shadowed_by_a_wall(tmp_path):
    assert run_lamp_case(tmp_path / 'wall', 'wall').irradiance[0] == 0
    assert run_lamp_case(tmp_path / 'open', 'plane').irradiance[0] > 0


def test_point_light_takes_the_glass_tint(tmp_path):
    result = run_lamp_case(tmp_path, 'glass')
    assert result.irradiance[0] == pytest.approx(PI / 4 * (np.float64([1, 0, .5]) @ vertex_cpu.LUMA), rel=1e-5)


@pytest.mark.parametrize('alpha,lit', [(0, True), (255, False)])
def test_cutout_hole_lets_lamp_light_through(tmp_path, alpha, lit):
    result = run_lamp_case(tmp_path, 'cutout', textures=[np.full((1, 1, 4), alpha, np.uint8)])
    assert (result.irradiance[0] > 0) == lit


def test_the_lamps_own_placement_does_not_block_it(tmp_path):
    assert run_lamp_case(tmp_path / 'own', 'own_box').irradiance[0] == pytest.approx(PI / 4, rel=1e-5)
    other = lamp_case('own_box')
    other['occluders'] = [(q, 5, extras) for q, _, extras in other['occluders']]
    pages = write_lamp_case(tmp_path / 'other', **other)
    assert vertex_cpu.load_and_run(tmp_path / 'other', pages).irradiance[0] == 0


def test_light_on_a_wall_is_not_put_out_by_the_wall(tmp_path):
    assert run_lamp_case(tmp_path / 'own', 'light_on_a_wall').irradiance[0] > 0
    pages = write_lamp_case(tmp_path / 'other', **lamp_case('light_on_a_wall', lights=[lamp((2.01, 0, 1), owner=4)]))
    assert vertex_cpu.load_and_run(tmp_path / 'other', pages).irradiance[0] == 0


@pytest.mark.parametrize('stack,lit', [(255, False), (300, True)])
def test_owner_skips_stop_at_the_skip_limit(tmp_path, stack, lit):
    # owner panes 0.002 apart with someone else's roof above them: the 256th skip ends the ray as a miss
    panes = [(floor(-1, 1, -1, 1, z=.1 + .002 * k), 0, {}) for k in range(stack)]
    pages = write_lamp_case(tmp_path, occluders=panes + [(floor(-1, 1, -1, 1, z=1.0), 1, {})],
                            queries=[((.3, .2, 0), UP, False)], lights=[lamp((.3, .2, 2), owner=0)])
    assert (vertex_cpu.load_and_run(tmp_path, pages).irradiance[0] > 0) == lit


def test_light_throughput_is_returned_for_flagged_queries(tmp_path):
    result = run_lamp_case(tmp_path / 'tool', 'glass', queries=[((0, 0, 0), UP, True), ((1, 0, 0), UP, False)])
    assert result.light_flat.shape == (2, 3) and result.light_flat.dtype == np.float32
    np.testing.assert_allclose(result.light_flat[0], np.float64([1, 0, .5]) / 4, rtol=1e-5)
    np.testing.assert_array_equal(result.light_flat[1], 0)
    assert run_lamp_case(tmp_path / 'ordinary', 'plane').light_flat is None
    pages = write_lamp_case(tmp_path / 'enhanced', mode='enhanced', **lamp_case('flagged'))
    assert vertex_cpu.load_and_run(tmp_path / 'enhanced', pages).light_flat is None


def test_enhanced_texel_gather_includes_lamps(tmp_path):
    # the lamp reaches only the upper sheet, which passes it on to the sheet below
    sheets = [floor(-20, 20, -20, 20, z=0), floor(-20, 20, -20, 20, z=-1)]
    kwargs = dict(receivers=sheets, flagged_receivers=(0, 1), pages=2, mode='enhanced', albedo_fallback=.5,
                  queries=[((0, 0, 0), (0, 0, 0), False)])
    lit = write_lamp_case(tmp_path / 'lit', lights=[lamp((0, 0, 2))], **kwargs)
    result = vertex_cpu.load_and_run(tmp_path / 'lit', lit)
    unlit = vertex_cpu.load_and_run(tmp_path / 'dark', write_lamp_case(tmp_path / 'dark', **kwargs))
    with np.load(tmp_path / 'lit' / 'scene.npz') as data:
        pos, page = data['texel_pos'].astype(np.float64), data['texel_page']
    d = np.linalg.norm(pos[page == 0] - [0, 0, 2], axis=1)
    np.testing.assert_allclose(result.texel_rgb[page == 0], np.tile(1 / d ** 2, (3, 1)).T, rtol=1e-4)
    assert result.texel_rgb[page == 1].max() > 0
    assert unlit.texel_rgb.max() == 0


def test_lamp_results_do_not_depend_on_thread_count_or_chunk(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    queries = [((x, y, 0), UP, i % 3 == 0) for i, (x, y) in enumerate(rng.uniform(-4, 4, (40, 2)))]
    lights = [lamp(rng.uniform(-3, 3, 3) + [0, 0, 2], owner=k) for k in range(4)]
    panes = [(floor(-1, 1, -1, 1, z=1), 9, dict(casts=False, tint=(.5, .7, .9)))]
    pages = write_lamp_case(tmp_path, occluders=panes, queries=queries, lights=lights)
    default = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = vertex_cpu.load_and_run(tmp_path, pages)
        numba.set_num_threads(default)
        many = vertex_cpu.load_and_run(tmp_path, pages)
        monkeypatch.setattr(vertex_cpu, 'CHUNK_RAYS', 7)
        split = vertex_cpu.load_and_run(tmp_path, pages)
    finally:
        numba.set_num_threads(default)
    assert one.irradiance.max() > 0 and one.light_flat.max() > 0
    for other in (many, split):
        for field in ('vector', 'irradiance', 'light_flat'):
            assert np.array_equal(getattr(one, field), getattr(other, field)), field
        assert one.rays == other.rays


def test_skip_minus_one_leaves_every_cast_as_it_was(tmp_path):
    # receivers carry owner -1, so only the skip >= 0 guard keeps a skip of -1 from passing through every receiver
    write_lamp_case(tmp_path, occluders=[(floor(-2, 2, -2, 2, z=1), 2, {})], queries=[((0, 0, 0), UP, False)])
    cs, bvh = load_cpu_scene(tmp_path)
    bv = (bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order)
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(300, 3)).astype(np.float32)
    origins = np.float32(rng.uniform(-1, 1, (300, 3)))
    limits = np.full(300, vertex_cpu.RAY_LENGTH)

    def escapes(skips):
        carried, escaped = np.empty((300, 3)), np.empty(300, bool)
        vertex_cpu._shadow(cs.arrays(), bv, origins, dirs, limits, carried, escaped, skips)
        return escaped

    plain = escapes(None)
    np.testing.assert_array_equal(escapes(np.full(300, -1, np.int64)), plain)
    np.testing.assert_array_equal(escapes(np.full(300, 7, np.int64)), plain)
    assert plain.any() and not plain.all()
    assert escapes(np.full(300, 2, np.int64)).sum() > plain.sum()


def test_skip_limit_copy_equals_vertex_gpu():
    assert vertex_cpu.SKIP_LIMIT == vertex_gpu.SKIP_LIMIT


def test_cpu_child_returns_the_lamp_throughput(tmp_path):
    pages = write_lamp_case(tmp_path, **lamp_case('glass', queries=[((0, 0, 0), UP, True), ((1, 0, 0), UP, False)]))
    expected = vertex_cpu.load_and_run(tmp_path, pages)
    result, _ = vertex_cpu.run_in_child(tmp_path, pages, 300)
    assert np.array_equal(result.light_flat, expected.light_flat) and result.light_flat[0].any()
    assert np.array_equal(result.irradiance, expected.irradiance)


def test_cpu_child_refuses_light_throughput_of_the_wrong_shape_or_without_flagged_queries(tmp_path, monkeypatch):
    np.savez(tmp_path / 'scene.npz', query_pos=np.zeros((2, 3)), edge_pos=np.zeros((0, 3)),
             query_ignore_normals=np.array([True, False]))
    (tmp_path / 'job.json').write_text(json.dumps({'scene': 'scene.npz', 'suns': [{}], 'ignore_normals_mode': 'tool'}))
    arrays = dict(vector=np.zeros((2, 3)), irradiance=np.zeros(2), sun_lit=np.zeros((2, 1), bool),
                  edge_sun_lit=np.zeros((0, 1), bool), texel_rgb=np.zeros((0, 3)), rays=np.int64(0),
                  vertex_sun_throughput=np.zeros((2, 1, 3)), seconds_compile=0., seconds_trace=0.,
                  threading_layer='workqueue', threads=np.int64(1))

    def child(*args, **kwargs):
        np.savez(tmp_path / 'cpu_out.npz', **arrays)
        return subprocess.CompletedProcess(args[0], 0, b'', b'')

    monkeypatch.setattr(subprocess, 'run', child)
    arrays['vertex_light_flat'] = np.zeros((2, 3))
    assert vertex_cpu.run_in_child(tmp_path, [], 60)[0].light_flat.shape == (2, 3)
    arrays['vertex_light_flat'] = np.zeros((1, 3))
    with pytest.raises(vertex_cpu.CpuPassError, match='short vertex arrays'):
        vertex_cpu.run_in_child(tmp_path, [], 60)


# ---------- scenery lights in the Blender pass

LAMP_CASES = ['plane', 'wall', 'glass', 'cutout', 'own_box', 'flagged', 'spot', 'light_on_a_wall']


def blender_lamp_pass(work):
    env = dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent))
    done = subprocess.run([str(LIGHTMAPPER_BLENDER), '-b', '--factory-startup', '--python-exit-code', '1',
                           '--python', DRIVER, '--', str(work)], capture_output=True, text=True, env=env,
                          stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    return np.load(work / 'vp.npz')


def assert_lamp_parity(blender, ours):
    assert_parity(blender, ours)
    np.testing.assert_allclose(ours.texel_rgb, blender['texel_rgb'], rtol=1e-4, atol=1e-6)
    if 'light_flat' in blender.files:
        np.testing.assert_allclose(ours.light_flat, blender['light_flat'], rtol=1e-4, atol=1e-6)
    else:
        assert ours.light_flat is None


@pytest.mark.blender
@pytest.mark.parametrize('name', LAMP_CASES)
def test_numba_matches_blender_pass_on_the_lamp_scenes(tmp_path, name):
    case = lamp_case(name)
    textures = [np.full((1, 1, 4), 0, np.uint8)] if name == 'cutout' else None
    pages = write_lamp_case(tmp_path, textures=textures, **case)
    ours = vertex_cpu.load_and_run(tmp_path, pages)
    blender = blender_lamp_pass(tmp_path)
    assert ours.irradiance.max() > 0 or name == 'wall'
    assert_lamp_parity(blender, ours)


@pytest.mark.blender
def test_numba_matches_blender_pass_on_the_lamp_throughput(tmp_path):
    case = lamp_case('glass', queries=[((0, 0, 0), UP, True), ((1, 0, 0), UP, False)])
    pages = write_lamp_case(tmp_path, **case)
    ours = vertex_cpu.load_and_run(tmp_path, pages)
    blender = blender_lamp_pass(tmp_path)
    assert ours.light_flat[0].max() > 0
    assert_lamp_parity(blender, ours)


@pytest.mark.blender
def test_numba_matches_blender_pass_on_the_owner_skip_limit(tmp_path):
    panes = [(floor(-1, 1, -1, 1, z=.1 + .002 * k), 0, {}) for k in range(300)]
    pages = write_lamp_case(tmp_path, occluders=panes + [(floor(-1, 1, -1, 1, z=1.0), 1, {})],
                            queries=[((.3, .2, 0), UP, False)], lights=[lamp((.3, .2, 2), owner=0)])
    ours = vertex_cpu.load_and_run(tmp_path, pages)
    assert ours.irradiance[0] > 0
    assert_lamp_parity(blender_lamp_pass(tmp_path), ours)


@pytest.mark.blender
def test_numba_matches_blender_pass_on_the_enhanced_lamp_texels(tmp_path):
    sheets = [floor(-20, 20, -20, 20, z=0), floor(-20, 20, -20, 20, z=-1)]
    pages = write_lamp_case(tmp_path, receivers=sheets, flagged_receivers=(0, 1), pages=2, mode='enhanced',
                            albedo_fallback=.5, queries=[((0, 0, 0), (0, 0, 0), False)],
                            lights=[lamp((0, 0, 2))])
    ours = vertex_cpu.load_and_run(tmp_path, pages)
    assert ours.texel_rgb.max() > 0
    assert_lamp_parity(blender_lamp_pass(tmp_path), ours)
