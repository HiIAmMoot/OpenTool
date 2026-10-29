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

from lightmapper import entry, frozen, vertex_cpu
from lightmapper.blender_job import write_job
from test_blender_bake import UP, build, floor, full_quad, sun
from test_vertex_gpu import CASES, edge_receivers, textured_room, write_bare_scene
from test_vertex_pass import (S, assert_asymmetric_twin, assert_twin_light, asymmetric_twin_scene, roof,
                              run_driver, twin_scene, unpaired_back_face_scene, wall_facing_minus_x)


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
    names = ('RAY_LENGTH', 'RAY_EPSILON', 'PASS_LIMIT', 'LUMA', 'GOLDEN_FRACTION', 'SEED_SPIN')
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
    assert (blender['tri'] >= 0).any() or case in ('glass_tint_half', 'emit_only_back')
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
    assert vertex_cpu._barycentric(*np.float64([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.25, 0.25, 0]]))         == pytest.approx((0.5, 0.25, 0.25))


def saved_pages(work):
    with np.load(work / 'pages.npz') as data:
        return [data[f'page_{i}'] for i in range(len(data.files))]


def blender_and_numba(tmp_path, **kwargs):
    blender = run_driver(tmp_path, **kwargs)
    ours = vertex_cpu.load_and_run(tmp_path, saved_pages(tmp_path))
    metrics = {'rays': ours.rays}
    for field in ('irradiance', 'vector'):
        diff = np.abs(getattr(ours, field) - blender[field])
        rel = diff / np.maximum(np.abs(blender[field]), 1e-6)
        metrics[field] = {'max_abs': float(diff.max(initial=0)), 'max_rel': float(rel.max(initial=0)),
                          'median_rel': float(np.median(rel)) if rel.size else 0.0}
    metrics['sun_lit_equal'] = np.array_equal(ours.sun_lit, blender['sun_lit'])
    metrics['edge_sun_lit_equal'] = np.array_equal(ours.edge_sun_lit, blender['edge_sun_lit'])
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


@pytest.mark.parametrize('field', ['query_ignore_normals', 'texel_pos'])
def test_flagged_queries_are_refused_until_ported(tmp_path, field):
    pages = write_bare_scene(tmp_path)
    with np.load(tmp_path / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    scene[field] = np.ones(1, bool) if field == 'query_ignore_normals' else np.ones((1, 3), np.float32)
    job = json.loads((tmp_path / 'job.json').read_text())
    with pytest.raises(vertex_cpu.CpuPassError, match='ignore-normals queries are not in the compiled pass yet'):
        vertex_cpu.run_vertex_pass(scene, job, [], pages)


def test_ordinary_pass_at_the_stock_query_limit(tmp_path, monkeypatch):
    pages = write_bare_scene(tmp_path, triangles=False)
    with np.load(tmp_path / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    count = 393216
    scene.update(query_pos=np.zeros((count, 3), np.float32),
                 query_normal=np.tile(np.float32(UP), (count, 1)), query_class=np.zeros(count, np.int16),
                 query_ignore_normals=np.zeros(count, bool))
    job = json.loads((tmp_path / 'job.json').read_text())
    job.update(cpu_vertex_rays=8, suns=[])
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
