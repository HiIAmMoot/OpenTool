import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lightmapper import entry, frozen, vertex_cpu


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
