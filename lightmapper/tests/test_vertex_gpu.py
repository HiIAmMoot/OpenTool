"""The vertex pass's numpy side against bake.py's CPU pass: formulas, copied constants, and the ray lists in real
Blender (vertex_driver.py)."""

import ast
import json

import numpy as np
import pytest

from lightmapper import vertex_gpu
from lightmapper.sun_facing import sun_facing_queries
from lightmapper.vertex_gpu import (GOLDEN_FRACTION, SEED_SPIN, edge_rays, gather_sums, hemisphere, query_frames,
                                    sun_rays)
from conftest import LIGHTMAPPER_BLENDER
from test_blender_bake import UP, build, floor, quad, sun
from test_vertex_pass import run_driver


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
    suns = [sun(UP), sun((0, 0, -1)), sun((1, 0.2, 0.1))]
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
