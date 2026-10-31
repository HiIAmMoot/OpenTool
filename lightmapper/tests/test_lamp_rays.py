import ast
from pathlib import Path

import numpy as np
import pytest

from lightmapper.formats.lights import ramp
from lightmapper.vertex_gpu import RAY_EPSILON, iter_lamp_rays


def lamp(position, kind=2, axis=(0, 0, -1), reach=100.0, cos_cutoff=0.5, cos_falloff=0.6, owner=-1):
    return {'position': position, 'axis': axis, 'strength': [1, 1, 1], 'kind': kind, 'reach': reach,
            'cos_cutoff': cos_cutoff, 'cos_falloff': cos_falloff, 'owner': owner, 'batch': 0}


def rays(points, lights, normals=None, flagged=None, usable=None, chunk=10 ** 9):
    points = np.float64(points)
    normals = np.tile([0.0, 0, 1], (len(points), 1)) if normals is None else np.float64(normals)
    flagged = np.zeros(len(points), bool) if flagged is None else np.asarray(flagged)
    usable = np.ones(len(points), bool) if usable is None else np.asarray(usable)
    return list(iter_lamp_rays(points, normals, flagged, usable, lights, chunk))


def joined(blocks):
    return {name: np.concatenate([getattr(b, name) for b in blocks]) for name in
            ('point', 'light', 'origin', 'direction', 'limit', 'skip', 'factor')}


def definitions(path, names):
    found = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if getattr(node, 'name', None) in names:
            found[node.name] = ast.dump(node)
    return found


def test_blender_carries_an_exact_copy_of_the_lamp_rule():
    # bake.py cannot import the package, so only this keeps its lamp rays from drifting from the GPU host's
    import lightmapper
    from lightmapper import vertex_gpu
    names = {'_unit', 'LampRays', '_lamp_factor', 'iter_lamp_rays'}
    bake = definitions(Path(lightmapper.__file__).parent / 'blender' / 'bake.py', names)
    host = definitions(vertex_gpu.__file__, names)
    assert set(bake) == set(host) == names
    for name in names:
        assert bake[name] == host[name], name


@pytest.mark.parametrize('chunk', [10 ** 9, 2])
def test_a_block_holds_one_light_and_each_point_once(chunk):
    # the passes add a block's rows with a fancy-indexed +=, which drops a point met twice
    points = [(x, 0, 0) for x in (0.0, 1.0, 2.0, 3.0, 4.0)]
    blocks = rays(points, [lamp((0, 0, 2)), lamp((1, 0, 3)), lamp((5, 0, 1))], chunk=chunk)
    assert len({int(b.light[0]) for b in blocks}) == 3
    for block in blocks:
        assert len(set(block.light.tolist())) == 1 and len(set(block.point.tolist())) == len(block.point)


def test_point_factor_is_cosine_over_distance_squared():
    x = np.array([0.0, 1.0, 3.0])
    got = joined(rays([(v, 0, 0) for v in x], [lamp((0, 0, 2))]))
    assert got['factor'] == pytest.approx(2 / (4 + x ** 2) ** 1.5)
    assert got['direction'][1] == pytest.approx(np.array([-1, 0, 2]) / np.sqrt(5))
    assert got['origin'][1] == pytest.approx(np.array([1, 0, 0]) + RAY_EPSILON * got['direction'][1])


def test_spot_factor_uses_tools_ramp():
    cutoff, falloff = 0.5, 0.6
    light = lamp((0, 0, 1), kind=3, axis=(0, 0, -1), cos_cutoff=cutoff, cos_falloff=falloff)
    angles = np.arccos([1.0, 0.7, 0.55, 0.45])
    points = [(np.tan(a), 0, 0) for a in angles]
    got = joined(rays(points, [light]))
    for point, factor in zip(got['point'], got['factor']):
        a = angles[point]
        d = 1 / np.cos(a)
        assert factor == pytest.approx(np.cos(a) * ramp(np.cos(a), cutoff, falloff) / d ** 2)
    assert set(got['point']) == {0, 1, 2}


def test_spot_ramp_is_linear_in_the_cosine_between_cutoff_and_falloff():
    light = lamp((0, 0, 1), kind=3, cos_cutoff=0.5, cos_falloff=0.9)
    # a height of 1 makes cos(angle) the cosine to the axis and 1 / d^2 its square
    cosines = np.array([0.6, 0.7, 0.8])
    got = joined(rays([(np.tan(np.arccos(c)), 0, 0) for c in cosines], [light]))
    np.testing.assert_allclose(got['factor'] / cosines ** 3, (cosines - 0.5) / 0.4)


def test_light_exactly_at_reach_makes_no_row_and_just_inside_does():
    assert rays([(0, 0, 0)], [lamp((0, 0, 2), reach=2.0)]) == []
    assert len(rays([(0, 0, 0)], [lamp((0, 0, 2), reach=2.0001)])) == 1


def test_a_falloff_not_wider_than_the_cutoff_leaves_a_step_at_the_cutoff():
    light = lamp((0, 0, 1), kind=3, cos_cutoff=0.8, cos_falloff=0.7)
    got = joined(rays([(np.tan(np.arccos(c)), 0, 0) for c in (0.79, 0.81)], [light]))
    assert got['point'].tolist() == [1]
    assert got['factor'][0] == pytest.approx(0.81 ** 3)


def test_flagged_point_takes_a_light_from_either_side():
    got = joined(rays([(0, 0, 0)] * 2, [lamp((0, 0, 2))], normals=[(0, 0, -1)] * 2, flagged=[True, False]))
    assert list(got['point']) == [0]
    assert got['factor'] == pytest.approx([0.25])


def test_light_beyond_reach_and_on_the_point_make_no_row():
    assert rays([(0, 0, 0), (0, 0, 1 - 5e-5)], [lamp((0, 0, 1), reach=0.9)]) == []
    assert rays([(0, 0, 1 - 5e-5)], [lamp((0, 0, 1))]) == []
    assert rays([(0, 0, 0), (0, 0, 0.5)], [lamp((0, 0, 1))], usable=[False, True])[0].point.tolist() == [1]


def test_limit_stops_short_of_the_light_and_never_goes_negative():
    got = joined(rays([(0, 0, 0), (0, 0, 0.9995)], [lamp((0, 0, 1))]))
    assert got['limit'][0] == pytest.approx(1 - 2 * RAY_EPSILON)
    assert got['limit'][1] == 0.0


def test_rows_carry_the_lights_owner_as_skip():
    got = joined(rays([(0, 0, 0)], [lamp((0, 0, 1), owner=7), lamp((0, 0, 2), owner=-1)]))
    assert got['skip'].tolist() == [7, -1]
    assert got['light'].tolist() == [0, 1]
    assert got['skip'].dtype == np.int64


def test_chunked_rows_equal_unchunked():
    rng = np.random.default_rng(3)
    points = rng.uniform(-3, 3, (50, 3))
    normals = rng.normal(size=(50, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    lights = [lamp(rng.uniform(-3, 3, 3), kind=k, axis=(0, 0, -1), cos_cutoff=0.2, cos_falloff=0.7, owner=i)
              for i, k in enumerate((2, 3, 2))]
    flagged = rng.random(50) < 0.3
    whole = rays(points, lights, normals, flagged, chunk=10 ** 9)
    parts = rays(points, lights, normals, flagged, chunk=7)
    assert len(whole) == 3 and all(len(b.point) <= 7 for b in parts) and len(parts) > 1
    a, b = joined(whole), joined(parts)
    for name in a:
        assert np.array_equal(a[name], b[name]), name
