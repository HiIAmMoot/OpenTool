"""Pin coherence magnitude, zero vectors for unlit queries, the scaled mode's normal fallbacks and byte-exact seam query sharing."""

import importlib
import importlib.util

import numpy as np
import pytest

from lightmapper.formats.vertices import ENV


def api():
    assert importlib.util.find_spec('lightmapper.radiosity') is not None, 'radiosity API is missing'
    return importlib.import_module('lightmapper.radiosity')


def test_coherence_single_direction():
    result = api().finalize(np.array([[0, 6, 8]]), np.array([10]),
                            np.array([[0, 0, 1]]), 'coherence', 99)
    np.testing.assert_allclose(result, [[0, 0.6, 0.8]], rtol=1e-7)
    assert result.dtype == np.float32


def test_coherence_single_direction_with_ambient_preserves_half_length():
    result = api().finalize(np.array([[0, 3, 4]]), np.array([10]),
                            np.array([[1, 0, 0]]), 'coherence', 99)
    np.testing.assert_allclose(result, [[0, 0.3, 0.4]], rtol=1e-7)


def test_coherence_opposite_equal_lights_give_a_zero_vector():
    result = api().finalize(np.array([[0, 0, 0]]), np.array([2]),
                            np.array([[0, 1, 0]]), 'coherence', 1)
    np.testing.assert_array_equal(result, [[0, 0, 0]])


def test_coherence_no_energy_gives_a_zero_vector_not_the_normal():
    result = api().finalize(np.array([[9, 2, 3], [9, 2, 3]]), np.array([0, -1.0]),
                            np.array([[0, 0.6, 0.8], [0, 1, 0]]), 'coherence', 1)
    np.testing.assert_array_equal(result, np.zeros((2, 3)))


def test_coherence_vector_threshold_is_strict():
    result = api().finalize(np.array([[0.999e-4, 0, 0], [1e-4, 0, 0]]),
                            np.ones(2), np.array([[0, 1, 0], [0, 1, 0]]), 'coherence', 1)
    np.testing.assert_allclose(result, [[0, 0, 0], [1e-4, 0, 0]], rtol=1e-7)


def test_scaled_opposite_equal_lights_fall_back_to_normal():
    result = api().finalize(np.array([[0, 0, 0]]), np.array([2]),
                            np.array([[0, 1, 0]]), 'scaled', 1)
    np.testing.assert_array_equal(result, [[0, 1, 0]])


def test_scaled_no_energy_normal_is_bit_exact_including_signed_zero():
    normal = np.array([[-0.0, 0.6, 0.8]], dtype=np.float32)
    result = api().finalize(np.array([[9, 2, 3]]), np.array([0]), normal, 'scaled', 1)
    assert result.tobytes() == normal.tobytes()


def test_scaled_energy_threshold_is_inclusive():
    result = api().finalize(np.array([[0.5, 0, 0], [0.5, 0, 0]]),
                            np.array([1e-6, 0.999e-6]),
                            np.array([[0, 1, 0], [0, 0, 1]]), 'scaled', 1)
    np.testing.assert_array_equal(result, [[0, 1, 0], [0, 0, 1]])


def test_scaled_vector_threshold_is_strict():
    result = api().finalize(np.array([[0.999e-4, 0, 0], [1e-4, 0, 0]]),
                            np.ones(2), np.array([[0, 1, 0], [0, 1, 0]]), 'scaled', 1)
    np.testing.assert_allclose(result, [[0, 1, 0], [1e-4, 0, 0]], rtol=1e-7)


def test_scaled_clamps_length_not_components_and_keeps_small_vectors():
    result = api().finalize(np.array([[0, 3, 4], [0.1, 0, 0]]), np.array([1, 99]),
                            np.array([[1, 0, 0], [0, 1, 0]]), 'scaled', 2)
    np.testing.assert_allclose(result, [[0, 0.6, 0.8], [0.2, 0, 0]], rtol=1e-7)
    assert np.linalg.norm(result[0]) == 1


@pytest.mark.parametrize('vector, expected', [
    ([1e200, 0, 0], [1, 0, 0]),
    ([1.7e308, -1.7e308, 0], [2**-0.5, -2**-0.5, 0]),
])
def test_scaled_large_finite_vector_keeps_direction(vector, expected):
    result = api().finalize(np.array([vector], dtype=np.float64), np.array([1]),
                            np.array([[0, 0, 1]]), 'scaled', 1)
    np.testing.assert_allclose(result, [expected], rtol=1e-7)


def test_scaled_nan_vector_uses_whole_normal():
    normal = np.array([[-0.0, 0.6, 0.8]], dtype=np.float32)
    result = api().finalize(np.array([[0.2, np.nan, 0.3]]), np.ones(1), normal, 'scaled', 1)
    assert np.isfinite(result).all()
    assert result.tobytes() == normal.tobytes()


def test_coherence_nan_vector_or_energy_gives_a_zero_vector():
    result = api().finalize(np.array([[0.2, np.nan, 0.3], [1, 0, 0]]), np.array([1, np.nan]),
                            np.array([[0, 0, 1], [0, 0, 1]]), 'coherence', 1)
    np.testing.assert_array_equal(result, np.zeros((2, 3)))


@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_coherence_nonfinite_vector_is_masked_as_a_whole(value):
    result = api().finalize(np.array([[0.2, value, 0.3]]), np.ones(1),
                            np.array([[0, 0, 1]]), 'coherence', 1)
    np.testing.assert_array_equal(result, [[0, 0, 0]])


def test_coherence_components_stay_within_one():
    result = api().finalize(np.array([[2, -3, 0.5]]), np.ones(1),
                            np.array([[0, 0, 1]]), 'coherence', 1)
    np.testing.assert_allclose(result, np.array([[2, -3, 0.5]]) / np.linalg.norm([2, -3, 0.5]), rtol=1e-6)


def test_seam_uvs_share_query_but_different_position_or_normal_does_not():
    env = np.zeros(4, dtype=ENV)
    env['position'] = [[1, 2, 3], [1, 2, 3], [2, 2, 3], [1, 2, 3]]
    env['normal'] = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]]
    env['texcoord'] = [[0, 0], [1, 1], [0.5, 0], [0, 0.5]]
    env['tangent'][1] = [1, 0, 0]
    pos, normal, classes, inverse = api().query_points(
        env, np.array([[0, 2, 3], [1, 2, 3]]), np.ones(2, dtype=np.int16))
    assert len(pos) == 3
    assert inverse[0] == inverse[1]
    assert len(set(inverse[[0, 2, 3]])) == 3
    np.testing.assert_array_equal(pos[inverse], env['position'])
    np.testing.assert_array_equal(normal[inverse], env['normal'])
    np.testing.assert_array_equal(classes, [1, 1, 1])
    assert classes.dtype == np.int16


def test_exact_bytes_distinguish_signed_zero_and_adjacent_float():
    env = np.zeros(4, dtype=ENV)
    env['normal'] = [0, 0, 1]
    env['position'][1, 0] = -0.0
    env['normal'][2, 0] = -0.0
    env['position'][3, 0] = np.nextafter(np.float32(0), np.float32(1))
    pos, normal, _, inverse = api().query_points(
        env, np.array([[0, 1, 2], [0, 2, 3]]), np.zeros(2, dtype=np.int16))
    assert len(pos) == len(set(inverse)) == 4
    assert pos[inverse].tobytes() == env['position'].tobytes()
    assert normal[inverse].tobytes() == env['normal'].tobytes()


def test_query_class_surface_tie_uses_lowest_signed_id():
    env = np.zeros(3, dtype=ENV)
    env['normal'] = [0, 0, 1]
    _, _, query_class, inverse = api().query_points(
        env, np.array([[0, 1, 2], [2, 1, 0]]), np.array([2, -1], dtype=np.int16))
    np.testing.assert_array_equal(query_class[inverse], [-1, -1, -1])


def test_query_class_shared_vertex_counts_distinct_surfaces_not_seam_copies():
    env = np.zeros(6, dtype=ENV)
    env['position'] = [[0, 0, 0], [0, 0, 0], [1, 0, 0],
                       [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    env['normal'] = [0, 0, 1]
    tris = np.array([[0, 2, 3], [0, 3, 4], [1, 4, 5]])
    tri_class = np.array([1, 1, -1], dtype=np.int16)
    _, _, query_class, inverse = api().query_points(env, tris, tri_class)
    assert inverse[0] == inverse[1]
    assert query_class[inverse[0]] == 1


def test_query_class_triangle_votes_once_across_all_its_seam_copies():
    env = np.zeros(4, dtype=ENV)
    env['normal'] = [0, 0, 1]
    env['position'][3, 0] = 1
    tris = np.array([[0, 1, 2], [0, 3, 3], [1, 3, 3]])
    _, _, query_class, inverse = api().query_points(
        env, tris, np.array([-1, 1, 1], dtype=np.int16))
    np.testing.assert_array_equal(query_class[inverse], [1, 1, 1, 1])


def test_query_class_untouched_query_is_minus_one():
    env = np.zeros(4, dtype=ENV)
    env['normal'] = [0, 0, 1]
    env['position'][:, 0] = [0, 1, 2, 0]
    _, _, query_class, inverse = api().query_points(
        env, np.array([[0, 0, 0]]), np.array([7], dtype=np.int16))
    np.testing.assert_array_equal(query_class[inverse], [7, -1, -1, 7])
    _, _, untouched, _ = api().query_points(
        env, np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.int16))
    np.testing.assert_array_equal(untouched, [-1, -1, -1])


def test_empty_inputs():
    pos, normal, classes, inverse = api().query_points(
        np.empty(0, dtype=ENV), np.empty((0, 3), dtype=np.intp),
        np.empty(0, dtype=np.int16))
    assert pos.shape == normal.shape == (0, 3)
    assert classes.shape == inverse.shape == (0,)
    for mode in ('coherence', 'scaled'):
        result = api().finalize(pos, np.empty(0), normal, mode, 1)
        assert result.shape == (0, 3)
        assert result.dtype == np.float32


def test_maximum_bsp_corner_count_grid_snapped_queries():
    # 131,072 surfaces can contribute 393,216 distinct output corners
    count = 131072 * 3
    env = np.zeros(count, dtype=ENV)
    env['position'][:, 0] = np.arange(count) % 1024
    env['position'][:, 1] = np.arange(count) // 1024
    env['normal'] = [0, 0, 1]
    tris = np.arange(count).reshape(-1, 3)
    tri_class = (np.arange(len(tris)) % 31 - 1).astype(np.int16)
    classes = np.repeat(tri_class, 3)
    pos, normal, query_class, inverse = api().query_points(env, tris, tri_class)
    assert len(pos) == count
    np.testing.assert_array_equal(pos[inverse], env['position'])
    np.testing.assert_array_equal(normal[inverse], env['normal'])
    np.testing.assert_array_equal(query_class[inverse], classes)
    result = api().finalize(np.tile([0, 0, 0.5], (count, 1)), np.ones(count),
                            normal, 'coherence', 1)
    np.testing.assert_array_equal(result, np.tile([0, 0, 0.5], (count, 1)))
