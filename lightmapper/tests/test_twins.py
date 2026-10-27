"""Exact-position pairing must not merge independent coincident layers or scale quadratically."""

import logging
import time

import numpy as np


def test_reversed_twins_pair_exact_reversed_triangles_only(caplog):
    from lightmapper.twins import reversed_twins

    a = np.float64([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    b = a + [3, 0, 0]
    c = a + [6, 0, 0]
    near = c[[1, 0, 2]].copy()
    near[0, 0] += 1e-6
    d = a + [9, 0, 0]
    corners = np.array([a, a[[1, 0, 2]], b, b[[1, 2, 0]], c, near, d, d[::-1], d])
    with caplog.at_level(logging.INFO):
        twin, perm = reversed_twins(corners)
    np.testing.assert_array_equal(twin, [1, 0, -1, -1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(perm[:2], [[1, 0, 2], [1, 0, 2]])
    np.testing.assert_array_equal(corners[0][perm[0]], corners[1])
    assert '1' in caplog.text and 'more than two' in caplog.text

    grid = np.zeros((20000, 3, 3))
    grid[:, :, 0] = np.arange(20000)[:, None] * 2
    grid[:, 1, 0] += 1
    grid[:, 2, 1] = 1
    # 20,000 singles plus 1,000 independent twin pairs, all grid-snapped
    pairs = grid[:1000] + [0, 4, 0]
    large = np.concatenate([grid, pairs, pairs[:, [2, 1, 0]]])
    start = time.perf_counter()
    twin, perm = reversed_twins(large)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, elapsed
    np.testing.assert_array_equal(twin[:20000], -1)
    np.testing.assert_array_equal(twin[20000:21000], np.arange(21000, 22000))
    np.testing.assert_array_equal(twin[21000:], np.arange(20000, 21000))
    np.testing.assert_array_equal(perm[20000:], np.tile([2, 1, 0], (2000, 1)))


def test_large_coincident_group_is_unpaired():
    from lightmapper.twins import reversed_twins

    corners = np.tile(np.float32([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]), (131072, 1, 1))
    corners[1::2] = corners[1::2, ::-1]
    twin, _ = reversed_twins(corners)
    np.testing.assert_array_equal(twin, -1)


def test_empty_twins_and_signed_zero():
    from lightmapper.twins import reversed_twins

    twin, perm = reversed_twins(np.empty((0, 3, 3)))
    assert twin.shape == (0,) and perm.shape == (0, 3)
    a = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    b = a[::-1].copy()
    b[b == 0] = -0.0
    twin, perm = reversed_twins(np.array([a, b]))
    np.testing.assert_array_equal(twin, [1, 0])
    np.testing.assert_array_equal(perm, [[2, 1, 0], [2, 1, 0]])
