"""Pin page row order, inclusive coverage, synchronous padding and clamp sampling."""

import importlib
import importlib.util

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('lightmapper.raster') is not None, 'raster API is missing'
    return importlib.import_module('lightmapper.raster')


@pytest.mark.parametrize('triangle', [[0, 1, 2], [2, 1, 0]])
def test_coverage_half_page_includes_edges_and_row_zero(triangle):
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    mask, cls = api().coverage(uv, np.array([triangle]), np.array([7]), 4, 4)
    y, x = np.indices((4, 4))
    np.testing.assert_array_equal(mask, x + y <= 3)
    assert mask.dtype == np.bool_
    assert cls.dtype == np.int16
    np.testing.assert_array_equal(cls[mask], 7)
    np.testing.assert_array_equal(mask[0], [True, True, True, True])
    np.testing.assert_array_equal(mask[-1], [True, False, False, False])


@pytest.mark.parametrize('reverse', [False, True])
def test_coverage_shared_edge_lowest_triangle_index_wins(reverse):
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    classes = np.array([11, 22], dtype=np.int16)
    if reverse:
        tris, classes = tris[::-1].copy(), classes[::-1].copy()
    mask, cls = api().coverage(uv, tris, classes, 4, 4)
    assert mask.all()
    np.testing.assert_array_equal(cls[0], [11, 11, 11, classes[0]])
    np.testing.assert_array_equal(cls[3], [classes[0], 22, 22, 22])
    np.testing.assert_array_equal(cls[np.arange(4), np.arange(3, -1, -1)], classes[0])


def test_coverage_rectangular_page_and_empty_input():
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    mask, cls = api().coverage(uv, np.array([[0, 1, 2]]), np.array([-1]), 8, 2)
    np.testing.assert_array_equal(mask, [[1, 1, 1, 1, 1, 1, 0, 0],
                                         [1, 1, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(cls[mask], -1)
    empty, empty_cls = api().coverage(uv, np.empty((0, 3), dtype=np.int32),
                                     np.empty(0, dtype=np.int16), 8, 2)
    assert empty.shape == empty_cls.shape == (2, 8)
    assert not empty.any()


def test_coverage_degenerate_edges_do_not_fill_bounding_box():
    uv = np.array([[0.125, 0.125], [0.875, 0.875], [0.625, 0.125]])
    mask, cls = api().coverage(uv, np.array([[0, 1, 0], [2, 2, 2]]),
                               np.array([3, 4]), 4, 4)
    expected = np.eye(4, dtype=bool)
    expected[0, 2] = True
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(cls[np.arange(4), np.arange(4)], 3)
    assert cls[0, 2] == 4


def test_coverage_maximum_bsp_surfaces_on_1024_page():
    # grid-snapped inputs exercise the full 131,072-surface BSP budget
    y, x = np.indices((257, 257))
    uv = np.column_stack((x.ravel(), y.ravel())).astype(np.float32) / 256
    y, x = np.indices((256, 256))
    a = (y * 257 + x).ravel()
    tris = np.stack((np.column_stack((a, a + 1, a + 257)),
                     np.column_stack((a + 1, a + 258, a + 257))), axis=1).reshape(-1, 3)
    classes = np.repeat((x + 3 * y).ravel() % 31, 2).astype(np.int16)
    assert len(tris) == 131072
    mask, cls = api().coverage(uv, tris, classes, 1024, 1024)
    assert mask.all()
    expected = np.repeat(np.repeat((x + 3 * y) % 31, 4, axis=0), 4, axis=1)
    np.testing.assert_array_equal(cls, expected)


def test_dilate_one_iteration_fills_only_eight_neighbour_ring():
    img = np.zeros((5, 5, 2), dtype=np.float32)
    img[2, 2] = [0.25, 1.75]
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    cls = np.full((5, 5), -99, dtype=np.int16)
    cls[2, 2] = -1
    result, filled, classes = api().dilate(img, mask, cls, 1)
    expected = np.zeros((5, 5), dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(filled, expected)
    np.testing.assert_array_equal(result[filled], np.tile([0.25, 1.75], (9, 1)))
    np.testing.assert_array_equal(classes[filled], -1)
    assert np.issubdtype(result.dtype, np.floating)


def test_dilate_mean_and_first_neighbour_in_raster_order():
    img = np.zeros((3, 3, 2), dtype=np.float64)
    mask = np.zeros((3, 3), dtype=bool)
    cls = np.zeros((3, 3), dtype=np.int16)
    for y, x, value, label in [(0, 2, [0.25, 2], 8),
                                (1, 0, [0.75, 4], 5),
                                (2, 2, [2, 12], 2)]:
        img[y, x], mask[y, x], cls[y, x] = value, True, label
    result, filled, classes = api().dilate(img, mask, cls, 1)
    np.testing.assert_allclose(result[1, 1], [1, 6])
    assert filled[1, 1]
    assert classes[1, 1] == 8
    np.testing.assert_array_equal(result[mask], img[mask])
    np.testing.assert_array_equal(classes[mask], cls[mask])


def test_dilate_multiple_iterations_reach_edges_synchronously():
    img = np.zeros((5, 7, 1), dtype=np.float32)
    img[0, 0, 0] = 0.375
    mask = np.zeros((5, 7), dtype=bool)
    mask[0, 0] = True
    cls = np.full((5, 7), 6, dtype=np.int16)
    result, filled, classes = api().dilate(img, mask, cls, 2)
    expected = np.zeros((5, 7), dtype=bool)
    expected[:3, :3] = True
    np.testing.assert_array_equal(filled, expected)
    np.testing.assert_array_equal(result[filled], 0.375)
    np.testing.assert_array_equal(classes[filled], 6)


@pytest.mark.parametrize('iterations', [0, 3])
def test_dilate_empty_mask_stays_empty(iterations):
    img = np.full((2, 4, 3), 0.125, dtype=np.float32)
    mask = np.zeros((2, 4), dtype=bool)
    cls = np.full((2, 4), -3, dtype=np.int16)
    result, filled, classes = api().dilate(img, mask, cls, iterations)
    np.testing.assert_array_equal(result, img)
    np.testing.assert_array_equal(filled, mask)
    np.testing.assert_array_equal(classes, cls)


def test_sample_bilinear_horizontal_ramp_at_texel_centres():
    img = np.array([[[0, 0.25], [1, 1.25], [2, 2.25], [3, 3.25]]], dtype=np.float32)
    uv = np.column_stack(((np.arange(4) + 0.5) / 4, np.full(4, 0.5)))
    np.testing.assert_array_equal(api().sample_bilinear(img, uv), img[0])


def test_sample_bilinear_horizontal_halfway_returns_mean():
    img = np.array([[[0], [2], [4], [6]]], dtype=np.float64)
    uv = np.array([[0.25, 0.5], [0.5, 0.5], [0.75, 0.5]])
    np.testing.assert_array_equal(api().sample_bilinear(img, uv), [[1], [3], [5]])


def test_sample_bilinear_rows_and_four_neighbour_weights():
    img = np.array([[[0], [4]], [[8], [16]]], dtype=np.float32)
    uv = np.array([[0.25, 0.25], [0.25, 0.75], [0.5, 0.5], [0.375, 0.625]])
    np.testing.assert_allclose(api().sample_bilinear(img, uv), [[0], [8], [7], [7.75]])


def test_sample_bilinear_clamps_page_edges():
    img = np.array([[[2], [4], [6]], [[12], [14], [16]]], dtype=np.float32)
    uv = np.array([[0, 0.25], [1, 0.25], [-2, -3], [2, 3], [0.5, 0], [0.5, 1]])
    np.testing.assert_array_equal(api().sample_bilinear(img, uv),
                                  [[2], [6], [2], [16], [4], [14]])


def test_sample_bilinear_empty_batch_and_single_texel():
    img = np.array([[[0.125, 2.5]]], dtype=np.float64)
    result = api().sample_bilinear(img, np.empty((0, 2)))
    assert result.shape == (0, 2)
    uv = np.array([[-1, 2], [0.5, 0.5], [0, 0]])
    np.testing.assert_array_equal(api().sample_bilinear(img, uv),
                                  [[0.125, 2.5], [0.125, 2.5], [0.125, 2.5]])
