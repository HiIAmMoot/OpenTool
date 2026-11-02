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

    scene = dict(lm_uv=uv, tris=tris, tri_page=np.zeros(len(tris), np.int16),
                 texel_pos=np.zeros((393216, 3), np.float32), texel_page=np.zeros(393216, np.int16),
                 texel_xy=np.tile(np.int32([[0, 0]]), (393216, 1)))
    counts = api().texel_ray_counts(scene, [(1024, 1024)], 28, 252)
    np.testing.assert_array_equal(counts, 28)


@pytest.mark.parametrize('target,want', [(0, [28, 28]), (252, [252, 28]), (512, [512, 57])])
def test_texel_ray_counts_top_up_edges_without_raising_the_interior(target, want):
    scene = dict(lm_uv=np.float32([[0, 0], [1/6, 0], [1/6, 1/3], [0, 1/3],
                                 [.5, 0], [1, 0], [1, 1], [.5, 1]]),
                 tris=np.int32([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]),
                 tri_page=np.zeros(4, np.int16), texel_pos=np.zeros((2, 3), np.float32),
                 texel_page=np.zeros(2, np.int16), texel_xy=np.int32([[0, 0], [3, 0]]))
    np.testing.assert_array_equal(api().texel_ray_counts(scene, [(2, 1)], 28, target), want)


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


# ---------- the finish stage's raster helpers keep their output (the old versions stay here as the reference)

def _old_dilate(img, mask, cls, iterations):
    img, mask, cls = img.copy(), mask.copy(), cls.copy()
    height, width, channels = img.shape
    for _ in range(iterations):
        result, filled, classes = img.copy(), mask.copy(), cls.copy()
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    continue
                y0, y1 = max(0, y - 1), min(height - 1, y + 1)
                x0, x1 = max(0, x - 1), min(width - 1, x + 1)
                count = 0
                for ny in range(y0, y1 + 1):
                    for nx in range(x0, x1 + 1):
                        if mask[ny, nx]:
                            if count == 0:
                                classes[y, x] = cls[ny, nx]
                            count += 1
                if count == 0:
                    continue
                filled[y, x] = True
                for channel in range(channels):
                    total = 0.0
                    for ny in range(y0, y1 + 1):
                        for nx in range(x0, x1 + 1):
                            if mask[ny, nx]:
                                total += img[ny, nx, channel]
                    result[y, x, channel] = total / count
        img, mask, cls = result, filled, classes
    return img, mask, cls


_old_dilate_compiled = None


def old_dilate(*args):
    global _old_dilate_compiled
    if _old_dilate_compiled is None:
        from numba import njit
        _old_dilate_compiled = njit(_old_dilate)
    return _old_dilate_compiled(*args)


def dilate_case(seed, shape, density, dtype, iterations):
    rng = np.random.default_rng(seed)
    height, width = shape
    img = rng.random((height, width, 3)).astype(dtype) * 10.0 ** rng.integers(-3, 3, (height, width, 1))
    mask = rng.random(shape) < density
    # (texels under the mask carry values too: the old dilate reads only masked ones, and must keep the rest)
    cls = rng.integers(-2, 5, shape).astype(np.int16)
    return img.astype(dtype), mask, cls, iterations


@pytest.mark.parametrize('shape', [(1, 1), (1, 9), (9, 1), (2, 2), (17, 23), (40, 40)])
@pytest.mark.parametrize('density', [0.0, 0.02, 0.2, 0.6, 1.0])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dilate_visiting_only_the_frontier_is_bit_identical_to_the_full_scan(shape, density, dtype):
    for iterations in (0, 1, 2, 3, 7, 60):
        img, mask, cls, _ = dilate_case(iterations + 5, shape, density, dtype, iterations)
        want = old_dilate(img, mask, cls, iterations)
        got = api().dilate(img, mask, cls, iterations)
        for have, expected in zip(got, want):
            assert have.dtype == expected.dtype
            np.testing.assert_array_equal(have, expected)


def test_dilate_leaves_its_inputs_alone():
    img, mask, cls, _ = dilate_case(1, (12, 12), 0.1, np.float32, 3)
    before = img.copy(), mask.copy(), cls.copy()
    api().dilate(img, mask, cls, 3)
    for kept, original in zip((img, mask, cls), before):
        np.testing.assert_array_equal(kept, original)


def _old_flagged_source(scene, page, width, height, rgb):
    from scipy.ndimage import distance_transform_edt
    rows = np.flatnonzero(scene['tri_page'] == page)
    owner, _ = api().triangle_weights(scene['lm_uv'], scene['tris'][rows], 3 * width, 3 * height)
    source = np.zeros((3 * height, 3 * width, 3), np.float32)
    chosen = scene['texel_page'] == page
    xy = scene['texel_xy'][chosen]
    source[xy[:, 1], xy[:, 0]] = rgb[chosen]
    if (owner >= 0).any():
        nearest = distance_transform_edt(owner < 0, return_distances=False, return_indices=True)
        source[owner < 0] = source[nearest[0][owner < 0], nearest[1][owner < 0]]
    return source


def flagged_scene():
    """Two charts on page 0 (two triangles each, a gap between them and a margin) and one on page 1."""
    uv = np.float32([[.1, .1], [.5, .1], [.1, .6], [.5, .6],
                     [.6, .2], [.95, .2], [.6, .9], [.95, .9],
                     [0, 0], [1, 0], [0, 1]])
    tris = np.int32([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6], [8, 9, 10]])
    xy = np.int32([[3, 3], [4, 4], [10, 9], [20, 13], [5, 1]])
    return dict(lm_uv=uv, tris=tris, tri_page=np.int16([0, 0, 0, 0, 1]), texel_page=np.int16([0, 0, 0, 0, 1]),
                texel_xy=xy)


@pytest.mark.parametrize('page', [0, 1, 2])
def test_flagged_source_with_one_index_list_is_bit_identical(page):
    scene = flagged_scene()
    rgb = np.random.default_rng(2).random((5, 3)).astype(np.float32)
    want = _old_flagged_source(scene, page, 8, 6, rgb)
    np.testing.assert_array_equal(api().flagged_source(scene, page, 8, 6, rgb), want)
    owner, _ = api().triangle_weights(scene['lm_uv'], scene['tris'][scene['tri_page'] == page], 24, 18)
    np.testing.assert_array_equal(api().flagged_source(scene, page, 8, 6, rgb, owner=owner), want)
    if page == 0:
        assert want.any()


def test_the_owner_map_a_caller_holds_is_not_computed_again(monkeypatch):
    raster = api()
    scene = flagged_scene()
    rgb = np.random.default_rng(2).random((5, 3)).astype(np.float32)
    owner, _ = raster.triangle_weights(scene['lm_uv'], scene['tris'][scene['tri_page'] == 0], 24, 18)
    want = raster.texel_contribution(scene, 0, 8, 6, rgb)
    monkeypatch.setattr(raster, 'triangle_weights', lambda *a: pytest.fail('owner map computed again'))
    got = raster.texel_contribution(scene, 0, 8, 6, rgb, owner=owner)
    raster.flagged_source(scene, 0, 8, 6, rgb, owner=owner)
    for have, expected in zip(got, want):
        np.testing.assert_array_equal(have, expected)


def _old_ordinary_transport(scene, page, base):
    height, width = base.shape[:2]
    rows = np.flatnonzero(scene['tri_page'] == page)
    owner, _ = api().triangle_weights(scene['lm_uv'], scene['tris'][rows], 3 * width, 3 * height)
    owned = (owner >= 0).reshape(height, 3, width, 3).sum(axis=(1, 3))
    xy = scene['texel_xy'][scene['texel_page'] == page] // 3
    flagged = np.zeros((height, width), np.int64)
    np.add.at(flagged, (xy[:, 1], xy[:, 0]), 1)
    ordinary = owned - flagged
    scale = np.where(flagged > 0, owned / np.maximum(ordinary, 1) * (ordinary > 0), 1.0)
    return base * scale[:, :, None]


@pytest.mark.parametrize('page', [0, 1, 2])
def test_ordinary_transport_in_float32_is_the_float64_product_rounded_once(page):
    scene = flagged_scene()
    base = np.random.default_rng(5).random((6, 8, 3)).astype(np.float32)
    old = _old_ordinary_transport(scene, page, base)
    assert old.dtype == np.float64
    np.testing.assert_array_equal(api().ordinary_transport(scene, page, base), old)
    narrow = api().ordinary_transport(scene, page, base, np.float32)
    assert narrow.dtype == np.float32 and narrow.tobytes() == old.astype(np.float32).tobytes()
