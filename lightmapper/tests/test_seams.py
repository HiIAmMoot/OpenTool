import numpy as np
import pytest
from lightmapper.formats.bsp import OutMaterial, OutPage
from lightmapper.formats.vertices import ENV, LM
from lightmapper.settings import Settings


def fixture(hard=False, groups=(0, 1)):
    pages = []
    for side in (0, 1):
        env, lm = np.zeros(3, ENV), np.zeros(3, LM)
        env['position'] = [[0, 0, 0], [8, 0, 0], [0, 8 if side == 0 else -8, 0]]
        env['normal'] = [0, 1, 0] if hard and side else [0, 0, 1]
        lm['texcoord'] = np.array([[2.5, 2.5], [10.5, 2.5], [2.5, 10.5]]) / 16
        pages.append(OutPage(16, 16, [OutMaterial(groups[side], env, lm, np.array([[0, 1, 2]]), np.array([side]))]))
    raw = [np.full((16, 16, 3), value, np.float32) for value in (0.2, 0.8)]
    masks = [np.ones((16, 16), bool) for _ in pages]
    classes = [np.zeros((16, 16), np.int16) for _ in pages]
    return pages, raw, masks, classes


def finish(hard=False, enabled=True, groups=(0, 1), joined=None):
    from lightmapper import pages as finishing
    assert hasattr(finishing, 'finish_pages'), 'cross-page finishing is missing'
    finish_pages = finishing.finish_pages
    pages, raw, masks, classes = fixture(hard, groups)
    s = Settings(denoise=False, stitch_seams=enabled)
    extra = {} if joined is None else {'joined_groups': joined}
    return finish_pages(pages, raw, masks, classes, {}, s, **extra), raw


def test_finish_stitches_different_raw_values():
    result, _ = finish()
    np.testing.assert_allclose(result[0][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_allclose(result[1][2, 2:11], 0.5, atol=1e-5)


def test_stitches_two_charts_merged_into_one_material():
    from types import SimpleNamespace
    from lightmapper.charts import _finish, ChartSettings
    from lightmapper.pages import finish_pages
    source, _, _, _ = fixture()
    pieces = []
    for i, page in enumerate(source):
        m = page.materials[0]
        pieces.append((0, SimpleNamespace(env=m.env), np.arange(3),
                       m.lm['texcoord'] * 16 + [16 * i, 0], m.tris, m.surface_ids))
    pages = _finish([pieces], [np.array([27, 11])], ChartSettings(padding=0))
    assert len(pages[0].materials) == 1
    assert len(pages[0].materials[0].env) == 6
    raw = np.full((16, 32, 3), 0.2, np.float32)
    raw[:, 16:] = 0.8
    result = finish_pages(pages, [raw], [np.ones((16, 32), bool)],
                          [np.zeros((16, 32), np.int16)], {}, Settings(denoise=False))
    np.testing.assert_allclose(result[0][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_allclose(result[0][2, 18:27], 0.5, atol=1e-5)


def test_hard_edge_left_alone():
    result, raw = finish(hard=True)
    for a, b in zip(result, raw):
        np.testing.assert_array_equal(a, b)


def test_joined_group_stitches_across_a_hard_crease():
    result, _ = finish(hard=True, groups=(0, 0), joined={0})
    np.testing.assert_allclose(result[0][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_allclose(result[1][2, 2:11], 0.5, atol=1e-5)


def test_plain_group_keeps_its_hard_crease_when_another_is_joined():
    result, raw = finish(hard=True, groups=(0, 0), joined={5})
    for a, b in zip(result, raw):
        np.testing.assert_array_equal(a, b)


def test_joined_exception_does_not_cross_groups():
    result, raw = finish(hard=True, groups=(0, 1), joined={0, 1})
    for a, b in zip(result, raw):
        np.testing.assert_array_equal(a, b)


def test_joined_group_still_stitches_to_a_neighbour_with_matching_normals():
    result, _ = finish(groups=(0, 1), joined={0})
    np.testing.assert_allclose(result[0][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_allclose(result[1][2, 2:11], 0.5, atol=1e-5)


def test_joined_group_still_stitches_to_a_same_shader_neighbour_with_matching_normals():
    result, _ = finish(groups=(0, 0), joined={0})
    np.testing.assert_allclose(result[0][2, 2:11], 0.5, atol=1e-5)


@pytest.mark.parametrize('order', [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_third_hard_member_does_not_suppress_smooth_seam(order):
    from lightmapper.pages import finish_pages
    pages, raw, masks, classes = fixture()
    hard_pages, hard_raw, hard_masks, hard_classes = fixture(hard=True)
    hard_pages[1].materials[0].env['position'][2] = [0, 0, 8]
    pages.append(hard_pages[1])
    raw.append(hard_raw[1])
    masks.append(hard_masks[1])
    classes.append(hard_classes[1])
    result = finish_pages([pages[i] for i in order], [raw[i] for i in order],
                          [masks[i] for i in order], [classes[i] for i in order], {}, Settings(denoise=False))
    np.testing.assert_allclose(result[order.index(0)][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_allclose(result[order.index(1)][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_array_equal(result[order.index(2)], raw[2])


def test_joined_group_keeps_its_smooth_seam_among_three_users_of_an_edge():
    from lightmapper.pages import finish_pages
    pages, raw, masks, classes = fixture()
    hard_pages, hard_raw, hard_masks, hard_classes = fixture(hard=True)
    hard_pages[1].materials[0].env['position'][2] = [0, 0, 8]
    hard_pages[1].materials[0].group = 2
    result = finish_pages(pages + hard_pages[1:], raw + hard_raw[1:], masks + hard_masks[1:],
                          classes + hard_classes[1:], {}, Settings(denoise=False), joined_groups={0})
    np.testing.assert_allclose(result[0][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_allclose(result[1][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_array_equal(result[2], hard_raw[1])


def test_joined_crease_stitches_when_a_third_surface_shares_the_edge():
    from lightmapper.pages import finish_pages
    pages, raw, masks, classes = fixture(hard=True, groups=(0, 0))
    other_pages, other_raw, other_masks, other_classes = fixture(hard=True, groups=(1, 1))
    third = other_pages[1].materials[0]
    third.env['normal'] = [1, 0, 0]
    third.env['position'][2] = [0, 0, 8]
    result = finish_pages(pages + other_pages[1:], raw + other_raw[1:], masks + other_masks[1:],
                          classes + other_classes[1:], {}, Settings(denoise=False), joined_groups={0})
    np.testing.assert_allclose(result[0][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_allclose(result[1][2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_array_equal(result[2], other_raw[1])


def test_multi_user_normal_chain_keeps_pair_predicates():
    from lightmapper.seams import _pair_forest
    normals = np.tile([[0., 0., 1.], [0., 0., 1.]], (3, 1, 1))
    normals[:, :, 0] = np.array([0., 0.0008, 0.0016])[:, None]
    pairs = list(_pair_forest(normals, np.zeros((3, 2, 2)), np.arange(3)))
    assert {tuple(sorted(pair)) for pair in pairs} == {(0, 1), (1, 2)}


def test_multi_user_continuous_uv_is_not_a_seam():
    from lightmapper.seams import _pair_forest
    normals = np.tile([[0., 0., 1.], [0., 0., 1.]], (33, 1, 1))
    assert list(_pair_forest(normals, np.zeros((33, 2, 2)), np.zeros(33))) == []


def test_multi_user_forest_at_stock_bsp_limit():
    from lightmapper.seams import _pair_forest
    count = 131072
    normals = np.tile([[0., 0., 1.], [0., 0., 1.]], (count, 1, 1))
    normals[-1] = [0., 1., 0.]
    pairs = list(_pair_forest(normals, np.zeros((count, 2, 2)), np.arange(count) % 2))
    assert len(pairs) == count - 2
    assert all(count - 1 not in pair for pair in pairs)


def test_away_from_seam_and_real_shadow_left_alone():
    from lightmapper.pages import finish_pages
    from lightmapper.raster import coverage
    pages, raw, masks, classes = fixture()
    raw[0][5:7, 3:5] = 0.01
    material = pages[0].materials[0]
    covered, _ = coverage(material.lm['texcoord'], material.tris, np.array([0]), 16, 16)
    assert covered[5:7, 3:6].all(), 'both sides of the shadow boundary must be covered'
    np.testing.assert_array_equal(raw[0][5:7, 3:5], 0.01)
    np.testing.assert_array_equal(raw[0][5:7, 5], 0.2)
    result = finish_pages(pages, raw, masks, classes, {}, Settings(denoise=False))
    for a, b in zip(result, raw):
        np.testing.assert_array_equal(a[4:], b[4:])


def test_setting_off_reproduces_old_finishing():
    result, raw = finish(enabled=False)
    for a, b in zip(result, raw):
        np.testing.assert_array_equal(a, b)


def test_stitch_after_denoise(monkeypatch):
    from lightmapper import pages as finishing
    pages, raw, masks, classes = fixture()
    monkeypatch.setattr(finishing, '_denoise', lambda page, timeout: page * page)
    result = finishing.finish_pages(pages, raw, masks, classes, {}, Settings(denoise=True))
    np.testing.assert_allclose(result[0][2, 2:11], 0.34, atol=1e-5)
    np.testing.assert_allclose(result[1][2, 2:11], 0.34, atol=1e-5)


def test_same_page_seam_and_reversed_edge():
    from lightmapper.seams import stitch_seams
    pages, raw, _, _ = fixture()
    second = pages[1].materials[0]
    second.lm['texcoord'][:, 1] += 0.5
    second.tris = second.tris[:, [1, 0, 2]]
    pages[0].materials.append(second)
    image = raw[0].copy()
    image[8:] = 0.8
    stitch_seams(pages[:1], [image])
    np.testing.assert_allclose(image[2, 2:11], 0.5, atol=1e-5)
    np.testing.assert_allclose(image[10, 2:11], 0.5, atol=1e-5)


def test_continuous_uv_shadow_edge_is_not_stitched():
    from lightmapper.seams import stitch_seams
    pages, raw, _, _ = fixture()
    pages[0].materials.append(pages[1].materials[0])
    image = raw[0].copy()
    image[3:] = 0.8
    before = image.copy()
    stitch_seams(pages[:1], [image])
    np.testing.assert_array_equal(image, before)


def test_different_density_and_fractional_uvs_agree_bilinearly():
    from lightmapper.seams import stitch_seams
    from lightmapper.raster import sample_bilinear
    pages, raw, _, _ = fixture()
    pages[1].materials[0].lm['texcoord'] *= 0.5
    stitch_seams(pages, raw)
    for t in np.linspace(0, 1, 33):
        samples = []
        for p, image in zip(pages, raw):
            uv = p.materials[0].lm['texcoord']
            point = uv[0] * (1 - t) + uv[1] * t
            samples.append(sample_bilinear(image, point[None])[0])
        np.testing.assert_allclose(samples[0], samples[1], atol=2e-5)


def test_seam_grouping_at_stock_bsp_limit():
    from lightmapper.seams import seam_edges
    from test_charts import grid
    pos, tris = grid(256)
    env, lm = np.zeros(len(pos), ENV), np.zeros(len(pos), LM)
    env['position'] = pos
    env['normal'] = [0, 0, 1]
    lm['texcoord'] = pos[:, :2] / 256
    page = OutPage(256, 256, [OutMaterial(0, env, lm, tris, np.arange(131072))])
    assert list(seam_edges([page])) == []
