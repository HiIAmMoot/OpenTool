"""Written sub-samples, including black ones, contribute equally to a page texel."""

import importlib.util
import sys
import types

import numpy as np
import pytest

from lightmapper.blender_job import BAKE_SCRIPT


@pytest.fixture
def bake_module(monkeypatch):
    # only Blender imports are stubbed; the numpy helper is the production implementation
    for name in ('bpy', 'mathutils', 'mathutils.bvhtree'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules['mathutils.bvhtree'].BVHTree = None
    spec = importlib.util.spec_from_file_location('lm_bake', BAKE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downsample_averages_only_written_subsamples(bake_module):
    assert hasattr(bake_module, 'downsample_written'), 'written-only downsample is missing'
    rgb = np.full((6, 9, 3), -1, np.float32)
    rgb[0, 0] = [3, 6, 9]
    rgb[2, 2] = [9, 12, 15]
    rgb[0, 6] = [0, 0, 0]
    rgb[1, 8] = [2, 4, 6]
    rgb[3:6, :3] = [0.25, 0.5, 0.75]
    page, written = bake_module.downsample_written(rgb)
    np.testing.assert_array_equal(page, [
        [[6, 9, 12], [0, 0, 0], [1, 2, 3]],
        [[0.25, 0.5, 0.75], [0, 0, 0], [0, 0, 0]]])
    np.testing.assert_array_equal(written, [[True, False, True], [True, False, False]])
    assert page.dtype == np.float32 and written.dtype == np.bool_


def test_downsample_default_max_page_grid_snapped(bake_module):
    assert hasattr(bake_module, 'downsample_written'), 'written-only downsample is missing'
    rgb = np.full((3072, 3072, 3), -1, np.float32)
    rgb[1::3, 1::3] = [0.5, 1, 2]
    page, written = bake_module.downsample_written(rgb)
    assert page.shape == (1024, 1024, 3) and written.all()
    np.testing.assert_array_equal(page, np.broadcast_to([0.5, 1, 2], page.shape))


def test_emit_pass_written_mask_is_compared_with_the_diffuse_one(bake_module):
    assert hasattr(bake_module, 'warn_if_masks_differ'), 'written-mask comparison is missing'
    same = np.zeros((2, 2), np.bool_)
    same[0, 0] = True
    assert bake_module.warn_if_masks_differ(same, same.copy(), 3) is False
    other = same.copy()
    other[1, 1] = True
    assert bake_module.warn_if_masks_differ(same, other, 3) is True


def test_texel_owning_one_subsample_ends_with_an_interior_texels_sample_count(bake_module):
    owned = np.array([[9, 1, 2, 3, 5, 8, 0]])
    tiers = bake_module.plan_extra_passes(owned, 28, 256)
    effective = bake_module.effective_samples(owned, 28, tiers)
    assert effective[0, 0] == 252
    # a tier's cost is set by its smallest count, so a larger one in the tier overshoots by up to 1.5x
    assert (effective[0, 1:6] >= 0.95 * 252).all() and (effective[0, 1:6] <= 1.6 * 252).all()
    assert effective[0, 6] == 0


def test_extra_passes_are_off_when_the_target_is_zero(bake_module):
    assert bake_module.plan_extra_passes(np.array([[1, 9]]), 28, 0) == []


def test_extra_pass_means_are_merged_by_sample_count(bake_module):
    owned = np.array([[1, 9]])
    base = np.float32([[[1, 1, 1], [5, 5, 5]]])
    tiers = bake_module.plan_extra_passes(owned, 28, 256)
    extra = [np.full((1, 2, 3), 9, np.float32) for _ in tiers]
    merged = bake_module.merge_extra(base, owned, 28, tiers, extra)
    mask = tiers[0][1]
    assert mask[0, 0] and not mask[0, 1]
    first = 28 * 1
    added = 1 * sum(spp for spp, m in tiers if m[0, 0])
    np.testing.assert_allclose(merged[0, 0], (1 * first + 9 * added) / (first + added), rtol=1e-6)
    np.testing.assert_array_equal(merged[0, 1], [5, 5, 5])


def test_plan_ends_for_one_sample_per_subtexel(bake_module):
    # a limit rounded down below the first count that already has enough would never advance
    owned = np.arange(1, 10).reshape(1, 9)
    tiers = bake_module.plan_extra_passes(owned, 1, 256)
    assert (bake_module.effective_samples(owned, 1, tiers) >= 0.95 * 256).all()


@pytest.mark.parametrize('samples', [64, 1024])
def test_edge_texels_reach_the_interior_count_whatever_the_samples(bake_module, samples):
    from lightmapper.settings import Settings

    base_spp = max(1, round(samples / 9))
    target = Settings(samples=samples).resolved_texel_samples()
    owned = np.arange(1, 10).reshape(1, 9)
    tiers = bake_module.plan_extra_passes(owned, base_spp, target)
    effective = bake_module.effective_samples(owned, base_spp, tiers)
    assert effective[0, 8] == 9 * base_spp
    assert (effective[0] >= 0.95 * 9 * base_spp).all()
    assert (effective[0] <= 1.6 * 9 * base_spp).all()


def test_merge_ignores_a_non_finite_value_in_an_unmasked_texel(bake_module):
    owned = np.array([[1, 9]])
    base = np.float32([[[1, 1, 1], [5, 5, 5]]])
    tiers = bake_module.plan_extra_passes(owned, 28, 256)
    extra = []
    for _ in tiers:
        image = np.full((1, 2, 3), 9, np.float32)
        # the unmasked texel's extra value comes from paths the mask threw away
        image[0, 1] = [np.inf, np.nan, -np.inf]
        extra.append(image)
    merged = bake_module.merge_extra(base, owned, 28, tiers, extra)
    assert np.isfinite(merged).all()
    np.testing.assert_array_equal(merged[0, 1], [5, 5, 5])


def test_merge_weights_come_from_effective_samples(bake_module, monkeypatch):
    owned = np.array([[1, 9]])
    base = np.float32([[[1, 1, 1], [5, 5, 5]]])
    tiers = bake_module.plan_extra_passes(owned, 28, 256)
    extra = [np.full((1, 2, 3), 9, np.float32) for _ in tiers]
    real = bake_module.effective_samples
    merged = bake_module.merge_extra(base, owned, 28, tiers, extra)
    monkeypatch.setattr(bake_module, 'effective_samples', lambda *args: 2 * real(*args))
    np.testing.assert_allclose(bake_module.merge_extra(base, owned, 28, tiers, extra), merged / 2, rtol=1e-6)
