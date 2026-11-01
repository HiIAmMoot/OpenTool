import numpy as np
import pytest

from lightmapper import raster
from lightmapper.charts import CapError
from lightmapper.combine import Placement, combine_arrays, combine_layout, plan_combine
from lightmapper.formats.bsp import OutMaterial, OutPage
from lightmapper.formats.vertices import LM

EXAMPLES = [
    ([(512, 512)] * 5, [(1024, 1024), (512, 512)]),
    ([(512, 512)] * 21, [(2048, 2048), (1024, 1024), (512, 512)]),
    ([(1024, 1024)] * 13, [(4096, 2048), (2048, 2048), (1024, 1024)]),
    ([(2048, 2048)] * 12, [(4096, 4096)] * 3),
    ([(512, 512)] * 6 + [(512, 128)], [(1024, 1024), (1024, 512), (512, 128)]),
    ([(512, 512)] * 2 + [(512, 128)], [(1024, 512), (512, 128)]),
    ([(256, 512)], [(256, 512)]),
]


def plan_sizes(plan):
    return sorted((p.width, p.height) for p in plan)


@pytest.mark.parametrize('sizes, expected', EXAMPLES)
def test_worked_examples(sizes, expected):
    plan = plan_combine(sizes, [1] * len(sizes), 4096)
    assert plan_sizes(plan) == sorted(expected)


def test_page_max_caps_the_combined_pages():
    sizes = [(512, 512)] * 5
    assert plan_sizes(plan_combine(sizes, [1] * 5, 512)) == [(512, 512)] * 5
    assert plan_sizes(plan_combine(sizes, [1] * 5, 1024)) == [(512, 512), (1024, 1024)]


@pytest.mark.parametrize('sizes, expected', EXAMPLES)
def test_written_texels_equal_baked_texels(sizes, expected):
    plan = plan_combine(sizes, [1] * len(sizes), 4096)
    assert sum(p.width * p.height for p in plan) == sum(w * h for w, h in sizes)
    assert sorted(i for p in plan for i, _, _ in p.sources) == list(range(len(sizes)))


def test_material_cap_leaves_a_run_uncombined():
    four = [(512, 512)] * 4
    assert plan_sizes(plan_combine(four, [600] * 4, 4096)) == [(1024, 512), (1024, 512)]
    assert plan_sizes(plan_combine(four, [1100] * 4, 4096)) == [(512, 512)] * 4
    assert plan_sizes(plan_combine(four, [500] * 4, 4096)) == [(1024, 1024)]


def test_too_many_written_pages_refuse():
    with pytest.raises(CapError):
        plan_combine([(512, 512)] * 128, [1] * 128, 512, max_pages=127)


def test_same_input_same_plan():
    sizes = [(512, 512)] * 6 + [(512, 128)] * 3
    assert plan_combine(sizes, [1] * 9, 4096) == plan_combine(sizes, [1] * 9, 4096)


def random_pages(rng, count=4, size=64):
    return [rng.random((size, size, 3), dtype=np.float32) for _ in range(count)]


def test_quadrants_are_the_source_pages_bit_for_bit():
    rng = np.random.default_rng(1)
    pages = random_pages(rng)
    plan = plan_combine([(64, 64)] * 4, [1] * 4, 4096)
    (out,) = combine_arrays(pages, plan)
    assert out.shape == (128, 128, 3)
    for i, ox, oy in plan[0].sources:
        assert np.array_equal(out[oy:oy + 64, ox:ox + 64], pages[i])


def out_material(rng, count=6):
    lm = np.zeros(count, LM)
    lm['incident_radiosity'] = rng.random((count, 3), dtype=np.float32)
    # a 2^-12 grid keeps the remap into the 128 page exact in float32
    lm['texcoord'] = rng.integers(4 * 64, 60 * 64 + 1, (count, 2)) / 4096
    return OutMaterial(0, np.zeros(count), lm, np.zeros((1, 3), np.int32), np.zeros(1, np.int32))


def test_remapped_uv_samples_the_same_value():
    rng = np.random.default_rng(2)
    arrays = random_pages(rng)
    pages = [OutPage(64, 64, [out_material(rng)]) for _ in range(4)]
    plan = plan_combine([(64, 64)] * 4, [1] * 4, 4096)
    (page,) = combine_layout(pages, plan)
    (combined,) = combine_arrays(arrays, plan)
    for k, material in enumerate(page.materials):
        old = pages[k].materials[0].lm['texcoord']
        got = raster.sample_bilinear(combined, material.lm['texcoord'])
        assert np.allclose(got, raster.sample_bilinear(arrays[k], old), atol=1e-6)


def test_nothing_to_combine_is_identity():
    rng = np.random.default_rng(3)
    sizes = [(64, 32), (128, 64), (32, 32)]
    pages = [OutPage(w, h, [out_material(rng)]) for w, h in sizes]
    arrays = [rng.random((h, w, 3), dtype=np.float32) for w, h in sizes]
    plan = plan_combine(sizes, [1] * 3, 4096)
    out = combine_layout(pages, plan)
    for page, original in zip(out, [pages[p.sources[0][0]] for p in plan]):
        assert (page.width, page.height) == (original.width, original.height)
        assert page.materials[0].lm.tobytes() == original.materials[0].lm.tobytes()
    for new, p in zip(combine_arrays(arrays, plan), plan):
        assert np.array_equal(new, arrays[p.sources[0][0]])


def test_incident_radiosity_rides_along():
    rng = np.random.default_rng(4)
    pages = [OutPage(64, 64, [out_material(rng)]) for _ in range(4)]
    plan = plan_combine([(64, 64)] * 4, [1] * 4, 4096)
    (page,) = combine_layout(pages, plan)
    for k, material in enumerate(page.materials):
        assert np.array_equal(material.lm['incident_radiosity'], pages[k].materials[0].lm['incident_radiosity'])
        assert material.lm.dtype == LM


@pytest.mark.parametrize('dtype', [np.uint16, np.uint32, np.bool_])
def test_combine_keeps_the_dtype(dtype):
    rng = np.random.default_rng(5)
    arrays = [(rng.random((16, 16)) * 100).astype(dtype) for _ in range(4)]
    plan = plan_combine([(16, 16)] * 4, [1] * 4, 4096)
    (out,) = combine_arrays(arrays, plan)
    assert out.dtype == np.dtype(dtype) and out.shape == (32, 32)
    for i, ox, oy in plan[0].sources:
        assert np.array_equal(out[oy:oy + 16, ox:ox + 16], arrays[i])
