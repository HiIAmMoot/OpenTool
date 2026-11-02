import numpy as np
import pytest

from lightmapper import raster
from lightmapper.charts import CapError, ChartSettings, _charts_separated, _pack_rectangles, ladder_shape, next_step
from lightmapper.combine import (Move, WrittenPage, chart_blocks, plan_repack, repack_arrays, repack_layout,
                                 written_shape)
from lightmapper.formats.bsp import SourceGroup
from lightmapper.formats.vertices import ENV, LM

PADDING = 4


def rect_pieces(sizes, sid0=0):
    # one chart per size, w x h texels, each its own group so a test can tell them apart
    pieces = []
    for i, (w, h) in enumerate(sizes):
        env = np.zeros(4, ENV)
        env['position'] = [[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]]
        tris = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
        group = SourceGroup(('senv', f'g{i}', 0, 0, -1), None, env, tris, np.arange(sid0 + 2 * i, sid0 + 2 * i + 2))
        pieces.append((i, group, np.arange(4), env['position'][:, :2].astype(np.float64), tris, group.surface_ids))
    return pieces


def bake_pages(sizes, side):
    pieces = rect_pieces(sizes)
    return _pack_rectangles(pieces, ChartSettings(page_max=side, padding=PADDING)), pieces


def where(written):
    # (written page, x, y, width, height) of each chart, found from its group
    found = {}
    for k, page in enumerate(written):
        for m in page.materials:
            texels = m.lm['texcoord'].astype(np.float64) * [page.width, page.height]
            low, high = texels.min(axis=0), texels.max(axis=0)
            found[m.group] = (k, *low, *(high - low))
    return found


# ---------- the ladder


@pytest.mark.parametrize('shape, step', [((2048, 2048), (2048, 1024)), ((2048, 1024), (1024, 1024)),
                                         ((1024, 1024), (1024, 512)), ((512, 256), (256, 256))])
def test_each_step_halves_the_longer_side(shape, step):
    assert next_step(shape) == step


@pytest.mark.parametrize('size, shape', [((1500, 600), (2048, 1024)), ((600, 1500), (2048, 1024)),
                                         ((1500, 1100), (2048, 2048)), ((1024, 512), (1024, 512)),
                                         ((1025, 10), (2048, 1024)), ((100, 90), (256, 128))])
def test_the_shape_is_the_smallest_step_that_holds_the_chart(size, shape):
    assert ladder_shape([size], 4096, 256) == shape


def test_a_square_drops_to_half_its_height_first():
    assert ladder_shape([(2000, 1000)], 4096, 256) == (2048, 1024)
    assert ladder_shape([(2000, 1100)], 4096, 256) == (2048, 2048)


# ---------- the blocks a bake page holds


def test_each_rectangle_chart_is_one_block_of_its_padded_size():
    pages, _ = bake_pages([(30, 20), (10, 40), (50, 5)], 128)
    blocks = chart_blocks(pages, PADDING)
    assert sorted((b.width, b.height) for b in blocks) == sorted([(38, 28), (18, 48), (58, 13)])


# ---------- the repack


def big_and_small():
    # three charts that need 1024 x 512 (none fits 512 x 512 either way round), then many small ones
    return [(700, 200)] * 3 + [(80, 80)] * 120


def test_charts_that_need_the_auto_shape_stay_whole_on_auto_pages():
    pages, pieces = bake_pages(big_and_small(), 1024)
    blocks = chart_blocks(pages, PADDING)
    shape = written_shape(blocks, 1024, auto=True)
    assert shape == (1024, 512)
    plan = plan_repack(blocks, shape)
    written = repack_layout(pages, plan)
    places = where(written)
    for i in range(3):
        k, x, y, w, h = places[i]
        assert (written[k].width, written[k].height) == (1024, 512)
        assert {round(w), round(h)} == {700, 200}


def test_smaller_charts_fill_the_free_space_on_auto_pages():
    pages, pieces = bake_pages(big_and_small(), 1024)
    blocks = chart_blocks(pages, PADDING)
    plan = plan_repack(blocks, (1024, 512))
    written = repack_layout(pages, plan)
    places = where(written)
    auto = {places[i][0] for i in range(3)}
    # (two big charts take 416 of the 512 rows on a page, and leave a 316-texel column beside them)
    on_auto = [i for i in range(3, 123) if places[i][0] in auto]
    assert len(on_auto) >= 50
    assert all(p.width <= 1024 and p.height <= 512 for p in written)
    assert _charts_separated(written, ChartSettings(page_max=1024, padding=PADDING), pieces)


def test_the_rest_go_on_auto_shape_pages_as_few_as_fit():
    pages, pieces = bake_pages([(700, 200)] + [(200, 200)] * 30, 1024)
    blocks = chart_blocks(pages, PADDING)
    plan = plan_repack(blocks, (1024, 512))
    written = repack_layout(pages, plan)
    assert all((p.width, p.height) == (1024, 512) for p in written[:-1])
    # (the last page holds one 208-texel chart, so it shrinks down the ladder to 256 x 256)
    assert (written[-1].width, written[-1].height) == (256, 256)
    # (eight 208-texel charts fit a 1024 x 512 page and five beside the big one: 5 + 3 x 8 holds 29 of 30)
    assert len(written) == 5


def test_a_fixed_max_resolution_is_the_shape():
    pages, _ = bake_pages([(100, 100)] * 4, 512)
    assert written_shape(chart_blocks(pages, PADDING), 512, auto=False) == (512, 512)


def test_written_pages_keep_the_gap_and_every_triangle():
    pages, pieces = bake_pages(big_and_small(), 1024)
    blocks = chart_blocks(pages, PADDING)
    written = repack_layout(pages, plan_repack(blocks, (1024, 512)))
    assert _charts_separated(written, ChartSettings(page_max=1024, padding=PADDING), pieces)
    assert sorted(int(i) for p in written for m in p.materials for i in m.surface_ids) == list(range(2 * 123))


def test_moved_charts_sample_the_same_light():
    pages, _ = bake_pages(big_and_small(), 1024)
    rng = np.random.default_rng(7)
    arrays = [rng.random((p.height, p.width, 3), dtype=np.float32) for p in pages]
    blocks = chart_blocks(pages, PADDING)
    plan = plan_repack(blocks, (1024, 512))
    written, moved = repack_layout(pages, plan), repack_arrays(arrays, plan)
    old = {m.group: (k, m) for k, p in enumerate(pages) for m in p.materials}
    for page, image in zip(written, moved):
        for m in page.materials:
            k, before = old[m.group]
            got = raster.sample_bilinear(image, m.lm['texcoord'])
            assert np.allclose(got, raster.sample_bilinear(arrays[k], before.lm['texcoord']), atol=1e-5)


def test_a_turned_chart_is_transposed_with_its_texels():
    # a tall chart on a wide page has to turn
    pages, _ = bake_pages([(40, 900)], 1024)
    blocks = chart_blocks(pages, PADDING)
    plan = plan_repack(blocks, written_shape(blocks, 1024, auto=True))
    assert plan[0].moves[0].swapped and (plan[0].width, plan[0].height) == (1024, 512)
    rng = np.random.default_rng(8)
    arrays = [rng.random((p.height, p.width, 3), dtype=np.float32) for p in pages]
    (page,), (image,) = repack_layout(pages, plan), repack_arrays(arrays, plan)
    got = raster.sample_bilinear(image, page.materials[0].lm['texcoord'])
    assert np.allclose(got, raster.sample_bilinear(arrays[0], pages[0].materials[0].lm['texcoord']), atol=1e-5)


def test_incident_radiosity_rides_along():
    pages, _ = bake_pages(big_and_small(), 1024)
    rng = np.random.default_rng(9)
    for p in pages:
        for m in p.materials:
            m.lm['incident_radiosity'] = rng.random((len(m.lm), 3), dtype=np.float32)
    written = repack_layout(pages, plan_repack(chart_blocks(pages, PADDING), (1024, 512)))
    old = {m.group: m for p in pages for m in p.materials}
    for p in written:
        for m in p.materials:
            assert m.lm.dtype == LM
            assert np.array_equal(m.lm['incident_radiosity'], old[m.group].lm['incident_radiosity'])


def test_same_input_same_plan():
    pages, _ = bake_pages(big_and_small(), 1024)
    assert plan_repack(chart_blocks(pages, PADDING), (1024, 512)) == \
        plan_repack(chart_blocks(pages, PADDING), (1024, 512))


def test_the_material_cap_holds_on_written_pages():
    pages, _ = bake_pages([(20, 20)] * 12, 128)
    plan = plan_repack(chart_blocks(pages, PADDING), (128, 128), max_materials=5)
    written = repack_layout(pages, plan)
    assert all(len(p.materials) <= 5 for p in written) and len(written) >= 3


def test_too_many_written_pages_refuse():
    pages, _ = bake_pages([(100, 100)] * 6, 128)
    with pytest.raises(CapError, match='lightmap pages'):
        plan_repack(chart_blocks(pages, PADDING), (128, 128), max_pages=5)


@pytest.mark.parametrize('dtype', [np.uint16, np.uint32, np.bool_])
def test_repack_keeps_the_dtype(dtype):
    pages, _ = bake_pages([(20, 20)] * 4, 64)
    rng = np.random.default_rng(5)
    arrays = [(rng.random((p.height, p.width)) * 100).astype(dtype) for p in pages]
    for out in repack_arrays(arrays, plan_repack(chart_blocks(pages, PADDING), (64, 64))):
        assert out.dtype == np.dtype(dtype)


def test_a_whole_page_move_is_identity():
    rng = np.random.default_rng(3)
    pages, _ = bake_pages([(20, 20)] * 4, 64)
    arrays = [rng.random((p.height, p.width, 3), dtype=np.float32) for p in pages]
    plan = [WrittenPage(p.width, p.height, (Move(i, 0, 0, p.width, p.height, 0, 0, False),)) for i, p in enumerate(pages)]
    for new, old in zip(repack_layout(pages, plan), pages):
        assert [m.lm.tobytes() for m in new.materials] == [m.lm.tobytes() for m in old.materials]
    for new, old in zip(repack_arrays(arrays, plan), arrays):
        assert np.array_equal(new, old)


def test_only_the_last_page_shrinks_and_only_down_the_ladder():
    pages, _ = bake_pages(big_and_small(), 1024)
    plan = plan_repack(chart_blocks(pages, PADDING), (1024, 512))
    assert len(plan) >= 2
    assert all((p.width, p.height) == (1024, 512) for p in plan[:-1])
    ladder = [(1024, 512), (512, 512), (512, 256), (256, 256), (256, 128), (128, 128), (128, 64), (64, 64)]
    last = (plan[-1].width, plan[-1].height)
    assert last in ladder
    # (one step smaller does not hold it)
    smaller = ladder[ladder.index(last) + 1]
    held = [b for b in chart_blocks(pages, PADDING) if (b.page, b.x, b.y) in
            {(m.page, m.x, m.y) for m in plan[-1].moves}]
    assert len(plan_repack(held, smaller)) > 1 or max(max(b.width, b.height) for b in held) > smaller[0]


def test_a_new_page_opens_only_for_charts_no_open_page_can_take():
    # (a small chart that fits the room left on page 1 goes there, though a bigger one already went to page 2)
    pages, _ = bake_pages([(1000, 400), (1000, 400), (60, 60)], 1024)
    plan = plan_repack(chart_blocks(pages, PADDING), (1024, 512))
    assert [len(p.moves) for p in plan] == [2, 1]
