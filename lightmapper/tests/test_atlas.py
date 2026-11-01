"""The edge top-up atlas: one small triangle per sub-texel, laid out so each bakes what the page path would."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import load_bake_module


DATA = Path(__file__).parent / 'data' / 'b9_spike.npz'
EPS = 0.02
SCENE_KEYS = ('positions', 'normals', 'lm_uv', 'tex_uv', 'tris', 'tri_page', 'tri_twin', 'tri_ignore_normals')


@pytest.fixture(scope='module')
def spike():
    with np.load(DATA) as data:
        return {key: data[key] for key in data.files}


def load(spike, prefix):
    scene = {key: spike[f'{prefix}_{key}'] for key in SCENE_KEYS}
    width, height = (int(v) for v in spike[f'{prefix}_page'])
    two_sided = scene['tri_ignore_normals'] & (str(spike[f'{prefix}_mode']) != 'enhanced')
    return scene, width, height, two_sided


def planned(bake_module, spike, prefix):
    scene, width, height, two_sided = load(spike, prefix)
    owner = bake_module.subtexel_owner(scene, 0, width, height)
    atlas = bake_module.plan_atlas(spike[f'{prefix}_written'], spike[f'{prefix}_mask'], owner, two_sided)
    return scene, width, height, atlas


@pytest.fixture(scope='module')
def placed_spike(spike):
    # one placement shared by the tests that read it (the greedy layout of 3545 slots takes seconds)
    out = {}
    with pytest.MonkeyPatch.context() as patch:
        module = load_bake_module(patch)
    for prefix in ('s', 's5'):
        scene, width, height, atlas = planned(module, spike, prefix)
        out[prefix] = (module, scene, width, height, module.place_atlas(scene, atlas, width, height))
    return out


# ---------- plain geometry for the checks (independent of the code under test)

def area(poly):
    poly = np.asarray(poly, np.float64)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip(poly, tri):
    tri = np.asarray(tri, np.float64)
    if (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0]) < 0:
        tri = tri[::-1]
    out = [np.asarray(p, np.float64) for p in poly]
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        src, out = out, []
        for k, p in enumerate(src):
            q = src[(k + 1) % len(src)]
            sp = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            sq = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                out.append(p + sp / (sp - sq) * (q - p))
        if not out:
            return np.zeros((0, 2))
    return np.array(out)


def footprint(col, row):
    return [(col - EPS, row - EPS), (col + 1 + EPS, row - EPS), (col + 1 + EPS, row + 1 + EPS), (col - EPS, row + 1 + EPS)]


def page_corners(module, scene, width, height):
    return module.flip_v(scene['lm_uv']).astype(np.float64)[scene['tris']] * [3 * width, 3 * height]


def barycentric(point, tri):
    a, b, c = tri
    u, v = np.linalg.solve(np.column_stack([b - a, c - a]), np.asarray(point) - a)
    return np.array([1 - u - v, u, v])


def scene_of(positions, lm_uv, normals=None, tex_uv=None, flagged=False):
    positions = np.asarray(positions, np.float32)
    count = len(positions) // 3
    return dict(positions=positions,
                normals=np.asarray(normals if normals is not None else np.tile([0, 0, 1], (len(positions), 1)), np.float32),
                lm_uv=np.asarray(lm_uv, np.float32),
                tex_uv=np.asarray(tex_uv if tex_uv is not None else lm_uv, np.float32),
                tris=np.arange(3 * count, dtype=np.int32).reshape(-1, 3), tri_page=np.zeros(count, np.int16),
                tri_twin=np.full(count, -1, np.int32), tri_ignore_normals=np.full(count, flagged))


def manual_atlas(bake_module, slots, owner, role=None, twin=None):
    count = len(slots)
    return bake_module.Atlas(slots=np.int64(slots), owner=np.int32(owner),
                             role=np.int8(role if role is not None else np.zeros(count)),
                             twin=np.int64(twin if twin is not None else np.full(count, -1)))


def corner_map(mesh, atlas):
    # which T' corner each mesh vertex is (the mesh may swap two corners to face the owner's normal)
    count = len(atlas.owner)
    page = (mesh['atlas_uv'].astype(np.float64).reshape(count, 3, 2) * atlas.side
            - atlas.pixels[:, None] + atlas.slots[:, None])
    distance = np.linalg.norm(page[:, :, None] - atlas.triangles[:, None], axis=3)
    assert distance.min(axis=2).max() < 1e-3
    return distance.argmin(axis=2)


# ---------- the scalar placement the vectorised one must equal (the layout the spike measured, frozen here)

REF_EPS = 0.02
REF_SLACK = 0.01


def ref_clip(poly, tri):
    sign = np.sign((tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0]))
    out = list(poly)
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        src, out = out, []
        for k in range(len(src)):
            p, q = src[k], src[(k + 1) % len(src)]
            sp = sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
            sq = sign * ((b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]))
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                out.append(p + sp / (sp - sq) * (q - p))
        if not out:
            return np.zeros((0, 2))
    return np.array(out)


def ref_needed_edges(clipped, tri):
    scale = max(1.0, np.abs(tri).max())
    need = []
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        d = b - a
        reach = 1e-9 * scale * np.hypot(*d)
        for k in range(len(clipped)):
            p, r = clipped[k], clipped[(k + 1) % len(clipped)]
            # (a side lying along the edge is the owner's edge cutting the footprint; a corner touching it is not)
            if (np.hypot(*(r - p)) > 1e-7 and abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) <= reach
                    and abs(d[0] * (r[1] - a[1]) - d[1] * (r[0] - a[0])) <= reach):
                need.append(i)
                break
    return need


def ref_slot_triangle(col, row, tri):
    # one triangle T' that equals the owner on the slot's widened footprint square and reaches no further than it must:
    # two triangles would sample only part of the footprint (Cycles keeps a jittered sample inside the triangle that
    # holds the pixel's centre), and a square past the owner's edges would sample area the page never does
    clipped = ref_clip(np.array([[col - REF_EPS, row - REF_EPS], [col + 1 + REF_EPS, row - REF_EPS],
                              [col + 1 + REF_EPS, row + 1 + REF_EPS], [col - REF_EPS, row + 1 + REF_EPS]]), tri)
    need = ref_needed_edges(clipped, tri)
    if len(need) == 3:
        return tri.copy(), 3
    if len(need) == 2:
        edges = {0: (0, 1), 1: (1, 2), 2: (2, 0)}
        shared = (set(edges[need[0]]) & set(edges[need[1]])).pop()
        others = [v for v in (0, 1, 2) if v != shared]
        corner = tri[shared]
        da, db = tri[others[0]] - corner, tri[others[1]] - corner
        ab = np.linalg.solve(np.column_stack([da, db]), (clipped - corner).T)
        # (the margin keeps the footprint's far corners inside T', whose far side then stays inside the owner)
        t = ab.sum(axis=0).max() * (1 + 1e-6) + 1e-6
        return np.array([corner, corner + t * da, corner + t * db]), 2
    if len(need) == 1:
        a, b = tri[need[0]], tri[(need[0] + 1) % 3]
        d = (b - a) / np.hypot(*(b - a))
        m = np.array([-d[1], d[0]])
        if (tri[(need[0] + 2) % 3] - a) @ m < 0:
            m = -m
        along, height = (clipped - a) @ d, (clipped - a) @ m
        lo, hi, top = along.min(), along.max(), max(height.max(), 0) + 1e-3
        return np.array([a + (lo - top) * d, a + (hi + top) * d, a + 0.5 * (lo + hi) * d + (0.5 * (hi - lo) + top) * m]), 1
    # (the widened footprint lies inside the owner, so the smallest right triangle around it does)
    x0, y0, side = col - REF_EPS, row - REF_EPS, 1 + 2 * REF_EPS
    return np.array([[x0, y0], [x0 + 2 * side + 0.01, y0], [x0, y0 + 2 * side + 0.01]]), 0


def ref_covers(tri, px, py):
    a, b, c = tri
    sign = np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    inside = np.ones(np.broadcast(px, py).shape, bool)
    for p, q in ((a, b), (b, c), (c, a)):
        inside &= sign * ((q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0])) / np.hypot(*(q - p)) >= -REF_SLACK
    return inside


def ref_page_corners(m, scene, width, height):
    return m.flip_v(scene['lm_uv']).astype(np.float64)[scene['tris']] * [3 * width, 3 * height]


def reference_place_atlas(m, scene, atlas, width, height):
    corners = ref_page_corners(m, scene, width, height)
    area = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
            - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0]))
    # (a zero-area owner has no barycentrics to place the slot's corners by)
    if (area[atlas.owner] == 0).any():
        return None
    count = len(atlas.slots)
    triangles = np.empty((count, 3, 2))
    needed = np.empty(count, np.int8)
    for k in range(count):
        # (a back copy repeats its front copy's footprint)
        source = atlas.twin[k] if atlas.role[k] == 2 else k
        if source != k:
            triangles[k], needed[k] = triangles[source], needed[source]
        else:
            triangles[k], needed[k] = ref_slot_triangle(*atlas.slots[k], corners[atlas.owner[k]])
    n_cols = int(math.ceil(math.sqrt(1.3 * count)))
    pixels = np.empty((count, 2), np.int64)
    occupied, covered = set(), set()
    cell = 0
    for k in range(count):
        relative = triangles[k] - atlas.slots[k]
        while True:
            ax, ay = 2 + 4 * (cell % n_cols), 2 + 4 * (cell // n_cols)
            cell += 1
            # (a cell under an earlier triangle would bake that slot's neighbour; one whose triangle reaches an earlier
            # slot's centre would overwrite it)
            if (ax, ay) in covered:
                continue
            tri = relative + [ax, ay]
            lo, hi = np.floor(tri.min(axis=0) - 1).astype(int), np.ceil(tri.max(axis=0) + 1).astype(int)
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            inside = ref_covers(tri, gx + 0.5, gy + 0.5)
            mine = set(zip(gx[inside].tolist(), gy[inside].tolist()))
            if any(pixel != (ax, ay) and pixel in occupied for pixel in mine):
                continue
            occupied.add((ax, ay))
            covered |= mine
            pixels[k] = ax, ay
            break
        if max(ax, ay) + 4 + 2 + 4 > m.MAX_ATLAS:
            return None
    side = int(pixels.max()) + 4 + 2 + 4 if count else 0
    return dataclasses.replace(atlas, triangles=triangles, needed=needed, pixels=pixels, side=side)




def skips_of(atlas):
    cols = int(np.ceil(np.sqrt(1.3 * len(atlas.slots))))
    cells = (atlas.pixels[:, 1] - 2) // 4 * cols + (atlas.pixels[:, 0] - 2) // 4
    return int(cells.max() + 1 - len(cells))


def assert_same_layout(module, scene, atlas, width, height):
    want = reference_place_atlas(module, scene, atlas, width, height)
    got = module.place_atlas(scene, atlas, width, height)
    np.testing.assert_array_equal(got.pixels, want.pixels)
    assert got.side == want.side and skips_of(got) == skips_of(want)
    np.testing.assert_array_equal(got.needed, want.needed)
    np.testing.assert_allclose(got.triangles, want.triangles, rtol=0, atol=1e-12)
    return got


def busy_page(module, seed, size=32):
    # shuffled small triangles of every orientation and a few slivers, overlapping, so owners compete
    rng = np.random.default_rng(seed)
    pixels = []
    for _ in range(12):
        start, angle, length = rng.uniform(4, 3 * size - 4, 2), rng.uniform(0, 2 * np.pi), rng.uniform(25, 60)
        along = np.array([np.cos(angle), np.sin(angle)])
        across = np.array([-along[1], along[0]])
        pixels.append(np.array([start, start + length * along, start + length * along + rng.uniform(0.3, 1.5) * across]))
    for _ in range(450):
        centre = rng.uniform(4, 3 * size - 4, 2)
        radius = rng.uniform(1.0, 7.0)
        angles = np.sort(rng.uniform(0, 2 * np.pi, 3))
        pixels.append(centre + radius * np.column_stack([np.cos(angles), np.sin(angles)]))
    pixels = np.clip(np.array(pixels), 0.5, 3 * size - 0.5)
    uv = np.column_stack([pixels[..., 0].ravel() / (3 * size), 1 - pixels[..., 1].ravel() / (3 * size)])
    positions = np.tile(np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (len(pixels), 1))
    scene = scene_of(positions, uv)
    owner = module.subtexel_owner(scene, 0, size, size)
    atlas = module.plan_atlas(owner >= 0, np.ones((size, size), bool), owner, np.zeros(len(pixels), bool))
    return scene, atlas, size


def test_vectorised_placement_equals_the_scalar_greedy_one(bake_module, spike):
    for prefix in ('s', 's5'):
        scene, width, height, atlas = planned(bake_module, spike, prefix)
        assert_same_layout(bake_module, scene, atlas, width, height)


def test_vectorised_placement_equals_the_scalar_one_on_a_busy_page(bake_module):
    for seed in (1, 2):
        scene, atlas, size = busy_page(bake_module, seed)
        assert len(atlas.slots) > 2500
        placed = assert_same_layout(bake_module, scene, atlas, size, size)
        assert skips_of(placed) > 0
        extent = (placed.triangles - atlas.slots[:, None]).max(axis=(1, 2))
        assert extent.max() > 12, 'the page holds slivers whose triangles leave the vectorised window'


def test_exact_per_cell_path_equals_the_scalar_one(bake_module, monkeypatch):
    # a margin of 0.6 sends nearly every slot to the cell-by-cell path that normally only rare slots take
    monkeypatch.setattr(bake_module, 'MARGIN', 0.6)
    calls = []
    real = bake_module._land
    monkeypatch.setattr(bake_module, '_land', lambda *args: calls.append(1) or real(*args))
    for seed in (1, 2):
        scene, atlas, size = busy_page(bake_module, seed)
        assert_same_layout(bake_module, scene, atlas, size, size)
    assert len(calls) > 3000


def test_exact_path_keeps_to_the_grid_at_the_first_row(bake_module):
    # a triangle reaching above and left of the first cell covers no cell there is
    start, cells = bake_module._land(np.float64([[-5, -5], [9, -5], [2, 6]]), 4, 0, bytearray(64), bytearray(64))
    assert start == 0 and sorted(cells) == [0, 1]


def test_place_atlas_says_why_it_returned_none(bake_module, spike, monkeypatch, capsys):
    scene, width, height, atlas = planned(bake_module, spike, 's5')
    monkeypatch.setattr(bake_module, 'MAX_ATLAS', 16)
    assert bake_module.place_atlas(scene, atlas, width, height) is None
    assert 'MAX_ATLAS' in capsys.readouterr().out
    flat = dict(scene, lm_uv=np.zeros_like(scene['lm_uv']))
    monkeypatch.setattr(bake_module, 'MAX_ATLAS', 8192)
    assert bake_module.place_atlas(flat, atlas, width, height) is None
    assert 'zero-area' in capsys.readouterr().out


# ---------- ownership

def stock_scenes():
    uv = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    small = dict(lm_uv=uv, tris=np.array([[0, 1, 2], [1, 3, 2], [0, 1, 2]]), tri_page=np.zeros(3, np.int16),
                 tri_ignore_normals=np.array([True, False, False]))
    size = 64
    x, y = np.meshgrid(np.arange(size + 1), np.arange(size + 1))
    grid = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32) / size
    a = (y[:-1, :-1] * (size + 1) + x[:-1, :-1]).ravel()
    tris = np.stack([np.column_stack([a, a + 1, a + size + 2]),
                     np.column_stack([a, a + size + 2, a + size + 1])], axis=1).reshape(-1, 3)
    limit = dict(lm_uv=grid, tris=tris, tri_page=np.zeros(len(tris), np.int16),
                 tri_ignore_normals=np.repeat(np.arange(size * size) % 2 == 0, 2))
    strips = dict(lm_uv=uv, tris=np.array([[0, 1, 2], [1, 3, 2]]), tri_page=np.zeros(2, np.int16),
                  tri_ignore_normals=np.array([True, False]))
    return [(small, 1), (limit, size), (strips, 1024)]


def test_subtexel_owner_covers_what_receiver_diffuse_scale_covers(bake_module):
    for scene, size in stock_scenes():
        owner = bake_module.subtexel_owner(scene, 0, size, size)
        coverage = bake_module.receiver_diffuse_scale(scene, 0, size, size, coverage=True)
        assert owner.shape == coverage.shape == (3 * size, 3 * size)
        np.testing.assert_array_equal(owner >= 0, coverage >= 0)
        assert (owner >= 0).all()


def test_subtexel_owner_matches_blenders_written_set(bake_module, spike):
    for prefix in ('s', 's5'):
        scene, width, height, _ = load(spike, prefix)
        owner = bake_module.subtexel_owner(scene, 0, width, height)
        assert owner.dtype == np.int32
        np.testing.assert_array_equal(owner >= 0, spike[f'{prefix}_written'])
    scene, width, height, _ = load(spike, 's')
    owner = bake_module.subtexel_owner(scene, 0, width, height)
    slots = spike['s_slots']
    np.testing.assert_array_equal(owner[slots[:, 0], slots[:, 1]], spike['s_slot_owner'])


# ---------- the plan

def test_plan_atlas_reproduces_the_spike_slots(bake_module, spike):
    for prefix, total, flagged in (('s', 3545, 730), ('s5', 633, 0)):
        scene, width, height, atlas = planned(bake_module, spike, prefix)
        front = len(spike[f'{prefix}_slots'])
        assert len(atlas.slots) == total and front == total - flagged
        np.testing.assert_array_equal(atlas.slots[:front], spike[f'{prefix}_slots'][:, ::-1])
        np.testing.assert_array_equal(atlas.owner[:front], spike[f'{prefix}_slot_owner'])
        want = scene['tri_ignore_normals'][atlas.owner[:front]]
        np.testing.assert_array_equal(atlas.role[:front], want.astype(np.int8))
        assert (atlas.role[front:] == 2).all()
        marked = np.flatnonzero(want)
        np.testing.assert_array_equal(atlas.slots[front:], atlas.slots[marked])
        np.testing.assert_array_equal(atlas.owner[front:], atlas.owner[marked])
        np.testing.assert_array_equal(atlas.twin[marked], front + np.arange(len(marked)))
        np.testing.assert_array_equal(atlas.twin[front:], marked)
        assert (atlas.twin[:front][~want] == -1).all()
        assert atlas.slots.dtype == np.int64 and atlas.owner.dtype == np.int32 and atlas.role.dtype == np.int8
        assert atlas.side == 0 and atlas.triangles is None


def test_enhanced_and_glow_atlases_have_one_copy(bake_module, spike):
    scene, width, height, _ = load(spike, 's')
    owner = bake_module.subtexel_owner(scene, 0, width, height)
    atlas = bake_module.plan_atlas(spike['s_written'], spike['s_mask'], owner, np.zeros(len(scene['tris']), bool))
    assert len(atlas.slots) == 2815
    assert (atlas.role == 0).all() and (atlas.twin == -1).all()


def test_unowned_written_subtexel_sends_the_tier_back_to_the_page_bake(bake_module, spike):
    scene, width, height, two_sided = load(spike, 's')
    owner = bake_module.subtexel_owner(scene, 0, width, height)
    col, row = spike['s_slots'][0][::-1]
    owner[row, col] = -1
    assert bake_module.plan_atlas(spike['s_written'], spike['s_mask'], owner, two_sided) is None


# ---------- the triangle of a slot

def check_slot_triangles(bake_module, scene, atlas, width, height):
    corners = page_corners(bake_module, scene, width, height)
    for k in range(len(atlas.slots)):
        col, row = atlas.slots[k]
        owner, tri = corners[atlas.owner[k]], atlas.triangles[k]
        square = footprint(col, row)
        inside_tri = clip(square, tri)
        want = area(clip(square, owner))
        assert abs(area(inside_tri) - want) <= 1e-9
        assert area(inside_tri) - area(clip(inside_tri, owner)) <= 1e-9


def test_slot_triangle_matches_the_owner_inside_the_footprint(bake_module, placed_spike):
    for prefix in ('s', 's5'):
        module, scene, width, height, atlas = placed_spike[prefix]
        check_slot_triangles(module, scene, atlas, width, height)
        assert set(np.unique(atlas.needed)) <= {0, 1, 2, 3}
    counts = np.bincount(placed_spike['s5'][4].needed, minlength=4)
    assert counts[0] == 0 and counts[1] == 157 and counts[2] == 334 and counts[3] == 142
    counts = np.bincount(placed_spike['s'][4].needed[:2815], minlength=4)
    assert counts.tolist() == [1422, 1281, 112, 0]


def constructed(bake_module, pixels, slots, width=8, height=8):
    # a lone triangle given in sub-texel pixels (y up, as Blender rows), flipped back to the file's v
    pixels = np.float64(pixels)
    uv = np.column_stack([pixels[:, 0] / (3 * width), 1 - pixels[:, 1] / (3 * height)])
    scene = scene_of([[0, 0, 0], [1, 0, 0], [0, 1, 0]], uv)
    atlas = manual_atlas(bake_module, slots, np.zeros(len(slots), np.int32))
    return scene, bake_module.place_atlas(scene, atlas, width, height)


def test_slot_triangle_handles_zero_to_three_needed_edges(bake_module):
    big = [(2.3, 2.2), (18.7, 3.1), (5.1, 19.4)]
    scene, atlas = constructed(bake_module, big, [(8, 8), (10, 2), (18, 3), (3, 8)])
    assert atlas.needed.tolist() == [0, 1, 2, 1]
    check_slot_triangles(bake_module, scene, atlas, 8, 8)
    corners = page_corners(bake_module, scene, 8, 8)[0]
    for k in np.flatnonzero(atlas.needed >= 2):
        for corner in atlas.triangles[k]:
            assert (barycentric(corner, corners) >= -1e-9).all() and (barycentric(corner, corners) <= 1 + 1e-9).all()
    # a triangle smaller than the sub-texel: the owner itself
    tiny = [(5.2, 5.3), (5.6, 5.4), (5.3, 5.8)]
    scene, atlas = constructed(bake_module, tiny, [(5, 5)])
    assert atlas.needed.tolist() == [3]
    check_slot_triangles(bake_module, scene, atlas, 8, 8)
    np.testing.assert_allclose(atlas.triangles[0], page_corners(bake_module, scene, 8, 8)[0])


def test_slot_triangle_covers_slivers_and_mirrored_charts(bake_module):
    # 0.05 m at 8 texels/m is 1.2 sub-texels wide; the second is the first as a mirrored chart
    sliver = [(2.0, 3.1), (20.5, 11.3), (20.5, 12.5)]
    for pixels in (sliver, sliver[::-1]):
        scene = scene_of([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.column_stack([np.float64(pixels)[:, 0] / 24,
                                                                           1 - np.float64(pixels)[:, 1] / 24]))
        owner = bake_module.subtexel_owner(scene, 0, 8, 8)
        rows, cols = np.nonzero(owner >= 0)
        assert len(rows) >= 5
        atlas = bake_module.place_atlas(scene, manual_atlas(bake_module, np.column_stack([cols, rows]),
                                                             np.zeros(len(rows), np.int32)), 8, 8)
        check_slot_triangles(bake_module, scene, atlas, 8, 8)
        assert set(np.unique(atlas.needed)) <= {1, 2, 3}


def test_no_slot_triangle_covers_another_slots_centre(placed_spike):
    for prefix in ('s', 's5'):
        module, scene, width, height, atlas = placed_spike[prefix]
        at = {tuple(p): k for k, p in enumerate(atlas.pixels)}
        assert len(at) == len(atlas.pixels)
        assert ((atlas.pixels - 2) % 4 == 0).all()
        for k in range(len(atlas.slots)):
            tri = atlas.triangles[k] - atlas.slots[k] + atlas.pixels[k]
            orient = np.sign((tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0]))

            def depth(x, y):
                return min(orient * ((q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])) / np.hypot(*(q - p))
                           for p, q in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
            assert depth(atlas.pixels[k][0] + 0.501, atlas.pixels[k][1] + 0.502) >= 0
            first = np.maximum(0, np.floor((tri.min(axis=0) - 3) / 4)).astype(int)
            last = np.ceil((tri.max(axis=0) - 1) / 4).astype(int)
            for i in range(first[0], last[0] + 1):
                for j in range(first[1], last[1] + 1):
                    other = at.get((2 + 4 * i, 2 + 4 * j))
                    if other is not None and other != k:
                        assert depth(2 + 4 * i + 0.5, 2 + 4 * j + 0.5) < -0.01, (k, other)


def test_place_atlas_reproduces_the_spike_layout(placed_spike, spike):
    for prefix, side in (('s', 280), ('s5', 136)):
        atlas = placed_spike[prefix][4]
        assert atlas.side == int(spike[f'{prefix}_v4o_side']) == side
        np.testing.assert_array_equal(atlas.pixels, spike[f'{prefix}_v4o_positions'])
        assert atlas.triangles.shape == (len(atlas.slots), 3, 2) and atlas.needed.dtype == np.int8


def test_place_atlas_refuses_past_max_atlas(bake_module, spike, monkeypatch):
    scene, width, height, atlas = planned(bake_module, spike, 's5')
    monkeypatch.setattr(bake_module, 'MAX_ATLAS', 16)
    assert bake_module.place_atlas(scene, atlas, width, height) is None


# ---------- the mesh

def test_atlas_uv_keeps_the_slot_footprint(placed_spike):
    for prefix in ('s', 's5'):
        module, scene, width, height, atlas = placed_spike[prefix]
        mesh = module.atlas_mesh(scene, atlas, width, height)
        count = len(atlas.slots)
        assert mesh['atlas_uv'].shape == (3 * count, 2) and mesh['atlas_uv'].dtype == np.float32
        page = (mesh['atlas_uv'].astype(np.float64).reshape(count, 3, 2) * atlas.side
                - atlas.pixels[:, None] + atlas.slots[:, None])
        order = corner_map(mesh, atlas)
        want = atlas.triangles[np.arange(count)[:, None], order]
        assert np.abs(page - want).max() < 1e-4


def expected_corners(module, scene, atlas, mesh, width, height):
    count = len(atlas.owner)
    order = corner_map(mesh, atlas)
    page = page_corners(module, scene, width, height)[atlas.owner]
    lam = np.array([[barycentric(atlas.triangles[k, order[k, i]], page[k]) for i in range(3)] for k in range(count)])
    return lam


def true_bake_corners(scene):
    corners = scene['positions'][scene['tris']].astype(np.float64)
    out = corners.astype(np.float32).astype(np.float64)
    for t in np.flatnonzero(scene['tri_twin'][:len(corners)] >= 0):
        c = corners[t]
        n = np.cross(c[1] - c[0], c[2] - c[0])
        n /= np.linalg.norm(n)
        out[t] = (c + separation(c[None])[0] * n).astype(np.float32)
    return out


def separation(corners):
    spacing = np.spacing(np.abs(corners).astype(np.float32)).max(axis=(1, 2))
    return np.maximum(1e-4, 4 * spacing.astype(np.float64))


def test_atlas_geometry_lies_on_its_owner_plane_at_the_offset(placed_spike):
    module, scene, width, height, atlas = placed_spike['s']
    mesh = module.atlas_mesh(scene, atlas, width, height)
    assert mesh['positions'].shape == (3 * len(atlas.slots), 3) and mesh['positions'].dtype == np.float32
    np.testing.assert_array_equal(mesh['tris'], np.arange(3 * len(atlas.slots)).reshape(-1, 3))
    np.testing.assert_array_equal(mesh['material_tri'], atlas.owner)
    lam = expected_corners(module, scene, atlas, mesh, width, height)
    bake = true_bake_corners(scene)[atlas.owner]
    normal = np.cross(bake[:, 1] - bake[:, 0], bake[:, 2] - bake[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    sign = np.where(atlas.role == 2, -1.0, 1.0)
    want = np.einsum('nkb,nbd->nkd', lam, bake) + (sign * separation(bake))[:, None, None] * normal[:, None]
    np.testing.assert_allclose(mesh['positions'].reshape(-1, 3, 3), want, rtol=1e-5, atol=2e-6)
    assert (atlas.role == 2).any()


def test_twin_geometry_takes_the_twin_offset(placed_spike):
    module, scene, width, height, atlas = placed_spike['s']
    mesh = module.atlas_mesh(scene, atlas, width, height)
    paired = scene['tri_twin'][atlas.owner] >= 0
    assert paired.sum() == 690
    raw = scene['positions'][scene['tris']].astype(np.float64)[atlas.owner[paired]]
    normal = np.cross(raw[:, 1] - raw[:, 0], raw[:, 2] - raw[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    height_above = np.einsum('nkd,nd->nk', mesh['positions'].reshape(-1, 3, 3)[paired].astype(np.float64) - raw[:, :1], normal)
    # the bake mesh moves a twin off its partner, and the slot sits the same distance again further out
    np.testing.assert_allclose(height_above, np.tile(2 * separation(raw)[:, None], 3), atol=2e-6)


def test_atlas_triangles_face_the_owner_normal(bake_module, placed_spike):
    for module, scene, width, height, atlas in (placed_spike['s'], placed_spike['s5']):
        mesh = module.atlas_mesh(scene, atlas, width, height)
        check_facing(scene, atlas, mesh)
    # a chart whose lightmap triangle runs the other way round than its 3D one
    scene = scene_of([[0, 0, 0], [3, 0, 0], [0, 3, 0]], [[0.1, 0.9], [0.1, 0.1], [0.9, 0.9]], flagged=True)
    owner = bake_module.subtexel_owner(scene, 0, 8, 8)
    atlas = bake_module.plan_atlas(owner >= 0, np.ones((8, 8), bool), owner, np.array([True]))
    atlas = bake_module.place_atlas(scene, atlas, 8, 8)
    assert (atlas.role == 2).any()
    check_facing(scene, atlas, bake_module.atlas_mesh(scene, atlas, 8, 8))


def check_facing(scene, atlas, mesh):
    bake = true_bake_corners(scene)[atlas.owner]
    normal = np.cross(bake[:, 1] - bake[:, 0], bake[:, 2] - bake[:, 0])
    p = mesh['positions'].astype(np.float64).reshape(-1, 3, 3)
    facing = np.einsum('nd,nd->n', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), normal)
    assert (facing > 0).all()


def test_atlas_normals_and_texture_uv_follow_the_owner(bake_module):
    positions = [[0, 0, 0], [3, 0, 1], [0, 3, 2]]
    normals = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    tex = [[0.0, 0.0], [2.0, 0.5], [0.25, 3.0]]
    scene = scene_of(positions, [[0.1, 0.1], [0.9, 0.2], [0.2, 0.8]], normals, tex)
    owner = bake_module.subtexel_owner(scene, 0, 8, 8)
    atlas = bake_module.place_atlas(scene, bake_module.plan_atlas(owner >= 0, np.ones((8, 8), bool), owner, np.array([False])), 8, 8)
    mesh = bake_module.atlas_mesh(scene, atlas, 8, 8)
    lam = expected_corners(bake_module, scene, atlas, mesh, 8, 8)
    order = corner_map(mesh, atlas)
    assert order.shape == (len(atlas.slots), 3)
    flipped = bake_module.flip_v(scene['tex_uv']).astype(np.float64)
    np.testing.assert_allclose(mesh['normals'].reshape(-1, 3, 3), lam @ np.float64(normals), atol=1e-5)
    np.testing.assert_allclose(mesh['tex_uv'].reshape(-1, 3, 2), lam @ flipped, atol=1e-5)
    # the three vertices differ, so a flat face normal could not satisfy this
    assert np.ptp(mesh['normals'][:, 0]) > 0.1


# ---------- reading the baked atlas back

def test_scatter_averages_both_halves(bake_module):
    slots = [(0, 0), (4, 1), (7, 2), (1, 5), (2, 7), (4, 1), (7, 2)]
    role = [0, 1, 1, 0, 0, 2, 2]
    twin = [-1, 5, 6, -1, -1, 1, 2]
    atlas = dataclasses.replace(manual_atlas(bake_module, slots, np.zeros(7), role, twin),
                                pixels=np.int64([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]]), side=8)
    values = np.full((8, 8, 3), -1, np.float32)
    values[0, 0] = [1, 2, 3]
    values[0, 1] = [2, 4, 6]
    values[0, 5] = [4, 0, 2]
    values[0, 2] = [8, 8, 8]
    values[0, 3] = [5, 5, 5]
    buffer, missing = bake_module.scatter_atlas(atlas, values, (9, 9))
    want = np.full((9, 9, 3), -1, np.float32)
    want[0, 0] = [1, 2, 3]
    want[1, 4] = [3, 2, 4]
    want[5, 1] = [5, 5, 5]
    np.testing.assert_array_equal(buffer, want)
    # the pair at (7, 2) lacks its back half and (2, 7) was never written
    assert buffer.dtype == np.float32 and missing == 2
    page, written = bake_module.downsample_written(buffer)
    reference = np.full((9, 9, 3), -1, np.float32)
    reference[0, 0], reference[1, 4], reference[5, 1] = want[0, 0], want[1, 4], want[5, 1]
    expected, expected_written = bake_module.downsample_written(reference)
    np.testing.assert_array_equal(page, expected)
    np.testing.assert_array_equal(written, expected_written)
