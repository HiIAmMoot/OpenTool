"""The edge top-up atlas: one small triangle per sub-texel, laid out so each bakes what the page path would."""

import dataclasses
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
