"""Sun light on faces whose plane turns away from it, from their lit corners, as tool's vertex pass gives it.

The bake lights a receiver only where its plane faces the sun. tool instead lights each corner whose smooth normal
faces the sun and whose shadow ray is clear, then interpolates the corners over the triangle, so a face that turns
away at a ridge still shades off smoothly from its lit corners. This adds that term to the faces the bake leaves
dark, in the same units as the baked page, and touches no other texel.

Ignore-normals receivers instead take RGB shadow throughput from every reached corner at full strength,
without the plane, cosine or boundary-distance tests. Their Cycles objects have no linked suns.
"""

from dataclasses import dataclass

import numpy as np

from .raster import triangle_weights


# sub-texels handled at a time: bounds the per-sub-texel temporaries, not the result
CHUNK = 1 << 18
# 25 ray epsilons clears the fold; the opposite-vertex fraction is capped so even narrow faces retain edge-local probes
EDGE_INSET = 0.025


@dataclass
class SunBoundaries:
    away: np.ndarray
    edge_lit: np.ndarray | None
    segments: list | None


def prepare_boundaries(scene, suns, reach, edge_sun_lit=None):
    """Mesh-sized boundary tables shared by every page of one bake."""
    if not suns:
        return SunBoundaries(np.empty((len(scene.tris), 0), bool), None, [])
    centre = np.array([sun.get('centre', sun['direction']) for sun in suns], np.float64)
    facing = _planes(scene, scene.tris) @ centre.T > 0
    lights = np.stack([np.isin(scene.tri_class, sun['classes']) for sun in suns], axis=1)
    away = ~facing & lights
    if scene.tri_ignore_normals is not None:
        away &= ~scene.tri_ignore_normals[:, None]
    if edge_sun_lit is None:
        return SunBoundaries(away, _lit_edges(scene, scene.tris, facing & lights), None)
    return SunBoundaries(away, None, _segment_tables(scene, suns, reach, scene.tris, edge_sun_lit, facing))


def sun_term(scene, suns, lit, page, width, height, reach, edge_sun_lit=None, boundaries=None):
    """(height, width, 3) light to add to page `page`'s raw float image; `suns` are the job's suns, `lit` the vertex
    pass's (query, sun) result."""
    term = np.zeros((height, width, 3), np.float32)
    rows = np.flatnonzero(scene.tri_page == page)
    if not len(rows) or not len(suns):
        return term
    tris = scene.tris[rows]
    if boundaries is None:
        boundaries = prepare_boundaries(scene, suns, reach, edge_sun_lit)
    corners = scene.positions[tris].astype(np.float64)
    smooth = scene.normals[tris].astype(np.float64)
    direction = np.array([sun['direction'] for sun in suns], np.float64)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    away = boundaries.away[rows]
    if not away.any():
        return term
    length = np.linalg.norm(smooth, axis=2, keepdims=True)
    unit = smooth / np.where(length > 0, length, 1)
    cosine = np.maximum(np.einsum('tck,sk->tcs', unit, direction), 0)
    visible = np.asarray(lit, np.float64)[scene.query_inverse[tris]]
    # tool's colour times the strength the page units carry: a sun of strength P*pi bakes to P*cos
    power = np.array([sun['color'] for sun in suns], np.float64) * np.array([sun['strength'] for sun in suns])[:, None] / np.pi
    corner = cosine * visible * away[:, None, :]
    # the bake averages a texel's 3x3 sub-texels that a triangle covers, so the term is built on that grid and
    # averaged the same way: a texel on a seam takes the added light for the sub-texels of the face that lacks it
    # and the baked light for the others
    owner, weights = triangle_weights(scene.lm_uv, tris, 3 * width, 3 * height)
    covered = owner >= 0
    count = covered.reshape(height, 3, width, 3).sum(axis=(1, 3))
    # only sub-texels of a triangle that is away for some sun can take a term; the rest would run the distance
    # math to be multiplied by zero
    wanted = covered & away.any(axis=1)[np.maximum(owner, 0)]
    tri = owner[wanted]
    weight = weights[wanted]
    sampled = boundaries.segments
    edge_lit = boundaries.edge_lit[rows] if boundaries.edge_lit is not None else None
    corners32 = corners.astype(np.float32)
    sub = np.zeros((len(tri), 3), np.float32)
    # chunked so the (n, 3, 3) corner gather and the distance temporaries stay a bounded size on a big page
    for start in range(0, len(tri), CHUNK):
        part = slice(start, start + CHUNK)
        t, w = tri[part], weight[part]
        around = corners32[t]
        point = np.einsum('nc,nck->nk', w, around)
        for k in range(len(suns)):
            value = np.einsum('nc,nc->n', w, corner[t, :, k].astype(np.float32))
            distance = _lit_distance(point, around, edge_lit[t, :, k] if edge_lit is not None else None,
                                     visible[t, :, k] >= 1)
            if sampled is not None:
                distance = np.minimum(distance, _segment_distance(point, around, rows[t], sampled[k]))
            sub[part] += (value * np.maximum(0, 1 - distance / np.float32(reach)))[:, None] * power[k].astype(np.float32)
    sub_image = np.zeros((3 * height, 3 * width, 3), np.float32)
    sub_image[wanted] = sub
    total = sub_image.reshape(height, 3, width, 3, 3).sum(axis=(1, 3), dtype=np.float64)
    return (total / np.maximum(count, 1)[:, :, None]).astype(np.float32)


def _planes(scene, tris):
    corners = scene.positions[tris].astype(np.float32).astype(np.float64)
    plane = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    smooth = scene.normals[tris].astype(np.float32).astype(np.float64)
    # (the plane takes the side of the smooth normals, as the bake's light linking does)
    return plane * np.where((plane * smooth.sum(axis=1)).sum(axis=1) < 0, -1, 1)[:, None]


def flagged_fraction(scene, page, width, height):
    fraction = np.zeros((height, width), np.float32)
    if scene.tri_ignore_normals is None or not scene.tri_ignore_normals.any():
        return fraction
    rows = np.flatnonzero(scene.tri_page == page)
    if not len(rows) or not scene.tri_ignore_normals[rows].any():
        return fraction
    owner, _ = triangle_weights(scene.lm_uv, scene.tris[rows], 3 * width, 3 * height)
    covered = owner >= 0
    flagged = covered & scene.tri_ignore_normals[rows][np.maximum(owner, 0)]
    count = covered.reshape(height, 3, width, 3).sum(axis=(1, 3))
    return (flagged.reshape(height, 3, width, 3).sum(axis=(1, 3)) / np.maximum(count, 1)).astype(np.float32)


def flat_sun_term(scene, suns, throughput, page, width, height):
    term = np.zeros((height, width, 3), np.float32)
    if not suns or scene.tri_ignore_normals is None:
        return term
    rows = np.flatnonzero(scene.tri_page == page)
    if not len(rows) or not scene.tri_ignore_normals[rows].any():
        return term
    tris = scene.tris[rows]
    owner, weights = triangle_weights(scene.lm_uv, tris, 3 * width, 3 * height)
    covered = owner >= 0
    wanted = covered & scene.tri_ignore_normals[rows][np.maximum(owner, 0)]
    count = covered.reshape(height, 3, width, 3).sum(axis=(1, 3))
    tri, weight = owner[wanted], weights[wanted]
    queries = scene.query_inverse[tris]
    sub = np.zeros((len(tri), 3), np.float32)
    for k, sun in enumerate(suns):
        power = np.asarray(sun['color'], np.float32) * np.float32(sun['strength'] / np.pi)
        corner = np.asarray(throughput[:, k], np.float32)[queries]
        corner *= np.isin(scene.tri_class[rows], sun['classes'])[:, None, None]
        for first in range(0, len(tri), CHUNK):
            part = slice(first, first + CHUNK)
            sub[part] += np.einsum('nc,nck->nk', weight[part], corner[tri[part]]) * power
    image = np.zeros((3 * height, 3 * width, 3), np.float32)
    image[wanted] = sub
    total = image.reshape(height, 3, width, 3, 3).sum(axis=(1, 3), dtype=np.float64)
    return (total / np.maximum(count, 1)[:, :, None]).astype(np.float32)


def flat_light_term(scene, light_flat, page, width, height):
    term = np.zeros((height, width, 3), np.float32)
    if scene.tri_ignore_normals is None:
        return term
    rows = np.flatnonzero(scene.tri_page == page)
    if not len(rows) or not scene.tri_ignore_normals[rows].any():
        return term
    tris = scene.tris[rows]
    owner, weights = triangle_weights(scene.lm_uv, tris, 3 * width, 3 * height)
    covered = owner >= 0
    wanted = covered & scene.tri_ignore_normals[rows][np.maximum(owner, 0)]
    count = covered.reshape(height, 3, width, 3).sum(axis=(1, 3))
    tri, weight = owner[wanted], weights[wanted]
    corner = np.asarray(light_flat, np.float32)[scene.query_inverse[tris]]
    sub = np.zeros((len(tri), 3), np.float32)
    for first in range(0, len(tri), CHUNK):
        part = slice(first, first + CHUNK)
        sub[part] = np.einsum('nc,nck->nk', weight[part], corner[tri[part]])
    image = np.zeros((3 * height, 3 * width, 3), np.float32)
    image[wanted] = sub
    total = image.reshape(height, 3, width, 3, 3).sum(axis=(1, 3), dtype=np.float64)
    return (total / np.maximum(count, 1)[:, :, None]).astype(np.float32)


def _lit_edges(scene, tris, facing):
    """(tris, 3, suns): whether each edge (corner c to c+1) is shared, by both endpoint positions, with a triangle
    that is plane-facing for the sun, which is where tool's mesh has light up to the edge."""
    _, ids = np.unique(scene.positions, axis=0, return_inverse=True)
    ids = ids.ravel().astype(np.int64)
    base = int(ids.max()) + 1

    def keys(t):
        a = ids[t]
        b = ids[np.roll(t, -1, axis=1)]
        return np.minimum(a, b) * base + np.maximum(a, b)

    own, all_keys = keys(tris), keys(scene.tris)
    lit = np.zeros(own.shape + (facing.shape[1],), bool)
    for k in range(facing.shape[1]):
        lit[:, :, k] = np.isin(own, all_keys[facing[:, k]])
    return lit


def _edge_keys(scene, tris=None):
    if tris is None:
        tris = scene.tris
    _, ids = np.unique(scene.positions, axis=0, return_inverse=True)
    ids = ids.ravel().astype(np.int64)
    a, b = ids[tris], ids[np.roll(tris, -1, axis=1)]
    return np.minimum(a, b) * (int(ids.max(initial=-1)) + 1) + np.maximum(a, b), a > b


def edge_queries(scene, suns, reach, facing=None):
    """Facing-side probes, ordered by welded edge then facing triangle; endpoints stay in the sample grid."""
    empty = dict(edge_pos=np.empty((0, 3), np.float32), edge_normal=np.empty((0, 3), np.float32),
                 edge_class=np.empty(0, np.int16), edge_active=np.empty((0, len(suns)), bool),
                 edge_keys=np.empty(0, np.int64), edge_offsets=np.zeros(1, np.int64))
    if not suns or not len(scene.tris):
        return empty
    keys, reverse = _edge_keys(scene)
    centre = np.array([s.get('centre', s['direction']) for s in suns], np.float64).reshape(-1, 3)
    if facing is None:
        facing = _planes(scene, scene.tris) @ centre.T > 0
    lights = np.stack([np.isin(scene.tri_class, s['classes']) for s in suns], axis=1) if suns else facing
    ignore_normals = (scene.tri_ignore_normals if scene.tri_ignore_normals is not None
                      else np.zeros(len(scene.tris), bool))
    adjacency = {}
    for t, row in enumerate(keys):
        for c, key in enumerate(row):
            adjacency.setdefault(int(key), []).append((t, c))
    positions, normals, classes, masks, group_keys, offsets = [], [], [], [], [], [0]
    for key, owners in sorted(adjacency.items()):
        away = np.any([~facing[t] & lights[t] & ~ignore_normals[t] for t, _ in owners], axis=0)
        if not away.any():
            continue
        seen = set()
        for t, c in owners:
            active = away & facing[t] & lights[t]
            if not active.any():
                continue
            ia, ib = scene.tris[t, c], scene.tris[t, (c + 1) % 3]
            if reverse[t, c]:
                ia, ib = ib, ia
            a, b = scene.positions[[ia, ib]].astype(np.float32)
            along = b - a
            length = float(np.linalg.norm(along))
            if length == 0:
                continue
            ic = scene.tris[t, (c + 2) % 3]
            target = scene.positions[ic]
            other = target - a
            inward = other - along * (other @ along / (length * length))
            altitude = float(np.linalg.norm(inward))
            if altitude == 0:
                continue
            fraction = np.float32(min(EDGE_INSET / altitude, 0.25))
            # seam copies share rays only when the opposite-vertex blend and its smooth normals agree
            signature = (fraction.tobytes(), target.tobytes(), scene.normals[[ia, ib, ic]].tobytes(), int(scene.tri_class[t]))
            if signature in seen:
                continue
            seen.add(signature)
            count = max(1, int(np.ceil(length / reach))) + 1
            f = np.linspace(0, 1, count, dtype=np.float32)[:, None]
            positions.append((1 - fraction) * (a + f * along) + fraction * target)
            normals.append((1 - fraction) * ((1 - f) * scene.normals[ia] + f * scene.normals[ib])
                           + fraction * scene.normals[ic])
            classes.append(np.full(count, scene.tri_class[t], np.int16))
            masks.append(np.broadcast_to(active, (count, len(suns))))
            group_keys.append(key)
            offsets.append(offsets[-1] + count)
    return dict(edge_pos=np.concatenate(positions).astype(np.float32) if positions else np.empty((0, 3), np.float32),
                edge_normal=np.concatenate(normals).astype(np.float32) if normals else np.empty((0, 3), np.float32),
                edge_class=np.concatenate(classes) if classes else np.empty(0, np.int16),
                edge_active=np.concatenate(masks) if masks else np.empty((0, len(suns)), bool),
                edge_keys=np.asarray(group_keys, np.int64), edge_offsets=np.asarray(offsets, np.int64))


def _segment_tables(scene, suns, reach, tris, lit, facing=None):
    edge = edge_queries(scene, suns, reach, facing)
    lit = np.asarray(lit, bool) & edge['edge_active']
    # one shared grid per welded edge, even when a non-manifold edge has several facing sides
    grids = {}
    for key, first, end in zip(edge['edge_keys'], edge['edge_offsets'][:-1], edge['edge_offsets'][1:]):
        segments = lit[first:end - 1] & lit[first + 1:end]
        if key in grids:
            grids[key] |= segments
        else:
            grids[key] = segments.copy()
    ordered = np.array([-1, *sorted(grids)], np.int64)
    counts = np.array([1, *[len(grids[key]) for key in ordered[1:]]], np.int64)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    # page triangles retain their original order, but need not be consecutive in the scene
    own_keys, own_reverse = _edge_keys(scene, tris)
    own_slot = np.searchsorted(ordered, own_keys)
    own_slot = np.minimum(own_slot, len(ordered) - 1)
    own_slot = np.where(ordered[own_slot] == own_keys, own_slot, 0)
    tables = []
    for k in range(len(suns)):
        before, after = [], []
        for key in ordered:
            enabled = np.zeros(1, bool) if key == -1 else grids[key][:, k]
            index = np.arange(len(enabled), dtype=np.int64)
            before.append(np.maximum.accumulate(np.where(enabled, index, -1)))
            after.append(np.minimum.accumulate(np.where(enabled, index, len(enabled))[::-1])[::-1])
        tables.append((own_slot, own_reverse, counts, starts, np.concatenate(before), np.concatenate(after)))
    return tables


def _segment_distance(point, corners, tri, table):
    slots, reverse, counts, starts, before, after = table
    best = np.full(len(point), np.inf, np.float32)
    for c in range(3):
        flipped = reverse[tri, c, None]
        a = np.where(flipped, corners[:, (c + 1) % 3], corners[:, c])
        b = np.where(flipped, corners[:, c], corners[:, (c + 1) % 3])
        along = b - a
        f = np.einsum('nk,nk->n', point - a, along) / np.maximum(np.einsum('nk,nk->n', along, along), 1e-30)
        slot = slots[tri, c]
        n = counts[slot]
        n32 = n.astype(np.float32)
        cell = np.minimum((np.clip(f, 0, 1) * n32).astype(np.int64), n - 1)
        address = starts[slot] + cell
        for index in (before[address], after[address]):
            valid = (index >= 0) & (index < n)
            lo = index.astype(np.float32) / n32
            hi = (index + 1).astype(np.float32) / n32
            near = a + np.clip(f, lo, hi)[:, None] * along
            best = np.minimum(best, np.where(valid, np.linalg.norm(point - near, axis=1), np.inf))
    return best


def _lit_distance(point, corners, edge_lit, corner_lit):
    """Per point, the distance to the nearest lit edge or lit corner of its triangle (inf with none)."""
    best = np.full(len(point), np.inf, np.float32)
    for c in range(3):
        a, b = corners[:, c], corners[:, (c + 1) % 3]
        if edge_lit is not None:
            along = b - a
            t = np.einsum('nk,nk->n', point - a, along) / np.maximum(np.einsum('nk,nk->n', along, along), 1e-30)
            near = a + np.clip(t, 0, 1)[:, None] * along
            best = np.minimum(best, np.where(edge_lit[:, c], np.linalg.norm(point - near, axis=1), np.inf))
        best = np.minimum(best, np.where(corner_lit[:, c], np.linalg.norm(point - a, axis=1), np.inf))
    return best
