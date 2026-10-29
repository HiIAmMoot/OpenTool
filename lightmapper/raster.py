"""Use one texel-centre convention for coverage, padding and page sampling.

Row zero is v=0, matching the first bitmap row. Coverage includes edges;
shared centres take the class of the lowest input triangle index.
"""

import math

import numpy as np
from numba import njit


def texel_queries(scene, page, width, height):
    rows = np.flatnonzero(scene.tri_page == page)
    flags = scene.tri_ignore_normals
    empty = dict(texel_pos=np.empty((0, 3), np.float32), texel_normal=np.empty((0, 3), np.float32),
                 texel_class=np.empty(0, np.int16), texel_page=np.empty(0, np.int16),
                 texel_xy=np.empty((0, 2), np.int32))
    if flags is None or not flags[rows].any():
        return empty
    owner, weights = triangle_weights(scene.lm_uv, scene.tris[rows], 3 * width, 3 * height)
    y, x = np.nonzero((owner >= 0) & flags[rows[np.maximum(owner, 0)]])
    tri = rows[owner[y, x]]
    corners, bary = scene.tris[tri], weights[y, x]
    return dict(texel_pos=np.einsum('ij,ijk->ik', bary, scene.positions[corners]).astype(np.float32),
                texel_normal=np.einsum('ij,ijk->ik', bary, scene.normals[corners]).astype(np.float32),
                texel_class=scene.tri_class[tri], texel_page=np.full(len(x), page, np.int16),
                texel_xy=np.column_stack([x, y]).astype(np.int32))


def texel_contribution(scene, page, width, height, rgb=None):
    """The gathered contribution and flagged share, using all owned samples rather than nine at chart edges."""
    rows = np.flatnonzero(scene['tri_page'] == page)
    owner, _ = triangle_weights(scene['lm_uv'], scene['tris'][rows], 3 * width, 3 * height)
    owned = (owner >= 0).reshape(height, 3, width, 3).sum(axis=(1, 3))
    chosen = scene['texel_page'] == page
    xy = scene['texel_xy'][chosen] // 3
    count = np.zeros((height, width), np.float32)
    np.add.at(count, (xy[:, 1], xy[:, 0]), 1)
    share = count / np.maximum(owned, 1)
    total = np.zeros((height, width, 3), np.float32)
    if rgb is not None:
        np.add.at(total, (xy[:, 1], xy[:, 0]), rgb[chosen])
        total /= np.maximum(owned, 1)[:, :, None]
    return total, share


def flagged_source(scene, page, width, height, rgb):
    """Keep each flagged sample's received light separate from neighbouring materials, including in the padding."""
    from scipy.ndimage import distance_transform_edt

    rows = np.flatnonzero(scene['tri_page'] == page)
    owner, _ = triangle_weights(scene['lm_uv'], scene['tris'][rows], 3 * width, 3 * height)
    source = np.zeros((3 * height, 3 * width, 3), np.float32)
    chosen = scene['texel_page'] == page
    xy = scene['texel_xy'][chosen]
    source[xy[:, 1], xy[:, 0]] = rgb[chosen]
    if (owner >= 0).any():
        # nearest owned samples extend each chart without averaging across chart or material boundaries
        nearest = distance_transform_edt(owner < 0, return_distances=False, return_indices=True)
        source[owner < 0] = source[nearest[0][owner < 0], nearest[1][owner < 0]]
    return source


# (copies of blender/bake.py's, which cannot be imported here; a test holds them equal)
FLAGGED_REACH = 3


def flagged_subtexels(shape, xy, rgb):
    """Each flagged sample's light at its own sub-texel, spread outward to the sub-texels FLAGGED_REACH steps away
    that no sample holds."""
    height, width = shape
    source = np.zeros((height, width, 3), np.float32)
    filled = np.zeros((height, width), bool)
    source[xy[:, 1], xy[:, 0]] = rgb
    filled[xy[:, 1], xy[:, 0]] = True
    for _ in range(FLAGGED_REACH):
        padded, padded_filled = np.pad(source, ((1, 1), (1, 1), (0, 0))), np.pad(filled, 1)
        grown, grown_filled = source.copy(), filled.copy()
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            near = padded_filled[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] & ~grown_filled
            grown[near] = padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width][near]
            grown_filled |= near
        source, filled = grown, grown_filled
    return source


def flagged_transport(scene, page, width, height, rgb):
    """The flagged samples' own light at sub-texels for the reflected stages, or None on a page without them."""
    chosen = scene['texel_page'] == page
    if not chosen.any():
        return None
    return flagged_subtexels((3 * height, 3 * width), scene['texel_xy'][chosen], rgb[chosen])


def ordinary_transport(scene, page, base):
    """The baked page over its ordinary sub-texels alone (the flagged ones bake black)."""
    height, width = base.shape[:2]
    rows = np.flatnonzero(scene['tri_page'] == page)
    owner, _ = triangle_weights(scene['lm_uv'], scene['tris'][rows], 3 * width, 3 * height)
    owned = (owner >= 0).reshape(height, 3, width, 3).sum(axis=(1, 3))
    xy = scene['texel_xy'][scene['texel_page'] == page] // 3
    flagged = np.zeros((height, width), np.int64)
    np.add.at(flagged, (xy[:, 1], xy[:, 0]), 1)
    ordinary = owned - flagged
    scale = np.where(flagged > 0, owned / np.maximum(ordinary, 1) * (ordinary > 0), 1.0)
    return base * scale[:, :, None]


def texel_ray_counts(scene, pages, base, target):
    counts = np.full(len(scene['texel_pos']), base, np.int64)
    if target <= 0:
        return counts
    for page, (width, height) in enumerate(pages):
        rows = scene['tri_page'] == page
        owner, _ = triangle_weights(scene['lm_uv'], scene['tris'][rows], 3 * width, 3 * height)
        owned = (owner >= 0).reshape(height, 3, width, 3).sum(axis=(1, 3))
        chosen = scene['texel_page'] == page
        xy = scene['texel_xy'][chosen] // 3
        counts[chosen] = np.maximum(base, -(-target // np.maximum(owned[xy[:, 1], xy[:, 0]], 1)))
    return counts


def coverage(uv, tris, tri_class, width, height):
    texels = np.asarray(uv, dtype=np.float64) * np.array([width, height])
    return _coverage(texels, np.asarray(tris), np.asarray(tri_class), width, height)


@njit
def _coverage(texels, tris, tri_class, width, height):
    mask = np.zeros((height, width), dtype=np.bool_)
    cls = np.zeros((height, width), dtype=np.int16)
    # serial input order avoids races at inclusive edges and fixes class ownership
    for index in range(len(tris)):
        a, b, c = tris[index]
        ax, ay = texels[a]
        bx, by = texels[b]
        cx, cy = texels[c]
        x0 = max(0, math.ceil(min(ax, bx, cx) - 0.5))
        x1 = min(width - 1, math.floor(max(ax, bx, cx) - 0.5))
        y0 = max(0, math.ceil(min(ay, by, cy) - 0.5))
        y1 = min(height - 1, math.floor(max(ay, by, cy) - 0.5))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if mask[y, x]:
                    continue
                px, py = x + 0.5, y + 0.5
                ab = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                bc = (cx - bx) * (py - by) - (cy - by) * (px - bx)
                ca = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
                if ((ab >= 0 and bc >= 0 and ca >= 0)
                        or (ab <= 0 and bc <= 0 and ca <= 0)):
                    mask[y, x] = True
                    cls[y, x] = tri_class[index]
    return mask, cls


def triangle_weights(uv, tris, width, height):
    """Per covered texel: the owning triangle (the same one `coverage` picks, -1 elsewhere) and the barycentric
    weights of the texel centre in its corners."""
    texels = np.asarray(uv, dtype=np.float64) * np.array([width, height])
    return _triangle_weights(texels, np.asarray(tris), width, height)


@njit
def _triangle_weights(texels, tris, width, height):
    owner = np.full((height, width), -1, dtype=np.int64)
    weights = np.zeros((height, width, 3), dtype=np.float32)
    for index in range(len(tris)):
        a, b, c = tris[index]
        ax, ay = texels[a]
        bx, by = texels[b]
        cx, cy = texels[c]
        x0 = max(0, math.ceil(min(ax, bx, cx) - 0.5))
        x1 = min(width - 1, math.floor(max(ax, bx, cx) - 0.5))
        y0 = max(0, math.ceil(min(ay, by, cy) - 0.5))
        y1 = min(height - 1, math.floor(max(ay, by, cy) - 0.5))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if owner[y, x] >= 0:
                    continue
                px, py = x + 0.5, y + 0.5
                ab = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                bc = (cx - bx) * (py - by) - (cy - by) * (px - bx)
                ca = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
                if ((ab >= 0 and bc >= 0 and ca >= 0)
                        or (ab <= 0 and bc <= 0 and ca <= 0)):
                    owner[y, x] = index
                    area = ab + bc + ca
                    if area == 0:
                        # (a sliver on a line has no area to divide by; its corners share the texel equally)
                        weights[y, x] = 1 / 3
                    else:
                        weights[y, x, 0] = bc / area
                        weights[y, x, 1] = ca / area
                        weights[y, x, 2] = ab / area
    return owner, weights


@njit
def dilate(img, mask, cls, iterations):
    img, mask, cls = img.copy(), mask.copy(), cls.copy()
    height, width, channels = img.shape
    for _ in range(iterations):
        # each ring reads only the previous iteration, not earlier writes in this one
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


def sample_bilinear(img, uv):
    height, width, _ = img.shape
    uv = np.asarray(uv)
    x = uv[:, 0] * width - 0.5
    y = uv[:, 1] * height - 0.5
    x_floor, y_floor = np.floor(x), np.floor(y)
    wx, wy = (x - x_floor)[:, None], (y - y_floor)[:, None]
    x0 = np.clip(x_floor, 0, width - 1).astype(np.intp)
    x1 = np.clip(x_floor + 1, 0, width - 1).astype(np.intp)
    y0 = np.clip(y_floor, 0, height - 1).astype(np.intp)
    y1 = np.clip(y_floor + 1, 0, height - 1).astype(np.intp)
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bottom = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    return top * (1 - wy) + bottom * wy
