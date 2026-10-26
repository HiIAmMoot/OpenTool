"""Use one texel-centre convention for coverage, padding and page sampling.

Row zero is v=0, matching the first bitmap row. Coverage includes edges;
shared centres take the class of the lowest input triangle index.
"""

import math

import numpy as np
from numba import njit


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
