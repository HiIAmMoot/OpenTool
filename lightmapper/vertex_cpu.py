"""Compiled CPU vertex tracing and frozen-build cache probe.

The BVH uses balanced median splits along the widest triangle-box centroid
axis, with at most four triangles per leaf. Intersection arithmetic follows
Blender 5.2's float32 watertight test, with lowest-index equal-distance hits.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter

import numba
from numba import njit, prange
import numpy as np


# copies keep numba's source-file cache sensitive to changes in ray constants
RAY_LENGTH = 1e5


@dataclass
class Bvh:
    lo: np.ndarray
    hi: np.ndarray
    first: np.ndarray
    count: np.ndarray
    order: np.ndarray


def build_bvh(corners: np.ndarray) -> Bvh:
    n = len(corners)
    capacity = max(0, 2 * n - 1)
    lo = np.empty((capacity, 3), np.float32)
    hi = np.empty((capacity, 3), np.float32)
    first = np.zeros(capacity, np.int32)
    count = np.zeros(capacity, np.int32)
    order = np.arange(n, dtype=np.int32)
    if not n:
        return Bvh(lo, hi, first, count, order)
    tri_lo, tri_hi = corners.min(axis=1), corners.max(axis=1)
    centers = tri_lo * np.float32(.5) + tri_hi * np.float32(.5)
    pending = [(0, 0, n)]
    used = 1
    while pending:
        node, start, end = pending.pop()
        indices = order[start:end]
        lo[node] = tri_lo[indices].min(axis=0)
        hi[node] = tri_hi[indices].max(axis=0)
        if end - start <= 4:
            first[node], count[node] = start, end - start
            continue
        node_centers = centers[indices]
        axis = np.argmax(node_centers.max(axis=0) - node_centers.min(axis=0))
        middle = (end - start) // 2
        partition = np.argpartition(node_centers[:, axis], middle)
        order[start:end] = indices[partition]
        first[node] = used
        pending.append((used + 1, start + middle, end))
        pending.append((used, start, start + middle))
        used += 2
    return Bvh(lo[:used].copy(), hi[:used].copy(), first[:used].copy(), count[:used].copy(), order)


@njit(cache=True)
def _normalize_direction(d):
    squared = (d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]
    result = np.zeros(3, np.float32)
    if squared > np.float32(1e-35):
        scale = np.float32(1) / np.sqrt(squared)
        for axis in range(3):
            result[axis] = d[axis] * scale
    return result


@njit(cache=True, error_model='numpy')
def _ray_triangle(o, d, a, b, c):
    x, y, z = abs(d[0]), abs(d[1]), abs(d[2])
    if x > y:
        kz = 0 if x > z else 2
    else:
        kz = 1 if y > z else 2
    if d[kz] == np.float32(0) or not np.isfinite(d[kz]):
        return np.float32(np.inf)
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3
    if d[kz] < np.float32(0):
        kx, ky = ky, kx
    sz = np.float32(1) / d[kz]
    sx, sy = d[kx] * sz, d[ky] * sz
    az, bz, cz = a[kz] - o[kz], b[kz] - o[kz], c[kz] - o[kz]
    ax, ay = (a[kx] - o[kx]) - sx * az, (a[ky] - o[ky]) - sy * az
    bx, by = (b[kx] - o[kx]) - sx * bz, (b[ky] - o[ky]) - sy * bz
    cx, cy = (c[kx] - o[kx]) - sx * cz, (c[ky] - o[ky]) - sy * cz
    u = cx * by - cy * bx
    v = ax * cy - ay * cx
    w = bx * ay - by * ax
    if ((u < np.float32(0) or v < np.float32(0) or w < np.float32(0))
            and (u > np.float32(0) or v > np.float32(0) or w > np.float32(0))):
        return np.float32(np.inf)
    det = (u + v) + w
    if det == np.float32(0) or not np.isfinite(det):
        return np.float32(np.inf)
    scaled_t = ((u * az + v * bz) + w * cz) * sz
    sign_t = -scaled_t if np.signbit(det) else scaled_t
    if sign_t < np.float32(0):
        return np.float32(np.inf)
    t = scaled_t * (np.float32(1) / det)
    return t if np.isfinite(t) else np.float32(np.inf)


@njit(cache=True)
def _box_entry(lo, hi, o, d, limit):
    entry, exit = np.float32(0), limit
    for axis in range(3):
        if d[axis] == np.float32(0):
            if o[axis] < lo[axis] or o[axis] > hi[axis]:
                return np.float32(np.inf)
        else:
            a = (lo[axis] - o[axis]) / d[axis]
            b = (hi[axis] - o[axis]) / d[axis]
            near, far = min(a, b), max(a, b)
            # slab rounding must not prune boundary hits or equal-distance twins
            near -= abs(near) * np.float32(1e-6)
            far += abs(far) * np.float32(1e-6)
            entry, exit = max(entry, near), min(exit, far)
            if entry > exit:
                return np.float32(np.inf)
    return entry


@njit(cache=True)
def _closest_hit(lo, hi, first, count, order, corners, o, d, limit):
    best_tri, best_t = np.int64(-1), np.float32(limit)
    if not len(lo) or not limit > np.float32(0):
        return best_tri, np.float32(np.inf)
    heading = _normalize_direction(d)
    if not np.all(np.isfinite(o)) or not np.all(np.isfinite(heading)) or not np.any(heading):
        return best_tri, np.float32(np.inf)
    # median splits bound depth below 32 for int32 triangle indices
    stack = np.empty(64, np.int32)
    stack[0] = 0
    size = 1
    while size:
        size -= 1
        node = stack[size]
        entry = _box_entry(lo[node], hi[node], o, heading, best_t)
        if not np.isfinite(entry) or entry > best_t:
            continue
        if count[node] > 0:
            for item in range(first[node], first[node] + count[node]):
                tri = order[item]
                t = _ray_triangle(o, heading, corners[tri, 0], corners[tri, 1], corners[tri, 2])
                if t < limit and (t < best_t or (t == best_t and tri < best_tri)):
                    best_tri, best_t = np.int64(tri), t
        else:
            left = first[node]
            right = left + 1
            left_t = _box_entry(lo[left], hi[left], o, heading, best_t)
            right_t = _box_entry(lo[right], hi[right], o, heading, best_t)
            if right_t < left_t:
                left, right = right, left
            stack[size], stack[size + 1] = right, left
            size += 2
    if best_tri < 0:
        return best_tri, np.float32(np.inf)
    return best_tri, best_t


@njit(parallel=True, cache=True)
def _closest_hits(lo, hi, first, count, order, corners, origins, dirs, limits):
    indices = np.empty(len(origins), np.int64)
    distances = np.empty(len(origins), np.float32)
    for ray in prange(len(origins)):
        indices[ray], distances[ray] = _closest_hit(lo, hi, first, count, order, corners,
                                                  origins[ray], dirs[ray], limits[ray])
    return indices, distances


def closest_hits(bvh, corners, origins, dirs, limits):
    return _closest_hits(bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order,
                         corners, origins, dirs, limits)


@njit(cache=True)
def brute_force_hits(corners, origins, dirs, limits):
    indices = np.full(len(origins), -1, np.int64)
    distances = np.full(len(origins), np.inf, np.float32)
    for ray in range(len(origins)):
        heading = _normalize_direction(dirs[ray])
        best_tri, best_t = np.int64(-1), limits[ray]
        for tri in range(len(corners)):
            t = _ray_triangle(origins[ray], heading, corners[tri, 0], corners[tri, 1], corners[tri, 2])
            if t < limits[ray] and (t < best_t or (t == best_t and tri < best_tri)):
                best_tri, best_t = np.int64(tri), t
        if best_tri >= 0:
            indices[ray], distances[ray] = best_tri, best_t
    return indices, distances


@njit(parallel=True, cache=True)
def _probe_kernel(values):
    result = np.empty_like(values)
    for i in prange(len(values)):
        result[i] = values[i] * values[i]
    return result


def probe_main(out: Path) -> int:
    values = np.arange(1_000_000, dtype=np.float64)
    started = perf_counter()
    _probe_kernel(values)
    seconds = perf_counter() - started
    stats = _probe_kernel.stats
    out.write_text(json.dumps({
        'threading_layer': numba.threading_layer(),
        'threads': numba.get_num_threads(),
        'cache_hits': int(sum(stats.cache_hits.values())),
        'cache_misses': int(sum(stats.cache_misses.values())),
        'seconds': seconds,
        'cache_dir': os.environ.get('NUMBA_CACHE_DIR'),
    }))
    return 0


def child_main(argv: list[str]) -> int:
    if argv[:1] == ['--probe']:
        return probe_main(Path(argv[1]))
    raise ValueError('expected --probe and an output path')
