"""Compiled CPU vertex tracing and frozen-build cache probe.

The BVH uses balanced median splits along the widest triangle-box centroid
axis, with at most four triangles per leaf. Intersection arithmetic follows
Blender 5.2's float32 watertight test. Lowest-index equal-distance hits are
OpenTool's rule; Blender retains its traversal-order hit.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
import zipfile

import numba
from numba import njit, prange
import numpy as np

from . import frozen
from .sun_facing import sun_facing_queries
from .vertex_gpu import VertexPassResult, ambient_table, edge_rays, hemisphere, query_frames, sphere, sun_rays


# copies keep numba's source-file cache sensitive to changes in ray constants; a test holds them equal to bake.py
RAY_LENGTH = 1e5
RAY_EPSILON = 1e-3
PASS_LIMIT = 8
LUMA = np.float64([0.2126, 0.7152, 0.0722])
GOLDEN_FRACTION = 0.618034
SEED_SPIN = 0.7548776662466927
# about 75 MB of directions per chunk, float64 and float32
CHUNK_RAYS = 2**21


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


@dataclass
class CpuScene:
    corners: np.ndarray
    corners32: np.ndarray
    tris: np.ndarray
    tex_uv: np.ndarray
    lm_uv: np.ndarray
    normals: np.ndarray
    twin: np.ndarray
    casts: np.ndarray
    cutout: np.ndarray
    glows: np.ndarray
    two_sided: np.ndarray
    has_two_sided: bool
    texture_of: np.ndarray
    page_of: np.ndarray
    class_of: np.ndarray
    tint: np.ndarray
    emission: np.ndarray
    texture_table: np.ndarray
    texture_texels: np.ndarray
    page_table: np.ndarray
    page_texels: np.ndarray
    flagged_table: np.ndarray
    flagged_texels: np.ndarray
    has_flagged: bool
    ambient: np.ndarray
    albedo_scale: float
    albedo_fallback: float
    n_recv: int

    def arrays(self):
        # one tuple, in field order, is what the compiled kernels take
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


def _packed(images):
    table = np.zeros((len(images), 3), np.int64)
    parts = []
    offset = 0
    for k, image in enumerate(images):
        if image is None:
            table[k] = (offset, 0, 0)
            continue
        height, width = image.shape[:2]
        table[k] = (offset, width, height)
        parts.append(image.reshape(height * width, -1))
        offset += height * width
    return table, parts


def cpu_scene(scene, job, textures, pages, flagged_pages=None):
    if flagged_pages is not None and len(flagged_pages) != len(pages):
        raise ValueError('flagged_pages must have one entry per page in pages')
    n_recv_vertices = len(scene['positions'])
    positions = np.concatenate([scene['positions'], scene['occ_positions']]).astype(np.float64)
    n_recv = len(scene['tris'])
    n_occ = len(scene['occ_tris'])
    tris = np.concatenate([scene['tris'], scene['occ_tris'] + n_recv_vertices]).astype(np.int64)
    corners = positions[tris]
    emission = np.concatenate([scene['tri_emission'], scene['occ_tri_emission']]).astype(np.float64)
    two_sided = np.zeros(n_recv + n_occ, np.bool_)
    has_two_sided = bool(job.get('ignore_normals_mode') == 'enhanced'
                         and scene.get('tri_ignore_normals', np.empty(0)).any())
    if has_two_sided:
        two_sided[:n_recv] = scene['tri_ignore_normals']
    class_of = np.concatenate([scene['tri_class'], np.full(n_occ, -1)]).astype(np.int64)
    texture_table, texture_parts = _packed([np.float32(t) / 255.0 for t in textures])
    page_table, page_parts = _packed([np.asarray(p, np.float64) for p in pages])
    flagged_table, flagged_parts = _packed([None if f is None else np.asarray(f, np.float64)
                                            for f in flagged_pages or []])
    return CpuScene(
        corners=corners, corners32=corners.astype(np.float32), tris=tris,
        tex_uv=np.concatenate([scene['tex_uv'], scene['occ_uv']]).astype(np.float64),
        # occluders have no page, so their lightmap UVs are never read
        lm_uv=np.concatenate([scene['lm_uv'], np.zeros((len(scene['occ_positions']), 2))]).astype(np.float64),
        normals=np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        twin=np.asarray(scene['tri_twin'], np.int64),
        casts=np.concatenate([scene['tri_casts_shadow'], scene['occ_tri_casts_shadow']]).astype(np.bool_),
        cutout=np.concatenate([scene['tri_cutout'], scene['occ_tri_cutout']]).astype(np.bool_),
        glows=np.concatenate([np.zeros(n_recv, np.bool_), emission[n_recv:].any(axis=1)]),
        two_sided=two_sided, has_two_sided=has_two_sided,
        texture_of=np.concatenate([scene['tri_texture'], scene['occ_tri_texture']]).astype(np.int64),
        page_of=np.concatenate([scene['tri_page'], np.full(n_occ, -1)]).astype(np.int64),
        class_of=class_of,
        tint=np.concatenate([scene['tri_tint'], scene['occ_tri_tint']]).astype(np.float64),
        emission=emission,
        texture_table=texture_table,
        texture_texels=np.concatenate(texture_parts) if texture_parts else np.empty((0, 4), np.float32),
        page_table=page_table,
        page_texels=np.concatenate(page_parts) if page_parts else np.empty((0, 3), np.float64),
        flagged_table=flagged_table,
        flagged_texels=np.concatenate(flagged_parts) if flagged_parts else np.empty((0, 3), np.float64),
        has_flagged=bool(flagged_pages is not None and has_two_sided),
        ambient=ambient_table(job, scene['query_class'], class_of),
        albedo_scale=float(job['albedo_scale']), albedo_fallback=float(job['albedo_fallback']), n_recv=n_recv)


@njit(cache=True)
def _barycentric(a, b, c, p):
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0[0] * v0[0] + v0[1] * v0[1]) + v0[2] * v0[2]
    d01 = (v0[0] * v1[0] + v0[1] * v1[1]) + v0[2] * v1[2]
    d11 = (v1[0] * v1[0] + v1[1] * v1[1]) + v1[2] * v1[2]
    d20 = (v2[0] * v0[0] + v2[1] * v0[1]) + v2[2] * v0[2]
    d21 = (v2[0] * v1[0] + v2[1] * v1[1]) + v2[2] * v1[2]
    denom = d00 * d11 - d01 * d01
    safe = 1.0 if denom == 0 else denom
    v = (d11 * d20 - d01 * d21) / safe
    w = (d00 * d21 - d01 * d20) / safe
    w0, w1, w2 = max((1 - v) - w, 0.0), max(v, 0.0), max(w, 0.0)
    total = (w0 + w1) + w2
    # a zero-area triangle has no meaningful weights, so its centre stands in
    if total > 0:
        return w0 / total, w1 / total, w2 / total
    return 1 / 3, 1 / 3, 1 / 3


@njit(cache=True)
def _interpolate(uv, tris, corners, tri, p):
    w0, w1, w2 = _barycentric(corners[tri, 0], corners[tri, 1], corners[tri, 2], p)
    a, b, c = tris[tri, 0], tris[tri, 1], tris[tri, 2]
    return ((uv[a, 0] * w0 + uv[b, 0] * w1) + uv[c, 0] * w2,
            (uv[a, 1] * w0 + uv[b, 1] * w1) + uv[c, 1] * w2)


@njit(cache=True)
def _sample(texels, offset, width, height, u, v, repeat):
    x, y = u * width - 0.5, v * height - 0.5
    x0, y0 = np.int64(np.floor(x)), np.int64(np.floor(y))
    fx, fy = x - x0, y - y0
    if repeat:
        x1, y1 = (x0 + 1) % width, (y0 + 1) % height
        x0, y0 = x0 % width, y0 % height
    else:
        x1, y1 = min(max(x0 + 1, 0), width - 1), min(max(y0 + 1, 0), height - 1)
        x0, y0 = min(max(x0, 0), width - 1), min(max(y0, 0), height - 1)
    result = np.empty(texels.shape[1], np.float64)
    for channel in range(texels.shape[1]):
        top = (np.float64(texels[offset + y0 * width + x0, channel]) * (1 - fx)
               + np.float64(texels[offset + y0 * width + x1, channel]) * fx)
        bottom = (np.float64(texels[offset + y1 * width + x0, channel]) * (1 - fx)
                  + np.float64(texels[offset + y1 * width + x1, channel]) * fx)
        result[channel] = top * (1 - fy) + bottom * fy
    return result


@njit(cache=True)
def _faces(normal, heading):
    return (normal[0] * heading[0] + normal[1] * heading[1]) + normal[2] * heading[2] < 0


@njit(cache=True)
def _cast(sc, bv, o32, d32, shadow, carried, limit=RAY_LENGTH):
    (corners, corners32, tris, tex_uv, lm_uv, normals, twin, casts, cutout, glows, two_sided, has_two_sided,
     texture_of, page_of, class_of, tint, emission, texture_table, texture_texels, page_table, page_texels,
     flagged_table, flagged_texels, has_flagged, ambient, albedo_scale, albedo_fallback, n_recv) = sc
    lo, hi, first, count, order = bv
    start = o32.copy()
    heading = d32.astype(np.float64)
    step = d32 * np.float32(RAY_EPSILON)
    remaining = limit
    for _ in range(PASS_LIMIT + 1):
        index, distance = _closest_hit(lo, hi, first, count, order, corners32, start, d32, np.float32(remaining))
        if index < 0:
            break
        unit = _normalize_direction(d32)
        location = np.empty(3, np.float32)
        for axis in range(3):
            location[axis] = start[axis] + unit[axis] * distance
        partner = twin[index]
        if partner >= 0 and not _faces(normals[index], heading) and _faces(normals[partner], heading):
            # the hit point stays put; interpolation recomputes weights in the facing twin's corner order
            index = partner
        if casts[index] or (not shadow and glows[index] and _faces(normals[index], heading)):
            hole = False
            if cutout[index] and texture_of[index] >= 0:
                u, v = _interpolate(tex_uv, tris, corners, index, location.astype(np.float64))
                k = texture_of[index]
                hole = _sample(texture_texels, texture_table[k, 0], texture_table[k, 1], texture_table[k, 2],
                               u, v, True)[3] < 0.5
            if not hole:
                return index, location
        if shadow and not casts[index]:
            for axis in range(3):
                carried[axis] *= tint[index, axis]
        start = location + step
        remaining -= np.float64(distance) + RAY_EPSILON
    return np.int64(-1), np.zeros(3, np.float32)


@njit(cache=True)
def _shade_rgb(sc, tri, hit32, dir64, cls, with_ambient, with_pages, with_emission):
    (corners, corners32, tris, tex_uv, lm_uv, normals, twin, casts, cutout, glows, two_sided, has_two_sided,
     texture_of, page_of, class_of, tint, emission, texture_table, texture_texels, page_table, page_texels,
     flagged_table, flagged_texels, has_flagged, ambient, albedo_scale, albedo_fallback, n_recv) = sc
    point = hit32.astype(np.float64)
    albedo = np.full(3, albedo_fallback)
    k = texture_of[tri]
    if k >= 0:
        u, v = _interpolate(tex_uv, tris, corners, tri, point)
        rgba = _sample(texture_texels, texture_table[k, 0], texture_table[k, 1], texture_table[k, 2], u, v, True)
        for channel in range(3):
            albedo[channel] = rgba[channel] * albedo_scale
    if glows[tri] and not casts[tri]:
        albedo[:] = 0
    facing = _faces(normals[tri], dir64)
    result = np.zeros(3)
    if tri >= n_recv:
        if with_ambient and cls >= -1 and cls + 1 < len(ambient):
            result += albedo * ambient[cls + 1]
        if with_emission and facing:
            result += emission[tri]
        return result[0], result[1], result[2]
    if not (facing or two_sided[tri]):
        return 0.0, 0.0, 0.0
    # with two-sided receivers only a facing hit takes ambient and emission
    side = 1.0 if (facing or not has_two_sided) else 0.0
    page_value = np.zeros(3)
    if with_pages:
        lu, lv = _interpolate(lm_uv, tris, corners, tri, point)
        p = page_of[tri]
        if page_table[p, 1] > 0:
            page_value = _sample(page_texels, page_table[p, 0], page_table[p, 1], page_table[p, 2], lu, lv, False)
        if has_flagged and two_sided[tri]:
            width, height = flagged_table[p, 1], flagged_table[p, 2]
            page_value = np.zeros(3)
            if width > 0:
                x = min(max(np.int64(np.floor(lu * width)), 0), width - 1)
                y = min(max(np.int64(np.floor(lv * height)), 0), height - 1)
                page_value = flagged_texels[flagged_table[p, 0] + y * width + x].copy()
    ambient_value = np.zeros(3)
    if with_ambient and class_of[tri] >= -1 and class_of[tri] + 1 < len(ambient):
        ambient_value = ambient[class_of[tri] + 1] * side
    glow = emission[tri] * side if with_emission else np.zeros(3)
    result = albedo * (page_value + ambient_value) + glow
    return result[0], result[1], result[2]


@njit(parallel=True, cache=True)
def _probe_rays(sc, bv, origins, dirs, classes):
    n = len(origins)
    tri = np.full(n, -1, np.int64)
    uv = np.full((n, 2), np.nan)
    rgb = np.full((n, 3), np.nan)
    for ray in prange(n):
        index, hit = _cast(sc, bv, origins[ray].astype(np.float32), dirs[ray].astype(np.float32),
                           False, np.ones(3))
        if index < 0:
            continue
        tri[ray] = index
        uv[ray, 0], uv[ray, 1] = _interpolate(sc[4], sc[2], sc[0], index, hit.astype(np.float64))
        r, g, b = _shade_rgb(sc, index, hit, dirs[ray], classes[ray], True, True, True)
        rgb[ray, 0], rgb[ray, 1], rgb[ray, 2] = r, g, b
    return tri, uv, rgb


def probe(cs, bvh, origins, dirs, classes):
    tri, uv, rgb = _probe_rays(cs.arrays(), (bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order),
                               np.asarray(origins, np.float64), np.asarray(dirs, np.float64),
                               np.asarray(classes, np.int64))
    return {'tri': tri, 'uv': uv, 'radiance': rgb @ LUMA, 'radiance_rgb': rgb}


@njit(parallel=True, cache=True)
def _gather(sc, bv, origins32, dirs32, dirs64, cls, active, rgb,
            with_ambient, with_pages, with_emission, sums):
    for q in prange(len(origins32)):
        x, y, z, total = 0.0, 0.0, 0.0, 0.0
        if active[q]:
            for ray in range(origins32.shape[1]):
                tri, hit = _cast(sc, bv, origins32[q, ray], dirs32[q, ray], False, np.ones(3))
                if tri < 0:
                    continue
                r, g, b = _shade_rgb(sc, tri, hit, dirs64[q, ray], cls[q],
                                     with_ambient, with_pages, with_emission)
                if rgb:
                    x += r
                    y += g
                    z += b
                else:
                    lum = (r * LUMA[0] + g * LUMA[1]) + b * LUMA[2]
                    x += lum * dirs64[q, ray, 0]
                    y += lum * dirs64[q, ray, 1]
                    z += lum * dirs64[q, ray, 2]
                    total += lum
        sums[q, 0], sums[q, 1], sums[q, 2], sums[q, 3] = x, y, z, total


@njit(parallel=True, cache=True)
def _shadow(sc, bv, origins32, dirs32, limits, carried, escaped):
    for ray in prange(len(origins32)):
        throughput = np.ones(3)
        tri, _ = _cast(sc, bv, origins32[ray], dirs32[ray], True, throughput, limits[ray])
        carried[ray] = throughput
        escaped[ray] = tri < 0


class CpuPassError(RuntimeError):
    pass


def _shadow_batches(scene, job, sc, bv, usable=None, n=None, edge=False):
    prefix = 'edge' if edge else 'query'
    count = len(scene.get(f'{prefix}_pos', ()))
    chunk = max(1, CHUNK_RAYS)
    for k, sun in enumerate(job['suns']):
        one_job = dict(job, suns=[sun])
        visible = (sun_facing_queries(scene, [sun])[:, 0]
                   if not edge and 'query_inverse' in scene else None)
        for first in range(0, count, chunk):
            stop = first + chunk
            local = {f'{prefix}_{key}': scene[f'{prefix}_{key}'][first:stop]
                     for key in ('pos', 'normal', 'class')}
            if edge:
                local['edge_active'] = scene['edge_active'][first:stop, k:k + 1]
                query, _, origins, dirs = edge_rays(local, one_job)
                cosine = None
            else:
                active = usable[first:stop]
                ignore = scene.get('query_ignore_normals')
                if ignore is not None:
                    local['query_ignore_normals'] = ignore[first:stop]
                if visible is not None:
                    active = active & (visible[first:stop] | local.get('query_ignore_normals', False))
                query, _, origins, dirs, cosine = sun_rays(local, one_job, active, n[first:stop])
            carried, escaped = np.empty((len(origins), 3)), np.empty(len(origins), bool)
            _shadow(sc, bv, origins.astype(np.float32), dirs.astype(np.float32),
                    np.full(len(origins), RAY_LENGTH), carried, escaped)
            yield k, query + first, dirs, cosine, carried, escaped
            # otherwise the suspended generator overlaps this payload with the next batch
            del query, origins, dirs, cosine, carried, escaped


def gather_texels(scene, job, textures, pages):
    count = len(scene.get('texel_pos', ()))
    rgb = np.zeros((count, 3), np.float32)
    if not count:
        return rgb, 0, pages
    from .raster import flagged_transport, ordinary_transport, texel_contribution, texel_ray_counts

    usable, n, tangent, bitangent, spin = query_frames(scene['texel_normal'], job.get('seed', 0))
    pos = scene['texel_pos'].astype(np.float64)
    cls = scene['texel_class'].astype(np.int64)
    cs = cpu_scene(scene, job, textures, pages)
    bvh = build_bvh(cs.corners32)
    bv = (bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order)
    probe = dict(scene, query_pos=scene['texel_pos'], query_normal=scene['texel_normal'],
                 query_class=scene['texel_class'], query_ignore_normals=np.ones(count, bool))
    probe.pop('query_inverse', None)
    shadow_rays = 0
    for k, query, dirs, cosine, carried, escaped in _shadow_batches(probe, job, cs.arrays(), bv, usable, n):
        sun = job['suns'][k]
        rows = np.flatnonzero(escaped)
        rgb[query[rows]] += np.float64(sun['color']) * sun['strength'] / np.pi * carried[rows]
        shadow_rays += len(query)
        del query, dirs, cosine, carried, escaped, rows
    counts = texel_ray_counts(scene, job['pages'], max(1, round(job.get('cpu_samples', job['samples']) / 9)),
                              job.get('cpu_texel_samples', job['texel_samples']))
    ids = np.flatnonzero(usable)

    def reflection(base, flagged_pages, with_pages, with_emission):
        source = cpu_scene(scene, job, textures, base, flagged_pages)
        sc = source.arrays()
        delta = np.zeros((count, 3), np.float32)
        for rays in np.unique(counts[ids]):
            rows = ids[counts[ids] == rays]
            local, phi, radius = sphere(int(rays))
            chunk = max(1, CHUNK_RAYS // int(rays))
            for first in range(0, len(rows), chunk):
                chosen = rows[first:first + chunk]
                angle = phi[None, :] + spin[chosen, None]
                direction = (radius[None, :, None] * np.cos(angle)[:, :, None] * tangent[chosen, None, :]
                             + radius[None, :, None] * np.sin(angle)[:, :, None] * bitangent[chosen, None, :]
                             + local[None, :, 2, None] * n[chosen, None, :])
                origin = (pos[chosen, None, :] + RAY_EPSILON * direction).astype(np.float32)
                sums = np.empty((len(chosen), 4))
                _gather(sc, bv, origin, direction.astype(np.float32), direction, cls[chosen], usable[chosen],
                        True, False, with_pages, with_emission, sums)
                delta[chosen] = 4 / rays * sums[:, :3]
        return delta

    def flagged(values):
        return [flagged_transport(scene, i, p.shape[1], p.shape[0], values) for i, p in enumerate(pages)]

    # original emission reaches the flagged sources before they spend a reflected stage
    rgb += reflection(pages, None, False, True)
    casts = shadow_rays + int(counts[ids].sum())
    stable, sources = [ordinary_transport(scene, i, p) for i, p in enumerate(pages)], flagged(rgb)
    for _ in range(max(0, int(job['bounces']))):
        delta = reflection(stable, sources, True, False)
        rgb += delta
        casts += int(counts[ids].sum())
        stable, sources = [np.zeros_like(p) for p in pages], flagged(delta)
    installed = [p + texel_contribution(scene, i, p.shape[1], p.shape[0], rgb)[0] for i, p in enumerate(pages)]
    return rgb, casts, installed


def run_vertex_pass(scene: dict, job: dict, textures: list, pages: list) -> VertexPassResult:
    texel_rgb, texel_rays, pages = gather_texels(scene, job, textures, pages)
    rays = int(job.get('cpu_vertex_rays', job['vertex_rays']))
    usable, n, tangent, bitangent, spin = query_frames(scene['query_normal'], job.get('seed', 0))
    queries = len(n)
    pos = scene['query_pos'].astype(np.float64)
    cls = scene['query_class'].astype(np.int64)
    cs = cpu_scene(scene, job, textures, pages)
    bvh = build_bvh(cs.corners32)
    sc, bv = cs.arrays(), (bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order)
    ignore_normals = scene.get('query_ignore_normals', np.zeros(queries, bool))
    vector, irradiance = np.zeros((queries, 3)), np.zeros(queries)
    chunk = max(1, CHUNK_RAYS // rays)
    for flagged, sample, solid_angle in ((False, hemisphere, np.pi), (True, sphere, 4 * np.pi)):
        ids = np.flatnonzero(ignore_normals == flagged)
        if not len(ids):
            continue
        local, phi, radius = sample(rays)
        for first in range(0, len(ids), chunk):
            rows = ids[first:first + chunk]
            angle = phi[None, :] + spin[rows, None]
            dirs = (radius[None, :, None] * np.cos(angle)[:, :, None] * tangent[rows, None, :]
                    + radius[None, :, None] * np.sin(angle)[:, :, None] * bitangent[rows, None, :]
                    + local[None, :, 2, None] * n[rows, None, :])
            origins = (pos[rows, None, :] + RAY_EPSILON * dirs).astype(np.float32)
            sums = np.empty((len(rows), 4))
            _gather(sc, bv, origins, dirs.astype(np.float32), dirs, cls[rows], usable[rows],
                    False, True, True, True, sums)
            vector[rows] = solid_angle / rays * sums[:, :3]
            irradiance[rows] = solid_angle / rays * sums[:, 3]
    flat = np.pi * (cs.ambient[cls + 1] @ LUMA) * usable
    vector += flat[:, None] * n
    irradiance += flat
    suns = job['suns']
    sun_lit = np.zeros((queries, len(suns)), bool)
    sun_throughput = (np.zeros((queries, len(suns), 3), np.float32)
                      if ignore_normals.any() and job.get('ignore_normals_mode', 'tool') == 'tool' else None)
    shadow_rays = 0
    for k, sun_query, sun_dir, sun_cos, carried, escaped in _shadow_batches(scene, job, sc, bv, usable, n):
        sun = suns[k]
        rows = np.flatnonzero(escaped)
        strength = ((np.float64(sun['color']) * carried[rows]) @ LUMA) * float(sun['strength'])
        vector[sun_query[rows]] += (strength * sun_cos[rows])[:, None] * sun_dir[rows]
        irradiance[sun_query[rows]] += strength * sun_cos[rows]
        sun_lit[sun_query[rows], k] = True
        if sun_throughput is not None:
            sun_throughput[sun_query[rows], k] = carried[rows]
        shadow_rays += len(sun_query)
        del sun_query, sun_dir, sun_cos, carried, escaped, rows, strength
    edge_sun_lit = np.zeros((len(scene.get('edge_pos', ())), len(suns)), bool)
    for k, edge, dirs, cosine, carried, escaped in _shadow_batches(scene, job, sc, bv, edge=True):
        edge_sun_lit[edge, k] = escaped
        shadow_rays += len(edge)
        del edge, dirs, cosine, carried, escaped
    total = texel_rays + int(usable.sum()) * rays + shadow_rays
    return VertexPassResult(vector.astype(np.float32), irradiance.astype(np.float32), sun_lit, edge_sun_lit, total,
                          sun_throughput, texel_rgb)


def load_and_run(work: Path, pages: list) -> VertexPassResult:
    work = Path(work)
    job = json.loads((work / 'job.json').read_text())
    with np.load(work / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    with np.load(work / 'textures.npz') as data:
        textures = [data[f'tex_{k}'] for k in range(len(data.files))]
    return run_vertex_pass(scene, job, textures, pages)


def run_in_child(work: Path, pages: list[np.ndarray], timeout: float | None) -> tuple[VertexPassResult, dict]:
    work = Path(work).resolve()
    np.savez(work / 'diffuse.npz', **{f'diffuse_{i}': p for i, p in enumerate(pages)})
    out = work / 'cpu_out.npz'
    # a crashed child must not leave an earlier run looking successful
    out.unlink(missing_ok=True)
    try:
        done = subprocess.run(frozen.child_command(frozen.CPU_VERTEX_FLAG, str(work)), cwd=work,
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                              creationflags=frozen.NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise CpuPassError(f'vertex: timed out after {timeout} s') from None
    except OSError as error:
        raise CpuPassError(f'vertex: process died (exit unavailable): {error}') from error
    tail = '\n'.join(done.stderr.decode(errors='replace').splitlines()[-40:])
    died = f'vertex: process died (exit {done.returncode}): {tail}'
    if done.returncode:
        raise CpuPassError(died)
    try:
        with np.load(out, allow_pickle=False) as data:
            result = VertexPassResult(np.array(data['vector'], np.float32), np.array(data['irradiance'], np.float32),
                                      np.array(data['sun_lit'], bool), np.array(data['edge_sun_lit'], bool),
                                      int(data['rays']), np.array(data['vertex_sun_throughput'], np.float32)
                                       if 'vertex_sun_throughput' in data.files else None,
                                      np.array(data['texel_rgb'], np.float32))
            timing = {'compile': float(data['seconds_compile']), 'trace': float(data['seconds_trace']),
                      'threading_layer': str(data['threading_layer']), 'threads': int(data['threads'])}
        job = json.loads((work / 'job.json').read_text())
        with np.load(work / job['scene'], allow_pickle=False) as data:
            q = len(data['query_pos'])
            edges = len(data['edge_pos']) if 'edge_pos' in data else 0
            texels = len(data['texel_pos']) if 'texel_pos' in data else 0
            needs_throughput = (job.get('ignore_normals_mode', 'tool') == 'tool'
                                and 'query_ignore_normals' in data and data['query_ignore_normals'].any())
        suns = len(job['suns'])
        if (result.vector.shape != (q, 3) or result.irradiance.shape != (q,)
                or result.sun_lit.shape != (q, suns) or result.edge_sun_lit.shape != (edges, suns)
                or (result.sun_throughput is not None) != bool(needs_throughput)
                or (result.sun_throughput is not None and result.sun_throughput.shape != (q, suns, 3))
                or result.texel_rgb.shape != (texels, 3)):
            raise ValueError('short vertex arrays')
        return result, timing
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise CpuPassError(f'{died}\n{error}') from error


def warm_up():
    scene = {
        'positions': np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        'tris': np.int32([[0, 1, 2]]), 'tex_uv': np.zeros((3, 2), np.float32),
        'lm_uv': np.zeros((3, 2), np.float32), 'tri_twin': np.int32([-1]),
        'tri_class': np.int16([0]), 'tri_page': np.int16([0]), 'tri_texture': np.int32([-1]),
        'tri_casts_shadow': np.ones(1, bool), 'tri_cutout': np.zeros(1, bool),
        'tri_tint': np.ones((1, 3), np.float32), 'tri_emission': np.zeros((1, 3), np.float32),
        'occ_positions': np.empty((0, 3), np.float32), 'occ_tris': np.empty((0, 3), np.int32),
        'occ_uv': np.empty((0, 2), np.float32), 'occ_tri_texture': np.empty(0, np.int32),
        'occ_tri_casts_shadow': np.empty(0, bool), 'occ_tri_cutout': np.empty(0, bool),
        'occ_tri_emission': np.empty((0, 3), np.float32), 'occ_tri_tint': np.empty((0, 3), np.float32),
        'query_pos': np.float32([[.25, .25, 1]]), 'query_normal': np.float32([[0, 0, -1]]),
        'query_class': np.int16([0]),
    }
    job = {'vertex_rays': 1, 'suns': [], 'ambient': {}, 'albedo_scale': 1., 'albedo_fallback': .5}
    pages = [np.zeros((1, 1, 3), np.float32)]
    run_vertex_pass(scene, job, [], pages)
    cs = cpu_scene(scene, job, [], pages)
    bvh = build_bvh(cs.corners32)
    origins, dirs = scene['query_pos'], scene['query_normal']
    limits = np.full(1, RAY_LENGTH, np.float32)
    closest_hits(bvh, cs.corners32, origins, dirs, limits)
    brute_force_hits(cs.corners32, origins, dirs, limits)
    probe(cs, bvh, origins, dirs, np.zeros(1, np.int64))
    _probe_kernel(np.ones(1, np.float64))


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
    work = Path(argv[0])
    with np.load(work / 'diffuse.npz', allow_pickle=False) as data:
        pages = [data[f'diffuse_{i}'] for i in range(len(data.files))]
    started = perf_counter()
    warm_up()
    seconds_compile = perf_counter() - started
    started = perf_counter()
    result = load_and_run(work, pages)
    seconds_trace = perf_counter() - started
    throughput = {'vertex_sun_throughput': result.sun_throughput} if result.sun_throughput is not None else {}
    np.savez(work / 'cpu_out.npz', vector=result.vector, irradiance=result.irradiance,
             sun_lit=result.sun_lit, edge_sun_lit=result.edge_sun_lit, rays=np.int64(result.rays),
             texel_rgb=result.texel_rgb, **throughput, seconds_compile=np.float64(seconds_compile),
             seconds_trace=np.float64(seconds_trace), threading_layer=numba.threading_layer(),
             threads=np.int64(numba.get_num_threads()))
    return 0
