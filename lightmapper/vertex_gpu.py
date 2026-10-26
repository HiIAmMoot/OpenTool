"""The per-vertex incident radiosity pass outside Blender: the hemisphere rays, query frames, ray sums and the sun and
edge ray lists that blender/bake.py's VertexPass builds, as float64 numpy so the GPU pass casts exactly what the CPU
pass does.

The constants are copies (bake.py cannot be imported outside Blender); a test parses bake.py and holds them equal.
"""

from dataclasses import dataclass

import numpy as np

from .sun_facing import sun_facing_queries


RAY_LENGTH, RAY_EPSILON, PASS_LIMIT = 1e5, 1e-3, 8
LUMA = np.float64([0.2126, 0.7152, 0.0722])
GOLDEN_FRACTION = 0.618034
SEED_SPIN = 0.7548776662466927


@dataclass
class VertexPassResult:
    vector: np.ndarray
    irradiance: np.ndarray
    # per query and sun, per edge probe and sun: the shadow ray reached the sun
    sun_lit: np.ndarray
    edge_sun_lit: np.ndarray
    rays: int


def radical_inverse(i):
    result, scale = 0.0, 0.5
    while i:
        result += scale * (i & 1)
        i >>= 1
        scale *= 0.5
    return result


def hemisphere(count):
    """Cosine-weighted directions about +z from a Hammersley set, with the polar radius and azimuth they came from."""
    i = np.arange(count)
    u1 = (i + 0.5) / count
    phi = 2 * np.pi * np.array([radical_inverse(int(k)) for k in i])
    r = np.sqrt(u1)
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], axis=1), phi, r


def query_frames(normals, seed):
    """(usable, n, tangent, bitangent, spin) per query; the spin turns every query's hemisphere by its own angle."""
    nrm = np.asarray(normals).astype(np.float64)
    length = np.linalg.norm(nrm, axis=1)
    # a zero or non-finite normal has no hemisphere to cast into: that query stays at zero
    usable = np.isfinite(length) & (length > 1e-12)
    n = np.where(usable[:, None], nrm / np.where(usable, length, 1.0)[:, None], [0.0, 0.0, 1.0])
    axis = np.where((np.abs(n[:, 0]) < 0.9)[:, None], [[1.0, 0, 0]], [[0, 1.0, 0]])
    tangent = np.cross(n, axis)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(n, tangent)
    ids = np.arange(len(n))
    spin = 2 * np.pi * ((ids * GOLDEN_FRACTION + seed * SEED_SPIN) % 1.0)
    return usable, n, tangent, bitangent, spin


def ambient_table(job, query_class, tri_class):
    """The class ambients indexed by class + 1: class -1 (indoor) has its own and must not wrap onto the last outdoor slot."""
    classes = [int(k) for k in job['ambient']] + [int(np.max(query_class, initial=0)), int(np.max(tri_class, initial=0))]
    table = np.zeros((max(classes) + 2, 3))
    for key, rgb in job['ambient'].items():
        table[int(key) + 1] = rgb
    return table


def gather_sums(lum, phi, r, h, n, tangent, bitangent, spin):
    """pi / rays x the sum of lum x direction and of lum, without building a direction per ray: the ray's azimuth is
    phi + spin, so its sine and cosine split into the spin-free sums over r cos phi and r sin phi."""
    rays = lum.shape[1]
    a, b, c = lum @ (r * np.cos(phi)), lum @ (r * np.sin(phi)), lum @ h
    cos_s, sin_s = np.cos(spin), np.sin(spin)
    vector = np.pi / rays * (((a * cos_s - b * sin_s)[:, None] * tangent)
                             + ((a * sin_s + b * cos_s)[:, None] * bitangent) + c[:, None] * n)
    return vector, np.pi / rays * lum.sum(axis=1)


def _unit(direction):
    d = np.float64(direction)
    return d / np.linalg.norm(d)


def sun_rays(scene, job, usable, n):
    """(query, sun, origin, direction, cosine) of every sun ray the CPU pass casts, ordered by sun then query."""
    pos = scene['query_pos'].astype(np.float64)
    visible = sun_facing_queries(scene, job['suns'])
    query, sun, cosine, direction = [], [], [], []
    for k, s in enumerate(job['suns']):
        d = _unit(s['direction'])
        c = n @ d
        rows = np.flatnonzero(usable & (c > 0) & np.isin(scene['query_class'], s['classes']) & visible[:, k])
        query.append(rows)
        sun.append(np.full(len(rows), k, np.int64))
        cosine.append(c[rows])
        direction.append(np.tile(d, (len(rows), 1)))
    query, sun = np.concatenate(query + [np.empty(0, np.int64)]), np.concatenate(sun + [np.empty(0, np.int64)])
    direction = np.concatenate(direction + [np.empty((0, 3))])
    return (query, sun, pos[query] + RAY_EPSILON * direction, direction,
            np.concatenate(cosine + [np.empty(0)]))


def edge_rays(scene, job):
    """(edge, sun, origin, direction) of every edge probe's sun ray the CPU pass casts, ordered by sun then edge."""
    edge_pos = scene.get('edge_pos', np.empty((0, 3), np.float32)).astype(np.float64)
    normals = scene.get('edge_normal', np.empty((0, 3), np.float32)).astype(np.float64)
    length = np.linalg.norm(normals, axis=1)
    usable = np.isfinite(length) & (length > 1e-12)
    normals = normals / np.where(usable, length, 1)[:, None]
    edge, sun, direction = [], [], []
    for k, s in enumerate(job['suns']):
        d = _unit(s['direction'])
        rows = np.flatnonzero(usable & (normals @ d > 0) & scene['edge_active'][:, k]
                              & np.isin(scene['edge_class'], s['classes']))
        edge.append(rows)
        sun.append(np.full(len(rows), k, np.int64))
        direction.append(np.tile(d, (len(rows), 1)))
    edge, sun = np.concatenate(edge + [np.empty(0, np.int64)]), np.concatenate(sun + [np.empty(0, np.int64)])
    direction = np.concatenate(direction + [np.empty((0, 3))])
    return edge, sun, edge_pos[edge] + RAY_EPSILON * direction, direction
