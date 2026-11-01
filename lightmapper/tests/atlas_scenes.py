"""The B9 spike's scenes (one 96 x 96 page of four regions, and a sliver region), built for write_job."""

import math

import numpy as np

from lightmapper.blender_job import Scene
from lightmapper.gather import Sun


PAGE = 96
# chart cell in texels, texels per metre, quads per side of a surface
CELL = 32
DENSITY = 8.0
SUB = 3


def grid(origin, es, et, length, n=SUB, length_t=None, n_t=None):
    s = np.linspace(0, length, n + 1)
    t = np.linspace(0, length if length_t is None else length_t, (n if n_t is None else n_t) + 1)
    return (np.float64(origin)[None, None] + s[:, None, None] * np.float64(es)[None, None]
            + t[None, :, None] * np.float64(et)[None, None])


def chart(cell, theta_deg, frac, length, n=SUB, swap=False, length_t=None, n_t=None):
    cx = (cell[0] + 0.5) * CELL + frac[0]
    cy = (cell[1] + 0.5) * CELL + frac[1]
    s = np.linspace(-length / 2, length / 2, n + 1) * DENSITY
    lt = length if length_t is None else length_t
    t = np.linspace(-lt / 2, lt / 2, (n if n_t is None else n_t) + 1) * DENSITY
    a, b = np.meshgrid(s, t, indexing='ij')
    if swap:
        a, b = b, a
    t = math.radians(theta_deg)
    x = cx + math.cos(t) * a - math.sin(t) * b
    y = cy + math.sin(t) * a + math.cos(t) * b
    return np.stack([x / PAGE, y / PAGE], axis=-1)


def flat(x0, x1, y0, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def spike_scene(slivers=False):
    """(Scene, region of each receiver triangle, [Sun]); regions R1-R4, or R5 with slivers."""
    receivers, occluders = [], []

    def surface(origin, es, et, length, cell, theta, frac, region, cls=0, ignore=False, P=None):
        es, et = np.float64(es), np.float64(et)
        receivers.append(dict(P=grid(origin, es, et, length) if P is None else P, normal=np.cross(es, et),
                              lm=chart(cell, theta, frac, length), cls=cls, ignore=ignore, region=region))

    def occluder(corners, emission=(0, 0, 0), casts=True):
        occluders.append(dict(corners=np.float64(corners), emission=emission, casts=casts))

    if slivers:
        # triangles smaller than a sub-texel, and long slivers, with a sun shadow across them
        up = np.float64([0, 0, 1])
        receivers.append(dict(P=grid((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.6, 15, 3.0, 75), normal=up,
                              lm=chart((0, 0), 13.3, (0.21, 0.37), 0.6, 15, length_t=3.0, n_t=75),
                              cls=0, ignore=False, region='R5'))
        receivers.append(dict(P=grid((2, 0, 0), (1, 0, 0), (0, 1, 0), 0.6, 12, 3.0, 1), normal=up,
                              lm=chart((1, 0), -21.7, (-0.33, 0.12), 0.6, 12, length_t=3.0, n_t=1),
                              cls=0, ignore=False, region='R5'))
        receivers.append(dict(P=grid((4, 0, 0), (1, 0, 0), (0, 1, 0), 3.0, 1, 0.6, 12), normal=up,
                              lm=chart((2, 0), 37.9, (0.07, -0.29), 3.0, 1, length_t=0.6, n_t=12),
                              cls=0, ignore=False, region='R5'))
        occluder(flat(-1, 7, 1.2, 1.6, 1.0))
    else:
        # R1: a floor meeting a wall facing +x, a hovering panel throwing a shadow across the floor's edge
        surface((0, 0, 0), (1, 0, 0), (0, 1, 0), 3, (0, 0), 11.3, (0.37, -0.21), 'R1')
        surface((0, 0, 0), (0, 1, 0), (0, 0, 1), 3, (1, 0), -17.9, (-0.43, 0.29), 'R1')
        occluder(flat(1.5, 5.0, 1.0, 1.8, 1.5))
        # R2: flagged floors (both_sides in tool mode) lit from behind by an emitter below, a strip shading part
        surface((20, 0, 0), (1, 0, 0), (0, 1, 0), 3, (2, 0), 23.7, (0.19, 0.41), 'R2', ignore=True)
        surface((20, 3.4, 0), (1, 0, 0), (0, 1, 0), 3, (1, 2), -13.7, (0.29, -0.17), 'R2', ignore=True)
        occluder(flat(18, 25, -2, 9, -1.0), emission=(4, 4, 4), casts=False)
        occluder(flat(21, 25, -2, 1.2, -0.4))
        # R3: a twin pair, a vertical sheet with faces +y and -y, a floor below bouncing light onto the -y face
        shared = grid((40, 0, 0), (0, 0, 1), (1, 0, 0), 3)
        surface(None, (0, 0, 1), (1, 0, 0), 3, (0, 1), 7.1, (0.23, 0.33), 'R3', P=shared)
        surface(None, (1, 0, 0), (0, 0, 1), 3, (1, 1), -29.3, (-0.31, 0.17), 'R3', P=shared.transpose(1, 0, 2).copy())
        occluder(flat(38, 45, -4, 4, 0.0))
        # R4: classes 0 and 1 on one page, the sun linked to class 0 only, a wall shading 0 and bouncing onto 1
        surface((60, 0, 0), (1, 0, 0), (0, 1, 0), 3, (2, 1), 31.1, (0.11, -0.37), 'R4', cls=0)
        surface((63.5, 0, 0), (1, 0, 0), (0, 1, 0), 3, (0, 2), -5.3, (-0.27, 0.13), 'R4', cls=1)
        occluder([(63.25, -1, 0), (63.25, 4, 0), (63.25, 4, 2.5), (63.25, -1, 2.5)])

    cols = {k: [] for k in ('p', 'n', 'lm', 'tris', 'cls', 'ignore', 'region')}
    base = 0
    for r in receivers:
        P, lm = r['P'], r['lm']
        m, mt = P.shape[0], P.shape[1]
        cols['p'].append(P.reshape(-1, 3))
        cols['lm'].append(lm.reshape(-1, 2))
        cols['n'].append(np.repeat(r['normal'][None], m * mt, axis=0))
        for i in range(m - 1):
            for j in range(mt - 1):
                a, b, c, d = (base + i * mt + j, base + (i + 1) * mt + j, base + (i + 1) * mt + j + 1,
                              base + i * mt + j + 1)
                for tri in ((a, b, c), (a, c, d)):
                    cols['tris'].append(tri)
                    cols['cls'].append(r['cls'])
                    cols['ignore'].append(r['ignore'])
                    cols['region'].append(r['region'])
        base += m * mt
    p = np.concatenate(cols['p'])
    tris = np.int32(cols['tris'])
    n = len(tris)
    op, otris, oemit, ocasts, obase = [], [], [], [], 0
    for o in occluders:
        op.append(o['corners'])
        otris += [(obase, obase + 1, obase + 2), (obase, obase + 2, obase + 3)]
        oemit += [o['emission']] * 2
        ocasts += [o['casts']] * 2
        obase += 4
    no = len(otris)
    scene = Scene(
        positions=p.astype(np.float32), normals=np.concatenate(cols['n']).astype(np.float32),
        lm_uv=np.concatenate(cols['lm']).astype(np.float32), tex_uv=np.zeros((len(p), 2), np.float32),
        tris=tris, tri_page=np.zeros(n, np.int16), tri_class=np.int16(cols['cls']),
        tri_casts_shadow=np.ones(n, np.bool_), tri_emission=np.zeros((n, 3), np.float32),
        tri_texture=np.full(n, -1, np.int32), tri_cutout=np.zeros(n, np.bool_), tri_tint=np.ones((n, 3), np.float32),
        occ_positions=np.concatenate(op).astype(np.float32), occ_uv=np.zeros((obase, 2), np.float32),
        occ_tris=np.int32(otris), occ_tri_texture=np.full(no, -1, np.int32), occ_tri_cutout=np.zeros(no, np.bool_),
        occ_tri_casts_shadow=np.bool_(ocasts), occ_tri_emission=np.float32(oemit),
        occ_tri_tint=np.ones((no, 3), np.float32),
        query_pos=np.float32([[0, 0, 5]]), query_normal=np.float32([[0, 0, 1]]), query_class=np.int16([0]),
        tri_ignore_normals=np.bool_(cols['ignore']),
    )
    direction = np.float64([0.6, 0.3, 0.75])
    return scene, cols['region'], [Sun(direction / np.linalg.norm(direction), 0.01, np.float64([1, 1, 1]), 3.0, (0,))]
