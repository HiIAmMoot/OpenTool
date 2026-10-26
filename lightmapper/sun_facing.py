"""Which vertex queries a sun reaches by the plane of a face around them, as blender/bake.py works it out.

bake.py runs inside Blender and cannot be imported from outside, so the two functions are copied here for the GPU
vertex pass; a test holds the copy equal to the original.
"""

import numpy as np


def sun_facing_planes(scene, suns):
    corners = scene['positions'][scene['tris']].astype(np.float64)
    plane = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    smooth = scene['normals'][scene['tris']].astype(np.float64).sum(axis=1)
    plane *= np.where((plane * smooth).sum(axis=1) < 0, -1, 1)[:, None]
    # (a sky light's samples all use its centre direction, so they share one receiver set)
    if not suns:
        return np.empty((len(plane), 0), bool)
    return np.stack([plane @ np.float64(sun.get('centre', sun['direction'])) > 0 for sun in suns], axis=1)


def sun_facing_queries(scene, suns):
    count = len(scene['query_pos'])
    if 'query_inverse' not in scene:
        # standalone point probes have no owning mesh faces; pipeline jobs always supply the relation
        return np.ones((count, len(suns)), bool)
    visible = np.zeros((count, len(suns)), bool)
    corners = scene['query_inverse'][scene['tris']]
    facing = sun_facing_planes(scene, suns)
    for n in range(len(suns)):
        # (assigning True through repeated indices is the same as an or over them)
        visible[corners.ravel()[np.repeat(facing[:, n], 3)], n] = True
    return visible
