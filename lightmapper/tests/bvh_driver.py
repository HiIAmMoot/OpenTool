"""Runs inside Blender to compare float32 soup rays with BVHTree."""

from pathlib import Path
import sys

import numpy as np
from mathutils.bvhtree import BVHTree


def main(work):
    with np.load(work / 'soup.npz') as data:
        corners, origins, dirs, limits = (data[key] for key in ('corners', 'origins', 'dirs', 'limits'))
    polygons = np.arange(len(corners) * 3).reshape(-1, 3)
    tree = BVHTree.FromPolygons(corners.reshape(-1, 3).tolist(), polygons.tolist(),
                                all_triangles=True, epsilon=0.0)
    index = np.full(len(origins), -1, np.int64)
    distance = np.full(len(origins), np.inf, np.float32)
    for i, (origin, direction, limit) in enumerate(zip(origins, dirs, limits)):
        _, _, hit, t = tree.ray_cast(origin.tolist(), direction.tolist(), float(limit))
        if hit is not None:
            index[i], distance[i] = hit, t
    np.savez(work / 'hits.npz', index=index, distance=distance)


if __name__ == '__main__':
    main(Path(sys.argv[sys.argv.index('--') + 1]))
