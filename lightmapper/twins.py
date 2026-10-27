"""Exact reversed triangle pairs shared by the CPU and GPU vertex passes, without Blender dependencies."""

import logging

import numpy as np


def reversed_twins(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    twin = np.full(len(corners), -1, np.int32)
    perm = np.tile(np.arange(3, dtype=np.int32), (len(corners), 1))
    groups = {}
    vertices = [tuple(map(tuple, triangle)) for triangle in np.asarray(corners).tolist()]
    for index, triangle in enumerate(vertices):
        key = tuple(sorted(triangle))
        groups.setdefault(key, []).append(index)
    ambiguous = 0
    for key, members in groups.items():
        if len(members) > 2:
            ambiguous += 1
            continue
        if len(members) != 2 or len(set(key)) != 3:
            continue
        a, b = members
        order = [vertices[a].index(corner) for corner in vertices[b]]
        inversions = sum(order[i] > order[j] for i in range(3) for j in range(i + 1, 3))
        if inversions % 2:
            twin[a], twin[b] = b, a
            # a target corner takes the weight of the source corner at the same position
            perm[a] = order
            perm[b] = np.argsort(order)
    if ambiguous:
        logging.getLogger(__name__).warning('%d position groups with more than two triangles left unpaired', ambiguous)
    return twin, perm, ambiguous
