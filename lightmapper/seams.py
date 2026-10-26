import logging

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


log = logging.getLogger(__name__)


def _pair_forest(normals, uvs, page_ids):
    # a spanning forest enforces every qualifying pair's equality without a
    # quadratic list of constraints when many faces share one smooth edge
    values = np.column_stack((normals.reshape(-1, 6), uvs.reshape(-1, 4), page_ids)).astype(np.float64)
    scale = np.array([1000.] * 6 + [10000.] * 4 + [1.])
    nodes, leaf = [], np.empty(len(values), np.int64)

    def build(ids, parent=-1):
        index = len(nodes)
        low, high = values[ids].min(axis=0), values[ids].max(axis=0)
        node = {'low': low, 'high': high, 'count': len(ids), 'parent': parent}
        nodes.append(node)
        if len(ids) <= 16:
            node['ids'] = ids
            leaf[ids] = index
        else:
            axis = int(np.argmax((high - low) * scale))
            order = ids[np.argsort(values[ids, axis], kind='stable')]
            middle = len(order) // 2
            node['children'] = (build(order[:middle], index), build(order[middle:], index))
        return index

    def refresh(index):
        node = nodes[index]
        if 'ids' in node:
            ids = node['ids']
            node['count'] = len(ids)
            if len(ids):
                node['low'], node['high'] = values[ids].min(axis=0), values[ids].max(axis=0)
        else:
            children = [nodes[c] for c in node['children'] if nodes[c]['count']]
            node['count'] = sum(c['count'] for c in children)
            if children:
                node['low'] = np.minimum.reduce([c['low'] for c in children])
                node['high'] = np.maximum.reduce([c['high'] for c in children])

    def take(index, value, found):
        node = nodes[index]
        if not node['count']:
            return
        low, high = node['low'], node['high']
        if np.any(high[:6] < value[:6] - 1e-3) or np.any(low[:6] > value[:6] + 1e-3):
            return
        # continuous coincident UVs are not graph edges; pruning their whole
        # box avoids repeated scans of a same-page fan that needs no correction
        if (low[10] == high[10] == value[10]
                and np.all(low[6:10] >= value[6:10] - 1e-4)
                and np.all(high[6:10] <= value[6:10] + 1e-4)):
            return
        if 'ids' in node:
            ids = node['ids']
            smooth = np.max(np.abs(values[ids, :6] - value[:6]), axis=1) <= 1e-3
            different = (values[ids, 10] != value[10]) | (np.max(np.abs(values[ids, 6:10] - value[6:10]), axis=1) > 1e-4)
            chosen = smooth & different
            found.extend(ids[chosen].tolist())
            node['ids'] = ids[~chosen]
        else:
            for child in node['children']:
                take(child, value, found)
        refresh(index)

    build(np.arange(len(values)))
    seen = np.zeros(len(values), bool)
    for seed in range(len(values)):
        if seen[seed]:
            continue
        index = int(leaf[seed])
        nodes[index]['ids'] = nodes[index]['ids'][nodes[index]['ids'] != seed]
        while index >= 0:
            refresh(index)
            index = nodes[index]['parent']
        seen[seed] = True
        pending = [seed]
        while pending and nodes[0]['count']:
            source = pending.pop()
            found = []
            take(0, values[source], found)
            seen[found] = True
            pending.extend(found)
            for target in found:
                yield source, target


def seam_edges(pages, joined_groups=()):
    positions, normals, uvs, page_ids, footprints, group_ids = [], [], [], [], [], []
    for p, page in enumerate(pages):
        for material in page.materials:
            corners = material.env['position'][material.tris].astype(np.float64)
            normal = material.env['normal'][material.tris]
            uv = material.lm['texcoord'][material.tris].astype(np.float64) * [page.width, page.height]
            world_area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
            ab, ac = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
            texel_area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
            footprint = world_area / np.maximum(texel_area, 1e-12)
            for a, b in ((0, 1), (1, 2), (2, 0)):
                positions.append(corners[:, [a, b]])
                normals.append(normal[:, [a, b]])
                uvs.append(uv[:, [a, b]])
                page_ids.append(np.full(len(corners), p, np.int32))
                group_ids.append(np.full(len(corners), material.group, np.int64))
                footprints.append(footprint)
    if not positions:
        return
    positions, normals, uvs = map(np.concatenate, (positions, normals, uvs))
    page_ids, footprints, group_ids = map(np.concatenate, (page_ids, footprints, group_ids))
    # a joined group's creases are not seams of its lighting (tool lays it out from texture uvs that run on across them),
    # so a second set of normals gives its edges one stand-in per group: equal within the group, beyond any real normal
    # across groups. It only adds pairs: the real normals still qualify a pair as before
    joined = np.isin(group_ids, list(joined_groups))
    normals = normals.astype(np.float64)
    stand_in = normals.copy()
    stand_in[joined] = (group_ids[joined] * 4.0 + 8.0)[:, None, None]
    # quantised keys are sorted, not probe-limited float-bit hashes; grid-snapped
    # maps have the same O(T log T) grouping cost as jittered geometry
    _, nodes = np.unique(np.round(positions.reshape(-1, 3), 4), axis=0, return_inverse=True)
    nodes = nodes.reshape(-1, 2)
    reverse = nodes[:, 0] > nodes[:, 1]
    nodes[reverse] = nodes[reverse, ::-1]
    normals[reverse] = normals[reverse, ::-1]
    stand_in[reverse] = stand_in[reverse, ::-1]
    uvs[reverse] = uvs[reverse, ::-1]
    order = np.lexsort((nodes[:, 1], nodes[:, 0]))
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(nodes[order], axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    all_starts = starts
    starts = starts[counts == 2]
    left, right = order[starts], order[starts + 1]
    smooth = np.max(np.abs(normals[left] - normals[right]), axis=(1, 2)) <= 1e-3
    smooth |= joined[left] & joined[right] & (group_ids[left] == group_ids[right])
    different = (page_ids[left] != page_ids[right]) | (np.max(np.abs(uvs[left] - uvs[right]), axis=(1, 2)) > 1e-4)
    valid = smooth & different & (nodes[left, 0] != nodes[left, 1])
    for a, b in zip(left[valid], right[valid]):
        yield (page_ids[a], uvs[a], footprints[a]), (page_ids[b], uvs[b], footprints[b])
    for start, count in zip(all_starts, counts):
        if count <= 2:
            continue
        users = order[start:start + count]
        if nodes[users[0], 0] == nodes[users[0], 1]:
            continue
        for a, b in _pair_forest(normals[users], uvs[users], page_ids[users]):
            a, b = users[a], users[b]
            yield (page_ids[a], uvs[a], footprints[a]), (page_ids[b], uvs[b], footprints[b])
        # (a repeated constraint costs lsqr nothing)
        mine = users[joined[users]]
        if len(mine) > 1:
            for a, b in _pair_forest(stand_in[mine], uvs[mine], page_ids[mine]):
                a, b = mine[a], mine[b]
                yield (page_ids[a], uvs[a], footprints[a]), (page_ids[b], uvs[b], footprints[b])


def _stencil(page, uv, offset):
    xy = uv - 0.5
    low = np.floor(xy).astype(np.int64)
    fraction = xy - low
    ids, weights = [], []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x = np.clip(low[:, 0] + dx, 0, page.width - 1)
        y = np.clip(low[:, 1] + dy, 0, page.height - 1)
        ids.append(offset + y * page.width + x)
        weights.append((fraction[:, 0] if dx else 1 - fraction[:, 0]) *
                       (fraction[:, 1] if dy else 1 - fraction[:, 1]))
    return np.array(ids).T, np.array(weights).T


def stitch_seams(pages, images, joined_groups=()):
    offsets = np.r_[0, np.cumsum([p.width * p.height for p in pages])]
    columns, coefficients, footprints = [], [], []
    for left, right in seam_edges(pages, joined_groups):
        length = max(np.linalg.norm(edge[1][1] - edge[1][0]) for edge in (left, right))
        # two constraints per longest-side texel also cover mismatched densities
        t = np.linspace(0, 1, max(2, int(np.ceil(length * 2)) + 1))[:, None]
        ids, weights, areas = [], [], []
        for sign, (p, uv, footprint) in zip((1, -1), (left, right)):
            idx, weight = _stencil(pages[p], uv[0] * (1 - t) + uv[1] * t, offsets[p])
            ids.append(idx)
            weights.append(sign * weight)
            areas.append(np.full(idx.shape, max(footprint, 1e-12)))
        columns.append(np.concatenate(ids, axis=1))
        coefficients.append(np.concatenate(weights, axis=1))
        footprints.append(np.concatenate(areas, axis=1))
    if not columns:
        return
    columns, coefficients, footprints = map(np.concatenate, (columns, coefficients, footprints))
    active = coefficients != 0
    pixel, inverse = np.unique(columns[active], return_inverse=True)
    rows = np.broadcast_to(np.arange(len(columns))[:, None], columns.shape)[active]
    matrix = coo_matrix((coefficients[active], (rows, inverse)), shape=(len(columns), len(pixel))).tocsr()
    # larger world footprints carry more evidence; the minimum-norm correction
    # therefore moves a coarse texel less than a fine texel at a density boundary
    area = np.bincount(inverse, footprints[active] * np.abs(coefficients[active]))
    area /= np.maximum(np.bincount(inverse, np.abs(coefficients[active])), 1e-12)
    scale = 1 / np.sqrt(area)
    weighted = matrix.multiply(scale).tocsr()
    values = np.empty((len(pixel), 3), np.float64)
    owners = np.searchsorted(offsets[1:], pixel, side='right')
    for p in np.unique(owners):
        mine = owners == p
        values[mine] = images[p].reshape(-1, 3)[pixel[mine] - offsets[p]]
    for channel in range(3):
        correction = lsqr(weighted, -(matrix @ values[:, channel]), atol=1e-8, btol=1e-8, iter_lim=200)[0]
        values[:, channel] += correction * scale
    for p in np.unique(owners):
        mine = owners == p
        images[p].reshape(-1, 3)[pixel[mine] - offsets[p]] = values[mine]
    log.info('stitched %d seam constraints on %d texels; max residual %.6g',
             len(columns), len(pixel), np.max(np.abs(matrix @ values)))
