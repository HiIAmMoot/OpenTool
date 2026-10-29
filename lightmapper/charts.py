"""Chart each source group with xatlas and pack the charts into power-of-two pages.

Ordinary groups are charted from their geometry. Groups whose shader asks for
simple parameterization keep their texture layout (islands scaled to world
density) and go through a second atlas, because xatlas cannot mix meshes and uv
meshes in one atlas; their chart rectangles share our final packing. Every stock cap is checked here: a layout over a cap is
split where a material can be, and refused otherwise, never truncated.
"""

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import xatlas_c
from .formats.bsp import OutMaterial, OutPage, SourceGroup
from .formats.vertices import ENV, LM
from .gather import SIMPLE_PARAMETERIZATION, ShaderRadiosity


logger = logging.getLogger(__name__)
# identical input bytes and settings can have incompatible layouts under a different packing algorithm
CACHE_VERSION = 11


def save_charts(path: Path, pages: list[OutPage], key: str):
    arrays = {'meta': json.dumps({'version': CACHE_VERSION, 'key': key,
                                  'pages': [[p.width, p.height, len(p.materials)] for p in pages]})}
    for i, page in enumerate(pages):
        for j, material in enumerate(page.materials):
            prefix = f'p{i}_m{j}_'
            for name, dtype in [('group', np.int32), ('env', ENV), ('lm', LM),
                                ('tris', np.uint16), ('surface_ids', np.int32)]:
                arrays[prefix + name] = np.asarray(getattr(material, name), dtype=dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # the old cache stays readable until a complete archive replaces it; close before replacing on Windows.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_charts(path: Path, key: str) -> list[OutPage] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['meta']))
            if meta['version'] != CACHE_VERSION or meta['key'] != key:
                return None
            pages = []
            for i, (width, height, count) in enumerate(meta['pages']):
                materials = []
                for j in range(count):
                    prefix = f'p{i}_m{j}_'
                    materials.append(OutMaterial(int(data[prefix + 'group']),
                        *(data[prefix + name].copy() for name in ('env', 'lm', 'tris', 'surface_ids'))))
                pages.append(OutPage(width, height, materials))
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        # an interrupted or damaged cache is disposable; gather and charts can rebuild it.
        return None
    return pages


class CapError(ValueError):
    pass


@dataclass
class ChartSettings:
    # the global texels_per_unit is the level 1 density; levels 0 (high) to 3 (turd) multiply it with
    # tool's final-quality start-edge ratios 2.4 / (0.9, 2.4, 4, 80)
    texels_per_unit: float = 2.0
    detail_level_scale: tuple = (2.667, 1.0, 0.6, 0.03)
    padding: int = 4
    page_max: int = 1024
    page_min: int = 8
    max_vertices: int = 64000
    max_surfaces: int = 20000
    max_materials_per_page: int = 2048
    max_pages: int = 127
    # xatlas lets a chart grow across a hard normal seam unless the seam weight tops 1000, and tool splits at every crease
    respect_normal_seams: bool = True

    def __post_init__(self):
        for name in ('page_max', 'page_min'):
            value = getattr(self, name)
            if value < 1 or value & (value - 1):
                raise ValueError(f'{name} must be a power of two, not {value}')
        if self.page_min > self.page_max:
            raise ValueError('page_min exceeds page_max')
        # (a settings file brings the list back as a list, which would not equal the default)
        self.detail_level_scale = tuple(float(v) for v in self.detail_level_scale)
        if len(self.detail_level_scale) != 4 or not all(v > 0 for v in self.detail_level_scale):
            raise ValueError(f'detail_level_scale needs four positive multipliers, not {self.detail_level_scale}')

    def density_scale(self, shader: ShaderRadiosity) -> float:
        if not 0 <= shader.detail_level < len(self.detail_level_scale):
            raise ValueError(f'radiosity detail level {shader.detail_level} is not 0 to 3')
        return self.detail_level_scale[shader.detail_level]


def next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def _in_triangle(px, py, a, b, c):
    d1 = (px - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (py - b[1])
    d2 = (px - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (py - c[1])
    d3 = (px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])
    return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))


def _island_overlaps(texels: np.ndarray, own: np.ndarray, tris: np.ndarray) -> bool:
    # one grid per island, sized to the island, so tiling uvs cost nothing extra
    low = np.floor(texels[own].min(axis=0)).astype(np.int64)
    size = np.ceil(texels[own].max(axis=0)).astype(np.int64) - low + 1
    grid = np.zeros((size[1], size[0]), np.uint8)
    for tri in tris:
        a, b, c = texels[tri[0]] - low, texels[tri[1]] - low, texels[tri[2]] - low
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(np.int64)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(np.int64)
        px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        inside = _in_triangle(px, py, a, b, c)
        cell = grid[y0:y1, x0:x1]
        cell[inside] = np.minimum(cell[inside] + 1, 2)
    return bool(grid.max() > 1)


def _simple_layout(group: SourceGroup, texels_per_unit: float, density: float):
    """Island uvs at world scale laid out in a row, or the reason the group cannot keep its layout."""
    uv = group.env['texcoord'].astype(np.float64)
    tris = group.tris
    # xatlas joins uv-mesh faces that share a uv value, not only a vertex index
    _, node = np.unique(uv, axis=0, return_inverse=True)
    node = node.reshape(-1)
    count = node.max() + 1
    edges = np.concatenate([node[tris[:, [0, 1]]], node[tris[:, [1, 2]]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    islands, component = connected_components(graph, directed=False)
    labels = component[node]
    a, b, c = (uv[tris[:, k]] for k in range(3))
    uv_area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    p, q, r = (group.env['position'][tris[:, k]].astype(np.float64) for k in range(3))
    world_area = 0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=1)
    if np.any((world_area > 1e-12) & (uv_area <= 1e-12)):
        return None, 'it has triangles with no texture area'
    tri_island = labels[tris[:, 0]]
    uv_sum = np.bincount(tri_island, uv_area, islands)
    world_sum = np.bincount(tri_island, world_area, islands)
    used = np.bincount(tri_island, minlength=islands) > 0
    if np.any(uv_sum[used] <= 0) or np.any(world_sum[used] <= 0):
        return None, 'it has islands with no texture or world area'
    scale = np.ones(islands)
    scale[used] = density * np.sqrt(world_sum[used] / uv_sum[used])
    scaled = uv * scale[labels][:, None]
    # (each island moves into a row of its own so no two share a uv value or overlap by accident)
    cursor = 0.0
    for island in np.flatnonzero(used):
        mine = labels == island
        low, high = scaled[mine].min(axis=0), scaled[mine].max(axis=0)
        scaled[mine] += [cursor - low[0], -low[1]]
        cursor += high[0] - low[0] + 1.0
        if _island_overlaps(scaled * texels_per_unit, mine, tris[tri_island == island]):
            return None, 'its texture layout overlaps'
    return scaled.astype(np.float32), None


def _split(vertex_count: int, tris: np.ndarray, s: ChartSettings):
    # greedy in triangle order; yields (triangle rows, local vertex ids) per material
    if vertex_count <= s.max_vertices and len(tris) <= s.max_surfaces:
        yield np.arange(len(tris)), np.unique(tris)
        return
    rows, seen = [], set()
    for t, tri in enumerate(tris):
        fresh = {int(v) for v in tri} - seen
        if rows and (len(seen) + len(fresh) > s.max_vertices or len(rows) + 1 > s.max_surfaces):
            yield np.array(rows), np.array(sorted(seen))
            rows, seen = [], set()
            fresh = {int(v) for v in tri}
        rows.append(t)
        seen |= fresh
    if rows:
        yield np.array(rows), np.array(sorted(seen))


def _run_atlas(entries, uv_mode: bool, s: ChartSettings, *, limit_charts=True, join_creases=False):
    # entries: (group index, group, scaled uvs or None, density multiplier); pieces keep xatlas texel uvs until page sizes are known
    atlas = xatlas_c.Atlas()
    # xatlas has one density for the whole atlas, so a group's own density is a scaled copy of its
    # positions (or world-scaled uvs). The scale is relative to the run's smallest multiplier,
    # which moves into the atlas density: no group shrinks, so a thin triangle never falls
    # under xatlas's zero-area cut
    lowest = min((d for _, _, _, d in entries), default=1.0)
    for _, group, uvs, density in entries:
        if uv_mode:
            atlas.add_uv_mesh(uvs / np.float32(lowest), group.tris)
        else:
            atlas.add_mesh(group.env['position'] * np.float32(density / lowest), group.tris, group.env['normal'])
    texels_per_unit = s.texels_per_unit * lowest
    # (xatlas starts a chart at texel 0 and lets it reach the last texel: _finish adds a margin
    # on both sides, so the atlas gets that much less)
    # an automatic-size reference cannot silently shrink a chart to page_max;
    # only the candidate is constrained to the requested page size
    atlas.generate(xatlas_c.PackSettings(resolution=s.page_max - 2 * s.padding if limit_charts else 0,
                                          texels_per_unit=texels_per_unit,
                                          padding=s.padding, bilinear=True, rotate_charts=True,
                                          brute_force=False,
                                          normal_seam_weight=1001.0 if s.respect_normal_seams and not join_creases else 4.0,
                                          # 64 texels keeps terrain pieces below the 128-page jump;
                                          # a boundary cap also stops long thin charts defeating the area cap
                                          max_chart_area=(64 / texels_per_unit) ** 2 if limit_charts and not uv_mode else 0,
                                          max_boundary_length=256 / texels_per_unit if limit_charts and not uv_mode else 0))
    page_count = max(atlas.page_count, 1)
    pages = [[] for _ in range(page_count)]
    extent = np.zeros((page_count, 2))
    for m, (index, group, _, _) in enumerate(entries):
        verts = atlas.vertices(m)
        local = atlas.indices(m)
        if len(local) != len(group.tris):
            raise RuntimeError('xatlas dropped triangles')
        unplaced = verts['atlas'][local] < 0
        if np.any(unplaced.any(axis=1) & ~unplaced.all(axis=1)):
            raise RuntimeError('xatlas left part of a triangle outside every page')
        # (xatlas charts no degenerate triangle: they get page 0 and uv 0, which covers no texel)
        tri_page = np.where(unplaced[:, 0], 0, verts['atlas'][local[:, 0]])
        labels = verts['chart'][local[:, 0]]
        order = np.argsort(labels, kind='stable')
        for rows in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1):
            page = tri_page[rows[0]]
            subsets = [rows] if uv_mode else _bounded_charts(verts['uv'], local, rows)
            for rows in subsets:
                tris = local[rows]
                for part, used in _split(len(verts), tris, s):
                    pick = verts[used]
                    extent[page] = np.maximum(extent[page], pick['uv'].max(axis=0))
                    pages[page].append((index, group, pick['xref'].astype(np.int64), pick['uv'].copy(),
                                        np.searchsorted(used, tris[part]).astype(np.int32), group.surface_ids[rows[part]]))
    return pages, extent


def _bounded_charts(uv, tris, rows):
    # xatlas's planar fast path bypasses its chart limits. Splitting its UV layout
    # at 120 texels leaves two four-texel margins in a 128 side, without rescaling faces
    pending = [rows]
    while pending:
        rows = pending.pop()
        corners = uv[tris[rows]]
        span = np.ptp(corners.reshape(-1, 2), axis=0)
        if span.max() <= 120 or len(rows) == 1:
            yield rows
            continue
        axis = int(np.argmax(span))
        order = np.argsort(corners[:, :, axis].mean(axis=1), kind='stable')
        middle = len(rows) // 2
        pending.extend((rows[order[middle:]], rows[order[:middle]]))


def _finish(pages, extent, s: ChartSettings) -> list[OutPage]:
    result = []
    for pieces, ext in zip(pages, extent):
        used = ext + 2 * s.padding
        w = min(max(next_pow2(math.ceil(used[0])), s.page_min), s.page_max)
        h = min(max(next_pow2(math.ceil(used[1])), s.page_min), s.page_max)
        assert used[0] <= w and used[1] <= h, 'a chart reaches into its page padding'
        by_group = {}
        for index, group, xref, uv, tris, surface_ids in pieces:
            lm = np.zeros(len(xref), LM)
            lm['texcoord'] = (uv + s.padding) / np.array([w, h], dtype=np.float32)
            by_group.setdefault(index, []).append((group.env[xref], lm, tris, surface_ids))
        materials = []
        for index, parts in by_group.items():
            offsets = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
            env = np.concatenate([p[0] for p in parts])
            lm = np.concatenate([p[1] for p in parts])
            # equal source vertices on separate charts need independent lightmap UVs
            tris = np.concatenate([p[2] + offset for p, offset in zip(parts, offsets)])
            surface_ids = np.concatenate([p[3] for p in parts])
            for rows, vertices in _split(len(env), tris, s):
                materials.append(OutMaterial(index, env[vertices], lm[vertices],
                    np.searchsorted(vertices, tris[rows]).astype(np.int32), surface_ids[rows]))
        result.append(OutPage(w, h, materials))
    return result


def _check_caps(pages, s):
    if len(pages) > s.max_pages:
        raise CapError(f'{len(pages)} lightmap pages needed, over max_pages ({s.max_pages}): '
                       'lower texels_per_unit or raise page_max')
    for page in pages:
        if len(page.materials) > s.max_materials_per_page:
            raise CapError(f'{len(page.materials)} materials on one page, over max_materials_per_page '
                           f'({s.max_materials_per_page}): lower texels_per_unit or raise page_max')


def _packing_areas(pages):
    groups, surfaces, areas = [], [], []
    for page in pages:
        for material in page.materials:
            uv = material.lm['texcoord'].astype(np.float64) * [page.width, page.height]
            a, b, c = (uv[material.tris[:, k]] for k in range(3))
            ab, ac = b - a, c - a
            areas.append(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]))
            groups.append(np.full(len(a), material.group))
            surfaces.append(material.surface_ids)
    order = np.lexsort((np.concatenate(surfaces), np.concatenate(groups)))
    return np.concatenate(areas)[order]


def _piece_areas(pieces):
    groups, surfaces, areas = [], [], []
    for group, _, _, uv, tris, surface_ids in pieces:
        corners = uv[tris].astype(np.float64)
        ab, ac = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        areas.append(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]))
        groups.append(np.full(len(tris), group))
        surfaces.append(surface_ids)
    order = np.lexsort((np.concatenate(surfaces), np.concatenate(groups)))
    return np.concatenate(areas)[order]


def _density_change(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        return math.inf, math.inf
    positive = reference > 1e-12
    if np.any(candidate[~positive] > 1e-6):
        return math.inf, math.inf
    median = float(np.median(np.abs(candidate[positive] / reference[positive] - 1))) if positive.any() else 0.0
    total = abs(float(candidate.sum() / reference.sum()) - 1) if reference.sum() > 1e-12 else 0.0
    return median, total


def _density_checked_pieces(entries, uv_mode, s, join_creases=False):
    reference = [piece for page in _run_atlas(entries, uv_mode, s, limit_charts=False, join_creases=join_creases)[0]
                 for piece in page]
    candidate = [piece for page in _run_atlas(entries, uv_mode, s, join_creases=join_creases)[0] for piece in page]
    median, total = _density_change(_piece_areas(reference), _piece_areas(candidate))
    # splitting can re-unfold a face, but cannot buy smaller pages by reducing
    # requested density: enforce both the typical face and the total area budget
    if median <= 0.01 and total <= 0.02:
        return candidate
    logger.info('rejecting chart candidate: median area change %.4f%%, total %.4f%%', median * 100, total * 100)
    return reference


class _FreeSpace:
    def __init__(self, capacity):
        self.base = next_pow2(max(1, capacity))
        self.tree = np.zeros(2 * self.base, np.int64)
        self.owners = [0] * (2 * self.base)

    def set(self, index, value, owner=1):
        index += self.base
        self.tree[index] = value
        self.owners[index] = owner if value else 0
        while index > 1:
            index //= 2
            value = max(self.tree[2 * index], self.tree[2 * index + 1])
            owners = self.owners[2 * index] | self.owners[2 * index + 1]
            if self.tree[index] == value and self.owners[index] == owners:
                break
            self.tree[index] = value
            self.owners[index] = owners

    def first(self, needed, allowed=None):
        def find(index):
            if self.tree[index] < needed or (allowed is not None and not self.owners[index] & allowed):
                return None
            if index >= self.base:
                return index - self.base
            left = find(2 * index)
            return left if left is not None else find(2 * index + 1)
        return find(1)

    def remaining(self, index):
        return int(self.tree[self.base + index])


def _piece_usage(piece, previous, s):
    batches, vertices, surfaces = previous
    if not batches:
        batches = 1
    if vertices + len(piece[2]) <= s.max_vertices and surfaces + len(piece[4]) <= s.max_surfaces:
        return batches, vertices + len(piece[2]), surfaces + len(piece[4])
    seen = set()
    for tri in piece[4]:
        fresh = set(map(int, tri)) - seen
        if surfaces and (vertices + len(fresh) > s.max_vertices or surfaces + 1 > s.max_surfaces):
            batches += 1
            vertices = surfaces = 0
            seen = set()
            fresh = set(map(int, tri))
        vertices += len(fresh)
        surfaces += 1
        seen.update(fresh)
    return batches, vertices, surfaces


def _shelf_pack(rectangles, order, width, height, transpose, s):
    row_space, page_space = _FreeSpace(len(rectangles)), _FreeSpace(s.max_pages)
    pages, extents, used_height, rows = [], [], [], []
    usage, material_counts = [], []
    for i in order:
        piece, uv, size = rectangles[i]
        w, h = size[::-1] if transpose else size
        w, h = max(1, w), max(1, h)
        if w > width or h > height:
            return None
        allowed = (1 << s.max_pages) - 1
        while True:
            row = row_space.first(w, allowed)
            page = rows[row][0] if row is not None else page_space.first(h, allowed)
            if page is None:
                break
            prior = usage[page].get(piece[0], (0, 0, 0))
            updated = _piece_usage(piece, prior, s)
            if material_counts[page] + updated[0] - prior[0] <= s.max_materials_per_page:
                break
            # a full draw-batch budget may still accept another chart of an existing group
            allowed &= ~(1 << page)
        if row is None:
            if page is None:
                if len(pages) >= s.max_pages:
                    return None
                page = len(pages)
                pages.append([])
                extents.append(np.zeros(2))
                used_height.append(0)
                usage.append({})
                material_counts.append(0)
            row = len(rows)
            rows.append((page, used_height[page]))
            used_height[page] += h
            row_space.set(row, width, 1 << page)
            page_space.set(page, height - used_height[page], 1 << page)
        page, y = rows[row]
        prior = usage[page].get(piece[0], (0, 0, 0))
        updated = _piece_usage(piece, prior, s)
        material_counts[page] += updated[0] - prior[0]
        if material_counts[page] > s.max_materials_per_page:
            return None
        usage[page][piece[0]] = updated
        free = row_space.remaining(row)
        moved = (uv[:, ::-1] if transpose else uv) + [width - free, y]
        pages[page].append((*piece[:3], moved, *piece[4:]))
        extents[page] = np.maximum(extents[page], moved.max(axis=0))
        row_space.set(row, free - w, 1 << page)
    return pages, extents


def _pack_rectangles(pieces, s):
    rectangles = []
    for piece in pieces:
        uv = piece[3]
        low = np.floor(uv.min(axis=0))
        size = np.ceil(uv.max(axis=0) - low).astype(int) + 2 * s.padding
        if np.any(size > s.page_max):
            raise CapError('a chart exceeds page_max including padding')
        rectangles.append((piece, uv - low, tuple(size)))
    # decreasing heights let every earlier shelf accept a later chart's height;
    # indexed first-fit fills those gaps without a quadratic scan of earlier rows
    best = None
    total_area = sum(w * h for _, _, (w, h) in rectangles)
    for transpose in (False, True):
        order = sorted(range(len(rectangles)), key=lambda i: (
            -rectangles[i][2][1 if not transpose else 0],
            -rectangles[i][2][0 if not transpose else 1], i))
        for width in (1 << k for k in range(s.page_min.bit_length() - 1, s.page_max.bit_length())):
            for height in (1 << k for k in range(s.page_min.bit_length() - 1, s.page_max.bit_length())):
                if width * height * s.max_pages < total_area:
                    continue
                packed = _shelf_pack(rectangles, order, width, height, transpose, s)
                if packed is not None:
                    pages, extents = packed
                    cost = sum(max(s.page_min, next_pow2(math.ceil(e[0] + 2 * s.padding))) *
                               max(s.page_min, next_pow2(math.ceil(e[1] + 2 * s.padding))) for e in extents)
                    # fewer textures and material splits outweigh extra texels;
                    # area only chooses between layouts with the same page count
                    if best is None or (len(pages), cost) < best[:2]:
                        best = (len(pages), cost, pages, extents)
    if best is None:
        raise CapError(f'packing exceeds max_pages ({s.max_pages}), page_max, or '
                       f'max_materials_per_page ({s.max_materials_per_page})')
    _, _, pages, extents = best
    return _finish(pages, extents, s)


def chart_and_pack(groups: list[SourceGroup], radiosity: list[ShaderRadiosity],
                   s: ChartSettings) -> list[OutPage]:
    if len(radiosity) != len(groups):
        raise ValueError('one shader radiosity per source group is required')
    if s.page_max - 2 * s.padding <= 0:
        raise ValueError('page_max must exceed twice the padding')
    ordinary, simple, fallback = [], [], []
    for index, (group, shader) in enumerate(zip(groups, radiosity)):
        if not len(group.tris):
            continue
        density = s.density_scale(shader)
        if shader.flags & SIMPLE_PARAMETERIZATION:
            uvs, reason = _simple_layout(group, s.texels_per_unit, density)
            if uvs is not None:
                simple.append((index, group, uvs, density))
                continue
            logger.warning('shader %s asks for simple parameterization but %s; charting it normally',
                           group.key[1], reason)
            fallback.append((index, group, None, density))
            continue
        ordinary.append((index, group, None, density))
    pieces = []
    if ordinary:
        pieces.extend(_density_checked_pieces(ordinary, False, s))
    if fallback:
        # (a separate atlas, because xatlas has one normal seam weight per generate call: tool lays these shaders
        # out from texture uvs that run on across creases, so their charts must not split there)
        pieces.extend(_density_checked_pieces(fallback, False, s, join_creases=True))
    if simple:
        pieces.extend(_density_checked_pieces(simple, True, s))
    if not pieces:
        return []
    pages = _pack_rectangles(pieces, s)
    _check_caps(pages, s)
    return pages
